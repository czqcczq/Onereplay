"""Diagonal empirical Fisher for the EWC baseline: collection, storage, loading.

EWC protects old knowledge with the same shape of penalty OneReplay uses, but
with a different weighting matrix:

    OneReplay   sum_l tr(DeltaW_l C_l DeltaW_l^T)      C_l = E[x x^T]
    EWC         sum_l sum_ij F_l,ij (DeltaW_l,ij)^2    F_l = diagonal Fisher

Both are estimated once on the frozen base model over the same old-knowledge
rows and held fixed for the whole run, so the comparison isolates the weighting
and nothing else.

The estimator implemented here is the sequence-level empirical Fisher:

    F_i = (1/N) sum_n (d L_n / d W_i)^2
    L_n = -sum_{t in assistant} log p(y_t | x, y_<t)

Three details decide whether the result is the quantity EWC actually asks for:

  sequence sum   L_n sums over the answer's tokens instead of averaging them.
                 The Fisher is built from the score of log p(y|x), which is a
                 sum over positions; a 1/T_n factor would produce a
                 length-normalized variant that is defensible but different,
                 and papers that do not say which one they used are not
                 reproducible.
  assistant only prompt and padding positions carry label -100, so F measures
                 how sensitive the model's *responses* are. That is what the
                 IFEval and Multi-IF retention metrics probe.
  one example    Per-example gradients must be squared before they are summed.
    per backward (sum_n g_n)^2 is not sum_n g_n^2, so ordinary gradient
                 accumulation would silently compute a different matrix. This
                 module therefore expects batch_size=1 and one backward call
                 per example.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from onereplay.data.old_knowledge import example_to_model_text, example_to_prompt_text


def load_fisher_file(path: str) -> dict[str, torch.Tensor]:
    """Load F matrices from disk and return only the Fisher dictionary."""

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "fishers" in payload:
        return payload["fishers"]
    if isinstance(payload, dict) and "covariances" in payload:
        raise ValueError(
            f"{path} holds covariance matrices, not Fisher matrices. Pass a cov file "
            "to --cov_path with --regularizer onereplay instead."
        )
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported Fisher file format: {path}")


def move_fishers_to_device(
    fishers: dict[str, torch.Tensor],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Move all F matrices once before training starts.

    The penalty reads every target layer's F at every optimization step, so the
    transfer is paid once here rather than inside the loop.
    """

    return {name: fisher.to(device=device, dtype=dtype) for name, fisher in fishers.items()}


def save_fisher_payload(
    output_path: str,
    fishers: dict[str, torch.Tensor],
    counts: dict[str, int],
    metadata: dict[str, Any],
) -> None:
    """Save F matrices, example counts, and the full estimator definition.

    metadata is not decoration. Two Fisher files that differ only in whether the
    loss was summed or averaged over answer tokens produce visibly different
    lambda scales and no error message, so every checkpoint has to be able to
    say which estimator produced the F it trained against.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fishers": {key: value.cpu() for key, value in fishers.items()},
        "counts": counts,
        "metadata": metadata,
    }
    torch.save(payload, output_path)


def select_target_weights(
    model: nn.Module,
    target_module_names: list[str],
) -> dict[str, torch.Tensor]:
    """Freeze everything except the target layers' weights and return them.

    Turning requires_grad off elsewhere does not stop the target layers from
    receiving gradients: autograd builds its graph through the activations, so
    layer 0 still gets a gradient as long as some parameter downstream of the
    embedding needs one. What it does save is the gradient buffer for the other
    1.7B parameters, which is the bulk of the memory this stage would otherwise
    need.

    Tied weights (Qwen3's small checkpoints share embed_tokens with lm_head) are
    deduplicated by storage pointer, the same way snapshot_reference_weights
    does it, so a tied layer does not contribute its gradient twice.
    """

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    module_dict = dict(model.named_modules())
    target_weights: dict[str, torch.Tensor] = {}
    seen_storage: set[int] = set()
    for module_name in target_module_names:
        weight = module_dict[module_name].weight
        pointer = weight.data_ptr()
        if pointer in seen_storage:
            print(f"Fisher: skipping {module_name}, its weight is tied to an earlier layer")
            continue
        seen_storage.add(pointer)
        weight.requires_grad_(True)
        target_weights[module_name] = weight
    return target_weights


def common_prefix_length(left: list[int], right: list[int]) -> int:
    """Length of the shared head of two token id lists.

    The assistant mask is derived by rendering the row twice, once with the
    answer and once without, and masking the shared prefix. Comparing token ids
    rather than trusting len(prompt_ids) matters because a chat template may
    inject tokens into the generation prompt that the full render does not have
    in the same place (Qwen3's thinking block is the usual culprit). A literal
    prefix comparison degrades gracefully in that case instead of masking the
    wrong span.
    """

    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def build_supervised_collate_fn(tokenizer, args: argparse.Namespace):
    """Tokenize one old-knowledge row into input_ids plus assistant-only labels.

    input_ids come from the same example_to_model_text that collect_cov feeds
    the model, so the token sequence F is estimated on is the sequence C was
    estimated on. Only the labels are new.

    Masking happens before truncation, not after. The prompt is only a token
    prefix of the untruncated sequence: cutting on the left shifts the answer
    forward, and a prefix comparison against the shortened sequence would then
    match nothing and leave the surviving prompt tokens supervised. Building the
    labels first and slicing both arrays together is correct for either side.
    Slicing also reproduces the tokenizer's own truncation exactly, so input_ids
    stay identical to what collect_cov forwarded.

    Truncation side itself is left at whatever collect_cov used, which is the
    tokenizer default (right) unless --truncation_side says otherwise. Right
    truncation cuts the answer first, so a row long enough to overflow can end
    up with no supervised token at all; those rows contribute a zero gradient
    and are counted in the run's diagnostics rather than dropped, since dropping
    them would change which rows F saw relative to C.

    This collator requires batch_size=1: the Fisher needs one backward per
    example, so there is nothing to pad.
    """

    truncation_side = getattr(args, "truncation_side", "")
    if truncation_side:
        tokenizer.truncation_side = truncation_side
    add_special_tokens = args.use_chat_template != 1

    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
        if len(examples) != 1:
            raise ValueError(
                f"Fisher collection needs batch_size=1 (got {len(examples)}); squaring a "
                "batch-summed gradient would estimate a different matrix"
            )
        example = examples[0]
        full_ids = tokenizer(
            example_to_model_text(example, tokenizer, args),
            add_special_tokens=add_special_tokens,
        )["input_ids"]
        prompt_ids = tokenizer(
            example_to_prompt_text(example, tokenizer, args),
            add_special_tokens=add_special_tokens,
        )["input_ids"]

        prefix = common_prefix_length(full_ids, prompt_ids)
        labels = [-100] * prefix + list(full_ids[prefix:])

        if len(full_ids) > args.max_len:
            keep = slice(-args.max_len, None)
            if getattr(tokenizer, "truncation_side", "right") != "left":
                keep = slice(0, args.max_len)
            full_ids = full_ids[keep]
            labels = labels[keep]

        return {
            "input_ids": torch.tensor([full_ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(full_ids), dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "prompt_tokens": prefix,
            "prompt_matched": int(prefix == len(prompt_ids)),
        }

    return collate_fn


def sequence_sum_nll(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Assistant-only -sum_t log p(y_t | x, y_<t) for one example.

    reduction="sum" rather than HuggingFace's token mean. The score behind the
    Fisher is grad log p(y|x) = sum_t grad log p(y_t | ...), so a 1/T_n factor
    would make this a length-normalized empirical Fisher instead. Summing also
    means a row with no supervised token returns 0.0 rather than the nan that a
    mean over zero elements produces.

    Logits are widened to fp32 before the softmax for the same reason recent
    transformers versions do it internally: the log-sum-exp over a 150k
    vocabulary is where bf16 loses the most.
    """

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    supervised_tokens = int((shift_labels != -100).sum())
    return loss, supervised_tokens


class DiagonalFisherAccumulator:
    """Sum per-example squared gradients for the target layers.

    Accumulation happens in fp32 no matter what dtype the model runs in. A bf16
    running sum would be fatal here rather than merely imprecise: after a few
    hundred additions the sum is large enough that bf16's 8-bit mantissa rounds
    each new term away entirely, and the tail of a 20000-term sum would simply
    not land. Widening cannot undo rounding that already happened inside the
    backward pass, though, which is why the collection script defaults to
    --use_bf16 0.

    Per-example T_n and ||g_n||^2 are kept so the run can report how strongly an
    example's contribution actually grows with its answer length instead of
    assuming it.
    """

    def __init__(
        self,
        target_weights: dict[str, torch.Tensor],
        device: torch.device | str | None = None,
    ) -> None:
        self.target_weights = target_weights
        self.sums = {
            name: torch.zeros(
                weight.shape,
                dtype=torch.float32,
                device=device if device is not None else weight.device,
            )
            for name, weight in target_weights.items()
        }
        self.num_examples = 0
        self.supervised_tokens: list[int] = []
        self.grad_sq_norms: list[float] = []

    def add_example(self, supervised_tokens: int) -> float:
        """Fold the gradients currently sitting on the target weights into F."""

        total = 0.0
        for name, weight in self.target_weights.items():
            if weight.grad is None:
                continue
            squared = weight.grad.detach().float().pow(2)
            accumulator = self.sums[name]
            accumulator += squared.to(accumulator.device)
            total += float(squared.sum())
        self.num_examples += 1
        self.supervised_tokens.append(int(supervised_tokens))
        self.grad_sq_norms.append(total)
        return total

    def add_empty(self) -> None:
        """Count a row that had no supervised token.

        Its score is genuinely zero (there is no answer to be sensitive to), so
        it belongs in N. Skipping it instead would rescale F by a constant,
        which lambda absorbs anyway, but would also make the reported N differ
        from the number of rows C saw.
        """

        self.num_examples += 1
        self.supervised_tokens.append(0)
        self.grad_sq_norms.append(0.0)

    def finalize(self) -> dict[str, torch.Tensor]:
        divisor = max(self.num_examples, 1)
        return {name: (total / divisor).cpu() for name, total in self.sums.items()}

    def memory_bytes(self) -> int:
        return sum(total.numel() * total.element_size() for total in self.sums.values())


def length_weighting_report(
    supervised_tokens: list[int],
    grad_sq_norms: list[float],
) -> dict[str, float]:
    """Measure how an example's Fisher contribution scales with its answer length.

    The sequence-sum score g_n = sum_t g_nt makes ||g_n||^2 grow with the number
    of supervised tokens T_n, but the exponent is not knowable in advance:
    uncorrelated per-token gradients give ||g_n||^2 ~ T_n, perfectly aligned
    ones give T_n^2, and real corpora sit somewhere between. Fitting
    log ||g_n||^2 against log T_n reports where this corpus actually landed.

    effective_sample_size is the usual (sum w)^2 / sum w^2 with w_n = ||g_n||^2.
    It answers "how many of the N examples is F really resting on", which is the
    number that decides whether the estimate needs more data, and it is a
    measurement rather than the assumption that long answers must dominate.
    """

    weights = np.asarray(grad_sq_norms, dtype=np.float64)
    lengths = np.asarray(supervised_tokens, dtype=np.float64)
    total_weight = float(weights.sum())
    sum_squared = float((weights**2).sum())

    report = {
        "examples": float(weights.size),
        "zero_supervision_rows": float(int((lengths <= 0).sum())),
        "mean_supervised_tokens": float(lengths.mean()) if lengths.size else 0.0,
        "median_supervised_tokens": float(np.median(lengths)) if lengths.size else 0.0,
        "max_supervised_tokens": float(lengths.max()) if lengths.size else 0.0,
        "effective_sample_size": (total_weight**2 / sum_squared) if sum_squared > 0 else 0.0,
    }
    report["ess_ratio"] = report["effective_sample_size"] / max(report["examples"], 1.0)

    usable = (weights > 0) & (lengths > 0)
    if int(usable.sum()) >= 2:
        slope, _intercept = np.polyfit(np.log(lengths[usable]), np.log(weights[usable]), 1)
        report["length_exponent"] = float(slope)
    else:
        report["length_exponent"] = float("nan")
    return report


def fisher_summary(fishers: dict[str, torch.Tensor]) -> dict[str, float]:
    """Scale statistics used to pick a starting lambda.

    F is built from squared gradients and lands orders of magnitude below the
    covariance matrices, so the OneReplay lambda grid does not transfer. The
    mean magnitude gives the sweep a starting decade instead of a guess.
    """

    if not fishers:
        return {"layers": 0.0, "mean": 0.0, "max": 0.0}
    totals = [float(fisher.double().sum()) for fisher in fishers.values()]
    counts = [fisher.numel() for fisher in fishers.values()]
    maxima = [float(fisher.max()) for fisher in fishers.values()]
    return {
        "layers": float(len(fishers)),
        "mean": sum(totals) / max(sum(counts), 1),
        "max": max(maxima),
    }


