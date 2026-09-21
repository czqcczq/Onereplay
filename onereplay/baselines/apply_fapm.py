"""Apply FAPM to a finished LoRA run and write the pruned model as a checkpoint.

FAPM (Forgetting-Aware Pruning, "Mitigating Catastrophic Forgetting in Large
Language Models with Forgetting-aware Pruning") is a *post-hoc* arm: it does not
change training at all. It takes the task vector of a finished fine-tune, keeps
only the entries that score highest under

    S = |dW| - mean(|W0|) * (|dW| / |W0|)

and throws the rest away, so the model that gets evaluated is W0 plus a sparse
version of what fine-tuning did. The score is |dW| discounted by how large the
update is *relative* to the pre-trained weight it sits on, which is the paper's
notion of "this coordinate was cheap to move, so moving it was probably the
model forgetting rather than learning".

The authors' code is kept as an unmodified clone under baseline/FAPM; the part
that constitutes the method is these five lines of their FAPM.py:

    tensor = weights_round1_lr1e_5[key] - weights[key]
    k = int(tensor.numel() * 0.1)
    lamda = 1.0 * weights[key].abs().mean()
    t = tensor.abs() - lamda*(tensor.abs() / weights[key].abs())
    indices = torch.argsort(t.view(-1), descending=True)[:k]

All five are reproduced below, including the two that look like accidents and
are not: ``int()`` truncates the budget rather than rounding it, and the top-k
is taken unconditionally, so exactly k coordinates survive per tensor even when
their score is negative. lamda is per-tensor and its coefficient is fixed at
1.0, which leaves ``--keep_ratio`` as the arm's single knob -- the same shape of
hyperparameter as OSFT's unfreeze rank ratio.

Four departures, all forced by this line being LoRA rather than full
fine-tuning, and all belonging in the paper's table note:

* The task vector only exists on the LoRA target modules. Their setting is full
  fine-tuning, where every matrix has a dW; here embed_tokens, the norms and
  lm_head have dW == 0 by construction, so FAPM's scope is decided by
  --target_modules before FAPM sees anything. The update is therefore doubly
  constrained: low-rank first, sparse second.
* dW is read from PEFT via ``LoraLayer.get_delta_weight`` rather than
  reconstructed as ``(alpha/r) * B @ A``. Same quantity, but the scaling
  convention (plain LoRA vs rsLoRA vs DoRA) is then whatever the adapter's own
  config says instead of whatever we assumed, and getting that wrong is the kind
  of error that produces a plausible number rather than a crash.
* Their script writes ``torch.save(weights, "pytorch_model.bin")``. A pruned
  task vector is no longer low-rank, so it cannot go back into a rank-r adapter;
  the output has to be a full checkpoint, and it is written with
  save_pretrained so eval/runner.py loads it through its full-checkpoint branch.
* Coordinates where W0 is exactly zero make |dW|/|W0| diverge. Upstream's
  arithmetic already handles the common half of that on its own: |dW| - inf is
  -inf, so such a coordinate sinks to the bottom of the sort and is never
  selected. What it does not handle is dW == 0 on the same coordinate, where
  0 - lamda * (0/0) is nan and argsort(descending=True) ranks nan *first* --
  upstream then spends that many slots of the budget on coordinates whose
  update is zero and keeps correspondingly fewer real ones. Taking the limit
  (S = -inf whenever W0 == 0) agrees with upstream everywhere except there and
  spends the whole budget on coordinates that actually move. It takes W0 == 0
  to trigger at all, which a trained checkpoint crossed with a LoRA task vector
  is unlikely to produce, so the manifest reports the count rather than leaving
  it to be assumed.

Usage:
    python -m onereplay.baselines.apply_fapm \
        --model_dir <models> --model_name Qwen3-8B \
        --adapter_path <results>/adapters/ds_vanilla_8b_... \
        --keep_ratio 0.1 \
        --save_path <results>/fapm_ckpt/ds_fapm_8b_...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.core.modeling import (  # noqa: E402
    is_lora_adapter_dir,
    join_model_path,
    load_causal_lm_and_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune a LoRA task vector with FAPM and save a full checkpoint."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="The vanilla LoRA run's output directory. FAPM prunes its task "
        "vector, so this must be the arm FAPM is meant to repair -- normally "
        "the unprotected run every other arm is also compared against.",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.1,
        help="Fraction of each matrix's task-vector entries to keep. The "
        "authors' value is 0.1. This is the arm's only hyperparameter; 1.0 "
        "reproduces the merged adapter and 0.0 reproduces the base model, "
        "which is what the two end-point self-checks assert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Where to write the full checkpoint. Required unless --save 0.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=1,
        help="0 computes the pruning and writes only the manifest. An 8B "
        "checkpoint is ~16GB, so this is how you look at what a keep_ratio "
        "does to the task vector without paying for the disk.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help="Defaults to <save_path>/fapm_manifest.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Where the per-matrix arithmetic runs; the model itself stays on "
        "CPU either way and only one matrix is on the accelerator at a time. "
        "auto picks cuda when it is available -- the top-k over a 50M-entry "
        "MLP matrix is the slow part and it is an order of magnitude faster "
        "there.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def fapm_prune(
    base_weight: torch.Tensor,
    delta: torch.Tensor,
    keep_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Zero every task-vector entry outside the top ``keep_ratio`` by FAPM score.

    Both tensors must already be float32 and on the same device. Returns the
    pruned task vector and the per-matrix numbers that go into the manifest.
    """

    magnitude = base_weight.abs()
    lamda = 1.0 * magnitude.mean()
    delta_magnitude = delta.abs()
    score = delta_magnitude - lamda * (delta_magnitude / magnitude)
    # An infinite discount is already -inf when dW != 0; this only changes the
    # dW == 0 corner, where upstream's nan would sort to the front and eat the
    # budget. See the module docstring.
    zero_base = magnitude == 0
    score = torch.where(zero_base, torch.full_like(score, float("-inf")), score)

    numel = delta.numel()
    budget = int(numel * keep_ratio)  # int(), not round(): upstream truncates.
    budget = max(0, min(budget, numel))

    if budget == numel:
        # Selecting everything is a full sort of a 50M-entry matrix for an
        # answer already known. This is the keep_ratio 1.0 self-check's path.
        pruned = delta.clone()
    else:
        pruned = torch.zeros_like(delta)
        if budget > 0:
            indices = torch.topk(score.view(-1), budget, sorted=False).indices
            flat_delta = delta.view(-1)
            pruned.view(-1).scatter_(0, indices, flat_delta.gather(0, indices))

    energy = float(delta.pow(2).sum())
    stats = {
        "numel": numel,
        "kept": budget,
        "zero_base": int(zero_base.sum()),
        "kept_energy": float(pruned.pow(2).sum() / energy) if energy > 0 else 0.0,
        "delta_norm": energy**0.5,
    }
    return pruned, stats


def single_adapter_name(peft_model) -> str:
    """The one adapter we are allowed to prune, or a hard error."""

    active = peft_model.active_adapters
    names = list(active) if isinstance(active, (list, tuple)) else [str(active)]
    if len(names) != 1:
        raise ValueError(
            f"expected exactly one active adapter, found {names}. FAPM prunes a "
            "single task vector; with several adapters attached it is ambiguous "
            "which one that is."
        )
    return names[0]


def main() -> None:
    args = parse_args()
    if args.save == 1 and not args.save_path:
        raise SystemExit("--save_path is required unless --save 0")
    if not 0.0 <= args.keep_ratio <= 1.0:
        raise SystemExit(f"--keep_ratio must be in [0, 1], got {args.keep_ratio}")
    if not is_lora_adapter_dir(args.adapter_path):
        raise SystemExit(
            f"{args.adapter_path} has no adapter_config.json, so it is not a LoRA "
            "run. This script reads the task vector off an adapter; a full "
            "fine-tuning checkpoint would need it computed against the base "
            "weights instead."
        )

    from peft import PeftModel
    from peft.tuners.lora import LoraLayer

    device = resolve_device(args.device)
    base_path = join_model_path(args.model_dir, args.model_name)
    print(f"base     : {base_path}")
    print(f"adapter  : {args.adapter_path}")
    print(f"keep     : {args.keep_ratio}")
    print(f"device   : {device} (per-matrix arithmetic; model stays on CPU)")

    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model_dir, args.model_name, args.use_bf16
    )
    peft_model = PeftModel.from_pretrained(model, args.adapter_path)
    adapter_name = single_adapter_name(peft_model)

    # Never merge. base_layer.weight is still W0 at this point, which is exactly
    # what the score needs, and get_delta_weight hands us dW without touching it.
    layers = [
        (name, module)
        for name, module in peft_model.named_modules()
        if isinstance(module, LoraLayer)
    ]
    if not layers:
        raise SystemExit(
            f"no LoRA layers found in {args.adapter_path}; nothing to prune."
        )
    print(f"layers   : {len(layers)} LoRA matrices, adapter '{adapter_name}'")

    started = time.time()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, (name, module) in enumerate(layers, start=1):
            weight = module.base_layer.weight
            base32 = weight.detach().to(device=device, dtype=torch.float32)
            delta32 = (
                module.get_delta_weight(adapter_name)
                .detach()
                .to(device=device, dtype=torch.float32)
            )
            if delta32.shape != base32.shape:
                raise SystemExit(
                    f"{name}: task vector {tuple(delta32.shape)} does not match "
                    f"the base weight {tuple(base32.shape)}."
                )
            pruned, stats = fapm_prune(base32, delta32, args.keep_ratio)
            weight.copy_((base32 + pruned).to(dtype=weight.dtype, device=weight.device))

            records.append({"layer": name, **stats})
            if index % 25 == 0 or index == len(layers):
                print(
                    f"  [{index}/{len(layers)}] {name} "
                    f"kept {stats['kept']}/{stats['numel']} "
                    f"energy {stats['kept_energy']:.4f}",
                    flush=True,
                )
            del base32, delta32, pruned

    elapsed = time.time() - started

    total_numel = sum(record["numel"] for record in records)
    total_kept = sum(record["kept"] for record in records)
    total_zero_base = sum(record["zero_base"] for record in records)
    # Energy is summed over squared norms, so the ratio is the share of the task
    # vector's squared magnitude that survived -- the number that says whether
    # a 10% budget threw away 90% of the update or 5% of it.
    total_energy = sum(record["delta_norm"] ** 2 for record in records)
    kept_energy = sum(
        record["kept_energy"] * record["delta_norm"] ** 2 for record in records
    )

    print()
    print(f"pruned {len(records)} matrices in {elapsed:.0f}s")
    print(f"  entries kept : {total_kept:,} / {total_numel:,}")
    print(
        "  energy kept  : "
        f"{(kept_energy / total_energy if total_energy > 0 else 0.0):.4f} "
        "of ||dW||^2"
    )
    print(f"  W0 == 0      : {total_zero_base:,} entries, never selected")

    manifest = {
        "method": "FAPM",
        "upstream": "baseline/FAPM/FAPM.py",
        "base_model": base_path,
        "adapter_path": str(args.adapter_path),
        "adapter_name": adapter_name,
        "keep_ratio": args.keep_ratio,
        "lamda_coefficient": 1.0,
        "zero_base_policy": "score=-inf (never selected)",
        "save_path": str(args.save_path) if args.save == 1 else "",
        "totals": {
            "matrices": len(records),
            "numel": total_numel,
            "kept": total_kept,
            "kept_fraction": total_kept / total_numel if total_numel else 0.0,
            "kept_energy": kept_energy / total_energy if total_energy > 0 else 0.0,
            "zero_base": total_zero_base,
            "seconds": elapsed,
        },
        "layers": records,
    }

    manifest_path = args.manifest_path
    if args.save == 1:
        unload = getattr(peft_model, "unload", None) or peft_model.base_model.unload
        merged = unload()
        directory = Path(args.save_path)
        directory.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(directory))
        # eval/runner.py reads the tokenizer out of the checkpoint directory for
        # full checkpoints, while the LoRA arms read it off the base model. This
        # is that same base tokenizer, so FAPM and the vanilla arm it is compared
        # against are tokenized identically.
        tokenizer.save_pretrained(str(directory))
        print(f"wrote checkpoint to {directory}")
        manifest_path = manifest_path or str(directory / "fapm_manifest.json")
    else:
        print("--save 0: checkpoint not written.")
        if not manifest_path:
            raise SystemExit("--save 0 needs --manifest_path")

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
