"""Covariance matrix load/save, identity ablation, and collection hooks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def load_covariance_file(path: str) -> dict[str, torch.Tensor]:
    """Load C matrices from disk and return only the covariance dictionary."""

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "covariances" in payload:
        return payload["covariances"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported covariance file format: {path}")


def to_identity_covariances(
    covariances: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Replace every C_l with an identity matrix of the same size.

    This is the key ablation control for OneReplay. It turns the penalty
    tr(DeltaW C DeltaW^T) into tr(DeltaW DeltaW^T) = ||DeltaW||_F^2, i.e. plain
    L2 shrinkage on the LoRA update with no old-knowledge structure. Comparing
    real C against this identity control isolates whether the benefit comes
    from the covariance directions or merely from shrinking DeltaW.

    Module-name keys are preserved so lookup_covariance still matches layers.

    One matrix is built per distinct (dim, dtype) and shared by every layer that
    asks for it. An identity is fully determined by those two numbers, so the
    per-layer copies the earlier version allocated were identical by
    construction: Qwen3-8B's seven projections span only 4096 and 12288, which is
    640 MiB of distinct content stored 252 times for 33.75 GiB. The control arm
    was paying the full covariance footprint to hold 2 matrices.
    """

    identity: dict[str, torch.Tensor] = {}
    by_shape: dict[tuple[int, torch.dtype], torch.Tensor] = {}
    for name, covariance in covariances.items():
        key = (int(covariance.shape[-1]), covariance.dtype)
        if key not in by_shape:
            by_shape[key] = torch.eye(key[0], dtype=key[1])
        identity[name] = by_shape[key]
    return identity


def move_covariances_to_device(
    covariances: dict[str, torch.Tensor],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Move all C matrices once before training starts.

    The OneReplay penalty reads every target layer's covariance at every
    optimization step. Keeping C on CPU would repeatedly copy large matrices to
    GPU inside the training loop, so this helper pays that transfer cost once.

    Entries that already share storage are moved once and keep sharing on the
    device. C describes a layer's input, and a block feeds q_proj, k_proj and
    v_proj from one layernorm output and gate_proj and up_proj from another, so
    three of the seven matrices per layer are copies -- 6.75 GiB of the 33.75 GiB
    that Qwen3-8B's full coverage costs. dedup_covariances.py makes that sharing
    explicit in the file; a per-key .to() would silently undo it here, since each
    call allocates its own destination.

    Keyed on data_ptr rather than id: torch.load rebuilds shared storage as
    distinct tensor objects, so identity does not survive a round trip through
    the file but the pointer does.
    """

    moved: dict[str, torch.Tensor] = {}
    on_device: dict[int, torch.Tensor] = {}
    for name, covariance in covariances.items():
        key = covariance.data_ptr()
        if key not in on_device:
            on_device[key] = covariance.to(device=device, dtype=dtype)
        moved[name] = on_device[key]
    return moved


def save_covariance_payload(
    output_path: str,
    covariances: dict[str, torch.Tensor],
    counts: dict[str, int],
    metadata: dict[str, Any],
) -> None:
    """Save C matrices, token counts, and collection settings together."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "covariances": {key: value.cpu() for key, value in covariances.items()},
        "counts": counts,
        "metadata": metadata,
    }
    torch.save(payload, output_path)


def make_covariance_hook(
    module_name: str,
    cov_sums: dict[str, torch.Tensor],
    counts: dict[str, int],
    token_mask_holder: dict[str, torch.Tensor | None],
    cov_normalization: str,
    cov_norm_eps: float,
    accum_on_device: bool = False,
):
    """Build a forward pre-hook that accumulates X^T X for one target layer.

    The hook sees the input hidden states of a Linear layer before W or LoRA is
    applied. This is exactly the x in DeltaW x.

    token_mask_holder["token_mask"] selects which positions of the batch enter
    the sum, and it is the single place the estimator's scope is decided. The
    caller puts the attention mask there to average over every non-padding
    token, or an assistant-only mask to restrict C to the answer span. The hook
    itself does not know which: it only drops the positions the mask zeroes, and
    counts what is left.

    With cov_normalization="none", the collected matrix is E[x x^T].

    With cov_normalization="base_output_norm", each token vector is replaced by:

        x' = x / max(||W x||_2, cov_norm_eps)

    and the collected matrix is E[x' x'^T]. This makes the later penalty
    measure relative output perturbation ||DeltaW x||^2 / ||W x||^2 instead of
    absolute output perturbation.

    accum_on_device keeps the running sums next to the activations instead of
    copying every batch's X^T X back to host memory. The default False path
    moves each X^T X to CPU, which costs one transfer per batch per layer:
    fine when C is a few tensors of a small hidden size, but it scales with the
    total size of C, not with the batch. Covering all seven projections of
    Qwen3-8B means 960 MiB per layer, 33.75 GiB per batch over PCIe, which
    dominates the forward pass by more than an order of magnitude. Accumulating
    on the GPU removes those transfers entirely at the cost of holding the full
    C in device memory for the duration of the run.
    """

    def hook(_module, inputs, output):
        hidden_states = inputs[0].detach()
        base_outputs = output.detach()
        if hidden_states.dim() == 2:
            flat_x = hidden_states
            flat_y = base_outputs
            flat_mask = None
        else:
            batch, seq_len, hidden_dim = hidden_states.shape
            flat_x = hidden_states.reshape(batch * seq_len, hidden_dim)
            flat_y = base_outputs.reshape(batch * seq_len, base_outputs.shape[-1])
            # "attention_mask" is this slot's pre-assistant_only name, still read
            # as a fallback: a caller that fills the old key and gets ignored
            # fails silently rather than loudly, because the shape guard below
            # would just leave flat_mask None and let padding into C.
            token_mask = token_mask_holder.get("token_mask")
            if token_mask is None:
                token_mask = token_mask_holder.get("attention_mask")
            flat_mask = None
            if token_mask is not None and token_mask.shape[:2] == (batch, seq_len):
                flat_mask = token_mask.reshape(batch * seq_len).bool().to(flat_x.device)

        if flat_mask is not None:
            flat_x = flat_x[flat_mask]
            flat_y = flat_y[flat_mask]
        if flat_x.numel() == 0:
            return

        flat_x = flat_x.float()
        if cov_normalization == "base_output_norm":
            # This is a forward hook, so output is already the frozen base
            # layer value W x. Reusing it avoids an extra large matrix multiply
            # for every target module.
            denom = flat_y.float().norm(dim=-1).clamp_min(float(cov_norm_eps)).unsqueeze(-1)
            flat_x = flat_x / denom

        xtx = flat_x.T @ flat_x
        if not accum_on_device:
            xtx = xtx.cpu()
        if module_name not in cov_sums:
            cov_sums[module_name] = xtx
            counts[module_name] = int(flat_x.shape[0])
        else:
            cov_sums[module_name] += xtx
            counts[module_name] += int(flat_x.shape[0])

    return hook


def register_covariance_hooks(
    model,
    target_module_names: list[str],
    token_mask_holder: dict[str, torch.Tensor | None],
    args: argparse.Namespace,
):
    """Attach hooks to every target Linear layer and return hook handles."""

    cov_sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []
    module_dict = dict(model.named_modules())
    accum_on_device = getattr(args, "cov_accum_device", "cpu") == "device"

    for module_name in target_module_names:
        module = module_dict[module_name]
        hook = make_covariance_hook(
            module_name,
            cov_sums,
            counts,
            token_mask_holder,
            args.cov_normalization,
            args.cov_norm_eps,
            accum_on_device=accum_on_device,
        )
        handles.append(module.register_forward_hook(hook))

    return cov_sums, counts, handles
