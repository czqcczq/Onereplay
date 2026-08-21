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
    """

    identity: dict[str, torch.Tensor] = {}
    for name, covariance in covariances.items():
        dim = covariance.shape[-1]
        identity[name] = torch.eye(dim, dtype=covariance.dtype)
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
    """

    return {
        name: covariance.to(device=device, dtype=dtype)
        for name, covariance in covariances.items()
    }


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
    attention_holder: dict[str, torch.Tensor | None],
    cov_normalization: str,
    cov_norm_eps: float,
):
    """Build a forward pre-hook that accumulates X^T X for one target layer.

    The hook sees the input hidden states of a Linear layer before W or LoRA is
    applied. This is exactly the x in DeltaW x.

    With cov_normalization="none", the collected matrix is E[x x^T].

    With cov_normalization="base_output_norm", each token vector is replaced by:

        x' = x / max(||W x||_2, cov_norm_eps)

    and the collected matrix is E[x' x'^T]. This makes the later penalty
    measure relative output perturbation ||DeltaW x||^2 / ||W x||^2 instead of
    absolute output perturbation.
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
            attention_mask = attention_holder.get("attention_mask")
            flat_mask = None
            if attention_mask is not None and attention_mask.shape[:2] == (batch, seq_len):
                flat_mask = attention_mask.reshape(batch * seq_len).bool().to(flat_x.device)

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
        if module_name not in cov_sums:
            cov_sums[module_name] = xtx.cpu()
            counts[module_name] = int(flat_x.shape[0])
        else:
            cov_sums[module_name] += xtx.cpu()
            counts[module_name] += int(flat_x.shape[0])

    return hook


def register_covariance_hooks(
    model,
    target_module_names: list[str],
    attention_holder: dict[str, torch.Tensor | None],
    args: argparse.Namespace,
):
    """Attach hooks to every target Linear layer and return hook handles."""

    cov_sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []
    module_dict = dict(model.named_modules())

    for module_name in target_module_names:
        module = module_dict[module_name]
        hook = make_covariance_hook(
            module_name,
            cov_sums,
            counts,
            attention_holder,
            args.cov_normalization,
            args.cov_norm_eps,
        )
        handles.append(module.register_forward_hook(hook))

    return cov_sums, counts, handles
