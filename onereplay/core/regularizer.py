"""OneReplay covariance regularizer (framework-agnostic).

The math used by OneReplay is one formula regardless of how the model is
adapted:

    regularizer = E_x ||DeltaW x||_2^2 = tr(DeltaW C DeltaW^T)

where C = E_x[x x^T] is estimated once from old-knowledge data. Only the
source of DeltaW differs between the two training modes:

  LoRA          y = W x + scale * B A x, so DeltaW = scale * B A and the trace
                collapses to scale^2 * tr((B^T B)(A C A^T)), which touches only
                rank x rank matrices.
  full finetune DeltaW = W - W0 against a frozen snapshot of the initial
                weights, evaluated as sum((DeltaW C) * DeltaW).

Both paths compute the same quantity; the LoRA form is an algebraic shortcut
that exploits the low-rank factorization, not a different penalty.

If C is collected with x' = x / ||W x||, the same training code instead
computes a relative-error penalty:

    E_x ||DeltaW x||_2^2 / ||W x||_2^2
"""

from __future__ import annotations

import torch
from torch import nn


def get_lora_weight_matrices(module: nn.Module, adapter_name: str = "default"):
    """Return A, B, and scale from one PEFT LoRA-wrapped linear module.

    PEFT stores A and B inside ModuleDicts. For a linear layer:
      A: [rank, d_in]
      B: [d_out, rank]
      scale: lora_alpha / rank
    """

    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        return None
    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
        return None

    lora_a = module.lora_A[adapter_name].weight
    lora_b = module.lora_B[adapter_name].weight

    if hasattr(module, "scaling") and adapter_name in module.scaling:
        scale = float(module.scaling[adapter_name])
    else:
        rank = lora_a.shape[0]
        alpha = module.lora_alpha[adapter_name]
        scale = float(alpha) / float(rank)
    return lora_a, lora_b, scale


def strip_peft_prefix(module_name: str) -> str:
    """Convert PEFT module names back to base-model style names.

    Example:
      base_model.model.model.layers.0.self_attn.q_proj
    becomes:
      model.layers.0.self_attn.q_proj
    """

    prefixes = (
        "base_model.model.",
        "base_model.",
    )
    for prefix in prefixes:
        if module_name.startswith(prefix):
            return module_name[len(prefix) :]
    return module_name


def lookup_covariance(
    covariances: dict[str, torch.Tensor],
    lora_module_name: str,
) -> tuple[str, torch.Tensor] | tuple[None, None]:
    """Find the covariance matrix that corresponds to one LoRA module.

    The exact module name can differ before and after PEFT wraps the model.
    We first try the stripped exact name, then suffix matching.
    """

    base_name = strip_peft_prefix(lora_module_name)
    if base_name in covariances:
        return base_name, covariances[base_name]

    matches = [key for key in covariances if base_name.endswith(key) or key.endswith(base_name)]
    if len(matches) == 1:
        key = matches[0]
        return key, covariances[key]
    return None, None


def lora_covariance_regularizer(
    model: nn.Module,
    covariances: dict[str, torch.Tensor],
    adapter_name: str = "default",
    normalize_by_layers: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute sum_l tr(DeltaW_l C_l DeltaW_l^T) for all LoRA layers.

    We avoid explicitly building the full DeltaW = scale * B @ A. Instead:

        tr((scale * B A) C (scale * B A)^T)
        = scale^2 * tr((B^T B) (A C A^T))

    The matrices inside the trace are only rank x rank, so this is much cheaper
    than multiplying a full d_out x d_in DeltaW.
    """

    total_reg: torch.Tensor | None = None
    used_layers = 0
    missing_layers: list[str] = []

    for module_name, module in model.named_modules():
        weights = get_lora_weight_matrices(module, adapter_name=adapter_name)
        if weights is None:
            continue

        lora_a, lora_b, scale = weights
        cov_key, covariance = lookup_covariance(covariances, module_name)
        if covariance is None:
            missing_layers.append(module_name)
            continue

        covariance = covariance.to(device=lora_a.device, dtype=torch.float32)
        a = lora_a.float()
        b = lora_b.float()

        # A C A^T measures how much the LoRA input projection responds to old
        # hidden-state directions. B^T B then weights that response by the LoRA
        # output projection.
        a_c_a_t = a @ covariance @ a.T
        b_t_b = b.T @ b
        layer_reg = (scale**2) * torch.sum(b_t_b * a_c_a_t.T)

        total_reg = layer_reg if total_reg is None else total_reg + layer_reg
        used_layers += 1

    if total_reg is None:
        device = next(model.parameters()).device
        total_reg = torch.zeros((), device=device, dtype=torch.float32)

    if normalize_by_layers and used_layers > 0:
        total_reg = total_reg / used_layers

    stats = {
        "used_layers": float(used_layers),
        "missing_layers": float(len(missing_layers)),
    }
    return total_reg, stats


def full_covariance_regularizer(
    model: nn.Module,
    covariances: dict[str, torch.Tensor],
    reference_weights: dict[str, torch.Tensor],
    normalize_by_layers: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute sum_l tr(DeltaW_l C_l DeltaW_l^T) for full fine-tuning.

    DeltaW = W - W0 against the frozen snapshot in reference_weights. Without a
    low-rank factorization to exploit, the trace is evaluated directly:

        tr(DeltaW C DeltaW^T) = sum((DeltaW @ C) * DeltaW)

    Only layers present in reference_weights are visited, so the snapshot
    doubles as the definition of which layers the penalty covers.
    """

    total_reg: torch.Tensor | None = None
    used_layers = 0
    missing_layers: list[str] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        reference = reference_weights.get(module_name)
        if reference is None:
            continue

        _, covariance = lookup_covariance(covariances, module_name)
        if covariance is None:
            missing_layers.append(module_name)
            continue

        weight = module.weight
        covariance = covariance.to(device=weight.device, dtype=torch.float32)
        delta = weight.float() - reference.to(device=weight.device, dtype=torch.float32)
        layer_reg = torch.sum((delta @ covariance) * delta)

        total_reg = layer_reg if total_reg is None else total_reg + layer_reg
        used_layers += 1

    if total_reg is None:
        device = next(model.parameters()).device
        total_reg = torch.zeros((), device=device, dtype=torch.float32)

    if normalize_by_layers and used_layers > 0:
        total_reg = total_reg / used_layers

    stats = {
        "used_layers": float(used_layers),
        "missing_layers": float(len(missing_layers)),
    }
    return total_reg, stats


def resolve_full_layers(
    model: nn.Module,
    covariances: dict[str, torch.Tensor],
    reference_weights: dict[str, torch.Tensor],
) -> tuple[list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]], list[str]]:
    """Pair every regularized Linear with its C and its frozen W0, once.

    full_covariance_regularizer re-walks named_modules() and re-runs the suffix
    matching in lookup_covariance on every call. That is a few milliseconds of
    Python per optimizer step, invisible next to the fp32 matmuls it precedes but
    not next to the analytic path, which is an order of magnitude cheaper. The
    pairing cannot change during a run -- C is frozen, W0 is frozen, and the
    module tree is fixed once the model is built -- so it is resolved once here
    and reused.
    """

    layers: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]] = []
    missing: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        reference = reference_weights.get(module_name)
        if reference is None:
            continue
        _, covariance = lookup_covariance(covariances, module_name)
        if covariance is None:
            missing.append(module_name)
            continue
        weight = module.weight
        layers.append(
            (
                module_name,
                weight,
                covariance.to(device=weight.device, dtype=torch.float32),
                reference.to(device=weight.device, dtype=torch.float32),
            )
        )
    return layers, missing


@torch.no_grad()
def full_covariance_grad_(
    layers: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]],
    scale: float,
    allow_tf32: bool = True,
    compute_dtype: torch.dtype = torch.float32,
) -> float:
    """Add scale * dR/dW straight into .grad and return the un-normalized R.

    R = sum_l tr(DeltaW_l C_l DeltaW_l^T) has an analytic gradient,

        dR/dDeltaW = DeltaW (C + C^T) = 2 DeltaW C    for symmetric C,

    and DeltaW C is exactly the product the forward pass already forms. Letting
    autograd rediscover it costs a second matmul of the same size, and keeps every
    layer's fp32 DeltaW alive in the graph until backward returns -- on Qwen3-1.7B
    that is 6.9 GB of temporaries for the 197 covered layers. Writing the gradient
    directly halves the arithmetic and lets each layer's temporaries die
    immediately, so the peak is one layer instead of all of them.

    R reads only weights, and weights are frozen across a gradient-accumulation
    window, so injecting once per window accumulates the same gradient as adding
    lambda*R to any single micro-batch's loss. The caller injects at the start of
    the window, matching where the autograd path adds it; see accumulate_grad for
    why that position is not free in bf16.

    C is stored fp32 and is not modified; allow_tf32 only lets the matmul round its
    inputs to 11 mantissa bits inside the tensor core. That is a ~5e-4 relative
    perturbation of C against a 7x throughput difference on H100, and it sits an
    order of magnitude below the 5e-3 penalty-ratio gap that two different
    sampling strategies of the *same* replay corpus already produce.
    """

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    total = 0.0
    try:
        for _, weight, covariance, reference in layers:
            delta = weight.to(compute_dtype) - reference.to(compute_dtype)
            delta_c = delta @ covariance.to(compute_dtype)
            total += float((delta_c * delta).sum())
            if scale != 0.0:
                if weight.grad is None:
                    weight.grad = torch.zeros_like(weight)
                weight.grad.add_(delta_c.to(weight.grad.dtype), alpha=2.0 * scale)
            del delta, delta_c
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return total


def _check_fisher_shape(module_name: str, fisher: torch.Tensor, delta: torch.Tensor) -> None:
    """Fail loudly when a covariance file was handed to the EWC path.

    Both files are dicts of tensors keyed by module name, so passing one where
    the other is expected costs nothing at load time and only shows up as a
    shape error deep inside a matmul, or worse, as a silently different penalty
    on a square layer. C is d_in x d_in and F is d_out x d_in, which separates
    them for every layer where the projection is not square.
    """

    if fisher.shape != delta.shape:
        raise ValueError(
            f"Fisher for {module_name} has shape {tuple(fisher.shape)} but DeltaW is "
            f"{tuple(delta.shape)}. A covariance file is d_in x d_in while a Fisher file "
            "is d_out x d_in, so check that --fisher_path is not pointing at a cov_*.pt."
        )


def lora_fisher_regularizer(
    model: nn.Module,
    fishers: dict[str, torch.Tensor],
    adapter_name: str = "default",
    normalize_by_layers: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute sum_l sum_ij F_l,ij (DeltaW_l,ij)^2 for all LoRA layers.

    Unlike the covariance penalty, this one does not collapse into rank x rank
    matrices. tr(DeltaW C DeltaW^T) contracts C against the row space of DeltaW,
    which the low-rank factorization can exploit; an elementwise weighting pairs
    every entry of F with a distinct entry of DeltaW, so DeltaW = scale * B A has
    to be materialized. That costs O(d_out * d_in * rank) per layer, which is the
    same order as the A C A^T contraction on the covariance path, so the two
    baselines stay comparable in per-step time.

    Penalizing the synthesized update rather than A and B separately is
    deliberate: the entries of DeltaW depend on the product of the two factors,
    so per-factor Fisher matrices would measure importance in coordinates that
    have no meaning on their own.
    """

    total_reg: torch.Tensor | None = None
    used_layers = 0
    missing_layers: list[str] = []

    for module_name, module in model.named_modules():
        weights = get_lora_weight_matrices(module, adapter_name=adapter_name)
        if weights is None:
            continue

        lora_a, lora_b, scale = weights
        _, fisher = lookup_covariance(fishers, module_name)
        if fisher is None:
            missing_layers.append(module_name)
            continue

        delta = scale * (lora_b.float() @ lora_a.float())
        fisher = fisher.to(device=delta.device, dtype=torch.float32)
        _check_fisher_shape(module_name, fisher, delta)
        layer_reg = torch.sum(fisher * delta * delta)

        total_reg = layer_reg if total_reg is None else total_reg + layer_reg
        used_layers += 1

    if total_reg is None:
        device = next(model.parameters()).device
        total_reg = torch.zeros((), device=device, dtype=torch.float32)

    if normalize_by_layers and used_layers > 0:
        total_reg = total_reg / used_layers

    stats = {
        "used_layers": float(used_layers),
        "missing_layers": float(len(missing_layers)),
    }
    return total_reg, stats


def full_fisher_regularizer(
    model: nn.Module,
    fishers: dict[str, torch.Tensor],
    reference_weights: dict[str, torch.Tensor],
    normalize_by_layers: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute sum_l sum_ij F_l,ij (W - W0)_ij^2 for full fine-tuning.

    Same penalty as the LoRA path with DeltaW read from a frozen W0 snapshot
    instead of the adapter factors, mirroring how full_covariance_regularizer
    relates to lora_covariance_regularizer.
    """

    total_reg: torch.Tensor | None = None
    used_layers = 0
    missing_layers: list[str] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        reference = reference_weights.get(module_name)
        if reference is None:
            continue

        _, fisher = lookup_covariance(fishers, module_name)
        if fisher is None:
            missing_layers.append(module_name)
            continue

        weight = module.weight
        delta = weight.float() - reference.to(device=weight.device, dtype=torch.float32)
        fisher = fisher.to(device=weight.device, dtype=torch.float32)
        _check_fisher_shape(module_name, fisher, delta)
        layer_reg = torch.sum(fisher * delta * delta)

        total_reg = layer_reg if total_reg is None else total_reg + layer_reg
        used_layers += 1

    if total_reg is None:
        device = next(model.parameters()).device
        total_reg = torch.zeros((), device=device, dtype=torch.float32)

    if normalize_by_layers and used_layers > 0:
        total_reg = total_reg / used_layers

    stats = {
        "used_layers": float(used_layers),
        "missing_layers": float(len(missing_layers)),
    }
    return total_reg, stats


class ReplayRegularizer:
    """Hold C on device and compute the OneReplay penalty each step.

    A thin wrapper so trainers and future verl integrations can call a single
    object without reloading C. When reference_weights is set the full
    fine-tuning path is used, otherwise the LoRA path. The underlying pure
    functions are unchanged.
    """

    def __init__(
        self,
        covariances: dict[str, torch.Tensor],
        adapter_name: str = "default",
        normalize_by_layers: bool = True,
        reference_weights: dict[str, torch.Tensor] | None = None,
        reg_impl: str = "autograd",
        allow_tf32: bool = True,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.covariances = covariances
        self.adapter_name = adapter_name
        self.normalize_by_layers = normalize_by_layers
        self.reference_weights = reference_weights
        # "autograd" builds the penalty into the loss graph and lets backward
        # derive dR/dW; "analytic" writes 2*lambda*DeltaW C into .grad itself.
        # Both are the same penalty. The flag exists so the equivalence check can
        # flip one argument instead of one git version, the same reason
        # --reg_once_per_update kept its 0 branch.
        self.reg_impl = reg_impl
        self.allow_tf32 = allow_tf32
        self.compute_dtype = compute_dtype
        self._layers: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor]] | None = None
        self._missing: list[str] = []

    @classmethod
    def from_path(
        cls,
        path: str,
        device: torch.device | str,
        identity: bool = False,
        normalize_by_layers: bool = True,
        adapter_name: str = "default",
        dtype: torch.dtype = torch.float32,
        reg_impl: str = "autograd",
        allow_tf32: bool = True,
        compute_dtype: torch.dtype = torch.float32,
    ) -> "ReplayRegularizer":
        from onereplay.core.covariance import (
            load_covariance_file,
            move_covariances_to_device,
            to_identity_covariances,
        )

        covariances = load_covariance_file(path)
        if identity:
            covariances = to_identity_covariances(covariances)
        covariances = move_covariances_to_device(covariances, device=device, dtype=dtype)
        return cls(
            covariances=covariances,
            adapter_name=adapter_name,
            normalize_by_layers=normalize_by_layers,
            reg_impl=reg_impl,
            allow_tf32=allow_tf32,
            compute_dtype=compute_dtype,
        )

    def set_reference_weights(self, reference_weights: dict[str, torch.Tensor]) -> None:
        """Switch to the full fine-tuning path using a frozen W0 snapshot."""

        self.reference_weights = reference_weights
        self._layers = None

    @property
    def injects_grad(self) -> bool:
        """Whether the trainer must call accumulate_grad instead of adding to the loss.

        Only the full fine-tuning path is covered. The LoRA path already collapses
        to rank x rank matrices and costs ~11 ms per update, 1.6% of a step, so
        there is nothing there worth a second code path.
        """

        return self.reg_impl == "analytic" and self.reference_weights is not None

    def accumulate_grad(self, model: nn.Module, replay_lambda: float) -> tuple[float, dict]:
        """Add lambda * dR/dW into .grad and return the reported R.

        Call this once per optimizer step, at the *start* of the accumulation
        window, before the first micro-batch's backward. Anywhere in the window
        gives the same DeltaW, but .grad is bf16 under full fine-tuning and only an
        empty buffer preserves a term running at ~1% of the task gradient. That is
        also where the autograd path adds it, which keeps the two comparable.

        The returned value is normalized the same way the autograd path normalizes
        it, so the logged replay_reg stays comparable across implementations.
        """

        if self._layers is None:
            self._layers, self._missing = resolve_full_layers(
                model, self.covariances, self.reference_weights or {}
            )
            print(
                f"analytic regularizer resolved {len(self._layers)} layers; "
                f"no penalty matrix for {len(self._missing)} layers; "
                f"tf32={int(self.allow_tf32)} dtype={self.compute_dtype}"
            )

        used = len(self._layers)
        divisor = used if (self.normalize_by_layers and used > 0) else 1
        total = full_covariance_grad_(
            self._layers,
            scale=replay_lambda / divisor,
            allow_tf32=self.allow_tf32,
            compute_dtype=self.compute_dtype,
        )
        stats = {"used_layers": float(used), "missing_layers": float(len(self._missing))}
        return total / divisor, stats

    def layer_keys(self) -> dict[str, torch.Tensor]:
        """Per-layer matrices, used as the definition of the penalty's coverage.

        snapshot_reference_weights only needs a dict keyed by module name to
        decide which layers to freeze W0 for, so both regularizers expose their
        matrices under one name and the caller stays agnostic.
        """

        return self.covariances

    def memory_bytes(self) -> int:
        """Bytes the covariance matrices occupy wherever they were moved.

        This is OneReplay's main fixed memory overhead over vanilla LoRA: one
        d_in x d_in matrix per target layer, resident for the whole run.
        """

        return sum(
            covariance.numel() * covariance.element_size()
            for covariance in self.covariances.values()
        )

    def reference_memory_bytes(self) -> int:
        """Bytes the W0 snapshot occupies; zero on the LoRA path.

        Full fine-tuning needs DeltaW = W - W0, so the initial weights of every
        regularized layer stay resident. LoRA reads DeltaW straight from B and
        A and pays nothing here.
        """

        if not self.reference_weights:
            return 0
        return sum(
            reference.numel() * reference.element_size()
            for reference in self.reference_weights.values()
        )

    def __call__(self, model: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
        if self.reference_weights is not None:
            return full_covariance_regularizer(
                model,
                self.covariances,
                self.reference_weights,
                normalize_by_layers=self.normalize_by_layers,
            )
        return lora_covariance_regularizer(
            model,
            self.covariances,
            adapter_name=self.adapter_name,
            normalize_by_layers=self.normalize_by_layers,
        )


class EWCRegularizer:
    """Hold F on device and compute the EWC penalty each step.

    The regularization baseline OneReplay is measured against. Same interface as
    ReplayRegularizer, so the trainer, the metrics schema and the lambda
    plumbing are shared and the only thing that changes between the two runs is
    which matrix weights DeltaW.

    F is estimated once on the frozen pre-trained weights and never updated.
    That is not a simplification of the method: EWC accumulates Fisher matrices
    across *task boundaries*, and this setting has exactly one, from the
    instruction-tuned base model to the new task. Keeping F fixed also keeps the
    comparison against C one-dimensional, since C is likewise estimated once and
    frozen; an F that adapted during training would confound "Fisher versus
    input covariance" with "adaptive versus static".
    """

    def __init__(
        self,
        fishers: dict[str, torch.Tensor],
        adapter_name: str = "default",
        normalize_by_layers: bool = True,
        reference_weights: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.fishers = fishers
        self.adapter_name = adapter_name
        self.normalize_by_layers = normalize_by_layers
        self.reference_weights = reference_weights

    @classmethod
    def from_path(
        cls,
        path: str,
        device: torch.device | str,
        normalize_by_layers: bool = True,
        adapter_name: str = "default",
        dtype: torch.dtype = torch.float32,
    ) -> "EWCRegularizer":
        from onereplay.core.fisher import load_fisher_file, move_fishers_to_device

        fishers = load_fisher_file(path)
        fishers = move_fishers_to_device(fishers, device=device, dtype=dtype)
        return cls(
            fishers=fishers,
            adapter_name=adapter_name,
            normalize_by_layers=normalize_by_layers,
        )

    def set_reference_weights(self, reference_weights: dict[str, torch.Tensor]) -> None:
        """Switch to the full fine-tuning path using a frozen W0 snapshot."""

        self.reference_weights = reference_weights

    def layer_keys(self) -> dict[str, torch.Tensor]:
        return self.fishers

    def memory_bytes(self) -> int:
        """Bytes the Fisher matrices occupy wherever they were moved.

        One d_out x d_in matrix per target layer. For attention projections
        under grouped-query attention this is smaller than the covariance path's
        d_in x d_in, so EWC's resident cost is not what separates the two
        methods; the collection stage is, since C needs forward passes only
        while F needs a backward pass per example.
        """

        return sum(fisher.numel() * fisher.element_size() for fisher in self.fishers.values())

    def reference_memory_bytes(self) -> int:
        """Bytes the W0 snapshot occupies; zero on the LoRA path."""

        if not self.reference_weights:
            return 0
        return sum(
            reference.numel() * reference.element_size()
            for reference in self.reference_weights.values()
        )

    def __call__(self, model: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
        if self.reference_weights is not None:
            return full_fisher_regularizer(
                model,
                self.fishers,
                self.reference_weights,
                normalize_by_layers=self.normalize_by_layers,
            )
        return lora_fisher_regularizer(
            model,
            self.fishers,
            adapter_name=self.adapter_name,
            normalize_by_layers=self.normalize_by_layers,
        )
