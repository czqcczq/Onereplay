"""Audit whether a C_mix is equal-weight in the sense its weights claim to be.

mix_covariances.py builds C_mix = sum_d w_d C_d, and the runs so far used
w = 0.5/0.5 on the grounds that C is a token mean, so unlike a sequence-summed
Fisher it carries no length bias and equal coefficients are already equal weight.
That argument is correct and it is also incomplete: it only covers scale. The
penalty is tr(DeltaW C DeltaW^T), a quadratic form, so a domain's influence is
set by *where* its mass sits as much as by how much of it there is, and two
covariance files can carry the same trace while guarding disjoint subspaces.

The measured numbers say this is the situation here rather than a hypothetical:
C_math and C_if agree on trace to within 9%, but their leading subspaces overlap
only 0.49 while their energy-weighted alignment is 0.79. The gap between those
two says the high-energy directions -- attention sinks, outlier feature
dimensions, the parts of the representation geometry every corpus excites -- are
shared, and the domain-specific structure lives in the low-energy tail. Trace is
dominated by the shared part (effective rank ~7 out of 2048, top eigenvalue ~35%
of the trace), so equalizing trace equalizes the part the two domains already
agree on and says nothing about the part that distinguishes them.

This script therefore reports four layers, cheapest first, and each one answers a
different question:

  provenance  did the two files see comparable token populations at all? C is a
              token mean, so row counts and corpus sizes divide out, but the
              *composition* of the average token does not. C_if collected at
              max_len 512 over FLAN, whose answers average 24 tokens, is mostly a
              covariance of prompt tokens; C_math collected at max_len 2048 over
              self-distilled chains of thought is mostly a covariance of answer
              tokens. That is a fact about what "equal weight" is equalizing, and
              it belongs in the writeup rather than in a footnote.

  scale       per-layer trace ratio r_l, and n_l = r_l / global_ratio. This is
              inspect_fisher's per-layer diagnostic applied to C. A wide n_l
              spread means no single scalar weight balances every layer, which
              bounds what any weight-tuning exercise can achieve.

  influence   tr(DeltaW C_d DeltaW^T) under a real DeltaW, plus the norms and the
              angle of the two penalty gradients 2 DeltaW C_d. Pass the *vanilla*
              adapter here. Using a C-regularized run's DeltaW biases the ratio:
              training actively minimized tr(DeltaW C_if DeltaW^T), so the
              denominator is artificially small and the ratio artificially large.
              The gradient angle is the cheap decisive number -- if the two
              domains pull DeltaW the same way, the mixing weight cannot matter
              much no matter what the other sections say.

  directions  the part with no Fisher analogue. Whiten by C_avg = mean_d C_d and
              both matrices become diagonal in one shared basis, with per-
              direction masses alpha (domain A) and beta (domain B) summing to 2.
              mu = beta / alpha is then a scale-free statement of who owns each
              direction, and a direction's share of the mix under weights w is
              w_b beta / (w_a alpha + w_b beta). Splitting the spectrum into
              A-unique, shared and B-unique lets the mix be judged on the
              protection it delivers to each domain's *distinctive* directions,
              which is the quantity 0.5/0.5 was never checked against.

The closing section collects every weight vector these sections imply, converts
each to the lambda that holds lambda * reg fixed against the incumbent mix, and
predicts the unique-direction balance each one would produce. Changing weights
without moving lambda changes the regularization strength at the same time, which
would confound the very comparison the new weights are meant to enable.

Usage:
    python -m onereplay.scripts.analyze_cov_mix \
        --covs results/cov/cov_flan_chat_20k_qv.pt,results/cov/cov_math_metamath30k_qv.pt \
        --labels if,math \
        --weights 0.5,0.5 \
        --probe_adapter results/adapters/cs_vanilla_seed1 \
        --ref_lambda 3e-2 \
        --layer_stride 4 \
        --json_out results/metrics/cov_mix_audit_ifmath.json

    # scale and provenance only, no adapter and no eigendecomposition, seconds:
    python -m onereplay.scripts.analyze_cov_mix --covs C_if.pt,C_math.pt \
        --labels if,math --skip_directions 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from onereplay.scripts.compare_cov_scale import load_lora_deltas  # noqa: E402

# A mismatch here means the two files describe different objects and no weight
# repairs that; the comparison has to be abandoned, not reweighted.
BLOCKING_KEYS = ("model_name", "target_modules", "cov_normalization", "cov_norm_eps")

# A mismatch here keeps the files comparable but changes which tokens entered the
# average, so it changes what the mix is equalizing rather than whether it can.
COMPOSITION_KEYS = (
    "max_len",
    "truncation_side",
    "use_chat_template",
    "include_target_in_chat",
    "enable_thinking",
    "concat_prompt_target",
    "require_target",
)

# Expected to differ -- printed for the record, never flagged.
CONTEXT_KEYS = (
    "dataset_name",
    "dataset_path",
    "data_files",
    "max_samples",
    "sample_strategy",
    "sample_shuffle",
    "sample_seed",
    "pool_rows",
    "pool_fingerprint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a covariance mix for real equal weight.")
    parser.add_argument(
        "--covs",
        type=str,
        required=True,
        help="Comma-separated covariance files. The first one is the reference domain.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated short names matching --covs; defaults to the file stems.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.5,0.5",
        help="The mix currently in use. Every alternative is reported relative to it.",
    )
    parser.add_argument(
        "--probe_adapter",
        type=str,
        default="",
        help=(
            "PEFT LoRA adapter supplying DeltaW for the influence section. Use the "
            "vanilla (unregularized) run: a C-regularized DeltaW was trained to make "
            "one of these penalties small and biases the ratio in that domain's favour."
        ),
    )
    parser.add_argument(
        "--ref_lambda",
        type=float,
        default=3e-2,
        help="lambda tuned on the incumbent mix; each candidate's equivalent is derived from it.",
    )
    parser.add_argument(
        "--subspace_rank",
        type=int,
        default=256,
        help=(
            "Directions retained from C_avg before whitening. C is extremely "
            "anisotropic, so the inverse square root is ill-conditioned on the full "
            "space; the retained fraction of each domain's trace is reported so this "
            "cut can be checked rather than assumed."
        ),
    )
    parser.add_argument(
        "--layer_stride",
        type=int,
        default=8,
        help="Only every k-th layer joins the direction section, where eigh dominates runtime.",
    )
    parser.add_argument(
        "--unique_tau",
        type=float,
        default=3.0,
        help=(
            "A direction is unique to a domain when that domain excites it tau times "
            "more than the other. Between 1/tau and tau it counts as shared."
        ),
    )
    parser.add_argument(
        "--eig_floor",
        type=float,
        default=1e-8,
        help="Eigenvalues of C_avg below this fraction of the largest are dropped as numerical noise.",
    )
    parser.add_argument("--skip_directions", type=int, default=0)
    parser.add_argument("--json_out", type=str, default="")
    parser.add_argument(
        "--strict",
        type=int,
        default=0,
        help="1 exits non-zero when the provenance section finds a blocking mismatch.",
    )
    return parser.parse_args()


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def load_payload(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "covariances" not in payload:
        raise SystemExit(f"{path} is not a covariance payload written by collect_cov.py")
    return payload


def percentiles(values: list[float] | np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    p0, p10, p50, p90, p100 = np.percentile(array, [0, 10, 50, 90, 100])
    return f"min={p0:.3f} P10={p10:.3f} median={p50:.3f} P90={p90:.3f} max={p100:.3f}"


def normalized(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values] if total > 0 else values


def format_weights(labels: list[str], weights: list[float]) -> str:
    return "  ".join(f"{label}={weight:.4f}" for label, weight in zip(labels, weights))


def trace_of(covariance: torch.Tensor) -> float:
    return float(torch.diagonal(covariance.double()).sum())


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def report_provenance(payloads: list[dict], labels: list[str], paths: list[str]) -> list[str]:
    """Print the collection settings side by side and return blocking mismatches.

    C divides by its own token count, so corpus size and row count cannot skew a
    mix. What survives that normalization is the makeup of the average token,
    which the truncation and templating flags control directly, so those are the
    settings a weight argument has to be stated against.
    """

    print("==== provenance: what each C actually averaged over ====")
    metas = [payload.get("metadata", {}) or {} for payload in payloads]
    for label, path, meta in zip(labels, paths, metas):
        if meta.get("type") == "weighted_covariance_mix":
            print(f"  NOTE {label}: {path} is already a mix; its counts are a floor, not a total")

    blocking: list[str] = []
    for group, keys in (("blocking", BLOCKING_KEYS), ("composition", COMPOSITION_KEYS),
                        ("context", CONTEXT_KEYS)):
        print(f"  -- {group} --")
        for key in keys:
            values = [meta.get(key) for meta in metas]
            if all(value is None for value in values):
                continue
            comparable = [tuple(v) if isinstance(v, list) else v for v in values]
            differs = len(set(map(repr, comparable))) > 1
            marker = "*" if differs and group != "context" else " "
            rendered = "  ".join(
                f"{label}={value!r}" for label, value in zip(labels, values)
            )
            print(f"   {marker} {key:<24} {rendered}")
            if differs and group == "blocking":
                blocking.append(f"{key} differs across inputs: {rendered}")

    print("  -- token population --")
    for label, payload, meta in zip(labels, payloads, metas):
        counts = payload.get("counts", {}) or {}
        if not counts:
            print(f"    {label:<8} no token counts stored")
            continue
        values = sorted(set(counts.values()))
        rows = meta.get("pool_rows") or meta.get("max_samples") or 0
        per_row = f"{values[-1] / rows:6.1f} tokens/row" if rows else "tokens/row unknown"
        spread = "" if len(values) == 1 else f" (varies across layers: {values[0]}..{values[-1]})"
        print(f"    {label:<8} {values[-1]:>12,} tokens over {rows:>7,} rows"
              f"  = {per_row}{spread}")
    print(
        "    tokens/row is the composition signal: a domain whose answers are short "
        "contributes a C dominated by prompt tokens, one whose answers are long "
        "contributes a C dominated by answer tokens, and 0.5/0.5 equalizes those two "
        "average tokens -- not two corpora and not two row counts."
    )

    if blocking:
        print("  BLOCKING: the inputs are not describing the same object:")
        for item in blocking:
            print(f"    - {item}")
    return blocking


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------


def report_scale(
    covariances: list[dict[str, torch.Tensor]],
    labels: list[str],
    shared: list[str],
) -> dict:
    """Per-layer trace ratios against the reference domain, plus n_l stability.

    tr(C) is the penalty a layer would see under an isotropic DeltaW, so the ratio
    of traces is the scale part of the influence question and nothing more. It is
    reported per layer because a global ratio only sets a usable scalar weight if
    the layers agree on it.
    """

    print()
    print("==== scale: trace per layer, against the reference domain ====")
    totals = [sum(trace_of(cov[name]) for name in shared) for cov in covariances]
    summary: dict = {"totals": dict(zip(labels, totals))}

    reference_total = totals[0]
    for index in range(1, len(covariances)):
        ratios = []
        for name in shared:
            reference_trace = trace_of(covariances[0][name])
            if reference_trace > 0:
                ratios.append(trace_of(covariances[index][name]) / reference_trace)
        if not ratios:
            continue
        global_ratio = totals[index] / reference_total if reference_total > 0 else float("nan")
        stability = [value / global_ratio for value in ratios]
        spread = max(stability) / min(stability) if min(stability) > 0 else float("inf")
        print(f"  -- {labels[index]} / {labels[0]} --")
        print(f"    global trace ratio : {global_ratio:.4f}x")
        print(f"    r_l across layers  : {percentiles(ratios)}")
        print(f"    n_l = r_l / global : {percentiles(stability)}")
        print(f"    spread (max/min)   : {spread:.1f}x")
        if spread <= 4:
            print("    -> tight: one scalar weight balances every layer within ~2x.")
        else:
            print(
                "    -> wide: the average balance is not what each layer gets, so a "
                "single scalar weight over-protects some layers and under-protects others."
            )
        summary[f"{labels[index]}_vs_{labels[0]}"] = {
            "global_trace_ratio": global_ratio,
            "r_l": {"min": min(ratios), "median": float(np.median(ratios)), "max": max(ratios)},
            "n_l_spread": spread,
        }

    weights = normalized([1.0 / total if total > 0 else 0.0 for total in totals])
    print(f"  equal-trace weights  : {format_weights(labels, weights)}")
    print(
        "    equal trace only means equal mass under an isotropic DeltaW, and most of "
        "that mass sits in directions both domains share."
    )
    summary["equal_trace_weights"] = dict(zip(labels, weights))
    return summary


# ---------------------------------------------------------------------------
# influence under a real DeltaW
# ---------------------------------------------------------------------------


def layer_influence(
    covariance: torch.Tensor,
    lora_a: torch.Tensor,
    kernel: torch.Tensor,
    scale: float,
) -> tuple[float, torch.Tensor]:
    """Return this layer's penalty and the projected covariance A C.

    Everything the influence section needs collapses to rank x rank matrices.
    With K = B^T B and P = A C, the penalty is scale^2 tr(K P A^T) -- the same
    shortcut lora_covariance_regularizer uses -- and the gradient inner products
    below are scale^2 tr(K P_j P_i^T), so no d_out x d_in matrix is ever formed.
    """

    projected = lora_a @ covariance
    penalty = float(scale**2 * torch.sum(kernel * (projected @ lora_a.T).T))
    return penalty, projected


def report_influence(
    covariances: list[dict[str, torch.Tensor]],
    labels: list[str],
    shared: list[str],
    adapter_path: str,
    deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]],
) -> dict:
    """Penalty and gradient geometry of each domain under one fixed DeltaW.

    The penalty ratio answers "how much of the mix's value does each domain
    supply"; the gradient angle answers whether that split has any consequence.
    Two domains whose gradients are nearly parallel steer DeltaW to the same
    place regardless of the weights, and no reweighting experiment on them is
    worth GPU time.
    """

    print()
    print("==== influence: penalty and gradient under a real DeltaW ====")
    joined = [name for name in shared if name in deltas]
    if not joined:
        print(f"  the adapter's layers do not match the covariance keys; example "
              f"adapter={next(iter(deltas))} cov={shared[0]}")
        return {}
    print(f"  adapter: {Path(adapter_path).name}   layers matched: {len(joined)} / {len(shared)}")

    penalties = [0.0 for _ in covariances]
    gradient_norms = [0.0 for _ in covariances]
    cosines: list[float] = []
    cosine_weights: list[float] = []
    for name in joined:
        lora_a, lora_b, scale = deltas[name]
        kernel = lora_b.T @ lora_b
        projections = []
        for index, covariance in enumerate(covariances):
            penalty, projected = layer_influence(covariance[name].float(), lora_a, kernel, scale)
            penalties[index] += penalty
            projections.append(projected)
            gradient_norms[index] += float(scale**2 * torch.sum(kernel * (projected @ projected.T).T))
        if len(projections) == 2:
            inner = float(scale**2 * torch.sum(kernel * (projections[1] @ projections[0].T).T))
            first = float(scale**2 * torch.sum(kernel * (projections[0] @ projections[0].T).T))
            second = float(scale**2 * torch.sum(kernel * (projections[1] @ projections[1].T).T))
            denominator = (first * second) ** 0.5
            if denominator > 0:
                cosines.append(inner / denominator)
                cosine_weights.append(denominator)

    # normalize_by_layers in the trainer divides by the layer count, so the numbers
    # printed here are on the same scale as the logged train_replay_reg.
    penalties = [value / len(joined) for value in penalties]
    gradient_norms = [(value / len(joined)) ** 0.5 for value in gradient_norms]
    for label, penalty, gradient in zip(labels, penalties, gradient_norms):
        print(f"    reg(C_{label:<6}) = {penalty:.6e}   ||DeltaW C||_F = {gradient:.6e}")
    if len(penalties) == 2 and penalties[0] > 0:
        print(f"    ratio {labels[1]}/{labels[0]} = {penalties[1] / penalties[0]:.4f}")
    weights = normalized([1.0 / value if value > 0 else 0.0 for value in penalties])
    print(f"  equal-penalty weights : {format_weights(labels, weights)}")

    summary = {
        "adapter": adapter_path,
        "layers": len(joined),
        "penalty": dict(zip(labels, penalties)),
        "equal_penalty_weights": dict(zip(labels, weights)),
    }
    if cosines:
        weighted = float(np.average(cosines, weights=cosine_weights))
        print(f"  gradient angle        : energy-weighted cos = {weighted:.4f}")
        print(f"    per layer           : {percentiles(cosines)}")
        if weighted > 0.95:
            print(
                "    -> the two domains pull DeltaW the same way. The mixing weight "
                "decides almost nothing here, and a retention sweep over weights will "
                "read as noise; spend the GPU time elsewhere."
            )
        else:
            print(
                "    -> the two domains pull DeltaW in measurably different directions, "
                "so the mixing weight genuinely selects which one gets protected."
            )
        summary["gradient_cosine"] = weighted
    return summary


# ---------------------------------------------------------------------------
# directions
# ---------------------------------------------------------------------------


def whitened_spectrum(
    first: torch.Tensor,
    second: torch.Tensor,
    subspace_rank: int,
    eig_floor: float,
    probe: tuple[torch.Tensor, torch.Tensor, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """Split both matrices over one shared basis and weigh each direction.

    Whitening by C_avg = (C_a + C_b) / 2 sends C_avg to the identity, and because
    the two matrices sum to 2 C_avg their whitened forms sum to 2I: they commute,
    share an eigenbasis, and alpha + beta = 2 exactly. One eigendecomposition
    therefore yields both spectra, and mu = beta / alpha is a scale-free statement
    of which domain owns a direction.

    Ownership is a ratio, but influence is not, so alpha alone cannot be summed:
    whitening deliberately removed the magnitudes, and at w = 0.5/0.5 every
    whitened direction carries mass exactly 1 -- the mix *is* C_avg, so any
    "protection" read in C_avg's own metric is a tautology rather than a
    measurement. The returned per-direction weight g restores the magnitude. With
    the basis V = U Lambda^-1/2 Q the penalty decomposes exactly as

        tr(DeltaW C_a DeltaW^T) = sum_j alpha_j * ||DeltaW V^-T e_j||^2

    so g_j = ||DeltaW V^-T e_j||^2 makes alpha_j g_j the share of the real penalty
    that direction j carries. Without a probe DeltaW the isotropic counterpart
    g_j = ||V^-T e_j||^2 is used instead, and then sum_j alpha_j g_j is exactly
    tr(C_a): the classes below become a decomposition of the trace.

    The retained subspace is capped because C's effective rank is a few dozen at
    most while its dimension is thousands; inverting the noise floor would
    manufacture directions no token ever excites. The capture fractions say how
    much of each domain's trace survived the cut.
    """

    average = 0.5 * (first + second)
    values, vectors = torch.linalg.eigh(average)
    keep = values > eig_floor * float(values[-1])
    values = values[keep]
    vectors = vectors[:, keep]
    if subspace_rank > 0 and values.numel() > subspace_rank:
        values = values[-subspace_rank:]
        vectors = vectors[:, -subspace_rank:]

    capture_first = float(
        (vectors.T @ first @ vectors).diagonal().sum() / torch.diagonal(first).sum()
    )
    capture_second = float(
        (vectors.T @ second @ vectors).diagonal().sum() / torch.diagonal(second).sum()
    )

    whitener = vectors * values.rsqrt()
    alpha, rotation = torch.linalg.eigh(whitener.T @ first @ whitener)
    # alpha + beta = 2 holds analytically; clamping only guards the ratio against
    # directions the retained subspace resolves to numerical zero on one side.
    alpha = alpha.clamp(1e-6, 2.0 - 1e-6)
    beta = 2.0 - alpha

    # Columns of V^-T = U Lambda^1/2 Q, whose squared norms are the isotropic g.
    if probe is None:
        weights = (values.unsqueeze(1) * rotation**2).sum(0)
    else:
        lora_a, lora_b, scale = probe
        projected = lora_a @ (vectors * values.sqrt()) @ rotation
        kernel = lora_b.T @ lora_b
        weights = (scale**2) * ((kernel @ projected) * projected).sum(0)

    return (
        alpha.numpy(),
        beta.numpy(),
        weights.numpy(),
        capture_first,
        capture_second,
        int(values.numel()),
    )


def unique_balance(
    mass_first: np.ndarray,
    mass_second: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    weights: tuple[float, float],
) -> tuple[float, float]:
    """Penalty mass the mix places on each domain's own distinctive directions."""

    mixed = weights[0] * mass_first + weights[1] * mass_second
    return float(mixed[first_mask].sum()), float(mixed[second_mask].sum())


def report_directions(
    covariances: list[dict[str, torch.Tensor]],
    labels: list[str],
    shared: list[str],
    args: argparse.Namespace,
    current_weights: list[float],
    deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]] | None,
) -> dict:
    """Split the spectrum into A-unique, shared and B-unique and weigh each.

    This is the section with no Fisher counterpart. An elementwise diagonal
    Fisher weights individual parameters, so "which directions does this domain
    own" is not a question that can be posed of it; a covariance defines a
    quadratic form, and the answer is its spectrum relative to the other domain's.

    The classification is scale-free but the masses are not: they are shares of
    tr(C_d), or of the real penalty when a probe DeltaW is available. That is what
    makes the balance a measurement rather than an artifact of the whitening --
    see whitened_spectrum for why the whitened masses alone would not be.
    """

    print()
    print("==== directions: who owns each direction, and what the mix gives them ====")
    picked = shared[:: max(args.layer_stride, 1)]
    if deltas is not None:
        picked = [name for name in picked if name in deltas]
    units = "penalty under the probe DeltaW" if deltas else "trace"
    print(f"  layers analyzed: {len(picked)} of {len(shared)} (stride {args.layer_stride}), "
          f"masses in units of {units}")
    if not picked:
        print("  no layer survived the adapter join; nothing to decompose")
        return {}

    masses = [[], []]
    captures = [[], []]
    ranks: list[int] = []
    for name in picked:
        alpha, beta, weights, capture_a, capture_b, rank = whitened_spectrum(
            covariances[0][name].double(),
            covariances[1][name].double(),
            args.subspace_rank,
            args.eig_floor,
            probe=None if deltas is None else tuple(
                item.double() if torch.is_tensor(item) else item for item in deltas[name]
            ),
        )
        masses[0].append(alpha * weights)
        masses[1].append(beta * weights)
        captures[0].append(capture_a)
        captures[1].append(capture_b)
        ranks.append(rank)

    mass_first = np.concatenate(masses[0])
    mass_second = np.concatenate(masses[1])
    ratio = mass_second / np.maximum(mass_first, 1e-300)
    print(f"  retained rank per layer: {int(np.median(ranks))} (median), capturing "
          f"{labels[0]}={np.mean(captures[0]):.1%} {labels[1]}={np.mean(captures[1]):.1%} of the trace")
    if min(np.mean(captures[0]), np.mean(captures[1])) < 0.8:
        print("    the cut is dropping a fifth or more of a domain's mass; raise --subspace_rank")

    tau = args.unique_tau
    second_mask = ratio > tau
    first_mask = ratio < 1.0 / tau
    shared_mask = ~(first_mask | second_mask)
    print(f"  mu = mass({labels[1]}) / mass({labels[0]}) per direction: {percentiles(ratio)}")
    print(f"  direction census at tau={tau:g}:")
    for label, mask in ((f"{labels[0]}-unique", first_mask), ("shared", shared_mask),
                        (f"{labels[1]}-unique", second_mask)):
        count = int(mask.sum())
        print(f"    {label:<16} {count:>6} directions ({count / ratio.size:.1%}), "
              f"holding {labels[0]}={mass_first[mask].sum() / mass_first.sum():.1%} and "
              f"{labels[1]}={mass_second[mask].sum() / mass_second.sum():.1%} of that domain's mass")

    first_unique = float(mass_first[first_mask].sum())
    second_unique = float(mass_second[second_mask].sum())
    gap = f"   ({second_unique / first_unique:.2f}x apart)" if first_unique > 0 else ""
    print(f"  distinctive mass       : {labels[0]}={first_unique:.4g}  "
          f"{labels[1]}={second_unique:.4g}{gap}")
    print(
        "    this is the number the trace ratio cannot see: the shared high-energy "
        "directions dominate the trace, so two domains can agree on trace while their "
        "distinctive subspaces differ by much more."
    )

    current = (current_weights[0], current_weights[1])
    protected_first, protected_second = unique_balance(
        mass_first, mass_second, first_mask, second_mask, current
    )
    imbalance = protected_second / protected_first if protected_first > 0 else float("inf")
    print(f"  under the current mix ({format_weights(labels[:2], current_weights[:2])}):")
    print(f"    protection to {labels[0]}-unique directions = {protected_first:.4g}")
    print(f"    protection to {labels[1]}-unique directions = {protected_second:.4g}")
    print(f"    imbalance = {imbalance:.3f}x in favour of "
          f"{labels[1] if imbalance > 1 else labels[0]}")

    # Solve w_a * (A1 - A2) + w_b * (B1 - B2) = 0 under w_a + w_b = 1, where the
    # index is the direction class and the letter the domain. This is the exact
    # convex weight that hands both distinctive subspaces the same protection.
    a_first, b_first = float(mass_first[first_mask].sum()), float(mass_second[first_mask].sum())
    a_second, b_second = float(mass_first[second_mask].sum()), float(mass_second[second_mask].sum())
    denominator = (a_first - a_second) - (b_first - b_second)
    equal_unique = None
    if abs(denominator) > 1e-12:
        candidate = (b_second - b_first) / denominator
        if 0.0 < candidate < 1.0:
            equal_unique = [candidate, 1.0 - candidate]
    if equal_unique is None:
        print("  no convex weight equalizes the two distinctive subspaces; the imbalance "
              "is structural and has to be reported rather than tuned away")
    else:
        print(f"  equal-unique weights   : {format_weights(labels[:2], equal_unique)}")
        print(
            "    this is the weight the 0.5/0.5 mix was never checked against: it "
            "equalizes protection on the directions that distinguish the two domains, "
            "not on the mass they share."
        )

    return {
        "layers": len(picked),
        "median_rank": int(np.median(ranks)),
        "capture": {labels[0]: float(np.mean(captures[0])), labels[1]: float(np.mean(captures[1]))},
        "tau": tau,
        "census": {
            f"{labels[0]}_unique": int(first_mask.sum()),
            "shared": int(shared_mask.sum()),
            f"{labels[1]}_unique": int(second_mask.sum()),
        },
        "distinctive_mass": {labels[0]: first_unique, labels[1]: second_unique},
        "total_mass": {
            labels[0]: float(mass_first.sum()) / len(picked),
            labels[1]: float(mass_second.sum()) / len(picked),
        },
        "current_imbalance": imbalance,
        "equal_unique_weights": (
            dict(zip(labels, equal_unique)) if equal_unique else None
        ),
        "_spectrum": (mass_first, mass_second, first_mask, second_mask),
    }


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


def report_candidates(
    labels: list[str],
    current_weights: list[float],
    scale_summary: dict,
    influence_summary: dict,
    direction_summary: dict,
    ref_lambda: float,
) -> dict:
    """Put every candidate weight on one line with its lambda and its balance.

    The penalty is linear in C, so a mix's penalty under a fixed DeltaW is the
    same linear combination of the per-domain penalties. That makes the lambda
    that holds lambda * reg fixed exactly computable here instead of requiring
    another probe run, which matters because a weight change that silently
    changes the regularization strength cannot be attributed to the weights.
    """

    print()
    print("==== candidates: weights, the lambda that keeps strength fixed, and the balance ====")
    candidates: list[tuple[str, list[float]]] = [("current", list(current_weights))]
    if "equal_trace_weights" in scale_summary:
        candidates.append(("equal-trace", [scale_summary["equal_trace_weights"][l] for l in labels]))
    if influence_summary.get("equal_penalty_weights"):
        candidates.append(
            ("equal-penalty", [influence_summary["equal_penalty_weights"][l] for l in labels])
        )
    if direction_summary.get("equal_unique_weights"):
        candidates.append(
            ("equal-unique", [direction_summary["equal_unique_weights"][l] for l in labels])
        )

    per_domain = influence_summary.get("penalty")
    reference_reg = None
    if per_domain:
        reference_reg = sum(w * per_domain[l] for w, l in zip(current_weights, labels))

    spectrum = direction_summary.get("_spectrum")
    rows = []
    for name, weights in candidates:
        line = f"  {name:<14} {format_weights(labels, weights)}"
        record = {"name": name, "weights": dict(zip(labels, weights))}
        if per_domain and reference_reg:
            mixed_reg = sum(w * per_domain[l] for w, l in zip(weights, labels))
            equivalent = ref_lambda * reference_reg / mixed_reg if mixed_reg > 0 else float("nan")
            line += f"   lambda={equivalent:.4g}"
            record["equivalent_lambda"] = equivalent
        if spectrum is not None and len(weights) >= 2:
            mass_first, mass_second, first_mask, second_mask = spectrum
            first, second = unique_balance(
                mass_first, mass_second, first_mask, second_mask, (weights[0], weights[1])
            )
            imbalance = second / first if first > 0 else float("inf")
            line += f"   unique-imbalance={imbalance:.3f}x"
            record["unique_imbalance"] = imbalance
        print(line)
        rows.append(record)

    if reference_reg:
        print(
            "  lambda is derived so that lambda * reg matches the current mix under the "
            "probe DeltaW. Without it a weight change moves strength and balance at once "
            "and neither effect can be read off the result."
        )
    if spectrum is not None:
        print(
            "  unique-imbalance is the prediction to falsify: train the candidates, then "
            "compare normalized retention (score - vanilla) / (base - vanilla) on each "
            "domain. Equal influence is a claim about that ratio, not about the matrices."
        )
    return {"candidates": rows, "ref_lambda": ref_lambda}


def main() -> None:
    args = parse_args()
    paths = parse_list(args.covs)
    if len(paths) < 2:
        raise SystemExit("--covs needs at least two covariance files")
    labels = parse_list(args.labels) or [Path(path).stem for path in paths]
    if len(labels) != len(paths):
        raise SystemExit("--labels must have as many entries as --covs")
    current_weights = [float(item) for item in parse_list(args.weights)]
    if len(current_weights) != len(paths):
        raise SystemExit("--weights must have as many entries as --covs")
    current_weights = normalized(current_weights)

    payloads = [load_payload(path) for path in paths]
    covariances = [payload["covariances"] for payload in payloads]
    shared = sorted(set.intersection(*[set(cov) for cov in covariances]))
    if not shared:
        raise SystemExit("the inputs share no layer names; they used different --target_modules")

    print(f"inputs   : {', '.join(f'{l}={p}' for l, p in zip(labels, paths))}")
    print(f"current  : {format_weights(labels, current_weights)}")
    print(f"layers   : {len(shared)} shared of {[len(cov) for cov in covariances]}")
    print()

    blocking = report_provenance(payloads, labels, paths)
    scale_summary = report_scale(covariances, labels, shared)

    deltas = load_lora_deltas(args.probe_adapter) if args.probe_adapter else None
    influence_summary: dict = {}
    if deltas is not None:
        influence_summary = report_influence(
            covariances, labels, shared, args.probe_adapter, deltas
        )
    else:
        print()
        print("==== influence: skipped, no --probe_adapter ====")
        print("  without a DeltaW the audit sees only scale, which is the part that was "
              "never in doubt. Point it at the vanilla adapter.")

    direction_summary: dict = {}
    if args.skip_directions == 1:
        print()
        print("==== directions: skipped by --skip_directions ====")
    elif len(paths) != 2:
        print()
        print("==== directions: skipped, the shared-basis decomposition is defined for two domains ====")
    else:
        direction_summary = report_directions(
            covariances, labels, shared, args, current_weights, deltas
        )
        # The decomposition is exact inside the retained subspace, so summing it
        # back has to reproduce the penalty the influence section measured
        # independently. A gap is the retained-subspace cut, not an approximation
        # in the split, which is why it is reported next to the capture fractions.
        totals = direction_summary.get("total_mass")
        penalties = influence_summary.get("penalty")
        if totals and penalties:
            print("  reconciliation with the influence section:")
            for label in labels:
                recovered = totals[label] / penalties[label] if penalties[label] else float("nan")
                print(f"    sum of {label} direction masses / reg(C_{label}) = {recovered:.4f}")

    candidate_summary = report_candidates(
        labels, current_weights, scale_summary, influence_summary,
        direction_summary, args.ref_lambda,
    )

    if args.json_out:
        direction_summary.pop("_spectrum", None)
        summary = {
            "inputs": dict(zip(labels, paths)),
            "current_weights": dict(zip(labels, current_weights)),
            "blocking": blocking,
            "scale": scale_summary,
            "influence": influence_summary,
            "directions": direction_summary,
            "candidates": candidate_summary,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    if blocking and args.strict == 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
