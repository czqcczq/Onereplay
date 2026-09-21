"""Does Adam-NSCL's null space exist on this covariance, and at which threshold?

Adam-NSCL keeps the eigenvectors of C whose eigenvalues satisfy

    lambda <= lambda_min * thres

and projects every update into their span. On the CNNs the paper was written for
this works because the feature covariances are genuinely rank-deficient: a
3x3x512 patch covariance of a ReLU network has exact zeros, and anything within
a factor of 10 of the floor is one of them. A transformer's input covariance has
no exact zeros. Its spectrum decays smoothly over eight or nine orders of
magnitude, so what the rule selects is decided by how far the *bottom* of that
decay spreads -- a quantity with no relation to the model, the task or the
hidden size, and one that cannot be guessed.

Both failure modes are silent in a training run. Too few directions kept and the
projected weights cannot move, so the arm is a frozen backbone with a trainable
embedding that still produces a falling loss curve. Too many and nothing is
constrained, so the arm is vanilla full fine-tuning under an Adam-NSCL label.
This script says which of those a given thres would produce, from the covariance
alone, before any GPU time is spent.

What to read
------------
kept_share    fraction of directions the update is allowed to move in. The
              method needs this well above 0 to learn anything.
energy_share  fraction of the old data's second-moment energy that the kept
              subspace still carries. The method needs this near 0, or the
              "null space" is not one and DeltaW is free to move directions the
              old task uses.
lr_shrink     1/sqrt(k). upstream's get_transforms divides the projector by its
              Frobenius norm, so every update is scaled by this on top of the
              learning rate. It is what --nscl_svd_lr has to compensate for, and
              it varies per layer with k.

A usable thres is one where kept_share is comfortably off 0 on *every* layer --
the minimum matters, not the mean, because one locked layer is a bottleneck the
others cannot route around -- while energy_share stays negligible. If no thres
does both, the method does not transfer to this covariance, and that is a result
worth having for the cost of this script rather than for the cost of a sweep.

Eigenvalues come from torch.linalg.svdvals, not from eigvalsh, because that is
what upstream's torch.svd returns and the difference changes the answer: eigvalsh
on a near-singular PSD matrix returns small *negative* values at the bottom, and
`lambda <= lambda_min * thres` with a negative floor selects a different set.

The covariance's overall scale is irrelevant here, so it does not matter that
collect_cov divides by the token count while upstream accumulates a bare sum:
the rule compares eigenvalues against the smallest eigenvalue, and both sides
scale together.

Usage
  python -m onereplay.baselines.inspect_nscl_nullspace \
      --cov_path .../cov/cov_flan_chat_20k_qv.pt

  # the per-layer table at the threshold you are considering, on GPU
  python -m onereplay.baselines.inspect_nscl_nullspace \
      --cov_path ... --detail_thres 100 --device cuda

  # machine-readable, for a plot
  python -m onereplay.baselines.inspect_nscl_nullspace \
      --cov_path ... --json_out results/nscl_nullspace.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from onereplay.core.covariance import load_covariance_file  # noqa: E402

DEFAULT_GRID = "1.001,2,5,10,30,100,1000,10000,100000,1000000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict what --nscl_thres would select, from the covariance alone."
    )
    parser.add_argument("--cov_path", type=str, required=True)
    parser.add_argument(
        "--thres_grid",
        type=str,
        default=DEFAULT_GRID,
        help=(
            "Comma-separated thresholds to evaluate. The default spans upstream's own "
            "values (10, 30) and the much larger ones a smoothly decaying spectrum "
            "needs before it selects anything."
        ),
    )
    parser.add_argument(
        "--detail_thres",
        type=float,
        default=-1.0,
        help="Also print the per-layer table at this threshold. Negative skips it.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "cuda is worth it: one 1536x1536 SVD is about a second on CPU and the file "
            "holds one per covered layer. Each matrix is moved over on its own, so the "
            "device only ever holds one."
        ),
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=0,
        help="Only look at the first N layers. 0 means all of them.",
    )
    parser.add_argument(
        "--require_usable",
        type=int,
        default=0,
        help=(
            "1 exits non-zero unless every threshold on the grid came out usable. For "
            "a job script to gate on before it spends a GPU: pass the single threshold "
            "the run will use as --thres_grid. Reading the verdict out of this script's "
            "stdout instead would be fragile, because the table formats thresholds with "
            "%%g and '1e5' does not appear in it as '1e5'."
        ),
    )
    parser.add_argument("--json_out", type=str, default="")
    return parser.parse_args()


def projector_cost(dims: dict[str, int], element_size: int) -> dict[str, float]:
    """What the projectors will cost, in bytes, for this covariance file.

    Two numbers, because they peak at different times and the larger one is not
    the one that stays.

    ``resident`` is one d_in x d_in projector per covered layer, alive for the
    whole run. ``setup_peak`` adds the eigenvectors: upstream's get_eigens fills
    an eigenvector matrix of the same size for *every* layer before
    get_transforms consumes any of them, so both sets exist at once.
    nscl.py drops the eigenvectors immediately afterwards, but the peak is what
    has to fit.

    Worth computing here rather than discovering it on the cluster, because the
    cost is quadratic in d_in and one wide layer dominates it: an MLP down_proj
    reading a 8960-wide intermediate activation is 34x the projector of a
    1536-wide attention projection.
    """

    resident = sum(dim * dim * element_size for dim in dims.values())
    return {
        "resident_gb": resident / 1024**3,
        "setup_peak_gb": 2 * resident / 1024**3,
        "widest_layer": max(dims, key=lambda name: dims[name]),
        "widest_dim": max(dims.values()),
        "widest_gb": max(dims.values()) ** 2 * element_size / 1024**3,
    }


def report_cost(dims: dict[str, int]) -> None:
    """Print the projector cost at both training precisions."""

    print("\nprojector memory (one d_in x d_in matrix per covered layer):")
    for label, element_size in (("float32", 4), ("bfloat16", 2)):
        cost = projector_cost(dims, element_size)
        print(
            f"  {label:<9} resident {cost['resident_gb']:6.2f} GiB, "
            f"setup peak {cost['setup_peak_gb']:6.2f} GiB "
            "(eigenvectors and projectors coexist)"
        )
    cost = projector_cost(dims, 4)
    widest = [name for name, dim in dims.items() if dim == cost["widest_dim"]]
    print(
        f"  widest input is {cost['widest_dim']} ({len(widest)} layers, e.g. "
        f"{cost['widest_layer']}), {cost['widest_gb']:.2f} GiB each in float32"
    )
    if cost["widest_dim"] >= 4 * min(dims.values()):
        narrow = projector_cost(
            {name: dim for name, dim in dims.items() if dim < cost["widest_dim"]}, 4
        )
        print(
            f"  dropping them would leave {narrow['resident_gb']:.2f} GiB resident. "
            "That is a different coverage than the penalty arms run with, so it has "
            "to be recorded as a difference rather than treated as a tuning knob."
        )


def warn_if_slow(dims: dict[str, int], device: str) -> None:
    """Say up front when the CPU path is going to take a long time.

    One spectrum costs O(d_in^3). The attention projections of a 1536-wide model
    are a second each on CPU; an MLP down_proj reading an 8960-wide intermediate
    activation is nearly two hundred times that much arithmetic, and there is one
    per layer. The run is not hung, but it can look like it for long enough that
    somebody kills it, so the estimate goes before the work rather than after.
    """

    if device != "cpu":
        return
    widest = max(dims.values())
    if widest < 4096:
        return
    heavy = sum(1 for dim in dims.values() if dim >= 4096)
    print(
        f"\n!! device=cpu with {heavy} layers at {widest}x{widest}. A spectrum costs "
        f"O(d^3), so those dominate and each takes minutes rather than seconds. "
        f"Pass --device cuda if a GPU is available; the matrices are moved over one "
        f"at a time, so only {widest * widest * 4 / 1024**3:.2f} GiB is ever resident "
        f"on it.",
        flush=True,
    )


def parse_grid(raw: str) -> list[float]:
    values = [float(item) for item in raw.replace(" ", "").split(",") if item]
    bad = [value for value in values if value < 1.0]
    if bad:
        raise ValueError(
            f"thresholds below 1 are not runnable: {bad}. The rule keeps eigenvalues at "
            "most thres times the smallest, so below 1 it can select nothing, and "
            "upstream then divides an all-zero projector by its own zero norm."
        )
    return sorted(values)


def layer_spectrum(covariance: torch.Tensor, device: str) -> torch.Tensor:
    """Descending singular values of one C, in float64 on the host.

    float64 because every quantity below is a ratio across the condition number,
    which runs past 1e8 on real covariances; in float32 the small end of the
    spectrum is the part that rounds away, and the small end is the whole
    question. The SVD itself runs in the input's precision on the requested
    device, then the values are widened -- widening afterwards does not recover
    lost digits, but it keeps the sums and ratios below from adding their own
    error on top.
    """

    matrix = covariance.to(device=device, dtype=torch.float32)
    values = torch.linalg.svdvals(matrix)
    return values.detach().to(device="cpu", dtype=torch.float64)


def selection(values: torch.Tensor, thres: float) -> dict[str, float]:
    """What upstream's rule would keep from this spectrum at this threshold."""

    # svdvals is descending, so [-1] is the floor the rule is anchored to.
    keep = values <= values[-1] * thres
    kept = int(keep.sum())
    total = int(values.numel())
    return {
        "kept": kept,
        "total": total,
        "kept_share": kept / max(total, 1),
        "energy_share": float(values[keep].sum() / values.sum()) if kept else 0.0,
        # The factor get_transforms' Frobenius normalization applies to every
        # update in this layer, on top of the learning rate.
        "lr_shrink": 1.0 / max(kept, 1) ** 0.5,
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate one threshold's per-layer selections into the row that matters."""

    shares = [row["kept_share"] for row in rows]
    energies = [row["energy_share"] for row in rows]
    shrinks = [row["lr_shrink"] for row in rows]
    ordered = sorted(shares)
    return {
        "kept_share_min": min(shares),
        "kept_share_median": ordered[len(ordered) // 2],
        "kept_share_max": max(shares),
        "energy_share_max": max(energies),
        "lr_shrink_min": min(shrinks),
        "lr_shrink_max": max(shrinks),
        # sqrt(k), the factor svd_lr would have to be raised by for the surviving
        # component of the update to move at the nominal rate.
        "lr_compensation_min": 1.0 / max(shrinks),
        "lr_compensation_max": 1.0 / min(shrinks),
        # One locked layer is enough to stop the network learning, so these are
        # counted rather than averaged away.
        "locked_layers": sum(1 for share in shares if share < 0.01),
        "vanilla_layers": sum(1 for share in shares if share > 0.99),
    }


def verdict(summary: dict[str, float], layers: int) -> str:
    """One word on whether this threshold is worth a training run."""

    if summary["locked_layers"] == layers:
        return "locked: every layer would be frozen"
    if summary["locked_layers"]:
        return f"locked in {summary['locked_layers']}/{layers} layers"
    if summary["vanilla_layers"] == layers:
        return "vanilla: nothing is constrained anywhere"
    if summary["energy_share_max"] > 1e-2:
        return f"leaky: a layer keeps {summary['energy_share_max']:.1%} of the old energy"
    return "usable"


def main() -> int:
    args = parse_args()
    grid = parse_grid(args.thres_grid)

    covariances = load_covariance_file(args.cov_path)
    names = sorted(covariances)
    if args.max_layers > 0:
        names = names[: args.max_layers]
    print(f"{len(names)} layers from {args.cov_path}, device={args.device}")
    dims = {name: int(covariances[name].shape[-1]) for name in names}
    report_cost(dims)
    warn_if_slow(dims, args.device)

    print("\ncomputing one spectrum per layer", flush=True)
    spectra: dict[str, torch.Tensor] = {}
    started = time.time()
    for index, name in enumerate(names, start=1):
        # Printed per layer, not every tenth. The cost is cubic in d_in, so one
        # wide layer can run minutes while the narrow ones run in a second, and
        # a progress line that only ticks every ten layers is indistinguishable
        # from a hang for as long as it takes to cross an MLP block.
        layer_started = time.time()
        spectra[name] = layer_spectrum(covariances[name], args.device)
        print(
            f"  [{index}/{len(names)}] {name} {dims[name]}x{dims[name]} "
            f"{time.time() - layer_started:6.1f}s  (elapsed {time.time() - started:.0f}s)",
            flush=True,
        )
    # The spectra are all that is needed from here on, and the file is the
    # largest thing in the process.
    del covariances

    conditions = []
    for name in names:
        values = spectra[name]
        smallest = float(values[-1])
        conditions.append(float(values[0] / smallest) if smallest > 0 else float("inf"))
    print(
        f"\ncondition number across layers: min {min(conditions):.3e}, "
        f"median {sorted(conditions)[len(conditions) // 2]:.3e}, max {max(conditions):.3e}"
    )
    print(
        "This is the whole story: the rule can only reach directions within thres of "
        "the floor, so a layer needs thres comparable to its condition number before "
        "it keeps a meaningful fraction."
    )

    print(
        f"\n{'thres':>10}  {'kept_share min/med/max':>26}  {'energy max':>11}  "
        f"{'step x':>17}  verdict"
    )
    print("-" * 108)
    table: list[dict] = []
    for thres in grid:
        rows = [selection(spectra[name], thres) for name in names]
        summary = summarize(rows)
        note = verdict(summary, len(names))
        print(
            f"{thres:>10g}  "
            f"{summary['kept_share_min']:>7.4f} /{summary['kept_share_median']:>8.4f} /"
            f"{summary['kept_share_max']:>8.4f}  "
            f"{summary['energy_share_max']:>11.3e}  "
            f"{summary['lr_shrink_min']:>7.4f}..{summary['lr_shrink_max']:<8.4f}  {note}"
        )
        table.append({"thres": thres, **summary, "verdict": note})

    usable = [row for row in table if row["verdict"] == "usable"]
    print()
    if usable:
        best = max(usable, key=lambda row: row["kept_share_min"])
        listed = ", ".join(format(row["thres"], "g") for row in usable)
        print(
            f"usable thresholds: {listed}. "
            f"The roomiest is {best['thres']:g}, where the tightest layer still keeps "
            f"{best['kept_share_min']:.1%} of its directions and the leakiest keeps "
            f"{best['energy_share_max']:.2e} of the old energy."
        )
        print(
            f"At that threshold the Frobenius normalization shrinks each step to "
            f"{best['lr_shrink_min']:.3g}x..{best['lr_shrink_max']:.3g}x of nominal, so "
            f"matching the effective step of the other arms would mean --nscl_svd_lr "
            f"about {best['lr_compensation_min']:.3g}x..{best['lr_compensation_max']:.3g}x "
            "their lr."
        )
        print(
            "Upstream does the opposite -- their svd_lr is half their model_lr, so they "
            "accept the shrink rather than compensate for it. Sweep both ends: the "
            "faithful setting and the step-matched one answer different questions, and "
            "an arm that simply never moved is not evidence about the method."
        )
    else:
        print(
            "No threshold on this grid is usable. Widen --thres_grid before concluding "
            "anything, but if the locked and vanilla rows meet with nothing in between, "
            "this covariance has no null space to train in and Adam-NSCL does not "
            "transfer to it. That is a reportable result, not a bug."
        )

    if args.detail_thres >= 1.0:
        print(f"\nper layer at thres={args.detail_thres:g}")
        print(f"{'layer':<52} {'kept':>7} {'total':>7} {'share':>8} {'energy':>11}")
        print("-" * 90)
        for name in names:
            row = selection(spectra[name], args.detail_thres)
            print(
                f"{name:<52} {row['kept']:>7d} {row['total']:>7d} "
                f"{row['kept_share']:>8.4f} {row['energy_share']:>11.3e}"
            )

    if args.json_out:
        payload = {
            "cov_path": args.cov_path,
            "layers": names,
            "condition_numbers": dict(zip(names, conditions)),
            "grid": table,
            "per_layer": {
                str(thres): {name: selection(spectra[name], thres) for name in names}
                for thres in grid
            },
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json_out}")

    if args.require_usable:
        unusable = [row for row in table if row["verdict"] != "usable"]
        if unusable:
            print(
                "\n--require_usable: "
                + "; ".join(f"thres={row['thres']:g} is {row['verdict']}" for row in unusable)
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
