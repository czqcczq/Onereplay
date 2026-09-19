"""Pick lambda by putting its cost and its benefit in one table.

The sweep summary the PBS scripts print shows only the cost side -- task_loss,
val_loss, and the penalty ratio. That is not enough to choose lambda, and on a
line where the penalty is cheap it is actively misleading: every lambda looks
free, so the table appears to say "take the largest one". What decides the
operating point is where the *retention* stops improving, and those numbers are
already in the same metrics files as old_val_loss_<domain> -- the sweep just
does not print them.

It also refuses to reproduce the summary's `lambda 反推` extrapolation, which is
only valid while R stays near R(lambda=0):

    lambda ~= target_ratio * task_loss / R(lambda=0)

R is the penalty evaluated on the current DeltaW, and the penalty's whole job is
to shrink DeltaW. On the finance mix line R(1e-2)/R(0) is 0.0011, so the
estimate is three orders of magnitude low and the suggested lambda does
essentially nothing. This script reports R/R(0) and says when the extrapolation
cannot be trusted.

Usage
-----
  python -m onereplay.scripts.analyze_lambda_sweep \
      --metrics_dir /scratch/.../results/qwen3-8b/metrics \
      --pattern 'fin_onereplay_mix-*_lam*_probe4800'
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Below this, a val_loss difference between two arms of one seed says nothing.
# The finance mix sweep shows why: lambda=1e-4, which barely penalizes anything,
# came out +0.0115 on val_loss while the ten-times-stronger 1e-3 came out
# -0.0075. Non-monotone at that scale means the scale is noise.
DEFAULT_VAL_NOISE = 0.010
# R/R(0) below this means the penalty has already collapsed DeltaW's energy, and
# any estimate that assumed R stays at R(0) is void.
EXTRAPOLATION_VALID_ABOVE = 0.5


@dataclass
class Point:
    """One lambda of the sweep."""

    lam: float
    name: str
    regularizer: str
    task_loss: float
    val_loss: float
    reg: float  # R
    lambda_reg: float  # lambda * R
    old_val: dict[str, float] = field(default_factory=dict)
    old_val_start: dict[str, float] = field(default_factory=dict)
    val_loss_start: float | None = None
    updates: int = 0

    @property
    def ratio(self) -> float:
        """lambda * R / task_loss, the penalty's share of the objective."""

        return self.lambda_reg / self.task_loss if self.task_loss else 0.0


def read_point(path: Path) -> Point | None:
    epochs: list[dict] = []
    baseline: dict = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("record_type")
            if kind == "epoch":
                epochs.append(row)
            elif kind == "baseline":
                baseline = row
    if not epochs:
        print(f"skip {path.name}: no epoch record")
        return None
    last = epochs[-1]
    old_val = {
        key[len("old_val_loss_"):]: float(value)
        for key, value in last.items()
        if key.startswith("old_val_loss_") and value is not None
    }
    if "old_val_loss" in last and last["old_val_loss"] is not None:
        old_val.setdefault("old", float(last["old_val_loss"]))
    start = {
        key[len("old_val_loss_"):]: float(value)
        for key, value in baseline.items()
        if key.startswith("old_val_loss_") and value is not None
    }
    if "old_val_loss" in baseline and baseline["old_val_loss"] is not None:
        start.setdefault("old", float(baseline["old_val_loss"]))
    return Point(
        lam=float(last.get("replay_lambda") or 0.0),
        name=path.stem,
        regularizer=str(last.get("regularizer") or "?"),
        task_loss=float(last.get("train_task_loss") or 0.0),
        val_loss=float(last.get("val_loss") or 0.0),
        reg=float(last.get("train_replay_reg") or 0.0),
        lambda_reg=float(last.get("train_lambda_reg") or 0.0),
        old_val=old_val,
        old_val_start=start,
        # The untrained checkpoint's own val_loss. Without it "d val is ~0 across
        # three decades of lambda" has two readings that look identical: every
        # lambda is free, or no arm is learning the new task in the first place.
        val_loss_start=(
            float(baseline["val_loss"]) if baseline.get("val_loss") is not None else None
        ),
        updates=int(last.get("total_updates") or 0),
    )


def cell(value, spec=".4f", width=11):
    return f"{'-' if value is None else format(value, spec):>{width}}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a lambda sweep's cost and benefit together")
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--pattern", default="*_lam*", help="Glob over run names inside --metrics_dir")
    parser.add_argument(
        "--val_noise",
        type=float,
        default=DEFAULT_VAL_NOISE,
        help=(
            "val_loss differences smaller than this are treated as noise. Estimate it from "
            "the sweep itself: the spread between two lambdas that are too weak to matter."
        ),
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    pattern = args.pattern if args.pattern.endswith(".jsonl") else f"{args.pattern}.jsonl"
    paths = sorted(metrics_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"nothing matched {metrics_dir / pattern}")
    points = [point for point in (read_point(path) for path in paths) if point is not None]
    if not points:
        raise SystemExit("every matched file was unusable")
    points.sort(key=lambda point: point.lam)

    zero = next((point for point in points if point.lam == 0.0), None)
    domains = sorted({name for point in points for name in point.old_val})
    print(f"{len(points)} point(s), regularizer={points[0].regularizer}, "
          f"protected domains={domains or ['(none recorded)']}")
    if zero is None:
        print(
            "!! no lambda=0 point in this glob. Without it there is no ruler for R/R(0) or for "
            "the deltas, and 'this lambda costs nothing' cannot be said at all. Re-run the "
            "sweep with 0 in the grid, or widen --pattern to include the vanilla probe."
        )
    if not domains:
        print(
            "!! no old_val_loss column in any of these files, so this sweep measured only what "
            "the penalty costs, never what it buys. Every lambda will look free and the table "
            "will seem to say 'take the largest'. Re-run with --old_val_jsonl / --prior_val_jsonl "
            "(95's OLD_VAL=1) before choosing."
        )

    print()
    print("---- cost and benefit together")
    header = f"{'lambda':>11}{'lamR/task':>11}{'R/R(0)':>10}{'d task':>11}{'d val':>11}"
    for domain in domains:
        header += f"{'old[' + domain + ']':>13}{'recovered':>11}"
    print(header)
    print("-" * len(header))
    for point in points:
        line = (
            f"{point.lam:>11.3e}{point.ratio:>11.4f}"
            + cell(point.reg / zero.reg if zero and zero.reg else None, ".4f", 10)
            + cell(point.task_loss - zero.task_loss if zero else None, "+.4f")
            + cell(point.val_loss - zero.val_loss if zero else None, "+.4f")
        )
        for domain in domains:
            value = point.old_val.get(domain)
            line += cell(value, ".4f", 13)
            # What fraction of vanilla's forgetting this lambda undid. 1.0 means
            # the old domain sits where the untouched model left it; 0 means it
            # forgot exactly as much as vanilla did.
            recovered = None
            start = point.old_val_start.get(domain)
            if zero is not None and value is not None and start is not None:
                forgotten = zero.old_val.get(domain, 0.0) - start
                if abs(forgotten) > 1e-9:
                    recovered = (zero.old_val[domain] - value) / forgotten
            line += cell(recovered, ".1%", 11)
        print(line)

    if zero is not None and zero.old_val_start:
        print()
        print("old_val at the untrained starting point (the 100%-recovered line): "
              + ", ".join(f"{name}={value:.4f}" for name, value in sorted(zero.old_val_start.items())))
        # recovered% divides by how much the unregularized run forgot. When that
        # is small the ratio is unstable, and when an arm lands below the starting
        # point the "fraction of forgetting undone" framing has been left behind
        # entirely. Both happen, and both silently make the column unreadable.
        for domain in domains:
            start = zero.old_val_start.get(domain)
            if start is None or domain not in zero.old_val:
                continue
            forgotten = zero.old_val[domain] - start
            best = min(point.old_val[domain] for point in points if domain in point.old_val)
            spread = zero.old_val[domain] - best
            if forgotten <= 0:
                print(f"  !! [{domain}] the unregularized run did not forget at all "
                      f"({zero.old_val[domain]:.4f} vs {start:.4f} at the start), so recovered% "
                      "has no denominator. Read the raw old_val column.")
            elif spread > forgotten:
                print(
                    f"  !! [{domain}] recovered% overshoots 100% because the best arm "
                    f"({best:.4f}) is *below* the starting checkpoint ({start:.4f}). That is no "
                    "longer 'retained some of what vanilla lost' -- the old domain improved "
                    "while the new one was learned. Report the raw loss and the transfer, not a "
                    "retention percentage."
                )
            elif forgotten < 0.1 * max(abs(start), 1e-9):
                print(
                    f"  !! [{domain}] vanilla only forgot {forgotten:.4f} from a base of "
                    f"{start:.4f}, so recovered% divides by a small number and swings wildly. "
                    "Compare the raw old_val column instead."
                )

    # ---- is the summary's extrapolation usable here ----
    print()
    print("---- can `lambda 反推` be trusted on this sweep?")
    if zero is None or not zero.reg:
        print("  cannot tell: no lambda=0 point to compare R against.")
    else:
        worst = min((point.reg / zero.reg for point in points if point.lam > 0), default=1.0)
        print(f"  smallest R/R(0) over the swept lambdas = {worst:.4f}")
        if worst < EXTRAPOLATION_VALID_ABOVE:
            print(
                f"  -> NO. The extrapolation lambda ~= target * task / R(0) assumes R stays near "
                f"R(0); here it falls to {worst:.4f} of it, so the suggested lambda is about "
                f"{1 / worst:.0f}x too small. Read the measured lamR/task column instead."
            )
        else:
            print("  -> yes, R barely moved, so the linear estimate is in the right decade.")

        # The gap between lambda=0 and the weakest swept lambda. R is quadratic in
        # DeltaW, so an unconstrained R(0) is a runaway value and a single decade
        # of lambda can cross most of the interesting range. If the first real
        # point already sits near zero, the grid never looked at the transition.
        weakest = min((point for point in points if point.lam > 0), key=lambda p: p.lam, default=None)
        if weakest is not None and zero.reg:
            first = weakest.reg / zero.reg
            if first < 0.05:
                print(
                    f"  !! the weakest lambda swept ({weakest.lam:.3e}) already collapses R to "
                    f"{first:.4f} of R(0). The whole transition from unconstrained to constrained "
                    f"happens below {weakest.lam:.3e}, where there is not a single measurement. "
                    "The low end is as unbracketed as the high end -- add points a decade or two "
                    "further down before concluding the curve is monotone."
                )

    # ---- where does the new task start paying ----
    # ---- did the new task move at all ----
    # This has to come before any statement about lambda being "free". If the
    # untrained checkpoint already sits at the same val_loss every arm reaches,
    # then d val ~= 0 everywhere means nothing is being learned, and the sweep
    # measures only how well each lambda preserves a model that never moved.
    print()
    print("---- did the new task actually get learned?")
    start_val = next((point.val_loss_start for point in points if point.val_loss_start), None)
    if start_val is None:
        print(
            "  unknown: no baseline val_loss in these files. Re-run the sweep with "
            "--eval_before_train 1 so every point records the untrained starting loss. "
            "Until then 'lambda is free' cannot be distinguished from 'nothing is learning'."
        )
    else:
        learned = start_val - min(point.val_loss for point in points)
        span = max(point.val_loss for point in points) - min(point.val_loss for point in points)
        print(f"  untrained val_loss = {start_val:.4f}; best arm = "
              f"{min(point.val_loss for point in points):.4f} (learned {learned:+.4f})")
        print(f"  spread across the swept lambdas = {span:.4f}")
        if learned <= 0:
            print(
                "  !! no arm beat the untrained checkpoint. The new task is not being learned, "
                "so nothing in this table is a lambda trade-off -- fix lr / steps / data first."
            )
        elif span < 0.1 * learned:
            print(
                f"  -> the new task moved {learned:.4f} while lambda moved val_loss only "
                f"{span:.4f} ({span / learned:.1%} of it). So lambda genuinely is cheap over this "
                "whole range, and the pick has to come from the retention column and the "
                "benchmark, not from val_loss."
            )
        else:
            print(f"  -> lambda moves val_loss by {span / learned:.1%} of what learning moved it, "
                  "so the trade-off is visible in val_loss and worth reading there.")

    print()
    print("---- reading it")
    if zero is not None:
        # One-sided on purpose. A lambda whose val_loss comes out *below* the
        # unregularized run has not charged anything for the new task -- the
        # penalty happened to also curb some overfitting. Treating that as a cost
        # (|d val| > noise) would mark the strongest lambda as expensive and hide
        # the fact that the grid never found an upper limit.
        free = [
            point for point in points
            if point.lam > 0 and (point.val_loss - zero.val_loss) <= args.val_noise
        ]
        hurt = [
            point for point in points
            if point.lam > 0 and (point.val_loss - zero.val_loss) > args.val_noise
        ]
        stiff = [
            point for point in points
            if point.lam > 0 and (point.task_loss - zero.task_loss) > args.val_noise
        ]
        print(f"  new task not charged (d val <= {args.val_noise:g}) at lambda = "
              + (", ".join(f"{point.lam:.3e}" for point in free) or "(none)"))
        if hurt:
            print("  new task measurably worse at lambda = "
                  + ", ".join(f"{point.lam:.3e}" for point in hurt))
        if stiff:
            print(f"  train loss visibly higher (d task > {args.val_noise:g}) at lambda = "
                  + ", ".join(f"{point.lam:.3e}" for point in stiff)
                  + "  <- plasticity is being spent even where val_loss still looks fine")

        # A weaker lambda cannot cost more than a stronger one. When it appears
        # to, the val_loss spread is noise and the threshold is too tight -- which
        # also means everything called "hurt" below that spread is unreadable.
        if hurt and free:
            strongest_free = max(point.lam for point in free)
            bogus = [point for point in hurt if point.lam < strongest_free]
            if bogus:
                implied = max(point.val_loss - zero.val_loss for point in bogus)
                print(
                    f"  !! lambda {', '.join(f'{p.lam:.3e}' for p in bogus)} looks worse than the "
                    f"stronger {strongest_free:.3e}, which cannot be a real penalty effect. "
                    f"The val_loss noise floor on one seed is at least {implied:.4f}, not "
                    f"{args.val_noise:g} -- re-read this section with "
                    f"--val_noise {implied * 1.5:.3f}, and do not call any lambda harmful on a "
                    "margin that small without a second seed."
                )

        if free and max(point.lam for point in free) == max(point.lam for point in points):
            print(
                "  !! the largest lambda swept is still free, so the upper end is NOT bracketed. "
                "The operating point cannot be both above the grid and inside it -- extend the "
                "grid upward (x3, x10) until something gives, or the pick is just 'the biggest "
                "one I happened to try'."
            )

    # A penalty that grows while R grows too is not a stronger constraint, it is
    # a diverging optimization. Worth calling out separately from "too strong".
    for earlier, later in zip(points, points[1:]):
        if later.lam > earlier.lam > 0 and later.reg > earlier.reg * 1.5:
            print(
                f"  !! R went UP from {earlier.reg:.3e} to {later.reg:.3e} between lambda "
                f"{earlier.lam:.3e} and {later.lam:.3e}. A stronger penalty should shrink "
                "DeltaW, so this point is diverging rather than merely over-regularized; "
                "treat it as broken and do not read its losses."
            )

    if domains and zero is not None:
        print()
        for domain in domains:
            usable = [point for point in points if point.lam > 0 and domain in point.old_val]
            if len(usable) < 2:
                continue
            best = min(usable, key=lambda point: point.old_val[domain])
            print(f"  [{domain}] best retention at lambda={best.lam:.3e} "
                  f"(old_val {best.old_val[domain]:.4f} vs vanilla {zero.old_val.get(domain, float('nan')):.4f})")
            # The knee: the smallest lambda already within 10% of the best gain,
            # which is the one to prefer because it keeps the most plasticity.
            gain_best = zero.old_val.get(domain, 0.0) - best.old_val[domain]
            if gain_best > 0:
                knee = next(
                    (
                        point for point in sorted(usable, key=lambda point: point.lam)
                        if (zero.old_val[domain] - point.old_val[domain]) >= 0.9 * gain_best
                    ),
                    best,
                )
                if knee.lam == max(point.lam for point in usable):
                    print(
                        f"  [{domain}] no knee inside the grid: retention is still improving at "
                        f"the largest lambda ({knee.lam:.3e}). That number is the grid edge, not "
                        "an operating point -- extend upward until retention flattens."
                    )
                else:
                    print(f"  [{domain}] knee at lambda={knee.lam:.3e}: within 10% of the best "
                          "retention while keeping more plasticity -- prefer this unless the "
                          "benchmark disagrees.")
    print()
    print("!! These are short-run numbers. DeltaW is still growing, so R keeps rising and "
          "task_loss keeps falling; the same lambda will sit at a higher lamR/task by the end "
          "of a full run. Confirm the pick with the benchmarks, not with loss alone.")


if __name__ == "__main__":
    main()
