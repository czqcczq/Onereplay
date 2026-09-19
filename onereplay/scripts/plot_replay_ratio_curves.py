"""Curves and a cost table for the replay-ratio sweep.

Reads the metrics jsonl the 96 script writes -- the record_type=probe rows for
the curves, the record_type=epoch rows for wall clock and peak memory -- and
answers the question the sweep exists for:

    how much extra data and compute does vanilla replay need before it
    matches what OneReplay gets for ~3%?

Five figures:

  old_val_curve       protected-domain loss against optimizer updates. The main
                      plot: does a larger ratio really hold the old domain
                      better, and by how much per unit of extra cost.
  new_val_curve       the new task on the same axis, which is what says whether
                      a large ratio bought retention by learning less.
  combined_own        (1 - w) * new + w * old with each arm's *own* w, i.e. the
                      objective that arm actually optimized.
  combined_fixed      the same with one w for every arm. Only this one may be
                      compared across arms; combined_own may not, because each
                      curve is measured against a different yardstick.
  efficiency          final protected-domain loss against measured cost, with
                      EWC and OneReplay as points. This is the figure the claim
                      is read off.

Arms are identified from the metrics fields rather than the file name, so a
renamed run still lands in the right series.

Usage:
  python -m onereplay.scripts.plot_replay_ratio_curves \
    --metrics_dir results/qwen3-8b/metrics --pattern 'dsratio_*' \
    --out_dir results_log/figs/ds_ratio
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # login nodes have no display
import matplotlib.pyplot as plt  # noqa: E402

# The probe curve names train.py writes. cs_val is the new task's validation
# split; the name predates DialogSum and is kept so older runs still parse.
NEW_CURVE = "probe_cs_val"
OLD_CURVE = "probe_old_val"


@dataclass
class Run:
    """One arm: its curves, its final numbers, and what it cost."""

    name: str
    path: Path
    arm: str  # vanilla | replay | ewc | onereplay
    ratio_r: float  # N_replay / N_new; 0 for the three baselines
    loss_weight: float  # replay's exact share of the gradient, or 0
    lam: float
    updates: list[int] = field(default_factory=list)
    old_curve: list[float] = field(default_factory=list)
    new_curve: list[float] = field(default_factory=list)
    train_sec: float = 0.0
    peak_memory_gb: float = 0.0
    resident_gb: float = 0.0
    final_val_loss: float | None = None
    final_old_val_loss: float | None = None
    new_per_update: int = 0
    replay_per_update: int = 0
    epochs: int = 0

    @property
    def label(self) -> str:
        if self.arm == "replay":
            return f"replay r={self.ratio_r:g} (w={self.loss_weight:.3f})"
        if self.arm == "vanilla":
            return "vanilla"
        return f"{self.arm} " + (f"\u03bb={self.lam:g}" if self.lam else "")

    @property
    def sort_key(self) -> tuple[int, float]:
        order = {"vanilla": 0, "replay": 1, "ewc": 2, "onereplay": 3}
        return order.get(self.arm, 9), self.ratio_r


def load_run(path: Path) -> Run | None:
    """Parse one metrics file into a Run, or None when it has no curve."""

    epochs: list[dict] = []
    probes: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("record_type")
            if kind == "epoch":
                epochs.append(row)
            elif kind == "probe":
                probes.append(row)

    if not epochs:
        print(f"skip {path.name}: no epoch record (the run did not finish an epoch)")
        return None
    if not probes:
        print(f"skip {path.name}: no probe record (this run was not given --probe_every_updates)")
        return None

    last = epochs[-1]
    lam = float(last.get("replay_lambda") or 0.0)
    replay_steps = int(last.get("replay_steps_per_update") or 0)
    replay_per_batch = int(last.get("replay_per_batch") or 0)
    if replay_steps > 0 or replay_per_batch > 0:
        arm = "replay"
    elif lam > 0:
        arm = str(last.get("regularizer") or "onereplay")
    else:
        arm = "vanilla"

    # replay_ratio_r was added with the step-level scheme; derive it for older
    # files so a batch-level run can still be dropped onto the same axes.
    ratio_r = last.get("replay_ratio_r")
    if ratio_r is None:
        new_per_batch = int(last.get("new_per_batch") or 0)
        ratio_r = replay_per_batch / new_per_batch if new_per_batch else 0.0
    loss_weight = last.get("replay_loss_weight")

    run = Run(
        name=path.stem,
        path=path,
        arm=arm,
        ratio_r=float(ratio_r or 0.0),
        # None means the scheme has no exact weight (batch-level mixing shares
        # one token average), so fall back to the row share and say so in the
        # table rather than silently plotting a number that is not the weight.
        loss_weight=float(loss_weight) if loss_weight is not None else 0.0,
        lam=lam,
        # Summed over epochs: train_sec is per-epoch and the comparison is over
        # the whole run. Probe time is already excluded by the trainer.
        train_sec=sum(float(row.get("train_sec") or 0.0) for row in epochs),
        peak_memory_gb=max(float(row.get("peak_memory_gb") or 0.0) for row in epochs),
        resident_gb=float(last.get("covariance_memory_gb") or 0.0),
        final_val_loss=last.get("val_loss"),
        final_old_val_loss=last.get("old_val_loss"),
        new_per_update=int(last.get("new_per_update") or 0),
        replay_per_update=int(last.get("replay_per_update") or 0),
        epochs=int(last.get("epoch") or 0),
    )
    if loss_weight is None and run.arm == "replay":
        run.loss_weight = run.ratio_r / (1 + run.ratio_r)

    for row in probes:
        if OLD_CURVE not in row or NEW_CURVE not in row:
            continue
        run.updates.append(int(row["update"]))
        run.old_curve.append(float(row[OLD_CURVE]))
        run.new_curve.append(float(row[NEW_CURVE]))
    if not run.updates:
        print(f"skip {path.name}: probe records carry neither {OLD_CURVE} nor {NEW_CURVE}")
        return None
    return run


def style_for(run: Run, index: int, replay_count: int):
    """Colour and line style: the sweep as a gradient, baselines as dashes."""

    if run.arm == "replay":
        shade = plt.cm.viridis(0.15 + 0.7 * (index / max(replay_count - 1, 1)))
        return {"color": shade, "linestyle": "-", "linewidth": 1.8}
    return {
        "vanilla": {"color": "black", "linestyle": "--", "linewidth": 2.0},
        "ewc": {"color": "tab:red", "linestyle": "-.", "linewidth": 2.0},
        "onereplay": {"color": "tab:blue", "linestyle": ":", "linewidth": 2.4},
    }.get(run.arm, {"color": "grey", "linestyle": "-"})


def plot_curves(runs: list[Run], values, title: str, ylabel: str, out_path: Path) -> None:
    figure, axes = plt.subplots(figsize=(8.5, 5.2))
    # Position in the ratio sweep decides the shade, so the gradient reads in
    # order of r. Keyed by name rather than by identity or equality: Run is a
    # dataclass, so two arms with the same numbers would compare equal.
    replay_names = [run.name for run in runs if run.arm == "replay"]
    for run in runs:
        index = replay_names.index(run.name) if run.name in replay_names else 0
        axes.plot(
            run.updates,
            values(run),
            label=run.label,
            **style_for(run, index, len(replay_names)),
        )
    axes.set_xlabel("optimizer updates (every arm puts the same new-task rows behind each)")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def plot_efficiency(runs: list[Run], out_path: Path, cost_by_time: bool) -> None:
    """Final protected-domain loss against what the arm cost to get it."""

    figure, axes = plt.subplots(figsize=(8.5, 5.2))
    vanilla = next((run for run in runs if run.arm == "vanilla"), None)

    def cost(run: Run) -> float:
        if cost_by_time and vanilla and vanilla.train_sec:
            return run.train_sec / vanilla.train_sec
        return 1.0 + run.ratio_r

    replay_runs = sorted(
        (run for run in runs if run.arm == "replay"), key=lambda run: run.ratio_r
    )
    if replay_runs:
        axes.plot(
            [cost(run) for run in replay_runs],
            [run.old_curve[-1] for run in replay_runs],
            "o-",
            color="tab:green",
            label="vanilla replay (ratio sweep)",
        )
        for run in replay_runs:
            axes.annotate(
                f"r={run.ratio_r:g}",
                (cost(run), run.old_curve[-1]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )

    for run in runs:
        if run.arm == "replay":
            continue
        style = style_for(run, 0, 1)
        axes.scatter(
            cost(run), run.old_curve[-1], marker="*", s=220,
            color=style["color"], zorder=5, label=run.label,
        )
        # The horizontal line is what makes the figure readable: where it cuts
        # the replay curve is the answer to "how much compute to match this".
        axes.axhline(
            run.old_curve[-1], color=style["color"], linestyle=style["linestyle"],
            linewidth=1.0, alpha=0.5,
        )

    axes.set_xlabel(
        "cost relative to vanilla ("
        + ("measured train_sec" if cost_by_time and vanilla else "1 + r, i.e. micro-batches")
        + ")"
    )
    axes.set_ylabel("protected-domain loss at the end of training (lower is better)")
    axes.set_title("How much extra compute does vanilla replay need to match the penalties?")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def write_table(runs: list[Run], out_path: Path, w_eval: float, pool_rows: int) -> None:
    vanilla = next((run for run in runs if run.arm == "vanilla"), None)
    lines = [
        "# replay ratio sweep",
        "",
        "Three ratio conventions, do not mix them:",
        "",
        "- `r = N_replay / N_new`, which is also the extra cost: the run takes (1 + r)",
        "  times the micro-batches of vanilla.",
        "- `w = r / (1 + r)` is replay's exact share of the accumulated gradient, which",
        "  under step-level mixing does not depend on answer length.",
        "- `pool seen` is what fraction of the replay pool the arm ever looked at. A small",
        "  r is two things at once -- less rehearsal *and* less of the pool -- so the sweep",
        "  is not a pure rehearsal-strength effect.",
        "",
        "`old_val` is token-weighted cross-entropy on FLAN rows behind the replay pool;",
        "`new_val` is the DialogSum validation split. Both come from the probe records, so",
        "they are on the same footing across arms. Only differences are meaningful.",
        "",
        "| arm | r | w | pool seen | old_val end | old_val delta | new_val end | train_sec | vs vanilla | peak GB | resident GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        consumed = run.replay_per_update * max(run.epochs, 1) * _updates_per_epoch(run)
        seen = f"{min(consumed / pool_rows, 1.0):.1%}" if pool_rows and consumed else "-"
        speed = (
            f"{run.train_sec / vanilla.train_sec:.2f}x"
            if vanilla and vanilla.train_sec
            else "-"
        )
        delta = run.old_curve[-1] - run.old_curve[0]
        lines.append(
            f"| {run.label} | {run.ratio_r:g} | {run.loss_weight:.4f} | {seen} | "
            f"{run.old_curve[-1]:.6f} | {delta:+.6f} | {run.new_curve[-1]:.6f} | "
            f"{run.train_sec:.0f} | {speed} | {run.peak_memory_gb:.2f} | {run.resident_gb:.2f} |"
        )

    anchors = {round(run.old_curve[0], 6) for run in runs}
    lines += [
        "",
        f"Combined loss is reported at a single w_eval = {w_eval:g} in "
        "`combined_fixed.png`. The other figure uses each arm's own w, which is the",
        "objective it optimized and therefore cannot be compared across arms.",
        "",
    ]
    if len(anchors) == 1:
        lines.append(
            f"Sanity check passed: every arm starts from the same probe anchor "
            f"({anchors.pop():.6f}). The adapter is zero-initialized, so all arms begin at "
            "the base model and the curves are directly subtractable."
        )
    else:
        lines.append(
            f"!! The arms do not share a probe anchor ({sorted(anchors)}). They are being "
            "scored on different rows or with a different budget, and the curves are not "
            "subtractable. Check --old_val_* against the replay flags."
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print()
    print("\n".join(lines))


def _updates_per_epoch(run: Run) -> int:
    """Recover updates per epoch from the probe axis; 0 when only one point."""

    if run.epochs <= 0 or not run.updates:
        return 0
    return run.updates[-1] // run.epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the replay ratio sweep")
    parser.add_argument("--metrics_dir", type=str, required=True)
    parser.add_argument(
        "--pattern",
        type=str,
        default="dsratio_*",
        help="Glob inside --metrics_dir; '.jsonl' is appended when absent.",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--w_eval",
        type=float,
        default=0.5,
        help=(
            "The single replay weight every arm is scored at in combined_fixed.png. "
            "Each arm optimized its own w, so only a shared one compares them."
        ),
    )
    parser.add_argument(
        "--pool_rows",
        type=int,
        default=20000,
        help="Replay pool size, for the 'pool seen' column. 0 leaves it blank.",
    )
    parser.add_argument(
        "--cost_by_time",
        type=int,
        default=1,
        help=(
            "1 puts measured train_sec on the efficiency plot's x axis, which is the "
            "only cost axis the penalty arms also live on. 0 uses 1 + r, which is exact "
            "but defined for the replay arms alone."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    pattern = args.pattern if args.pattern.endswith(".jsonl") else f"{args.pattern}.jsonl"
    paths = sorted(metrics_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"no metrics file matched {metrics_dir / pattern}")

    runs = [run for run in (load_run(path) for path in paths) if run is not None]
    if not runs:
        raise SystemExit("every matched file was skipped; see the reasons above")
    runs.sort(key=lambda run: run.sort_key)
    print(f"loaded {len(runs)} run(s): {', '.join(run.label for run in runs)}")

    # Curves that do not share an x axis cannot be read together, and the whole
    # design of the sweep was to make them share one. Say so loudly rather than
    # drawing a plot that looks fine.
    axes_seen = {tuple(run.updates) for run in runs}
    if len(axes_seen) > 1:
        print(
            "warning: the arms have different probe axes, so the curves are not "
            "pointwise comparable. Lengths: "
            + ", ".join(f"{run.label}={len(run.updates)}" for run in runs)
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_curves(
        runs,
        lambda run: run.old_curve,
        "Protected domain (FLAN rows behind the replay pool)",
        "token-weighted cross-entropy (lower = better retained)",
        out_dir / "old_val_curve.png",
    )
    plot_curves(
        runs,
        lambda run: run.new_curve,
        "New task (DialogSum validation split)",
        "token-weighted cross-entropy (lower = better learned)",
        out_dir / "new_val_curve.png",
    )
    plot_curves(
        runs,
        lambda run: [
            (1 - run.loss_weight) * new + run.loss_weight * old
            for new, old in zip(run.new_curve, run.old_curve)
        ],
        "Combined loss at each arm's own w -- NOT comparable across arms",
        "(1 - w) * new + w * old, w from the arm itself",
        out_dir / "combined_own.png",
    )
    plot_curves(
        runs,
        lambda run: [
            (1 - args.w_eval) * new + args.w_eval * old
            for new, old in zip(run.new_curve, run.old_curve)
        ],
        f"Combined loss at a shared w_eval = {args.w_eval:g}",
        f"({1 - args.w_eval:g}) * new + ({args.w_eval:g}) * old",
        out_dir / "combined_fixed.png",
    )
    plot_efficiency(runs, out_dir / "efficiency.png", bool(args.cost_by_time))
    write_table(runs, out_dir / "summary.md", args.w_eval, args.pool_rows)


if __name__ == "__main__":
    main()
