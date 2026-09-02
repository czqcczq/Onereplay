"""Is the penalty spread over the layers it covers, or owned by one of them?

The OneReplay penalty is a plain sum over layers, rescaled by their count:

    R = (1 / L) * sum_l tr(DeltaW_l C_l DeltaW_l^T)

Nothing in that expression equalizes the layers. Each C_l carries the units of
its own input activations, so a layer whose hidden states run an order of
magnitude larger contributes two orders of magnitude more penalty. Past some gap
the sum stops being a penalty on L layers and becomes a penalty on one, with the
rest as spectators, and no downstream metric would report that: R still has a
sensible value, lambda still tunes it, the loss curve still moves.

The gap is not hypothetical once full fine-tuning is on the table. The LoRA runs
covered q_proj and v_proj only, whose inputs are post-LayerNorm residual states
of comparable scale across depth. Full fine-tuning adds down_proj, whose input is
the MLP intermediate activation, and that is exactly where massive activations
live: a few coordinates in the early layers run hundreds of times larger than
everything else. C is a second moment, so it squares whatever x does.

Measuring this needs no DeltaW. For a DeltaW whose entries are i.i.d. with
variance s^2,

    E[tr(DeltaW C DeltaW^T)] = s^2 * d_out * tr(C)

so under the update Adam actually produces -- per-element steps of comparable
size across layers -- a layer's expected share of the penalty is proportional to
d_out * tr(C). d_out needs the model; without it the script falls back to tr(C)
alone, which is enough because d_out spans a factor of 6 across the projections
while the effect being looked for spans several orders of magnitude.

Two numbers answer two different questions:

  share       how much of the scalar R a layer accounts for. A layer at 99%
              means lambda was tuned against that layer and nothing else.
  mean_diag   tr(C_l) / d_in_l, the penalty per unit of update energy. The
              injected gradient is 2 * (lambda / L) * DeltaW_l C_l, so this is
              what sets how hard a layer is actually held in place. Its spread
              across layers is the spread in protection strength.

Usage
  # the whole diagnosis, CPU, needs enough RAM to hold the C file (~6.6 GiB full)
  python -m onereplay.scripts.inspect_cov_layers \
      --cov_path .../cov/cov_flan_chat_20k_full.pt

  # exact d_out weighting, and the tie_word_embeddings check
  python -m onereplay.scripts.inspect_cov_layers \
      --cov_path ... --base_model_dir .../models/Qwen3-1.7B

  # also weight by the DeltaW a real run produced
  python -m onereplay.scripts.inspect_cov_layers \
      --cov_path ... --base_model_dir .../models/Qwen3-1.7B \
      --trained_model_dir .../checkpoints/full/regimpl_A_step3000_seed1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from onereplay.core.covariance import load_covariance_file  # noqa: E402
from onereplay.core.regularizer import lookup_covariance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the OneReplay penalty is dominated by a few layers."
    )
    parser.add_argument("--cov_path", type=str, required=True)
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="",
        help=(
            "Full path to the base checkpoint. Optional: it supplies d_out per layer "
            "for the exact contribution weighting, and reports tie_word_embeddings."
        ),
    )
    parser.add_argument(
        "--trained_model_dir",
        type=str,
        default="",
        help=(
            "A finished full fine-tune. With it the script also reports the split "
            "under the real DeltaW instead of an isotropic one. Needs --base_model_dir."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cuda makes the real-DeltaW pass much faster; the C-only pass is seconds either way.",
    )
    parser.add_argument(
        "--power_iterations",
        type=int,
        default=30,
        help=(
            "Power-iteration steps for lambda_max. The measured spectra are very "
            "top-heavy (top-1 around a third of the trace), so this converges quickly."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many of the largest layers to list. --full 1 lists every layer.",
    )
    parser.add_argument("--full", type=int, default=0)
    parser.add_argument(
        "--dominance_fail",
        type=float,
        default=0.5,
        help="Fail when the single largest layer owns at least this share of R.",
    )
    parser.add_argument(
        "--dominance_warn",
        type=float,
        default=10.0,
        help="Warn when the largest layer's share exceeds this multiple of an even split.",
    )
    parser.add_argument("--out_json", type=str, default="")
    return parser.parse_args()


def top_eigenvalue(matrix: torch.Tensor, iterations: int, generator: torch.Generator) -> float:
    """Largest eigenvalue of a symmetric PSD matrix, by power iteration.

    torch.linalg.eigvalsh on a 6144 x 6144 matrix costs minutes of CPU once it is
    repeated over 197 layers, and returns 6143 numbers nobody reads. Only the
    leading value is needed here, to separate a C whose mass sits on one direction
    from one that is merely large everywhere.
    """

    dimension = matrix.shape[-1]
    vector = torch.randn(dimension, generator=generator, dtype=torch.float32)
    vector = vector / vector.norm()
    value = 0.0
    for _ in range(iterations):
        product = matrix @ vector
        norm = float(product.norm())
        if norm == 0.0:
            return 0.0
        vector = product / norm
        value = float(vector @ (matrix @ vector))
    return value


def require_local_model_dir(path: str, flag: str) -> None:
    """Fail on a missing local directory instead of letting transformers reach the Hub.

    An unset ${MODEL_DIR} expands to an empty string, so the path becomes
    "/Qwen3-1.7B", which from_pretrained treats as a Hub repo id and reports three
    frames deep as an HFValidationError about alphanumeric characters, never
    naming the flag that was wrong. The pbs scripts set these variables
    themselves, so an interactive session is exactly where this happens.
    """

    if not path:
        raise SystemExit(f"{flag} is empty; pass an absolute path to the model directory")
    if not (Path(path) / "config.json").is_file():
        raise SystemExit(
            f"{flag}={path!r} is not a local model directory (no config.json inside).\n"
            "If you used ${MODEL_DIR}, note that it is set inside the pbs scripts rather\n"
            "than in an interactive shell, so it expands to nothing here."
        )


def check_local_model_dir(path: str, flag: str) -> None:
    """Reject a path that is not a local checkpoint, before transformers sees it.

    from_pretrained falls back to treating its argument as a Hub repo id, so an
    unset ${MODEL_DIR} expanding to an empty string arrives as "/Qwen3-1.7B" and
    surfaces as an HFValidationError about alphanumeric characters, three frames
    deep and without naming the flag that was wrong. The shell variables the pbs
    scripts define do not exist in an interactive session, which makes that the
    likeliest way to call this script incorrectly.
    """

    if not path:
        raise SystemExit(f"{flag} is empty; pass an absolute path to the checkpoint")
    if not (Path(path) / "config.json").is_file():
        raise SystemExit(
            f"{flag}={path!r} is not a local model directory (no config.json inside).\n"
            "If the path looks truncated, an unset shell variable expanded to an empty\n"
            "string: MODEL_DIR and friends are set inside the pbs scripts, not in an\n"
            "interactive shell."
        )


def load_linear_shapes(model_dir: str) -> tuple[dict[str, int], bool | None]:
    """Per-layer d_out and the weight-tying flag, without keeping the model.

    Tying matters here for a reason unrelated to shapes: when lm_head shares
    storage with embed_tokens, training lm_head also moves the embedding, but the
    penalty only ever sees lm_head's input covariance. The embedding side of that
    shared tensor is then completely unconstrained, which is worth knowing before
    reading any coverage claim.
    """

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, local_files_only=True
    )
    out_features = {
        name: int(module.out_features)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    tied = getattr(model.config, "tie_word_embeddings", None)
    del model
    return out_features, tied


def load_linear_weights(model_dir: str) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, local_files_only=True
    )
    weights = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    del model
    return weights


def collect_layer_stats(
    covariances: dict[str, torch.Tensor],
    out_features: dict[str, int],
    iterations: int,
    seed: int,
) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    rows: list[dict] = []
    for name in sorted(covariances):
        covariance = covariances[name].float()
        dimension = int(covariance.shape[-1])
        trace = float(torch.diagonal(covariance).sum())
        rows.append(
            {
                "module": name,
                "kind": name.rsplit(".", 1)[-1],
                "d_in": dimension,
                "d_out": int(out_features.get(name, 0)),
                "trace": trace,
                "mean_diag": trace / max(dimension, 1),
                "lambda_max": top_eigenvalue(covariance, iterations, generator),
            }
        )
    return rows


def add_shares(rows: list[dict], use_d_out: bool, key: str = "share") -> None:
    """Expected share of R under an isotropic DeltaW, per the identity in the header."""

    for row in rows:
        weight = float(row["d_out"]) if use_d_out and row["d_out"] else 1.0
        row["contribution"] = weight * row["trace"]
    total = sum(row["contribution"] for row in rows) or 1.0
    for row in rows:
        row[key] = row["contribution"] / total


def real_delta_shares(
    base_dir: str,
    trained_dir: str,
    covariances: dict[str, torch.Tensor],
    device: str,
) -> dict[str, float]:
    """Per-layer tr(DeltaW C DeltaW^T) under the update a real run produced.

    The isotropic estimate above answers "which layer would dominate", this one
    answers "which layer did". They can differ: training moves DeltaW away from
    the directions the penalty charges most for, which is the penalty working as
    intended and also a bias that flatters whichever layer is being penalized
    hardest. Read them together, not either alone.
    """

    base = load_linear_weights(base_dir)
    trained = load_linear_weights(trained_dir)

    regs: dict[str, float] = {}
    for name, base_weight in base.items():
        trained_weight = trained.get(name)
        if trained_weight is None:
            continue
        _, covariance = lookup_covariance(covariances, name)
        if covariance is None:
            continue
        delta = (trained_weight.float() - base_weight.float()).to(device)
        matrix = covariance.float().to(device)
        regs[name] = float(((delta @ matrix) * delta).sum())
        del delta, matrix

    del base, trained
    return regs


def format_row(row: dict, share_keys: tuple[str, ...]) -> str:
    shares = "".join(f"{100.0 * row.get(key, 0.0):>10.4f}" for key in share_keys)
    return (
        f"  {row['module']:<40} {row['d_in']:>6} {row['trace']:>12.4e} "
        f"{row['mean_diag']:>12.4e} {row['lambda_max']:>12.4e}{shares}"
    )


def print_table(rows: list[dict], share_keys: tuple[str, ...], title: str) -> None:
    header = "".join(f"{key:>10}" for key in share_keys)
    print(f"\n==== {title} ====")
    print(
        f"  {'module':<40} {'d_in':>6} {'trace':>12} "
        f"{'mean_diag':>12} {'lambda_max':>12}{header}"
    )
    for row in rows:
        print(format_row(row, share_keys))


def print_by_kind(rows: list[dict]) -> None:
    """Group by projection name, because the suspicion is about a projection type.

    down_proj is the one whose input is the MLP intermediate activation rather
    than a residual-stream state, and it is also the one whose C is 9x larger per
    layer under a 3x expansion ratio. If the spread is structural rather than a
    single bad layer, it shows up here.
    """

    kinds: dict[str, list[dict]] = {}
    for row in rows:
        kinds.setdefault(row["kind"], []).append(row)

    print("\n==== by projection type ====")
    print(
        f"  {'kind':<14} {'layers':>7} {'sum share':>11} {'median mean_diag':>18} "
        f"{'max mean_diag':>15} {'max/median':>12}"
    )
    for kind in sorted(kinds, key=lambda name: -sum(r["share"] for r in kinds[name])):
        group = kinds[kind]
        diagonals = sorted(row["mean_diag"] for row in group)
        median = diagonals[len(diagonals) // 2]
        largest = diagonals[-1]
        print(
            f"  {kind:<14} {len(group):>7} {100.0 * sum(r['share'] for r in group):>10.4f}% "
            f"{median:>18.4e} {largest:>15.4e} {largest / max(median, 1e-30):>12.1f}x"
        )


def report(level: str, name: str, detail: str) -> None:
    print(f"[{level:<4}] {name}: {detail}")


def verdict(rows: list[dict], args: argparse.Namespace) -> str:
    """Print the summary and return "ok", "concentrated", or "dominated".

    Three states rather than a boolean because the middle one is the interesting
    answer. A largest share of, say, 40% is under any reasonable fail bar and is
    still 80x an even split over 197 layers, which is not a penalty on 197 layers.
    Collapsing that into "passed" would report the opposite of what it means.
    """

    layers = len(rows)
    even = 1.0 / max(layers, 1)
    ordered = sorted(rows, key=lambda row: -row["share"])
    top1 = ordered[0]
    top5_share = sum(row["share"] for row in ordered[:5])
    tail_share = 1.0 - top1["share"]

    diagonals = sorted(row["mean_diag"] for row in rows)
    median_diag = diagonals[len(diagonals) // 2]
    spread = diagonals[-1] / max(diagonals[0], 1e-30)

    print("\n==== verdict ====")
    print(f"       layers covered            : {layers} (an even split is {100.0 * even:.3f}% each)")
    print(
        f"       largest layer             : {top1['module']} at {100.0 * top1['share']:.4f}% "
        f"({top1['share'] / even:.1f}x an even split)"
    )
    print(f"       top-5 layers together     : {100.0 * top5_share:.4f}%")
    print(f"       all other {layers - 1:>3} layers      : {100.0 * tail_share:.4f}%")
    print(
        f"       mean_diag spread          : {spread:.3e}x from smallest to largest layer, "
        f"median {median_diag:.4e}"
    )

    dominated = top1["share"] >= args.dominance_fail
    concentrated = top1["share"] >= args.dominance_warn * even

    report(
        "FAIL" if dominated else "OK  ",
        "no single layer owns the penalty",
        f"largest share {100.0 * top1['share']:.4f}% against a "
        f"{100.0 * args.dominance_fail:.0f}% bar",
    )
    report(
        "WARN" if concentrated else "OK  ",
        "penalty is spread evenly enough",
        f"largest share is {top1['share'] / even:.1f}x an even split, "
        f"bar is {args.dominance_warn:.0f}x",
    )

    if dominated or concentrated:
        print(
            "\n       What this means: lambda was tuned against the layers at the top of\n"
            "       that list, and the layers at the bottom receive a penalty gradient\n"
            "       smaller by the same factor, so the regularizer barely reaches them.\n"
            "       R, lambda*R/task_loss and the loss curves all stay well-behaved while\n"
            "       this is true, so none of them can rule it out.\n"
            "\n       Two fixes, both already reachable from the current code:\n"
            "         1. collect_cov.py --cov_normalization base_output_norm, which\n"
            "            estimates E[(x/||Wx||)(x/||Wx||)^T] so the penalty measures\n"
            "            relative output perturbation and the layer scales cancel.\n"
            "            Needs a re-collection, and lambda has to be re-calibrated.\n"
            "         2. Divide each C_l by its own mean diagonal at load time, which\n"
            "            equalizes the contributions above by construction and needs no\n"
            "            re-collection, but changes what the penalty means.\n"
            "       Re-run this script after either one; the largest share should land\n"
            f"       near {100.0 * even:.3f}%."
        )
    if dominated:
        return "dominated"
    return "concentrated" if concentrated else "ok"


def main() -> None:
    args = parse_args()

    print(f"==== covariance file ====\n       {args.cov_path}")
    covariances = load_covariance_file(args.cov_path)
    resident = sum(value.numel() * value.element_size() for value in covariances.values())
    print(f"       {len(covariances)} matrices, {resident / 1024**3:.3f} GiB resident")

    out_features: dict[str, int] = {}
    tied: bool | None = None
    if args.base_model_dir:
        check_local_model_dir(args.base_model_dir, "--base_model_dir")
        if args.trained_model_dir:
            check_local_model_dir(args.trained_model_dir, "--trained_model_dir")
        out_features, tied = load_linear_shapes(args.base_model_dir)
        covered = [name for name in covariances if name in out_features]
        print(
            f"       matched d_out for {len(covered)} of {len(covariances)} matrices; "
            f"tie_word_embeddings={tied}"
        )
        if tied:
            print(
                "       note: lm_head shares storage with embed_tokens, so training it also\n"
                "       moves the embedding while the penalty only sees lm_head's input C."
            )
    else:
        print("       no --base_model_dir: weighting by tr(C) alone, d_out is not applied")

    rows = collect_layer_stats(covariances, out_features, args.power_iterations, args.seed)
    use_d_out = bool(out_features) and all(row["d_out"] for row in rows)
    add_shares(rows, use_d_out=use_d_out)
    share_keys: tuple[str, ...] = ("share",)

    if args.trained_model_dir:
        if not args.base_model_dir:
            raise SystemExit("--trained_model_dir needs --base_model_dir")
        regs = real_delta_shares(
            args.base_model_dir, args.trained_model_dir, covariances, args.device
        )
        total = sum(regs.values()) or 1.0
        for row in rows:
            row["real_share"] = regs.get(row["module"], 0.0) / total
        share_keys = ("share", "real_share")
        print(
            f"       real DeltaW from {args.trained_model_dir}, "
            f"{len(regs)} layers, sum tr(dW C dW^T)={total:.6e}"
        )

    ordered = sorted(rows, key=lambda row: -row["share"])
    if args.full == 1:
        print_table(ordered, share_keys, f"every layer, by share ({len(ordered)})")
    else:
        print_table(ordered[: args.top], share_keys, f"largest {min(args.top, len(ordered))} layers")
        print_table(ordered[-5:], share_keys, "smallest 5 layers")

    print_by_kind(rows)
    status = verdict(rows, args)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cov_path": args.cov_path,
            "layers": len(rows),
            "weighted_by_d_out": use_d_out,
            "tie_word_embeddings": tied,
            "rows": ordered,
        }
        Path(args.out_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.out_json}")

    # Printed rather than raised with a message, so the line lands after the report
    # on stdout instead of ahead of it on stderr.
    if status == "dominated":
        print(
            "\nPENALTY IS LAYER-DOMINATED: see the note above before trusting any "
            "full fine-tune result",
            flush=True,
        )
        raise SystemExit(1)
    if status == "concentrated":
        print(
            "\nPENALTY IS CONCENTRATED: under the fail bar, but far from an even split. "
            "Read the shares above before treating this as full-model protection."
        )
        return
    print("\nALL CHECKS PASSED: the penalty is spread across the layers it covers")


if __name__ == "__main__":
    main()
