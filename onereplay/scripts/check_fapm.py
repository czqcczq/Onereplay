"""Correctness self-check for the FAPM baseline.

Run this once before spending cluster time on a FAPM arm. It answers the only
question that matters about a transcribed baseline -- is the arm labelled FAPM
actually computing FAPM -- and it answers it against the authors' own code
rather than against our reading of the paper.

  1. upstream   fapm_prune agrees entry-for-entry with a literal transcription
                of baseline/FAPM/FAPM.py, over many random matrices and several
                keep ratios. This is the decisive test: it compares against
                their arithmetic, not against a restatement of it.
  2. budget     exactly int(numel * keep_ratio) entries survive, including when
                the surviving scores are negative. Upstream takes the top-k
                unconditionally and so must we.
  3. zero_base  characterizes the one deviation. Where W0 == 0 and dW != 0,
                upstream's score is already -inf and the two agree; where W0 and
                dW are both zero, upstream produces a nan that sorts first and
                burns that many slots of the budget, and we do not.
  4. endpoints  keep_ratio 1.0 reproduces the merged adapter and 0.0 reproduces
                the base model. These bracket every other ratio, so a bug in the
                selection, the write-back or the save has to show up in one of
                them.
  5. loadable   the written checkpoint is not mistaken for an adapter and comes
                back through eval/runner.py's full-checkpoint branch, which is
                the path the PBS script will evaluate it on.

Checks 1-3 are pure arithmetic and need no checkpoint; 4-5 run the real CLI as
a subprocess on a throwaway adapter. Use the smallest model of the family you
plan to prune: check 4 writes two full checkpoints into a temporary directory,
so it costs twice the checkpoint in scratch space and two model loads.

Usage:
    python -m onereplay.scripts.check_fapm \
        --model_dir /home/weiliu1/huggingface/models/ --model_name Qwen3-1.7B

    python -m onereplay.scripts.check_fapm --skip_model 1   # checks 1-3 only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from onereplay.core.modeling import (  # noqa: E402
    build_lora_model,
    is_lora_adapter_dir,
    join_model_path,
    load_causal_lm_and_tokenizer,
    set_seed,
)
from onereplay.scripts.apply_fapm import fapm_prune  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FAPM baseline self-check")
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--skip_model",
        type=int,
        default=0,
        help="Run only the arithmetic checks, which need no checkpoint.",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="",
        help="Where check 4 writes its two throwaway checkpoints. Defaults to "
        "the system temp directory, which on a cluster node is often too small "
        "for a full model.",
    )
    return parser.parse_args()


def upstream_prune(
    base: torch.Tensor,
    delta: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    """baseline/FAPM/FAPM.py's five load-bearing lines, applied to one matrix.

    Transcribed rather than imported: their script is a top-level loop over a
    state_dict with hard-coded paths, so there is no function to call. Keeping
    the transcription here, next to the test that uses it, is what makes the
    comparison auditable.
    """

    tensor = delta.clone()
    k = int(tensor.numel() * ratio)
    lamda = 1.0 * base.abs().mean()
    t = tensor.abs() - lamda * (tensor.abs() / base.abs())
    indices = torch.argsort(t.view(-1), descending=True)[:k]
    mask = torch.zeros_like(tensor)
    mask.view(-1)[indices] = 1
    tensor.mul_(mask)
    return tensor


def check_upstream_agreement(trials: int = 20) -> bool:
    """Check 1: identical output to upstream wherever upstream is well defined."""

    ratios = (0.0, 0.05, 0.1, 0.37, 1.0)
    mismatches = []
    for trial in range(trials):
        base = torch.randn(64, 96)
        delta = torch.randn(64, 96) * 0.02
        for ratio in ratios:
            ours, _ = fapm_prune(base, delta, ratio)
            if not torch.equal(ours, upstream_prune(base, delta, ratio)):
                mismatches.append((trial, ratio))
    return record(
        "upstream",
        not mismatches,
        f"{trials} matrices x {len(ratios)} ratios"
        if not mismatches
        else f"disagrees at {mismatches[:5]}",
    )


def check_budget() -> bool:
    """Check 2: the budget is truncated, spent in full, and spent on negatives."""

    base = torch.randn(40, 50)
    delta = torch.randn(40, 50) * 0.02
    problems = []
    for ratio in (0.03, 0.1, 0.5, 0.97):
        pruned, stats = fapm_prune(base, delta, ratio)
        expected = int(delta.numel() * ratio)
        if stats["kept"] != expected:
            problems.append(f"ratio {ratio}: budget {stats['kept']} != int() {expected}")
        if int((pruned != 0).sum()) != expected:
            problems.append(f"ratio {ratio}: {int((pruned != 0).sum())} survivors, want {expected}")

    # The score is |dW| * (1 - mean(|W0|)/|W0|), so every coordinate sitting on a
    # below-average weight scores negative. Upstream keeps them anyway once the
    # budget is large enough, and a "only keep positive scores" reading of the
    # paper would quietly keep fewer.
    magnitude = base.abs()
    score = delta.abs() - magnitude.mean() * (delta.abs() / magnitude)
    pruned, stats = fapm_prune(base, delta, 0.97)
    kept_scores = score[pruned != 0]
    if not bool((kept_scores < 0).any()):
        problems.append("no negative-score coordinate survived a 0.97 budget")
    return record(
        "budget",
        not problems,
        "int() truncation, full spend, negatives kept" if not problems else "; ".join(problems),
    )


def check_zero_base() -> bool:
    """Check 3: pin down the single deviation from upstream."""

    problems = []

    # W0 == 0 with dW != 0: upstream's own arithmetic gives -inf, so we must
    # agree with it rather than merely happen to exclude the coordinate.
    base = torch.randn(8, 8)
    base[0, 0] = 0.0
    delta = torch.randn(8, 8) * 0.01
    ours, stats = fapm_prune(base, delta, 0.5)
    if not torch.equal(ours, upstream_prune(base, delta, 0.5)):
        problems.append("dW != 0 on a zero weight: should still agree with upstream")
    if stats["zero_base"] != 1:
        problems.append(f"counted {stats['zero_base']} zero-weight coordinates, want 1")

    # W0 == 0 and dW == 0: upstream's nan sorts first and eats the budget.
    zeros = 40
    base = torch.randn(64, 64)
    delta = torch.randn(64, 64) * 0.01
    base.view(-1)[:zeros] = 0.0
    delta.view(-1)[:zeros] = 0.0
    ours, stats = fapm_prune(base, delta, 0.1)
    theirs = upstream_prune(base, delta, 0.1)
    ours_kept = int((ours != 0).sum())
    theirs_kept = int((theirs != 0).sum())
    if ours_kept != stats["kept"]:
        problems.append(f"we kept {ours_kept} real coordinates, want the full {stats['kept']}")
    if theirs_kept != stats["kept"] - zeros:
        problems.append(
            f"upstream kept {theirs_kept} real coordinates; the nan-eats-the-budget "
            f"effect this check exists to document did not reproduce"
        )
    return record(
        "zero_base",
        not problems,
        f"agrees except on dW == 0, where upstream loses {zeros} slots"
        if not problems
        else "; ".join(problems),
    )


def build_throwaway_adapter(args: argparse.Namespace, directory: Path) -> Path:
    """A LoRA adapter with a nonzero task vector, saved where the CLI can read it.

    lora_B is zero at initialization, which would make dW identically zero and
    every end-point check vacuously true. Randomizing it is what gives the
    checks something to be wrong about.
    """

    model, _ = load_causal_lm_and_tokenizer(args.model_dir, args.model_name, args.use_bf16)
    peft_model = build_lora_model(
        model,
        args.target_modules.split(","),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
    )
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0.0, 0.01)
    adapter_path = directory / "adapter"
    peft_model.save_pretrained(str(adapter_path))
    del peft_model, model
    return adapter_path


def run_cli(args: argparse.Namespace, adapter_path: Path, ratio: float, out: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "onereplay.scripts.apply_fapm",
            "--model_dir", args.model_dir,
            "--model_name", args.model_name,
            "--use_bf16", str(args.use_bf16),
            "--adapter_path", str(adapter_path),
            "--keep_ratio", str(ratio),
            "--device", "cpu",
            "--save", "1",
            "--save_path", str(out),
        ],
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


def check_endpoints(args: argparse.Namespace) -> bool:
    """Checks 4 and 5: the two ratios whose answer is known independently."""

    from onereplay.eval.runner import load_eval_model

    base_path = join_model_path(args.model_dir, args.model_name)
    root = Path(args.tmp_dir) if args.tmp_dir else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(root) if root else None) as scratch:
        directory = Path(scratch)
        adapter_path = build_throwaway_adapter(args, directory)

        # keep_ratio 1.0 must reproduce the adapter merged the ordinary way.
        merged_reference, _ = load_causal_lm_and_tokenizer(
            args.model_dir, args.model_name, args.use_bf16
        )
        from peft import PeftModel

        merged_reference = PeftModel.from_pretrained(
            merged_reference, str(adapter_path)
        ).merge_and_unload()
        reference = {
            name: parameter.detach().clone()
            for name, parameter in merged_reference.named_parameters()
        }
        del merged_reference

        full = directory / "keep1"
        run_cli(args, adapter_path, 1.0, full)
        if is_lora_adapter_dir(str(full)):
            return record("loadable", False, f"{full} looks like an adapter to eval")
        model, tokenizer = load_eval_model(
            args.model_dir, args.model_name, args.use_bf16, str(full)
        )
        record("loadable", tokenizer is not None, "eval's full-checkpoint branch read it back")

        # bf16 rounding: we add in float32 and round once, PEFT's merge adds in
        # bf16. Those differ by up to an ulp, which is why this is a tolerance
        # and not torch.equal.
        worst, worst_name = 0.0, ""
        for name, parameter in model.named_parameters():
            target = reference.get(name)
            if target is None:
                continue
            gap = float((parameter.detach().float() - target.float()).abs().max())
            if gap > worst:
                worst, worst_name = gap, name
        scale = max(
            float(value.float().abs().max()) for value in reference.values()
        )
        tolerance = scale * (2**-7 if args.use_bf16 == 1 else 1e-5)
        ok_full = worst <= tolerance
        record(
            "endpoint_1.0",
            ok_full,
            f"max |ours - merged| = {worst:.3e} <= {tolerance:.3e} ({worst_name})",
        )
        del model, reference

        base_reference, _ = load_causal_lm_and_tokenizer(
            args.model_dir, args.model_name, args.use_bf16
        )
        base_weights = {
            name: parameter.detach().clone()
            for name, parameter in base_reference.named_parameters()
        }
        del base_reference

        empty = directory / "keep0"
        run_cli(args, adapter_path, 0.0, empty)
        model, _ = load_eval_model(args.model_dir, args.model_name, args.use_bf16, str(empty))
        differing = [
            name
            for name, parameter in model.named_parameters()
            if name in base_weights and not torch.equal(parameter.detach(), base_weights[name])
        ]
        ok_empty = not differing
        record(
            "endpoint_0.0",
            ok_empty,
            "identical to the base model"
            if ok_empty
            else f"{len(differing)} tensors moved, e.g. {differing[:3]}",
        )
        return ok_full and ok_empty


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    print("checking FAPM against baseline/FAPM/FAPM.py\n", flush=True)
    check_upstream_agreement()
    check_budget()
    check_zero_base()

    if args.skip_model == 1:
        print("\n--skip_model 1: end-point checks not run.")
    else:
        print(
            f"\nend-point checks against {join_model_path(args.model_dir, args.model_name)}\n",
            flush=True,
        )
        check_endpoints(args)

    failed = [name for name, ok, _ in _results if not ok]
    print()
    if failed:
        print(f"{len(failed)} check(s) failed: {failed}")
        return 1
    print(f"all {len(_results)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
