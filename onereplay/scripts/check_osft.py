"""Preflight and correctness self-check for the OSFT baseline.

Run this once per environment before spending cluster time on an OSFT arm. It
answers the only question that matters about a vendored baseline: is the arm
labelled OSFT actually running OSFT, and is it running it on the same model the
other arms run on?

The checks are ordered so that the cheap environment ones fail first:

  1. env      python / torch / transformers, and whether osft_utils imports.
  2. commit   the clone is at the commit onereplay/core/osft.py was written for.
  3. patterns upstream's path-substring lookup resolved the architecture we meant.
  4. parity   at step 0 the decomposed model and the plain model give the same
              logits. This is the load-bearing check: it proves U S V^T rebuilt
              the checkpoint rather than some rotation of it, so an OSFT run
              starts from exactly the weights every other arm starts from.
  5. frozen   after real optimizer steps the high-rank factors have not moved,
              and the trainable factors have.
  6. leak     U_low stays orthogonal to U_high and V_low to V_high, computed
              here in float32 rather than by calling the projection being tested.
  7. save     a reconstructed checkpoint reloads as a plain causal LM and gives
              the same logits as the in-memory decomposed model, which is what
              onereplay.scripts.evaluate will read.
  8. degenerate  --osft_unfreeze_rank_ratio 1.0 freezes nothing, so the arm
              collapses onto vanilla full fine-tuning. Its loss curve has to
              match a vanilla run's, and that is the end-to-end guard against an
              integration bug; this check only confirms the rank bookkeeping.

Usage:
    python -m onereplay.scripts.check_osft \
        --model_dir /home/weiliu1/huggingface/models/ --model_name Qwen3-1.7B \
        --osft_unfreeze_rank_ratio 0.25

Prefer the smallest checkpoint of the family you plan to train, not the 8B: the
load path holds the base model and the decomposed model at once, so peak host
memory is roughly twice the checkpoint before the SVD factors are even built.

For the glue-only checks that need neither GPU nor checkpoint, run
test_code/check_osft_integration.py instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from onereplay.core.modeling import join_model_path, set_seed  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSFT integration self-check")
    parser.add_argument("--model_dir", type=str, default="/home/weiliu1/huggingface/models/")
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--osft_unfreeze_rank_ratio", type=float, default=0.25)
    parser.add_argument("--osft_target_patterns", type=str, default="")
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Optimizer steps to take on random tokens before the frozen/leak checks.",
    )
    parser.add_argument(
        "--check_save",
        type=int,
        default=1,
        help=(
            "1 also writes a reconstructed checkpoint to a temporary directory and "
            "reloads it. Costs one checkpoint's worth of disk and a second model "
            "load; skip only if the node is tight on scratch space."
        ),
    )
    parser.add_argument(
        "--parity_tolerance",
        type=float,
        default=0.0,
        help=(
            "Max allowed absolute logit difference for checks 4 and 7. 0 derives it "
            "from the dtype, which is the right default: bf16 has 8 mantissa bits, so "
            "a factored matmul cannot reproduce a dense one exactly and demanding "
            "equality would fail on arithmetic rather than on correctness."
        ),
    )
    return parser.parse_args()


def check_environment() -> bool:
    import transformers

    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}")
    print(f"transformers {transformers.__version__}")
    print(f"cuda         {torch.cuda.is_available()}")

    ok = True
    if sys.version_info < (3, 10):
        ok = record("env: python >= 3.10", False, f"found {sys.version.split()[0]}")
    try:
        import rich  # noqa: F401
    except ImportError:
        ok = record(
            "env: rich installed",
            False,
            "mini_trainer.utils imports rich.logging; pip install rich",
        )
    try:
        import huggingface_hub  # noqa: F401
        from safetensors.torch import save_file  # noqa: F401
    except ImportError as error:
        ok = record("env: safetensors + huggingface_hub", False, str(error))
    return ok


def check_upstream_import() -> bool:
    from onereplay.core import osft

    try:
        upstream = osft.load_upstream()
    except Exception as error:  # noqa: BLE001 - the message is the whole point
        return record("upstream: import mini_trainer.osft_utils", False, str(error))

    module = upstream["osft_utils"]
    required = [
        "create_osft_model_class",
        "optim_wrapper",
        "reconstruct_weight_matrix",
        "get_model_config",
        "_build_osft_kwargs",
        "_set_osft_dtypes",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        return record("upstream: expected symbols present", False, f"missing {missing}")
    return record(
        "upstream: import mini_trainer.osft_utils",
        True,
        f"from {upstream['source_dir']}",
    )


def check_commit() -> bool:
    """The glue was written against one commit; say so if the clone moved."""

    import subprocess

    from onereplay.core import osft

    repo = osft.upstream_source_dir().parent
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return record("commit: clone is a git checkout", False, str(error))

    if head != osft.UPSTREAM_COMMIT:
        return record(
            "commit: matches the one core/osft.py was written for",
            False,
            f"clone is at {head[:10]}, expected {osft.UPSTREAM_COMMIT[:10]}. Re-read "
            "core/osft.py's call order against their setup_model_for_training before "
            "trusting results, then update UPSTREAM_COMMIT.",
        )
    return record("commit: matches the one core/osft.py was written for", True, head[:10])


def check_patterns(model_path: str, raw_patterns: str) -> bool:
    from onereplay.core.osft import load_upstream, parse_target_patterns, resolved_target_patterns

    osft_utils = load_upstream()["osft_utils"]
    resolved = resolved_target_patterns(model_path, parse_target_patterns(raw_patterns))

    from transformers import AutoConfig

    model_type = str(getattr(AutoConfig.from_pretrained(model_path), "model_type", "")).lower()
    table_key = None
    for identifier, key in osft_utils.MODEL_NAME_MAPPINGS.items():
        if identifier in model_type:
            table_key = key
            break

    if raw_patterns.strip():
        return record("patterns: explicit list used", True, f"{resolved}")
    if table_key is None:
        return record(
            "patterns: architecture recognised",
            False,
            f"model_type {model_type!r} is not in upstream's table; pass "
            "--osft_target_patterns explicitly",
        )
    expected = osft_utils.MODEL_CONFIGS[table_key]["patterns"]
    if list(resolved) != list(expected):
        return record(
            "patterns: path lookup agrees with the architecture",
            False,
            f"model_type {model_type!r} implies {table_key!r} patterns {expected}, but the "
            f"path resolved to {resolved}. Their lookup scans the *path* and tests 'opt' "
            "before 'qwen', so a directory name collided. Pass --osft_target_patterns.",
        )
    return record("patterns: path lookup agrees with the architecture", True, f"{table_key}")


def load_plain(model_path: str, dtype, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    return model.to(device).eval()


def load_osft(model_path: str, args, dtype, device, unfreeze_rank_ratio: float):
    from onereplay.core.osft import build_osft_model, parse_target_patterns

    model = build_osft_model(
        model_path,
        unfreeze_rank_ratio=unfreeze_rank_ratio,
        target_patterns=parse_target_patterns(args.osft_target_patterns),
        torch_dtype=dtype,
    )
    return model.to(device).eval()


def dtype_tolerance(dtype, override: float) -> float:
    if override > 0:
        return override
    # Logits run to ~20 in magnitude, and the factored path sums two matmuls
    # where the dense path does one, so the error budget is a few multiples of
    # the dtype's relative resolution at that scale, not of 1.0.
    return {torch.bfloat16: 0.5, torch.float16: 0.1}.get(dtype, 1e-3)


def max_logit_gap(first, second, input_ids) -> float:
    with torch.no_grad():
        a = first(input_ids=input_ids).logits.float()
        b = second(input_ids=input_ids).logits.float()
    return float((a - b).abs().max())


def check_step0_parity(plain, osft_model, input_ids, tolerance: float) -> bool:
    """The decomposition must not move the model before training starts."""

    gap = max_logit_gap(plain, osft_model, input_ids)
    return record(
        "parity: decomposed model == plain model at step 0",
        gap <= tolerance,
        f"max |dlogit| = {gap:.4g} (tolerance {tolerance:g})",
    )


def take_steps(model, input_ids, steps: int, lr: float = 1e-4) -> None:
    from onereplay.core.osft import wrap_optimizer

    model.train()
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=lr
    )
    optimizer = wrap_optimizer(optimizer, model)
    for _ in range(steps):
        optimizer.zero_grad()
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        optimizer.step()
    model.eval()


def check_frozen_and_moved(model, before: dict[str, torch.Tensor]) -> bool:
    """High-rank factors must be untouched; low-rank ones must have moved."""

    from onereplay.core.osft import _osft_modules

    frozen_drift = 0.0
    trainable_drift = 0.0
    with torch.no_grad():
        for index, module in enumerate(_osft_modules(model)):
            for field, tensor in (
                ("U_high", module.osft_U_high),
                ("S_high", module.osft_S_high),
                ("V_high", module.osft_V_high),
            ):
                key = f"{index}.{field}"
                if key in before:
                    frozen_drift = max(
                        frozen_drift, float((tensor.float() - before[key]).abs().max())
                    )
            for field, tensor in (
                ("U_low", module.osft_params.U_low),
                ("S_low", module.osft_params.S_low),
                ("V_low", module.osft_params.V_low),
            ):
                key = f"{index}.{field}"
                if key in before:
                    trainable_drift = max(
                        trainable_drift, float((tensor.float() - before[key]).abs().max())
                    )

    ok = record(
        "frozen: high-rank factors unchanged after optimizer steps",
        frozen_drift == 0.0,
        f"max drift {frozen_drift:.4g}",
    )
    ok = (
        record(
            "trainable: low-rank factors moved",
            trainable_drift > 0.0,
            f"max drift {trainable_drift:.4g}",
        )
        and ok
    )
    return ok


def snapshot_factors(model) -> dict[str, torch.Tensor]:
    from onereplay.core.osft import _osft_modules

    snapshot: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for index, module in enumerate(_osft_modules(model)):
            snapshot[f"{index}.U_high"] = module.osft_U_high.detach().float().clone()
            snapshot[f"{index}.S_high"] = module.osft_S_high.detach().float().clone()
            snapshot[f"{index}.V_high"] = module.osft_V_high.detach().float().clone()
            snapshot[f"{index}.U_low"] = module.osft_params.U_low.detach().float().clone()
            snapshot[f"{index}.S_low"] = module.osft_params.S_low.detach().float().clone()
            snapshot[f"{index}.V_low"] = module.osft_params.V_low.detach().float().clone()
    return snapshot


def check_leak(model, dtype) -> bool:
    from onereplay.core.osft import max_frozen_subspace_leak

    leak = max_frozen_subspace_leak(model)
    # The factors are stored in the training dtype, so the projection can only
    # zero the overlap to that dtype's resolution: bf16's unit roundoff is ~2^-8.
    # The measurement itself is done in float32, whose own epsilon (~1e-7) sits
    # far below these budgets, so a real leak cannot hide inside it.
    budget = {torch.bfloat16: 5e-2, torch.float16: 5e-3}.get(dtype, 1e-4)
    return record(
        "leak: trainable factors stay orthogonal to the frozen subspace",
        leak <= budget,
        f"max overlap {leak:.4g} (budget {budget:g} for {dtype})",
    )


def check_save_roundtrip(model, input_ids, dtype, device, tolerance: float) -> bool:
    """The reconstructed checkpoint must be the model we just trained."""

    from onereplay.core.osft import save_osft_checkpoint

    directory = Path(tempfile.mkdtemp(prefix="osft_roundtrip_"))
    try:
        save_osft_checkpoint(model, str(directory))
        if not (directory / "config.json").is_file():
            return record("save: wrote config.json", False, str(directory))
        reloaded = load_plain(str(directory), dtype, device)
        architectures = None
        import json

        with open(directory / "config.json", encoding="utf-8") as handle:
            architectures = json.load(handle).get("architectures")
        if architectures and any("WithOSFT" in name for name in architectures):
            return record(
                "save: config.json names the base architecture",
                False,
                f"architectures={architectures}",
            )
        gap = max_logit_gap(model, reloaded, input_ids)
        del reloaded
        return record(
            "save: reconstructed checkpoint reloads to the same model",
            gap <= tolerance,
            f"max |dlogit| = {gap:.4g} (tolerance {tolerance:g}), architectures={architectures}",
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def check_degenerate_full_ft(model_path: str, args, dtype, device) -> bool:
    """unfreeze_rank_ratio 1.0 has to leave nothing frozen.

    That is the setting under which OSFT is vanilla full fine-tuning, so it is
    the arm's own control: a run at 1.0 should reproduce the vanilla loss curve
    you already have in results_log. Here we only confirm the bookkeeping, which
    is cheap; the curve comparison is the real test and belongs in a PBS job.
    """

    from onereplay.core.osft import describe_osft

    model = load_osft(model_path, args, dtype, device, unfreeze_rank_ratio=1.0)
    summary = describe_osft(model, 1.0)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # floor(min(shape) * 0.0) == 0, so every direction should be trainable.
    return record(
        "degenerate: unfreeze_rank_ratio 1.0 freezes nothing",
        summary["osft_rank_high_total"] == 0,
        f"frozen rank total {summary['osft_rank_high_total']} over "
        f"{summary['osft_matrices']} matrices",
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    model_path = join_model_path(args.model_dir, args.model_name)
    dtype = torch.bfloat16 if args.use_bf16 == 1 else torch.float32
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tolerance = dtype_tolerance(dtype, args.parity_tolerance)
    print(f"checking OSFT against {model_path} on {device} in {dtype}\n", flush=True)

    if not check_environment():
        print("\nenvironment is not ready; fix the above before continuing")
        return 1
    if not check_upstream_import():
        return 1
    check_commit()
    if not Path(model_path).exists():
        record("model: checkpoint exists", False, f"{model_path} not found")
        return 1
    check_patterns(model_path, args.osft_target_patterns)

    print("\nloading the plain model for the step-0 comparison", flush=True)
    plain = load_plain(model_path, dtype, device)
    vocab_size = int(plain.config.vocab_size)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids = torch.randint(
        0, vocab_size, (2, 64), generator=generator, dtype=torch.long
    ).to(device)

    print("\nloading the OSFT model", flush=True)
    osft_model = load_osft(model_path, args, dtype, device, args.osft_unfreeze_rank_ratio)

    check_step0_parity(plain, osft_model, input_ids, tolerance)
    del plain
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\ntaking {args.steps} optimizer steps on random tokens", flush=True)
    before = snapshot_factors(osft_model)
    take_steps(osft_model, input_ids, args.steps)
    check_frozen_and_moved(osft_model, before)
    del before
    check_leak(osft_model, dtype)

    if args.check_save == 1:
        print("\nround-tripping a reconstructed checkpoint", flush=True)
        check_save_roundtrip(osft_model, input_ids, dtype, device, tolerance)
    del osft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nchecking the degenerate setting", flush=True)
    check_degenerate_full_ft(model_path, args, dtype, device)

    failed = [name for name, ok, _ in _results if not ok]
    print("\n" + "=" * 72)
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
        print(
            "\nDo not launch an OSFT arm until these pass. A failure here is an "
            "integration problem, which is exactly the thing that would otherwise "
            "show up as an unexplainable number in the results table."
        )
        return 1
    print(
        "OSFT is wired correctly. The remaining end-to-end check is a training run at "
        "--osft_unfreeze_rank_ratio 1.0, whose loss curve must match the vanilla arm's."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
