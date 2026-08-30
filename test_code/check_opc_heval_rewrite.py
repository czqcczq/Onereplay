"""Check that the HumanEval rewrite does not change a problem's behaviour.

    python test_code/check_opc_heval_rewrite.py --num 80 --seed 1

A rewrite is only "wrong" if the original code already passes its own asserts
and the reconstructed program (stub + original body) no longer does. Other
outcomes are not rewrite bugs:

  source_fail   the dataset's own `code` already fails `testcase`
  no_tests      the row has no asserts, so we cannot score behaviour
  fallback      build_heval_view returned None; that row stays bare

AST parse of stub+body is a necessary but weaker check and is reported
separately. Execution is the actual guarantee.

Execution uses a fresh `python -I -c` subprocess per program, not the
multiprocessing sandbox in code_exec: on Windows that sandbox re-imports this
module on every spawn (slow) and cannot interrupt a program that blocks on
input(). A subprocess gives a hard, OS-level timeout and stdin=DEVNULL turns
input() into an immediate EOFError.
"""

from __future__ import annotations

import argparse
import ast
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onereplay.scripts.prepare_opc_ccode import build_heval_view  # noqa: E402
from test_code.preview_opc_heval import load_local_opc  # noqa: E402

_OK = "__ONEREPLAY_OK__"


def run_asserts(program: str, tests: list[str], timeout: float) -> tuple[bool, str]:
    """Run program + its asserts in an isolated subprocess with a hard timeout."""

    source = program + "\n" + "\n".join(tests) + f"\nprint({_OK!r})\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", source],
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except OSError as error:
        return False, f"spawn failed: {error}"
    if proc.returncode == 0 and _OK in proc.stdout:
        return True, ""
    last = (proc.stderr.strip().splitlines() or [""])[-1]
    return False, last or f"exit={proc.returncode}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="codedata_check/data/opencoder_educational",
    )
    parser.add_argument("--num", type=int, default=80, help="How many convertible rows to score.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--max_scan",
        type=int,
        default=800,
        help="Give up after this many random rows if --num conversions are not found.",
    )
    parser.add_argument(
        "--show_broken",
        type=int,
        default=5,
        help="Print this many rewrite-broke examples, if any.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = load_local_opc(args.data_dir)
    print(f"loaded {table.num_rows} rows; scoring up to {args.num} convertible examples", flush=True)
    rng = random.Random(args.seed)
    order = list(range(table.num_rows))
    rng.shuffle(order)

    counts = {
        "scanned": 0,
        "fallback": 0,
        "no_tests": 0,
        "source_fail": 0,
        "rewrite_pass": 0,
        "rewrite_broke": 0,
        "ast_fail": 0,
    }
    broken: list[dict] = []

    for index in order:
        if counts["scanned"] >= args.max_scan:
            break
        scored = (
            counts["rewrite_pass"]
            + counts["rewrite_broke"]
            + counts["source_fail"]
            + counts["no_tests"]
        )
        if scored >= args.num:
            break

        counts["scanned"] += 1
        row = {name: table[name][index].as_py() for name in table.column_names}
        instruction = str(row.get("instruction") or "")
        code = str(row.get("code") or "")
        entry_point = str(row.get("entry_point") or "")
        tests = [str(test) for test in (row.get("testcase") or []) if str(test).strip()]

        built = build_heval_view(code, entry_point, instruction)
        if built is None:
            counts["fallback"] += 1
            continue

        stub, body = built
        rebuilt = stub + body + "\n"
        try:
            ast.parse(rebuilt)
        except (SyntaxError, ValueError):
            counts["ast_fail"] += 1
            broken.append(
                {
                    "index": index,
                    "entry_point": entry_point,
                    "reason": "reconstructed module does not parse",
                    "error": "",
                    "instruction": instruction,
                    "rebuilt": rebuilt,
                }
            )
            continue

        if not tests:
            counts["no_tests"] += 1
            print(f"  [{scored + 1}/{args.num}] index={index} no_tests", flush=True)
            continue

        original_ok, original_error = run_asserts(code, tests, args.timeout)
        if not original_ok:
            counts["source_fail"] += 1
            print(f"  [{scored + 1}/{args.num}] index={index} source_fail: {original_error[:80]}", flush=True)
            continue

        rebuilt_ok, rebuilt_error = run_asserts(rebuilt, tests, args.timeout)
        if rebuilt_ok:
            counts["rewrite_pass"] += 1
            print(f"  [{scored + 1}/{args.num}] index={index} rewrite_pass", flush=True)
            continue
        counts["rewrite_broke"] += 1
        print(f"  [{scored + 1}/{args.num}] index={index} REWRITE_BROKE: {rebuilt_error[:120]}", flush=True)
        if len(broken) < args.show_broken:
            broken.append(
                {
                    "index": index,
                    "entry_point": entry_point,
                    "reason": "original passed, rewrite failed",
                    "error": rebuilt_error,
                    "instruction": instruction,
                    "rebuilt": rebuilt,
                }
            )

    comparable = counts["rewrite_pass"] + counts["rewrite_broke"]
    print(f"scanned {counts['scanned']} random rows from {args.data_dir}")
    print(f"  fallback (stays bare)     : {counts['fallback']}")
    print(f"  no testcase               : {counts['no_tests']}")
    print(f"  original already fails    : {counts['source_fail']}")
    print(f"  AST of stub+body failed   : {counts['ast_fail']}")
    print(f"  original pass, rewrite pass: {counts['rewrite_pass']}")
    print(f"  original pass, rewrite FAIL: {counts['rewrite_broke']}")
    if comparable:
        rate = counts["rewrite_broke"] / comparable
        print(
            f"\nrewrite error rate among originally-correct rows: "
            f"{counts['rewrite_broke']}/{comparable} = {rate:.1%}"
        )
        print(
            "This is the only number that means 'we turned a correct problem into a wrong one'."
        )
    else:
        print("\nno originally-correct rows with tests in this sample")

    for item in broken:
        print("\n" + "=" * 80)
        print(f"BROKEN  index={item['index']}  {item['entry_point']}")
        print(item["reason"])
        if item["error"]:
            print(f"error: {item['error']}")
        print("\n[instruction]")
        print(item["instruction"])
        print("\n[rebuilt program]")
        print(item["rebuilt"][:2000])


if __name__ == "__main__":
    main()
