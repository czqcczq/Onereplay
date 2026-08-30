"""Smoke test for the APPS stdin judge. No APPS data, no model, no GPU.

    python test_code/smoke_apps_exec.py

Checks that the runner actually distinguishes correct from incorrect, and that
each failure mode lands in the bucket the metric expects. If this passes locally
it will pass on the cluster; if it fails inside the Singularity container the
problem is process spawning, not the judge logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onereplay.eval.apps_exec import (  # noqa: E402
    evaluate_stdin_program,
    extract_apps_code,
    looks_truncated,
    outputs_match,
)

SUM_TWO = "a = int(input())\nb = int(input())\nprint(a + b)\n"
MAIN_GUARD = (
    "import sys\n\n"
    "def solve():\n"
    "    data = sys.stdin.read().split()\n"
    "    print(int(data[0]) + int(data[1]))\n\n"
    'if __name__ == "__main__":\n'
    "    solve()\n"
)

CASES: list[tuple[str, dict, dict]] = [
    (
        "correct, input()",
        {"code": SUM_TWO, "inputs": ["2\n3\n", "10\n20\n"], "outputs": ["5\n", "30\n"]},
        {"passed": True, "pass_ratio": 1.0},
    ),
    (
        "correct, __main__ guard + sys.stdin.read",
        {"code": MAIN_GUARD, "inputs": ["2\n3\n"], "outputs": ["5\n"]},
        {"passed": True, "pass_ratio": 1.0},
    ),
    (
        "partially correct",
        {"code": SUM_TWO, "inputs": ["2\n3\n", "1\n1\n"], "outputs": ["5\n", "3\n"]},
        {"passed": False, "pass_ratio": 0.5, "error": "wrong answer"},
    ),
    (
        "syntax error",
        {"code": "print(", "inputs": ["1\n"], "outputs": ["1\n"]},
        {"passed": False, "pass_ratio": 0.0, "error_startswith": "SyntaxError"},
    ),
    (
        "runtime error",
        {"code": "print(1 / 0)\n", "inputs": ["1\n"], "outputs": ["1\n"]},
        {"passed": False, "pass_ratio": 0.0, "error_startswith": "ZeroDivisionError"},
    ),
    (
        "infinite loop is killed",
        {"code": "while True:\n    pass\n", "inputs": ["1\n"], "outputs": ["1\n"]},
        {"passed": False, "pass_ratio": 0.0, "error": "timeout"},
    ),
    (
        "empty completion",
        {"code": "", "inputs": ["1\n"], "outputs": ["1\n"]},
        {"passed": False, "pass_ratio": 0.0, "error": "empty completion"},
    ),
    (
        "trailing whitespace is tolerated",
        {"code": "print(input().strip())\n", "inputs": ["hi\n"], "outputs": ["  hi  "]},
        {"passed": True, "pass_ratio": 1.0},
    ),
]

failures: list[str] = []

for label, payload, want in CASES:
    got = evaluate_stdin_program(
        payload["code"], payload["inputs"], payload["outputs"], timeout=5.0, max_tests=0
    )
    problems = []
    if got["passed"] != want["passed"]:
        problems.append(f"passed={got['passed']} want {want['passed']}")
    if abs(got["pass_ratio"] - want["pass_ratio"]) > 1e-9:
        problems.append(f"pass_ratio={got['pass_ratio']} want {want['pass_ratio']}")
    if "error" in want and got["error"] != want["error"]:
        problems.append(f"error={got['error']!r} want {want['error']!r}")
    if "error_startswith" in want and not got["error"].startswith(want["error_startswith"]):
        problems.append(f"error={got['error']!r} want prefix {want['error_startswith']!r}")

    status = "ok  " if not problems else "FAIL"
    print(f"[{status}] {label:<42} {got['num_passed']}/{got['num_tests']} {got['error']!r}")
    if problems:
        failures.append(f"{label}: {'; '.join(problems)}")

print()

# max_tests truncation must actually truncate, since strict pass@1 depends on it.
truncated = evaluate_stdin_program(
    SUM_TWO, ["2\n3\n"] * 50, ["5\n"] * 50, timeout=5.0, max_tests=3
)
if truncated["num_tests"] != 3:
    failures.append(f"max_tests: num_tests={truncated['num_tests']} want 3")
print(f"[{'ok  ' if truncated['num_tests'] == 3 else 'FAIL'}] max_tests=3 caps 50 cases")

# Code extraction: last fenced block wins, not the longest one.
response = (
    "Let me try.\n```python\n# a long first draft that is wrong\nprint(0)\nprint(0)\n```\n"
    "That was wrong. Final answer:\n```python\nprint(1)\n```"
)
extracted = extract_apps_code(response)
if extracted != "print(1)":
    failures.append(f"extract: got {extracted!r} want 'print(1)'")
print(f"[{'ok  ' if extracted == 'print(1)' else 'FAIL'}] extract takes the LAST fenced block")

if not looks_truncated("```python\nprint(1)\n```"):
    pass
else:
    failures.append("looks_truncated: false positive on a closed block")
if not looks_truncated("```python\nprint(1)"):
    failures.append("looks_truncated: missed an unclosed block")
print(f"[{'ok  ' if not failures or 'looks_truncated' not in str(failures) else 'FAIL'}] "
      "looks_truncated flags unclosed fences")

if not outputs_match("1 2\n3\n", "1  2\n\n3"):
    failures.append("outputs_match: whitespace normalisation broken")

print()
if failures:
    print(f"{len(failures)} failure(s):")
    for line in failures:
        print(f"  - {line}")
    raise SystemExit(1)
print("APPS 判分链路 OK")
