"""End-to-end check of rejudge_code_eval on synthetic responses.jsonl files.

Builds a two-task HumanEval-shaped reference set plus a responses file whose
completions carry the exact damage seen in the real runs, then confirms the
fixed judge recovers the rows the old one lost.

Run: python test_code/smoke_rejudge_code_eval.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onereplay.scripts.rejudge_code_eval import rejudge_run  # noqa: E402


def main() -> int:
    reference = {
        "T/1": {
            "task_id": "T/1",
            "prompt": (
                "def add_tail(numbers):\n"
                '    """ Return the list with its own sum appended. """\n'
            ),
            "entry_point": "add_tail",
            "test": (
                "def check(candidate):\n"
                "    assert candidate([1, 2]) == [1, 2, 3]\n"
                "    assert candidate([]) == [0]\n"
            ),
        },
        "T/2": {
            "task_id": "T/2",
            "prompt": 'def double(x):\n    """ Return twice x. """\n',
            "entry_point": "double",
            "test": "def check(candidate):\n    assert candidate(3) == 6\n",
        },
    }

    responses = [
        # De-indented multi-line body: first line flat, rest keeps its indent.
        {
            "task_id": "T/1",
            "passed": False,
            "error": "IndentationError('unexpected indent', ...)",
            "completion": "result = list(numbers)\n    result.append(sum(numbers))\n    return result\n",
        },
        # De-indented one-liner.
        {
            "task_id": "T/2",
            "passed": False,
            "error": "SyntaxError(\"'return' outside function\", ...)",
            "completion": "return x * 2\n",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "fake_run"
        run_dir.mkdir()
        response_path = run_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for row in responses:
                file.write(json.dumps(row) + "\n")

        report = rejudge_run("humaneval", response_path, reference, timeout=10.0)

    print(json.dumps({k: v for k, v in report.items() if k != "_rejudged"}, indent=2))

    failures = []
    if report["old_passed"] != 0:
        failures.append(f"expected 0 old passes, got {report['old_passed']}")
    if report["new_passed"] != 2:
        failures.append(f"expected 2 new passes, got {report['new_passed']}")
    if report["recovered"] != 2:
        failures.append(f"expected 2 recovered, got {report['recovered']}")
    if report["lost"] != 0:
        failures.append(f"expected 0 lost, got {report['lost']}")
    if report["fidelity"] == "exact":
        failures.append("fidelity should flag the missing raw field")

    if failures:
        print("\nFAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
