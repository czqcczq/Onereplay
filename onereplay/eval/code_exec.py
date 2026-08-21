"""Sandboxed execution helpers shared by the HumanEval and MBPP metrics."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

DANGEROUS_PATTERNS = (
    "import os",
    "import sys",
    "import subprocess",
    "from os",
    "from sys",
    "from subprocess",
    "open(",
    "exec(",
    "eval(",
    "__import__",
    "socket",
    "requests",
    "urllib",
    "shutil",
    "pathlib",
    "pickle",
)


def has_dangerous_code(code: str) -> bool:
    """Reject completions that try to access files, processes, network, or eval."""

    lowered = code.lower()
    return any(pattern in lowered for pattern in DANGEROUS_PATTERNS)


def cleanup_completion(text: str) -> str:
    """Remove chatty wrappers and stop at the next top-level definition."""

    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        text = max(parts, key=len).replace("python", "", 1).strip()
    stop_markers = ["\nclass ", "\ndef ", "\nif __name__", "\n# Example", "\nprint("]
    for marker in stop_markers:
        index = text.find(marker)
        if index >= 0:
            text = text[:index].rstrip()
    return text.rstrip() + "\n"


def _run_entry_point_test(program: str, entry_point: str, test_code: str, queue: mp.Queue) -> None:
    """Execute one HumanEval-style test program inside a subprocess."""

    try:
        namespace: dict[str, Any] = {}
        exec(program, namespace)
        exec(test_code, namespace)
        namespace["check"](namespace[entry_point])
        queue.put({"passed": True, "error": ""})
    except BaseException as exc:  # noqa: BLE001 - report model/runtime failures.
        queue.put({"passed": False, "error": repr(exc)})


def _run_assert_tests(program: str, tests: list[str], queue: mp.Queue) -> None:
    """Execute one MBPP-style program and its asserts inside a subprocess."""

    try:
        namespace: dict[str, Any] = {}
        exec(program, namespace)
        for test in tests:
            exec(str(test), namespace)
        queue.put({"passed": True, "error": ""})
    except BaseException as exc:  # noqa: BLE001 - report model/runtime failures.
        queue.put({"passed": False, "error": repr(exc)})


def _join_with_timeout(process: mp.Process, queue: mp.Queue, timeout: float) -> tuple[bool, str]:
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return False, "timeout"
    if queue.empty():
        return False, "no result"
    result = queue.get()
    return bool(result["passed"]), str(result["error"])


def evaluate_entry_point_program(
    program: str,
    entry_point: str,
    test_code: str,
    timeout: float,
) -> tuple[bool, str]:
    """Run one HumanEval program with timeout protection."""

    if has_dangerous_code(program):
        return False, "blocked dangerous code pattern"
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_run_entry_point_test, args=(program, entry_point, test_code, queue)
    )
    return _join_with_timeout(process, queue, timeout)


def evaluate_assert_program(program: str, tests: list[str], timeout: float) -> tuple[bool, str]:
    """Run one MBPP program with timeout protection."""

    if has_dangerous_code(program):
        return False, "blocked dangerous code pattern"
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_run_assert_tests, args=(program, tests, queue))
    return _join_with_timeout(process, queue, timeout)
