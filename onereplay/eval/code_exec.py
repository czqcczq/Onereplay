"""Sandboxed execution helpers shared by the HumanEval and MBPP metrics.

Two answer shapes have to be judged, and they need different cleanup:

* MBPP asks for a whole program, so imports and every top-level ``def`` must
  survive while trailing demo scaffolding is dropped.
* HumanEval asks for a function body that gets appended to the stub, so the
  leading indentation is load-bearing and must not be stripped.

An earlier single ``cleanup_completion`` did the opposite on both counts: it
truncated at the first ``\\ndef ``, which beheaded MBPP programs that open with
an import, and it called ``strip()``, which de-indented HumanEval bodies into
``'return' outside function``. Both silently scored correct answers as failures.
"""

from __future__ import annotations

import ast
import multiprocessing as mp
import re
import textwrap
from typing import Any

# Execution already happens in a subprocess with a timeout, so this list only
# needs to cover reaching outside that subprocess: the filesystem, other
# processes, and the network. `sys`, `pathlib` and `pickle` used to be here too
# and rejected ordinary solutions that merely imported them.
DANGEROUS_PATTERNS = (
    "import os",
    "from os",
    "import subprocess",
    "from subprocess",
    "open(",
    "exec(",
    "eval(",
    "__import__",
    "socket",
    "requests",
    "urllib",
    "shutil",
)

_FENCE = "```"
_INDENT = "    "
_FENCE_LANGUAGES = {"", "python", "py", "python3"}
_CODE_START = re.compile(r"^(?:import\s|from\s|def\s|class\s|async\s+def\s|@)")

# Top-level nodes a standalone program is allowed to keep. Everything else is
# demo scaffolding: bare expressions (`print(...)`), `if __name__` guards, and
# the model's own asserts, which may encode guesses the real tests contradict.
_PROGRAM_KEEP = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
)


def has_dangerous_code(code: str) -> bool:
    """Reject completions that try to access files, processes, or network."""

    lowered = code.lower()
    return any(pattern in lowered for pattern in DANGEROUS_PATTERNS)


def _compiles(source: str) -> bool:
    try:
        compile(source, "<program>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def _extract_code_block(text: str) -> str:
    """Pull code out of markdown fences without touching its indentation."""

    if _FENCE not in text:
        return text
    block = max(text.split(_FENCE), key=len)
    lines = block.split("\n")
    if lines and lines[0].strip().lower() in _FENCE_LANGUAGES:
        lines = lines[1:]
    return "\n".join(lines)


def _trim_blank_edges(text: str) -> str:
    """Drop blank leading/trailing lines, preserving the first line's indent."""

    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _node_segment(lines: list[str], node: ast.stmt) -> str:
    """Slice a node's original source, decorators included."""

    start = node.lineno
    for decorator in getattr(node, "decorator_list", None) or []:
        start = min(start, decorator.lineno)
    end = getattr(node, "end_lineno", None) or node.lineno
    return "\n".join(lines[start - 1 : end])


def _largest_parsable(text: str) -> tuple[ast.Module | None, str]:
    """Find the longest chunk that parses, dropping prose head and cut-off tail."""

    lines = text.split("\n")
    starts = [0]
    for index, line in enumerate(lines):
        if _CODE_START.match(line):
            if index:
                starts.append(index)
            break
    for start in starts:
        body = lines[start:]
        for end in range(len(body), 0, -1):
            candidate = "\n".join(body[:end])
            try:
                tree = ast.parse(candidate)
            except (SyntaxError, ValueError):
                continue
            return tree, candidate
    return None, ""


def cleanup_program_completion(text: str) -> str:
    """Reduce an MBPP-style answer to the program it defines.

    Keeps imports, assignments and every top-level definition in source order;
    drops prose, ``print`` demos, ``if __name__`` guards and model-written
    asserts. Text that never parses is returned as-is so the judge reports the
    genuine SyntaxError instead of hiding it.
    """

    text = _trim_blank_edges(_extract_code_block(text))
    tree, source = _largest_parsable(text)
    if tree is None:
        return text.rstrip() + "\n"
    source_lines = source.split("\n")
    kept = [
        segment
        for node in tree.body
        if isinstance(node, _PROGRAM_KEEP)
        for segment in [_node_segment(source_lines, node)]
        if segment.strip()
    ]
    if not kept:
        return source.rstrip() + "\n"
    return "\n\n".join(kept).rstrip() + "\n"


def cleanup_body_completion(text: str) -> str:
    """Clean a HumanEval-style answer while preserving indentation.

    Only fences and blank edges go; the completion may be a bare body, a
    re-written function, or a body plus helpers, and it is not this function's
    job to decide which. `assemble_entry_point_program` resolves that by
    checking what compiles against the stub.
    """

    return _trim_blank_edges(_extract_code_block(text)).rstrip() + "\n"


def assemble_entry_point_program(prompt: str, completion: str) -> str:
    """Attach a HumanEval completion to its stub, normalizing indentation.

    Models answer in three shapes: the indented body the prompt asks for, the
    whole function re-written (valid because the stub's docstring already serves
    as its body, so the second definition just wins), or the body flush against
    the left margin. All three are accepted -- the variant that compiles is
    used, preferring the completion as written. A cut-off tail is trimmed line
    by line as a last resort.

    Indentation normalization is a judging policy and belongs in the write-up:
    without it, a model that follows the instruction to return only the body is
    scored 0 while one that ignores it and repeats the signature passes.
    """

    if not prompt.endswith("\n"):
        prompt += "\n"
    lines = completion.split("\n")
    variants = [completion]
    if lines and lines[0][:1] not in ("", " ", "\t"):
        # `strip()` upstream only ever ate the *first* line's indent, so putting
        # it back alone is its exact inverse; indenting everything is the case
        # where the model wrote the whole body flush left.
        variants.append("\n".join([_INDENT + lines[0]] + lines[1:]))
        variants.append(textwrap.indent(completion, _INDENT))
    for variant in variants:
        program = prompt + variant.rstrip() + "\n"
        if _compiles(program):
            return program
    for variant in variants:
        variant_lines = variant.split("\n")
        for end in range(len(variant_lines) - 1, 0, -1):
            program = prompt + "\n".join(variant_lines[:end]).rstrip() + "\n"
            if _compiles(program):
                return program
    return prompt + completion.rstrip() + "\n"


# Kept so older callers keep working; new code should pick the explicit variant.
cleanup_completion = cleanup_program_completion


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
