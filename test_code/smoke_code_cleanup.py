"""Check the code judge on the failure shapes that showed up in real runs.

The HumanEval cases are verbatim completions from
results/humaneval/cs_onereplay_balanced_lam3e-2_seed1_regonce/responses.jsonl,
where 26/164 rows died of `'return' outside function` or `unexpected indent`
purely because the judge stripped the body's leading indentation.

Run: python test_code/smoke_code_cleanup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onereplay.eval.code_exec import (  # noqa: E402
    assemble_entry_point_program,
    cleanup_body_completion,
    cleanup_program_completion,
    has_dangerous_code,
)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
        return
    failures.append(f"{name}{': ' + detail if detail else ''}")
    print(f"  FAIL {name}{': ' + detail if detail else ''}")


def compiles(source: str) -> bool:
    try:
        compile(source, "<program>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------
# HumanEval: bodies whose first line lost its indent to strip()
# --------------------------------------------------------------------------
HEVAL_CASES = [
    (
        "HumanEval/2 single-line body",
        'def truncate_number(number: float) -> float:\n    """ Docstring. """\n',
        "return number - math.floor(number)\n",
    ),
    (
        "HumanEval/4 first line flat, rest indented",
        'def mean_absolute_deviation(numbers) -> float:\n    """ Docstring. """\n',
        "mean = sum(numbers) / len(numbers)\n"
        "    return sum(abs(x - mean) for x in numbers) / len(numbers)\n",
    ),
    (
        "HumanEval/5 multi-line nested body",
        'def intersperse(numbers, delimeter):\n    """ Docstring. """\n',
        "result = []\n"
        "    for i, num in enumerate(numbers):\n"
        "        result.append(num)\n"
        "        if i < len(numbers) - 1:\n"
        "            result.append(delimeter)\n"
        "    return result\n",
    ),
    (
        "HumanEval/7 flat one-liner",
        'def filter_by_substring(strings, substring):\n    """ Docstring. """\n',
        "return [s for s in strings if substring in s]\n",
    ),
]

print("== HumanEval: de-indented bodies now compile ==")
for name, prompt, completion in HEVAL_CASES:
    program = assemble_entry_point_program(prompt, cleanup_body_completion(completion))
    check(name, compiles(program), repr(program[-90:]))

print("== HumanEval: shapes that already worked must not regress ==")
REWRITTEN = "def truncate_number(number: float) -> float:\n    return number % 1.0\n"
STUB = 'def truncate_number(number: float) -> float:\n    """ Docstring. """\n'
program = assemble_entry_point_program(STUB, cleanup_body_completion(REWRITTEN))
check("re-written signature still compiles", compiles(program))
check("re-written body is used verbatim", "return number % 1.0" in program)

INDENTED = "    return number % 1.0\n"
program = assemble_entry_point_program(STUB, cleanup_body_completion(INDENTED))
check("properly indented body untouched", compiles(program))

FENCED = "```python\n    return number % 1.0\n```"
program = assemble_entry_point_program(STUB, cleanup_body_completion(FENCED))
check("fenced body keeps indentation", compiles(program) and "```" not in program)

WHOLE_BODY_FLAT = "total = 0\ntotal = number % 1.0\nreturn total\n"
program = assemble_entry_point_program(STUB, cleanup_body_completion(WHOLE_BODY_FLAT))
check("fully flush-left body is re-indented", compiles(program))

TRAILING_PROSE = "    return number % 1.0\n\nThis works because the modulo keeps the fraction.\n"
program = assemble_entry_point_program(STUB, cleanup_body_completion(TRAILING_PROSE))
check("trailing prose trimmed", compiles(program))

# --------------------------------------------------------------------------
# MBPP: programs that used to be beheaded at the first '\ndef '
# --------------------------------------------------------------------------
print("== MBPP: imports before def survive ==")
BEHEADED = "import math\n\ndef is_square(n):\n    return int(math.isqrt(n)) ** 2 == n\n"
cleaned = cleanup_program_completion(BEHEADED)
check("import kept", "import math" in cleaned)
check("def kept", "def is_square" in cleaned)
check("program compiles", compiles(cleaned))

PROSE_HEAD = (
    "Here is the solution:\n\n"
    "import math\n\n"
    "def is_square(n):\n"
    "    return int(math.isqrt(n)) ** 2 == n\n"
)
cleaned = cleanup_program_completion(PROSE_HEAD)
check("prose head dropped", "Here is" not in cleaned and compiles(cleaned))
check("prose head keeps import", "import math" in cleaned)

SCAFFOLD = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "print(add(1, 2))\n"
    'if __name__ == "__main__":\n'
    "    print(add(3, 4))\n"
    "assert add(1, 1) == 3\n"
)
cleaned = cleanup_program_completion(SCAFFOLD)
check("demo print dropped", "print(" not in cleaned)
check("__main__ guard dropped", "__main__" not in cleaned)
check("wrong self-assert dropped", "assert" not in cleaned)
check("target def kept", "def add" in cleaned and compiles(cleaned))

HELPERS = (
    "from collections import Counter\n"
    "\n"
    "def _tally(items):\n"
    "    return Counter(items)\n"
    "\n"
    "def most_common(items):\n"
    "    return _tally(items).most_common(1)[0][0]\n"
)
cleaned = cleanup_program_completion(HELPERS)
check("helper def kept", "_tally" in cleaned)
check("second def kept", "most_common" in cleaned and compiles(cleaned))

DECORATED = (
    "from functools import lru_cache\n"
    "\n"
    "@lru_cache(maxsize=None)\n"
    "def fib(n):\n"
    "    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
)
cleaned = cleanup_program_completion(DECORATED)
check("decorator kept", "@lru_cache" in cleaned and compiles(cleaned))

CONSTANT = "LIMIT = 100\n\ndef under_limit(n):\n    return n < LIMIT\n"
cleaned = cleanup_program_completion(CONSTANT)
check("module constant kept", "LIMIT = 100" in cleaned and compiles(cleaned))

BROKEN = "this is not python at all\n"
cleaned = cleanup_program_completion(BROKEN)
check("unparsable text surfaces as-is", not compiles(cleaned))

# --------------------------------------------------------------------------
# Blacklist
# --------------------------------------------------------------------------
print("== blacklist ==")
for allowed in ("import sys", "from pathlib import Path", "import pickle"):
    check(f"allowed: {allowed}", not has_dangerous_code(f"{allowed}\n"))
for blocked in ("import os", "import subprocess", "urllib.request", "open('f')"):
    check(f"blocked: {blocked}", has_dangerous_code(f"{blocked}\n"))

print()
if failures:
    print(f"{len(failures)} check(s) failed:")
    for item in failures:
        print(f"  - {item}")
    raise SystemExit(1)
print("All checks passed.")
