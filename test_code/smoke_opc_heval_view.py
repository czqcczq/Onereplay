"""Edge cases for the HumanEval-style rewrite in prepare_opc_ccode.

    python test_code/smoke_opc_heval_view.py

Every case states what it expects and why. `convert` means a stub was produced;
`fallback` means build_heval_view returned None and the row would be written in
the bare style instead. A fallback is never a bug in itself -- the pool keeps
its row count either way -- but a fallback on case 1-5 or 10 would mean the
rewrite is losing rows it should be able to handle.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onereplay.scripts.prepare_opc_ccode import build_heval_view  # noqa: E402

CASES: list[tuple[str, str, str, str, bool]] = [
    (
        "plain function",
        "def is_palindrome(s):\n"
        "    s = ''.join(c for c in s if c.isalnum()).lower()\n"
        "    return s == s[::-1]\n",
        "is_palindrome",
        "Write a function to check if a given string is a palindrome.",
        True,
    ),
    (
        "imports and a helper before the target",
        "import math\n"
        "\n"
        "def _area(r):\n"
        "    return math.pi * r * r\n"
        "\n"
        "def total_area(radii):\n"
        "    return sum(_area(r) for r in radii)\n",
        "total_area",
        "Write a function to sum the areas of circles with the given radii.",
        True,
    ),
    (
        "helper defined AFTER the target",
        "def total_area(radii):\n"
        "    return sum(_area(r) for r in radii)\n"
        "\n"
        "def _area(r):\n"
        "    return 3.14159 * r * r\n",
        "total_area",
        "Write a function to sum the areas of circles.",
        True,
    ),
    (
        "trailing demo prints must be dropped from the preamble",
        "from collections import defaultdict\n"
        "\n"
        "def num_rolls(d, f):\n"
        "    return d * f\n"
        "\n"
        "print(num_rolls(2, 6))\n"
        "print(num_rolls(3, 6))\n",
        "num_rolls",
        "Write a function to count dice roll combinations.",
        True,
    ),
    (
        "__main__ guard must be dropped from the preamble",
        "def solve(n):\n"
        "    return n * 2\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    print(solve(5))\n",
        "solve",
        "Write a function to double a number.",
        True,
    ),
    (
        "trailing assert self-tests must not land before the def",
        "def trailing_zeros(n):\n"
        "    count = 0\n"
        "    i = 5\n"
        "    while n // i >= 1:\n"
        "        count += n // i\n"
        "        i *= 5\n"
        "    return count\n"
        "\n"
        "assert trailing_zeros(5) == 1\n"
        "assert trailing_zeros(25) == 6\n",
        "trailing_zeros",
        "Write a function to count trailing zeros of n factorial.",
        True,
    ),
    (
        "type hints, defaults, kwonly",
        "from typing import List, Optional\n"
        "\n"
        "def top_k(values: List[int], k: int = 3, *, reverse: bool = True) -> Optional[List[int]]:\n"
        "    if not values:\n"
        "        return None\n"
        "    return sorted(values, reverse=reverse)[:k]\n",
        "top_k",
        "Write a function to return the top k values from a list.",
        True,
    ),
    (
        "async def",
        "async def fetch_all(urls):\n    return [u.upper() for u in urls]\n",
        "fetch_all",
        "Write a coroutine that uppercases every url.",
        True,
    ),
    (
        "decorated function",
        "import functools\n"
        "\n"
        "@functools.lru_cache(maxsize=None)\n"
        "def fib(n):\n"
        "    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
        "fib",
        "Write a memoized function to compute the nth Fibonacci number.",
        True,
    ),
    (
        "target already has a docstring (must be replaced, not duplicated)",
        'def add(a, b):\n    """Old docstring."""\n    return a + b\n',
        "add",
        "Write a function to add two numbers.",
        True,
    ),
    (
        "instruction contains a triple quote",
        "def strip_quotes(s):\n    return s.strip('\\\"')\n",
        "strip_quotes",
        'Write a function that removes """ from a string.',
        True,
    ),
    (
        "instruction ends with a backslash",
        "def escape(s):\n    return s.replace('a', 'b')\n",
        "escape",
        "Write a function to escape a path such as C:\\\\tmp\\\\",
        True,
    ),
    (
        "entry_point is a method on a class -> fallback",
        "class Solver:\n    def solve(self, n):\n        return n * 2\n",
        "solve",
        "Write a function to double a number.",
        False,
    ),
    (
        "entry_point not present in code -> fallback",
        "def other(n):\n    return n\n",
        "missing_name",
        "Write a function.",
        False,
    ),
    (
        "unparseable code -> fallback",
        "def broken(:\n    return 1\n",
        "broken",
        "Write a function.",
        False,
    ),
    (
        "body is only a docstring -> fallback (nothing left to complete)",
        'def noop():\n    """does nothing"""\n',
        "noop",
        "Write a function that does nothing.",
        False,
    ),
]


def main() -> None:
    failures = 0
    for name, code, entry_point, instruction, should_convert in CASES:
        built = build_heval_view(code, entry_point, instruction)
        converted = built is not None
        status = "convert " if converted else "fallback"
        ok = converted == should_convert

        detail = ""
        if converted:
            stub, body = built
            # The pool is only sound if the stub plus a body is real Python and
            # the target function survives with its name intact.
            try:
                tree = ast.parse(stub + body + "\n")
                names = {
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if entry_point not in names:
                    ok = False
                    detail = f" !! entry_point {entry_point!r} missing from rebuilt module"
            except SyntaxError as error:
                ok = False
                detail = f" !! rebuilt module does not parse: {error}"

            # A HumanEval stub only carries imports, constants and helper defs
            # before the target. Anything else in the preamble is demo
            # scaffolding (print, assert self-test, __main__ guard) that would
            # run before the function exists.
            allowed = (
                ast.Import,
                ast.ImportFrom,
                ast.Assign,
                ast.AnnAssign,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            )
            stub_tree = ast.parse(stub + "    pass\n")
            target_index = next(
                i
                for i, node in enumerate(stub_tree.body)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == entry_point
            )
            for node in stub_tree.body[:target_index]:
                if not isinstance(node, allowed):
                    ok = False
                    detail = f" !! preamble has stray {type(node).__name__} (demo scaffolding)"
            for node in stub_tree.body[target_index + 1 :]:
                ok = False
                detail = f" !! statement after the stub def: {type(node).__name__}"

        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {status}  {name}{detail}")

    print("\n---- rendered stub for case 4 (type hints / defaults / kwonly) ----")
    _, code, entry_point, instruction, _ = CASES[3]
    built = build_heval_view(code, entry_point, instruction)
    if built:
        stub, body = built
        print(stub + "<<< model completes from here >>>")
        print("---- reference body kept as `targets` (overwritten by self-distill) ----")
        print(body)

    print("\n---- rendered stub for case 3 (helper defined after target) ----")
    _, code, entry_point, instruction, _ = CASES[2]
    built = build_heval_view(code, entry_point, instruction)
    if built:
        print(built[0] + "<<< model completes from here >>>")

    print(f"\n{len(CASES) - failures}/{len(CASES)} cases behaved as expected")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
