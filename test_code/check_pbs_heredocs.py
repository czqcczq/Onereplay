"""Compile the inline Python heredocs inside PBS scripts.

    python test_code/check_pbs_heredocs.py onereplay/pbs/50_opc_code_data_cov.pbs ...

`bash -n` validates the shell around them but treats a heredoc as opaque text,
so a syntax error inside one only surfaces hours into a cluster job.
"""

from __future__ import annotations

import re
import sys

BLOCK = re.compile(r"<<'PY'\n(.*?)\nPY\n", re.DOTALL)


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: check_pbs_heredocs.py <pbs file> [...]")

    failures = 0
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            blocks = BLOCK.findall(handle.read())
        print(f"{path}: {len(blocks)} heredoc block(s)")
        for index, source in enumerate(blocks):
            try:
                compile(source, f"{path}#{index}", "exec")
            except SyntaxError as error:
                failures += 1
                print(f"   block {index}: SYNTAX ERROR line {error.lineno}: {error.msg}")
            else:
                print(f"   block {index}: OK ({len(source.splitlines())} lines)")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
