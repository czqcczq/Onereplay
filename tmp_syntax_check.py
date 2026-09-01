"""Run `bash -n` on the gold-ablation PBS scripts through WSL.

The repo checks out .pbs with CRLF on Windows; bash chokes on the carriage
returns, so each file is rewritten LF-only into a temp copy inside the
workspace (WSL can only reach paths under /mnt/<drive>).
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
TMP = REPO / "tmp_bashcheck"
TARGETS = [
    "onereplay/pbs/57_opc_code_gold_data.pbs",
    "onereplay/pbs/59_opc_code_gold_onereplay.pbs",
    "onereplay/pbs/60_opc_code_gold_ewc.pbs",
    "onereplay/pbs/61_opc_code_gold_replay.pbs",
    "onereplay/pbs/62_math_gold_data.pbs",
    "onereplay/pbs/63_math_gold_onereplay.pbs",
    "onereplay/pbs/64_math_gold_ewc.pbs",
    "onereplay/pbs/65_math_gold_replay.pbs",
]


def wsl_path(path: Path) -> str:
    drive, rest = str(path).split(":", 1)
    return f"/mnt/{drive.lower()}{rest.replace(chr(92), '/')}"


def main() -> int:
    TMP.mkdir(exist_ok=True)
    failures = 0
    for rel in TARGETS:
        src = REPO / rel
        if not src.exists():
            print(f"MISSING  {rel}")
            failures += 1
            continue
        dst = TMP / Path(rel).name
        dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
        done = subprocess.run(
            ["wsl", "bash", "-n", wsl_path(dst)],
            capture_output=True,
            text=True,
        )
        if done.returncode == 0:
            print(f"ok       {rel}")
        else:
            print(f"FAIL     {rel}\n{done.stdout}{done.stderr}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
