"""Temporary: LF-normalize the new PBS scripts and run `bash -n` on each."""

import os
import re
import subprocess
import sys
import tempfile

FILES = [
    "onereplay/pbs/63_math_gold_onereplay.pbs",
    "onereplay/pbs/64_math_gold_ewc.pbs",
    "onereplay/pbs/65_math_gold_replay.pbs",
]


def wsl_path(win_path):
    win_path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(win_path)
    return "/mnt/" + drive[0].lower() + rest.replace("\\", "/")


def check(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        raw = handle.read()
    lf = raw.replace("\r\n", "\n")
    n_lines = lf.count("\n") + (0 if lf.endswith("\n") else 1)
    crlf = raw.count("\r\n")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, encoding="utf-8", newline=""
    )
    tmp.write(lf)
    tmp.close()
    try:
        proc = subprocess.run(
            ["bash", "-n", wsl_path(tmp.name)],
            capture_output=True,
        )
        def clean(blob):
            # WSL prints a UTF-16 localhost-proxy notice on stderr; drop anything
            # that is not plain ASCII so the report stays readable on a GBK console.
            text = blob.decode("utf-8", "ignore")
            return "".join(ch for ch in text if ch.isascii()).strip()

        out = clean(proc.stdout)
        err = clean(proc.stderr)
        status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        print(f"{path}: lines={n_lines} crlf_pairs={crlf} bash -n => {status}")
        if out:
            print("  stdout:", out)
        if err:
            print("  stderr:", err)
    finally:
        os.unlink(tmp.name)

    # heredoc pairing: every `<<'TAG'` needs a matching bare `TAG` line
    opens = re.findall(r"<<'([A-Za-z_][A-Za-z0-9_]*)'", lf)
    for tag in sorted(set(opens)):
        n_open = opens.count(tag)
        n_close = len(re.findall(rf"^{tag}\s*$", lf, re.M))
        flag = "OK" if n_open == n_close else "MISMATCH"
        print(f"  heredoc <<'{tag}': open={n_open} close={n_close} {flag}")
    return proc.returncode


rc = 0
for f in FILES:
    rc |= check(f)
sys.exit(rc)
