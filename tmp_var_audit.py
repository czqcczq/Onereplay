"""Temporary: list every ${VAR} referenced in the new PBS scripts that is never
assigned in the same file and has no :- default, plus grep for code-domain leftovers."""

import re

FILES = [
    "onereplay/pbs/63_math_gold_onereplay.pbs",
    "onereplay/pbs/64_math_gold_ewc.pbs",
    "onereplay/pbs/65_math_gold_replay.pbs",
]

# Provided by PBS / the shell / the container, never assigned by these scripts.
EXTERNAL = {
    "PBS_JOBID", "PBS_O_WORKDIR", "PYTHONPATH", "USER", "HOME", "PATH", "IFS",
    "SECONDS", "HOSTNAME", "SHELL", "TERM", "PWD", "BASH_SOURCE", "FUNCNAME",
    "RANDOM", "LINENO", "OLDPWD",
}

# Loop / read / local variables introduced by the shell itself.
LOOP_VARS = {
    "_v", "_init", "m", "v", "f", "r", "spec", "specs", "extra", "name",
    "adapter", "mp", "ip", "cp", "run", "fmix", "variant", "arm", "weights",
    "kind", "short", "out", "lam", "s", "n", "i", "w", "text", "path",
}

CODE_LEFTOVERS = [
    "humaneval", "HumanEval", "mbpp", "MBPP", "opcedu", "opc_edu", "OPC",
    "ccode", "CODE_DATA_DIR", "CODE_MAX_NEW_TOKENS", "TIMEOUT", "code_exec",
]

ASSIGN = re.compile(r"^\s*(?:export\s+|local\s+|declare\s+-\w+\s+)?([A-Za-z_]\w*)=", re.M)
ARRAY_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\+?=\(", re.M)
FOR_VAR = re.compile(r"^\s*for\s+([A-Za-z_]\w*)\s+in\b", re.M)
READ_VAR = re.compile(r"\bread\s+(?:-\w+\s+)*([A-Za-z_]\w*)", re.M)
LOCAL_DECL = re.compile(r"^\s*local\s+(.+)$", re.M)
FUNC_ARR = re.compile(r"^\s*([A-Za-z_]\w*)\+=\(", re.M)

REF = re.compile(r"\$\{([A-Za-z_]\w*)([^}]*)\}|\$([A-Za-z_]\w*)")


def strip_heredocs(text):
    """Drop quoted-heredoc bodies: they are Python, not shell, so ${...} inside
    them is either absent or not expanded by bash."""

    out, skip_until = [], None
    for line in text.split("\n"):
        if skip_until is None:
            match = re.search(r"<<'([A-Za-z_]\w*)'", line)
            out.append(line)
            if match:
                skip_until = match.group(1)
        else:
            if line.strip() == skip_until:
                skip_until = None
    return "\n".join(out)


for path in FILES:
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()
    body = strip_heredocs(raw)
    # comments do not need to resolve, but keep them out of the reference scan
    code = "\n".join(
        line for line in body.split("\n") if not line.lstrip().startswith("#")
    )

    assigned = set(ASSIGN.findall(code)) | set(ARRAY_ASSIGN.findall(code))
    assigned |= set(FOR_VAR.findall(code)) | set(READ_VAR.findall(code))
    assigned |= set(FUNC_ARR.findall(code))
    for decl in LOCAL_DECL.findall(code):
        for token in decl.split():
            assigned.add(token.split("=")[0])
    # FORWARD_VARS lists names that are only ever read via ${!_v:-}
    forward = re.search(r"FORWARD_VARS=\((.*?)\n  \)", code, re.S)
    forward_names = set(forward.group(1).split()) if forward else set()

    missing = {}
    for match in REF.finditer(code):
        var = match.group(1) or match.group(3)
        modifier = match.group(2) or ""
        if var in EXTERNAL or var in LOOP_VARS or var in assigned:
            continue
        if modifier.startswith(":-") or modifier.startswith(":+") or modifier.startswith("-"):
            continue
        if var.startswith("#") or var in {"1", "2", "3"}:
            continue
        missing.setdefault(var, code[:match.start()].count("\n") + 1)

    print(f"\n=== {path} ===")
    print(f"assigned names: {len(assigned)}   FORWARD_VARS entries: {len(forward_names)}")
    unforwarded = sorted(n for n in assigned if n.isupper() and n not in forward_names)
    print("uppercase assigned but NOT in FORWARD_VARS:", unforwarded)
    if missing:
        print("!! referenced without assignment or default:")
        for var, line in sorted(missing.items()):
            print(f"   {var}  (first use around line {line})")
    else:
        print("referenced-but-unassigned: none")

    hits = []
    for needle in CODE_LEFTOVERS:
        for num, line in enumerate(raw.split("\n"), 1):
            if needle in line:
                hits.append((needle, num, line.strip()))
    if hits:
        print("!! code-domain leftovers:")
        for needle, num, line in hits:
            print(f"   [{needle}] L{num}: {line[:120]}")
    else:
        print("code-domain leftovers: none")
