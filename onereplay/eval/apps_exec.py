"""Sandboxed execution for stdin/stdout programming problems (APPS).

Why this is separate from code_exec.py
--------------------------------------
code_exec.py judges MBPP/HumanEval, where the model writes a *function* and the
tests are assert statements or a check() harness. APPS is competitive
programming: the program reads a test case from stdin and prints the answer, so
the whole notion of "call the entry point" does not apply. Its cleanup rules are
incompatible too -- cleanup_completion() truncates at the first '\\nclass ',
'\\ndef ', '\\nif __name__' and '\\nprint(', all four of which are normal and
required in an APPS solution.

The runner below is adapted from PettingLLMs (pettingllms/multi_agent_env/code/
code_worker.py, MIT). Their version is wrapped in Ray actors for parallel RL
rollouts; that layer is stripped here. Compared with the official APPS harness
(apps/eval/testing_util.py) this approach has no `pyext` dependency, does not
install a SIGALRM handler in the parent, and never monkeypatches os/shutil/
subprocess in the calling process -- isolation comes from actually forking a
fresh interpreter per test case, which is what makes it safe to run inside a
long-lived evaluation job.

Judging is whitespace-normalised exact match. That is stricter than the official
APPS harness, which falls back through six comparison strategies including
float tolerance via np.allclose. Absolute scores will therefore sit below
published APPS numbers; comparisons between runs remain valid.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any

MAX_OUTPUT_BYTES = 1 << 20

# Executed as a separate process. Reads the whole of stdin, exposes it both as
# sys.stdin and through a fake input(), then execs the candidate program with
# __name__ == "__main__" so that `if __name__ == "__main__":` blocks fire.
#
# Exceptions are deliberately NOT caught: letting the child die with a non-zero
# exit code keeps the traceback on stderr, which is what lets the caller bucket
# failures by exception type (SyntaxError vs IndexError vs EOFError). Catching
# them and printing to stdout would also corrupt the output comparison.
_RUNNER_SOURCE = '''
import io
import sys
import typing


def _main():
    data = sys.stdin.read()
    lines = iter(data.splitlines())

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError("No more input")

    sys.stdin = io.StringIO(data)
    context = {
        "__name__": "__main__",
        "input": fake_input,
        "List": typing.List,
        "Tuple": typing.Tuple,
        "Dict": typing.Dict,
        "Set": typing.Set,
        "Optional": typing.Optional,
    }
    with open("script.py", encoding="utf-8") as handle:
        source = handle.read()
    try:
        exec(compile(source, "script.py", "exec"), context)
    except SystemExit:
        pass


_main()
'''


def extract_apps_code(text: str) -> str:
    """Pull the program out of a chat response.

    Takes the LAST fenced block rather than the longest one: the prompt asks for
    a single ```python block, but a model that reasons in the open will emit
    draft attempts first and the final one is the answer. This is the opposite
    of cleanup_completion()'s "longest block" rule, which is right for MBPP
    (one short function) and wrong here.
    """

    import re

    for pattern in (r"```python\s*(.*?)```", r"```\s*(.*?)```"):
        blocks = re.findall(pattern, text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
    return text.strip()


def looks_truncated(text: str) -> bool:
    """Heuristic: the generation ran out of token budget mid-answer.

    An unbalanced number of fences means a block was opened and never closed.
    Used only as a diagnostic, never to fail a problem.
    """

    return text.count("```") % 2 == 1


def outputs_match(actual: str, expected: str) -> bool:
    """Whitespace-normalised exact match, as used by PettingLLMs' test_if_eq."""

    return " ".join(actual.split()) == " ".join(expected.split())


def _kill_process_group(proc: subprocess.Popen) -> None:
    if os.name == "posix" and hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _read_capped(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return handle.read(MAX_OUTPUT_BYTES).decode(errors="replace")
    except Exception:
        return ""


def run_program_on_stdin(
    code: str,
    stdin_text: str,
    timeout: float,
    tmp_root: str = "",
) -> tuple[str | None, str]:
    """Run one program against one stdin payload in a throwaway subprocess.

    Returns (stdout, error). stdout is None whenever the program did not exit
    cleanly, in which case error carries a short reason: 'timeout' or the last
    line of the traceback (e.g. 'SyntaxError: invalid syntax').
    """

    workdir = tempfile.mkdtemp(prefix="apps_exec_", dir=tmp_root or None)
    try:
        script_path = os.path.join(workdir, "script.py")
        runner_path = os.path.join(workdir, "runner.py")
        stdin_path = os.path.join(workdir, "stdin.txt")
        stdout_path = os.path.join(workdir, "stdout.txt")
        stderr_path = os.path.join(workdir, "stderr.txt")

        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(code)
        with open(runner_path, "w", encoding="utf-8") as handle:
            handle.write(_RUNNER_SOURCE)
        with open(stdin_path, "w", encoding="utf-8") as handle:
            handle.write(stdin_text)

        env = dict(os.environ)
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "OMP_NUM_THREADS": "1",
            }
        )
        # A candidate program may spawn children or ignore SIGTERM; start_new_session
        # puts it in its own process group so a timeout can take the whole tree down.
        popen_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        with open(stdin_path, "rb") as fin, open(stdout_path, "wb") as fout, open(
            stderr_path, "wb"
        ) as ferr:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-u", runner_path],
                stdin=fin,
                stdout=fout,
                stderr=ferr,
                cwd=workdir,
                env=env,
                **popen_kwargs,
            )
            try:
                returncode = proc.wait(timeout=max(timeout, 1.0))
            except subprocess.TimeoutExpired:
                _kill_process_group(proc)
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                return None, "timeout"

        if returncode == 0:
            return _read_capped(stdout_path), ""

        stderr_text = _read_capped(stderr_path).strip()
        reason = stderr_text.splitlines()[-1].strip() if stderr_text else f"exit {returncode}"
        return None, reason
    except Exception as exc:  # harness failure, not a candidate failure
        return None, f"harness error: {exc}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def evaluate_stdin_program(
    code: str,
    test_inputs: list[str],
    test_outputs: list[str],
    timeout: float,
    tmp_root: str = "",
    max_tests: int = 0,
) -> dict[str, Any]:
    """Run a program against a problem's test cases.

    max_tests > 0 keeps only the first N cases. Some APPS problems ship over a
    hundred cases and each costs a process launch plus up to `timeout` seconds,
    so the full set is not affordable across several runs. Truncating makes
    strict pass@1 easier than the official APPS number, so it must be identical
    for every run being compared and reported alongside the score.

    Returns strict pass (all evaluated cases matched), the pass ratio, and the
    first error seen -- the ratio is the higher-resolution signal: it is
    continuous per problem, so a paired test on it detects much smaller effects
    than McNemar on the binary strict outcome.
    """

    total = min(len(test_inputs), len(test_outputs))
    if max_tests > 0:
        total = min(total, max_tests)
    if total == 0:
        return {
            "passed": False,
            "pass_ratio": 0.0,
            "num_tests": 0,
            "num_passed": 0,
            "error": "no test cases",
        }

    if not code.strip():
        return {
            "passed": False,
            "pass_ratio": 0.0,
            "num_tests": total,
            "num_passed": 0,
            "error": "empty completion",
        }

    num_passed = 0
    first_error = ""
    for index in range(total):
        stdout, error = run_program_on_stdin(
            code, test_inputs[index], timeout, tmp_root=tmp_root
        )
        if stdout is None:
            if not first_error:
                first_error = error
            continue
        if outputs_match(stdout, test_outputs[index]):
            num_passed += 1
        elif not first_error:
            first_error = "wrong answer"

    return {
        "passed": num_passed == total,
        "pass_ratio": num_passed / total,
        "num_tests": total,
        "num_passed": num_passed,
        "error": "" if num_passed == total else first_error,
    }
