"""Run-wide system message for chat rendering.

Qwen2.5-Math's chat template injects
``Please reason step by step, and put your final answer within \\boxed{}.``
whenever the message list carries no system turn. On the math line that is the
intended "default chat template + CoT prompting" recipe, but the IF and
commonsense lines render every training prompt and every evaluation prompt
through the same helpers, so they would silently inherit a math instruction.

The fix has to reach four call sites that never see each other: training
(data/chat.py), evaluation (eval/generation.py), and the C/F collectors (which
already take --system_prompt of their own). Threading an argument through all
of them means every metric would have to remember to forward it, so it lives
here as run-wide state that train.py and evaluate.py set once at startup --
the same shape as configure_decoding.

The default is the empty string, which reproduces the behavior from before this
module existed: no system turn, template decides.
"""

from __future__ import annotations

_SYSTEM_PROMPT = ""


def configure_system_prompt(system_prompt: str = "") -> None:
    """Set the system turn prepended to every rendered conversation."""

    global _SYSTEM_PROMPT
    _SYSTEM_PROMPT = (system_prompt or "").strip()


def current_system_prompt() -> str:
    """The system turn in effect, or an empty string when none is set."""

    return _SYSTEM_PROMPT


def with_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return messages with the configured system turn in front.

    A conversation that already carries its own system turn is returned
    untouched, so a metric with a task-specific system message keeps it.
    """

    if not _SYSTEM_PROMPT:
        return messages
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": _SYSTEM_PROMPT}, *messages]


__all__ = ["configure_system_prompt", "current_system_prompt", "with_system"]
