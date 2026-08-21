"""Self-test: evaluating the penalty once per optimizer step keeps the gradient.

The penalty tr(dW C dW^T) reads only weights, and weights are frozen across a
gradient-accumulation window, so computing it on every micro-batch returns the
same number accumulation_steps times over. Skipping those repeats must not
change what the optimizer sees:

    old   sum_i grad(L_i / A + lambda * R / A) = mean_i grad(L_i) + lambda*grad(R)
    new   grad(L_1 / A + lambda * R) + sum_{i>1} grad(L_i / A)  -- same value

This checks that claim directly on the accumulated .grad rather than on a loss
curve, using a toy Linear in float64 so ordinary rounding stays near 1e-16 and
any real discrepancy stands out. Runs on CPU in seconds.

It also pins down the one intentional behavior change: on a final window that
is shorter than accumulation_steps, the old code applied a fraction of the
penalty and the new code applies all of it. The new behavior is the correct one
(one full penalty per update), but it does alter that single update, so the
check measures it instead of letting it surface later as a surprise.

Usage: python -m onereplay.scripts.verify_reg_once_per_update
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from onereplay.trainers.base import BaseTrainer  # noqa: E402

IN_FEATURES = 6
OUT_FEATURES = 4
BATCH_SIZE = 2
ACCUMULATION_SIZE = 16  # 8 micro-batches per update at BATCH_SIZE=2
LAMBDA = 3e-2


class CountingRegularizer:
    """Stand-in for ReplayRegularizer: weights only, and it counts its calls.

    Same algebraic form as full_covariance_regularizer, sum((dW @ C) * dW),
    with W0 = 0 so dW is just the weight.
    """

    def __init__(self, covariance: torch.Tensor) -> None:
        self.covariance = covariance
        self.calls = 0

    def __call__(self, model: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
        self.calls += 1
        weight = model.weight
        reg = torch.sum((weight @ self.covariance) * weight)
        return reg, {"used_layers": 1.0, "missing_layers": 0.0}


class RecordingSGD(torch.optim.SGD):
    """Plain SGD that snapshots the accumulated gradient at every step.

    SGD without momentum keeps the comparison direct: any gradient difference
    shows up in the parameters scaled by lr, with no optimizer state to blur it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.snapshots: list[list[torch.Tensor]] = []

    def step(self, *args, **kwargs):  # type: ignore[override]
        self.snapshots.append(
            [
                parameter.grad.detach().clone()
                for group in self.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
        )
        return super().step(*args, **kwargs)


class ToyTrainer(BaseTrainer):
    def compute_task_loss(self, prepared: dict[str, torch.Tensor]) -> torch.Tensor:
        return nn.functional.mse_loss(self.model(prepared["input_ids"]), prepared["labels"])


def build_batches(num_steps: int) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(7)
    batches = []
    for _ in range(num_steps):
        inputs = torch.randn(BATCH_SIZE, IN_FEATURES, generator=generator, dtype=torch.float64)
        labels = torch.randn(BATCH_SIZE, OUT_FEATURES, generator=generator, dtype=torch.float64)
        batches.append(
            {
                "input_ids": inputs,
                "labels": labels,
                "attention_mask": torch.ones(BATCH_SIZE, 3, dtype=torch.long),
            }
        )
    return batches


def run(reg_once: int, num_steps: int) -> dict[str, object]:
    torch.manual_seed(0)
    model = nn.Linear(IN_FEATURES, OUT_FEATURES).double()
    covariance = torch.eye(IN_FEATURES, dtype=torch.float64) * 1.7 + 0.3
    regularizer = CountingRegularizer(covariance)
    optimizer = RecordingSGD(model.parameters(), lr=0.1)

    trainer = ToyTrainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        regularizer=regularizer,
        replay_lambda=LAMBDA,
        batch_size=BATCH_SIZE,
        accumulation_size=ACCUMULATION_SIZE,
        log_every=0,
        reg_once_per_update=reg_once,
    )
    task_loss, mean_reg = trainer.train_one_epoch(build_batches(num_steps))
    return {
        "snapshots": optimizer.snapshots,
        "calls": regularizer.calls,
        "task_loss": task_loss,
        "mean_reg": mean_reg,
        "weights": [parameter.detach().clone() for parameter in model.parameters()],
    }


def max_difference(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return max(
        float((one - other).abs().max()) for one, other in zip(left, right, strict=True)
    )


def check_full_windows() -> list[str]:
    """Whole windows only: gradients and weights must match to rounding."""

    failures: list[str] = []
    accumulation_steps = ACCUMULATION_SIZE // BATCH_SIZE
    num_steps = accumulation_steps * 3

    old = run(reg_once=0, num_steps=num_steps)
    new = run(reg_once=1, num_steps=num_steps)

    if old["calls"] != num_steps:
        failures.append(f"old path called the penalty {old['calls']} times, expected {num_steps}")
    if new["calls"] != 3:
        failures.append(f"new path called the penalty {new['calls']} times, expected 3")

    if len(old["snapshots"]) != len(new["snapshots"]):
        failures.append(
            f"update counts differ: {len(old['snapshots'])} vs {len(new['snapshots'])}"
        )
        return failures

    worst = 0.0
    for index, (old_grads, new_grads) in enumerate(zip(old["snapshots"], new["snapshots"])):
        difference = max_difference(old_grads, new_grads)
        worst = max(worst, difference)
        if difference > 1e-12:
            failures.append(f"update {index + 1} gradient differs by {difference:.3e}")

    weight_difference = max_difference(old["weights"], new["weights"])
    if weight_difference > 1e-12:
        failures.append(f"final weights differ by {weight_difference:.3e}")

    reg_difference = abs(old["mean_reg"] - new["mean_reg"])
    if reg_difference > 1e-9:
        failures.append(
            f"reported mean replay_reg differs by {reg_difference:.3e} "
            f"({old['mean_reg']} vs {new['mean_reg']}); the logged value should reuse "
            "the one computed for the current window"
        )

    print(
        f"full windows: {num_steps} steps, {len(new['snapshots'])} updates, "
        f"penalty calls {old['calls']} -> {new['calls']} "
        f"({old['calls'] / max(new['calls'], 1):.0f}x fewer), "
        f"worst gradient diff {worst:.3e}, weight diff {weight_difference:.3e}"
    )
    return failures


def check_partial_final_window() -> list[str]:
    """A short final window is the one update that legitimately changes."""

    failures: list[str] = []
    accumulation_steps = ACCUMULATION_SIZE // BATCH_SIZE
    remainder = 3
    num_steps = accumulation_steps + remainder

    old = run(reg_once=0, num_steps=num_steps)
    new = run(reg_once=1, num_steps=num_steps)

    if len(old["snapshots"]) != 2 or len(new["snapshots"]) != 2:
        failures.append(
            f"expected 2 updates, got {len(old['snapshots'])} and {len(new['snapshots'])}"
        )
        return failures

    first = max_difference(old["snapshots"][0], new["snapshots"][0])
    if first > 1e-12:
        failures.append(f"the full first window should still match, differs by {first:.3e}")

    second = max_difference(old["snapshots"][1], new["snapshots"][1])
    if second <= 1e-12:
        failures.append(
            "the short final window was expected to differ (old applies "
            f"{remainder}/{accumulation_steps} of the penalty, new applies all of it) "
            "but the gradients matched; the fix may not be active"
        )

    print(
        f"partial final window: {num_steps} steps = {accumulation_steps} + {remainder}; "
        f"first update diff {first:.3e} (must match), "
        f"final update diff {second:.3e} (expected, {remainder}/{accumulation_steps} "
        "of the penalty vs all of it)"
    )
    return failures


def main() -> None:
    failures = check_full_windows() + check_partial_final_window()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print(
        "OK: identical gradients on whole windows with 8x fewer penalty evaluations; "
        "only a short final window differs, as intended"
    )


if __name__ == "__main__":
    main()
