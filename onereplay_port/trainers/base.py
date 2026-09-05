"""Shared training loop with OneReplay regularization injection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from onereplay.core.profiling import (
    PhaseTimer,
    current_memory_gb,
    format_cost_summary,
    peak_memory_stats,
    reset_peak_memory,
)


# =============================================================================
# PORT NOTE 文件状态：[REFERENCE-ONLY] 不搬这个文件
#
# onereplay/trainers/base.py 的逐字节副本，只加注释、未改代码行。放在这里是为了对照，
# 逻辑要在 litgpt/pretrain.py 的 fit() 里重写，不要把这个类搬过去——它自带 probe /
# evaluate_loss / jsonl 日志一整套，和 LitGPT 的 validate() + Logger 会打架。
#
# 真正要移植的只有**注入时序**。下面是逐行对应表（右侧是 litgpt/pretrain.py 行号）：
#
#   base.py                                        litgpt/pretrain.py
#   ---------------------------------------------  --------------------------------
#   for step, batch in enumerate(loader, start=1)  for train_data in train_iterator  L336
#   step（1 起）                                    state["iter_num"]，L345 先自增
#   accumulation_steps = accum_size // batch_size  train.gradient_accumulation_iters(
#     （SFT 里是 8）                                  devices, num_nodes)  args.py L56-60
#                                                    = 512/(devices*nodes)/4，单卡 = 128
#   window_open = (step-1) % accum == 0   L265     state["iter_num"] % accum == 1
#     ★ 注意 iter_num 在 L345 已自增，所以窗口开头是**余 1**、不是余 0
#   regularizer.accumulate_grad(...)      L268     插在 L351 之后、L352 之前
#   task_loss / accumulation_steps        L276     fabric.backward(loss / accum)   L355
#     （两边都除，只是除的位置不同，等价）
#   loss.backward()                       L281     fabric.backward(...)            L355
#   step % accum == 0（窗口末尾）          L283     not is_accumulating             L359
#     ★ is_accumulating = iter_num % accum != 0    L351，语义与 window_open **相反**
#   optimizer.step()                      L285     optimizer.step()                L361
#   optimizer.zero_grad()                 L286     optimizer.zero_grad()           L362
#   reset_peak_memory(...)                L248     无对应物，要自己加
#
# base.py 里**没有**对应物的三处，全都要处理：
#
#   1. fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)   L360
#      max_norm 在 pretrain.py L62 默认 1.0，且 validate_args（L524）要求必填，
#      所以裁剪**一定是开着的**。惩罚梯度会一起被裁，等于 lambda 被隐式乘上一个随 step
#      变化的系数，而且 onereplay 臂和 baseline 臂的系数还不一样——这会直接污染成本/效果对比。
#      处理方案见 core/regularizer.py 里 accumulate_grad 的批注（我倾向注入放在 clip 之后）。
#
#   2. with fabric.no_backward_sync(model, enabled=is_accumulating)       L352
#      FSDP 下窗口内抑制梯度 all-reduce。注入必须在这个 with 块**外面**：块内写 .grad
#      会不会被 reduce 取决于 strategy，别赌。写在外面语义明确。
#
#   3. 尾部残窗。base.py L283 有 `or step == total_steps`，会把最后一个不满的窗口也 step 掉；
#      LitGPT 在 L337-338 直接 break，残窗丢弃。对结果无影响，但如果按"余 1"判断注入时机，
#      break 可能发生在已注入、未 step 之间，那一次惩罚梯度就白算了（也不会被用掉）。
#      再一个改用"在 not is_accumulating 分支里注入"的理由。
#
# 还有一处语义差异要留意：这里的 reg_once_per_update / reg_scale（L217-218）是为了让
# "每窗口注入一次"和"每 micro-batch 注入 1/accum"两种写法可切换、便于等价性校验。
# 移植时这个开关值得保留——accum 从 8 变 128，等价性验证比以前更有必要做一次。
# =============================================================================


class BaseTrainer:
    """Common loop: prepare_batch -> task_loss -> + lambda * reg -> step."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
        regularizer: Any | None = None,
        replay_lambda: float = 0.0,
        batch_size: int = 8,
        accumulation_size: int = 64,
        log_every: int = 500,
        metrics_path: str = "",
        max_steps: int = 0,
        profile: int = 0,
        reg_once_per_update: int = 1,
        probe_loaders: dict[str, Any] | None = None,
        probe_every: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.regularizer = regularizer
        self.replay_lambda = float(replay_lambda)
        self.batch_size = batch_size
        self.accumulation_size = accumulation_size
        self.log_every = log_every
        self.metrics_path = metrics_path
        # max_steps > 0 stops an epoch early; used by the refactor parity check.
        self.max_steps = max_steps
        # 0 restores the old behavior of penalizing on every micro-batch. Kept
        # so the equivalence check can flip one flag instead of one git version.
        self.reg_once_per_update = reg_once_per_update
        # profile=1 adds per-phase timers. They need cuda synchronize, which
        # itself costs throughput, so leave this off for production runs and
        # turn it on for a short cost-measurement run.
        self.timer = PhaseTimer(device, enabled=bool(profile))
        self.last_epoch_cost: dict[str, Any] = {}
        # Fixed held-out sets scored every probe_every micro-batches. Their
        # cost is subtracted from the training clock rather than folded into
        # it, so a probed run still reports the same ms/step as a bare one.
        self.probe_loaders = probe_loaders or {}
        self.probe_every = probe_every
        self.global_step = 0
        self.probe_sec_total = 0.0

    def profiled_devices(self) -> list[torch.device | str]:
        """Devices whose peak memory belongs to this trainer's cost."""

        return [self.device]

    def _covariance_memory_gb(self) -> float:
        """Resident size of the covariance matrices, 0 when no regularizer."""

        memory_bytes = getattr(self.regularizer, "memory_bytes", None)
        return memory_bytes() / 1024**3 if callable(memory_bytes) else 0.0

    def _reference_memory_gb(self) -> float:
        """Resident size of the W0 snapshot; 0 on the LoRA path."""

        memory_bytes = getattr(self.regularizer, "reference_memory_bytes", None)
        return memory_bytes() / 1024**3 if callable(memory_bytes) else 0.0

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Turn a raw batch into a form ready for compute_task_loss.

        SFT: identity. OPD: student on-policy rollout + teacher scoring.
        """

        return batch

    def compute_task_loss(self, prepared: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def training_step(
        self, batch: dict[str, Any], with_reg: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        """Task loss for this micro-batch, plus the penalty when one is due.

        The penalty is returned separately instead of folded into a single loss
        because the two terms carry different scales: the task loss is one
        sample of an average over the accumulation window, while the penalty is
        a single term per optimizer step. train_one_epoch combines them.

        with_reg=False skips the penalty entirely, which is the whole point of
        computing it once per window rather than once per micro-batch.
        """

        with self.timer.track("prepare_batch"):
            prepared = self.prepare_batch(batch)
        with self.timer.track("task_loss"):
            task_loss = self.compute_task_loss(prepared)

        stats = {"task_loss": float(task_loss.detach().cpu())}
        if not (with_reg and self.regularizer is not None):
            return task_loss, None, stats

        with self.timer.track("replay_reg"):
            reg, reg_stats = self.regularizer(self.model)
        stats["replay_reg"] = float(reg.detach().cpu())
        stats["used_layers"] = float(reg_stats.get("used_layers", 0.0))
        stats["missing_layers"] = float(reg_stats.get("missing_layers", 0.0))
        return task_loss, reg, stats

    def evaluate_loss(self, valid_loader) -> float:
        """Compute mean validation task loss (no regularizer)."""

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in valid_loader:
                prepared = self.prepare_batch(batch)
                task_loss = self.compute_task_loss(prepared)
                total_loss += float(task_loss.detach().cpu())
                total_batches += 1
        self.model.train()
        return total_loss / max(total_batches, 1)

    def evaluate_probe(self, loader) -> float:
        """Token-weighted mean cross-entropy over a fixed probe set.

        evaluate_loss averages over batches, which is fine for a fixed
        Commonsense split whose rows all carry ~7 supervised tokens. FLAN rows
        run from a couple of tokens to a couple of hundred, so a batch mean
        would let short answers dominate and would move when the batch size
        changes. Weighting by supervised tokens makes the number an honest
        per-token loss and keeps the two FLAN curves subtractable.

        The count follows the shift the causal LM loss applies internally:
        position 0 never has a target, so it is not one of the terms averaged.
        """

        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in loader:
                prepared = self.prepare_batch(batch)
                task_loss = self.compute_task_loss(prepared)
                supervised = int((prepared["labels"][..., 1:] != -100).sum())
                if supervised == 0:
                    continue
                total_loss += float(task_loss.detach().cpu()) * supervised
                total_tokens += supervised
        return total_loss / max(total_tokens, 1)

    def run_probes(self, epoch: int) -> float:
        """Score every probe set, log one record, return the seconds it took."""

        if not self.probe_loaders:
            return 0.0

        was_training = self.model.training
        self.model.eval()
        start = time.time()
        values = {name: self.evaluate_probe(loader) for name, loader in self.probe_loaders.items()}
        probe_sec = time.time() - start
        if was_training:
            self.model.train()

        self.probe_sec_total += probe_sec
        accumulation_steps = max(self.accumulation_size // self.batch_size, 1)
        record = {
            "record_type": "probe",
            "global_step": self.global_step,
            # The x axis the curves are plotted on. global_step counts
            # micro-batches, which is twice as many on a batch-mixed replay run
            # for the same amount of new-task data.
            "update": self.global_step // accumulation_steps,
            "epoch": epoch,
            "probe_sec": probe_sec,
            **{f"probe_{name}": value for name, value in values.items()},
        }
        summary = " ".join(f"{name}={value:.6f}" for name, value in values.items())
        print(
            f"probe update {record['update']} (step {self.global_step}) "
            f"{summary} ({probe_sec:.1f}s)",
            flush=True,
        )
        self._append_jsonl(record)
        return probe_sec

    # PORT NOTE [REFERENCE-ONLY] 这个方法是全文重点，对应 pretrain.py 的 fit() L336-375。
    #
    # 读的时候只看三样东西：L265-271 的注入时机、L276-281 的 loss 组装、L283-286 的 step 边界。
    # 其余（epoch 均值、window 吞吐、probe）在 LitGPT 里由 throughput/Logger 承担。
    #
    # 一个概念差异：预训练没有 epoch。LitGPT 按 max_iters / max_tokens 走单层循环
    # （train_iterator 只在 L379 暴露一个 epoch 字段用于日志）。所以
    # "train_one_epoch 返回均值、train() 逐 epoch 记录"这套结构没有对应物，
    # 成本统计要改成按固定 iter 窗口（比如每 log_iter_interval）汇总。
    #
    # L219-231 那段长注释（bf16 下 .grad 只有 8 位尾数、所以必须注入在空 buffer 上）
    # 在 bf16-mixed 下不再成立：.grad 是 fp32，窗口内任意位置注入等价。
    # 这段推理值得保留在论文附录里解释旧实现，但不要作为新实现的约束。
    def train_one_epoch(self, train_loader, epoch: int = 0) -> tuple[float, float]:
        """Train one epoch; return mean task_loss and mean replay_reg."""

        accumulation_steps = max(self.accumulation_size // self.batch_size, 1)
        # The penalty tr(dW C dW^T) reads only model weights, and weights stay
        # frozen until the optimizer steps at the end of the window. So every
        # micro-batch in a window would compute the identical number, and
        # accumulation_steps - 1 of those computations are pure waste.
        #
        # Adding lambda*R once, *outside* the /accumulation_steps division,
        # accumulates the same gradient as adding lambda*R/accumulation_steps on
        # every micro-batch, at 1/accumulation_steps of the cost.
        reg_once = self.reg_once_per_update == 1
        reg_scale = 1.0 if reg_once else 1.0 / accumulation_steps
        # The analytic path writes dR/dW into .grad itself, so the penalty never
        # enters the loss and there is nothing for backward to traverse. Since the
        # weights do not move inside a window, DeltaW is the same at either end of
        # it and the injection could go anywhere -- but not with the same result in
        # bf16. Full fine-tuning keeps parameters in bf16, so .grad is bf16 too,
        # with 8 mantissa bits. Injected last, the penalty gradient would land on a
        # buffer already holding all accumulation_steps micro-batches of task
        # gradient; at the ~1% relative size the penalty runs at, that is only ~2x
        # the buffer's resolution and most of it would quantize away. Injected
        # first, it lands on an empty buffer, which is exactly where autograd puts
        # it today (reg_once adds lambda*R to the first micro-batch's loss). Same
        # position, same rounding, so the only difference left between the two
        # implementations is the matmul precision this test is about.
        inject_reg = self.regularizer is not None and getattr(
            self.regularizer, "injects_grad", False
        )
        reg_stats: dict[str, float] = {}
        total_steps = len(train_loader)
        if self.max_steps > 0:
            total_steps = min(total_steps, self.max_steps)
        self.model.train()
        self.optimizer.zero_grad()
        total_task_loss = 0.0
        total_replay_reg = 0.0
        total_samples = 0
        total_tokens = 0
        done_steps = 0

        self.timer.reset()
        reset_peak_memory(self.profiled_devices())
        epoch_start = time.time()
        # A running average keeps amortizing the cold start over the whole epoch,
        # so on a short run it never reveals the steady-state cost. Report the
        # speed of each log window alongside it.
        window_start = epoch_start
        window_first_step = 1
        # The penalty is only evaluated on some steps now, so the reported value
        # and the epoch mean reuse the one computed for the current window.
        window_reg = 0.0

        for step, batch in enumerate(train_loader, start=1):
            if step > total_steps:
                break
            num_samples = batch["input_ids"].shape[0] if "input_ids" in batch else 1
            if "attention_mask" in batch:
                total_tokens += int(batch["attention_mask"].sum())
            window_open = (step - 1) % accumulation_steps == 0
            if inject_reg and window_open:
                with self.timer.track("replay_reg"):
                    window_reg, reg_stats = self.regularizer.accumulate_grad(
                        self.model, self.replay_lambda
                    )
            reg_due = not inject_reg and (not reg_once or window_open)
            task_loss, reg, stats = self.training_step(batch, with_reg=reg_due)
            if inject_reg:
                stats["replay_reg"] = window_reg
                stats.update(reg_stats)
            loss = task_loss / accumulation_steps
            if reg is not None:
                loss = loss + self.replay_lambda * reg_scale * reg
                window_reg = stats["replay_reg"]
            with self.timer.track("backward"):
                loss.backward()

            if step % accumulation_steps == 0 or step == total_steps:
                with self.timer.track("optimizer"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_task_loss += stats["task_loss"] * num_samples
            total_replay_reg += window_reg * num_samples
            total_samples += num_samples
            done_steps = step
            if step == 1 and "used_layers" in stats:
                print(
                    f"regularizer covers {int(stats['used_layers'])} layers; "
                    f"no penalty matrix for {int(stats['missing_layers'])} layers"
                )
            if self.log_every > 0 and (step % self.log_every == 0 or step == total_steps):
                now = time.time()
                window_steps = step - window_first_step + 1
                window_ms = (now - window_start) / max(window_steps, 1) * 1000
                print(
                    f"step {step}/{total_steps} "
                    f"task_loss={stats['task_loss']:.6f} "
                    f"replay_reg={window_reg:.6e} "
                    f"{(now - epoch_start) / step * 1000:.0f}ms/step "
                    f"win={window_ms:.1f}ms/step "
                    f"mem={current_memory_gb(self.device):.2f}GiB",
                    flush=True,
                )
                window_start = now
                window_first_step = step + 1

            self.global_step += 1
            if self.probe_every > 0 and self.global_step % self.probe_every == 0:
                probe_sec = self.run_probes(epoch)
                # Push both clocks forward by what the probe cost, so ms/step,
                # the win= field and train_sec stay pure training time. Without
                # this the steady-state percentile the cost tables are built on
                # would be polluted in whichever window a probe fell into.
                epoch_start += probe_sec
                window_start += probe_sec

        train_sec = time.time() - epoch_start
        self.last_epoch_cost = {
            "train_sec": train_sec,
            "train_steps": done_steps,
            "train_samples": total_samples,
            "train_tokens": total_tokens,
            "sec_per_step": train_sec / max(done_steps, 1),
            "samples_per_sec": total_samples / max(train_sec, 1e-9),
            "tokens_per_sec": total_tokens / max(train_sec, 1e-9),
            "covariance_memory_gb": self._covariance_memory_gb(),
            "reference_memory_gb": self._reference_memory_gb(),
            **peak_memory_stats(self.profiled_devices()),
            **self.timer.as_record(),
        }

        return total_task_loss / max(total_samples, 1), total_replay_reg / max(total_samples, 1)

    def train(
        self,
        train_loader,
        epochs: int,
        val_loader=None,
        extra_record: dict[str, Any] | None = None,
        save_path: str = "",
        tokenizer=None,
    ) -> list[dict[str, Any]]:
        """Run epochs, optionally evaluate and save the final adapter."""

        records: list[dict[str, Any]] = []
        # The anchor for every probe curve. On the self-distilled targets this
        # point is the loss of the untouched base model against its own
        # answers, so it is near zero and every later value reads directly as
        # "how far the model has drifted from W0".
        self.run_probes(epoch=0)
        for epoch in range(epochs):
            start_time = time.time()
            train_loss, replay_reg = self.train_one_epoch(train_loader, epoch=epoch + 1)

            eval_start = time.time()
            val_loss = self.evaluate_loss(val_loader) if val_loader is not None else None
            eval_sec = time.time() - eval_start if val_loader is not None else None

            record = {
                "record_type": "epoch",
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "train_task_loss": train_loss,
                "train_replay_reg": replay_reg,
                "train_lambda_reg": self.replay_lambda * replay_reg,
                "val_loss": val_loss,
                "elapsed_sec": time.time() - start_time,
                "replay_lambda": self.replay_lambda,
                "eval_sec": eval_sec,
                **self.last_epoch_cost,
            }
            if self.probe_loaders:
                # elapsed_sec is wall clock and still contains the probes;
                # train_sec and sec_per_step already have them removed.
                record["probe_sec_total"] = self.probe_sec_total
            if extra_record:
                record.update(extra_record)
            print(json.dumps(record, ensure_ascii=False, indent=2))
            print(format_cost_summary(record), flush=True)
            self._append_jsonl(record)
            records.append(record)

        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(save_path)
            # PEFT writes an adapter, a plain model writes full weights. The
            # eval loader tells them apart by looking for adapter_config.json.
            print(f"saved final model to {save_path}")
        return records

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        if not self.metrics_path:
            return
        Path(self.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
