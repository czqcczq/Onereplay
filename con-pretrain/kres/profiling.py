"""训练循环的时间与显存计量。这是 LitGPT 给不了、而成本表必须有的东西。

LitGPT 自带的 throughput 日志（`pretrain.py` 的 metrics 块）报 tokens/s、flops 和一个
memory 项，但没有 peak allocated 的 C / W₀ 分解，也没有分阶段计时。而"本方法比 replay
贵多少"正是这次工作的核心指标之一，所以这份要自己做。

两类测量的成本完全不同：

- **显存峰值**读的是 CUDA allocator 自己的计数器，免费，所以一直开着。
- **分阶段计时**不免费。GPU launch 是异步的，要给一个阶段计时就得
  `torch.cuda.synchronize()`，那会把流水线排空、拖慢训练。所以 phase timer 默认关闭，
  只在专门测成本的 run 里打开。

三个口径问题，每个都会让成本表得出一个看起来合理但错的数：

1. **多卡不能求和。** 原实现把各卡峰值相加。FSDP 是每卡一个进程、每个进程只看到自己那
   张卡，求和既不是"单卡能不能装下"（那要看 max），也不是全局占用（每个进程还会各自打
   印一遍）。这里改成报 max，并且只在 rank 0 打印。审稿人关心的是"能不能塞进一张卡"。

2. **compile 会污染前若干步。** LitGPT 默认 `torch.compile`，前几步是编译时间，混进均值
   里会让 per-step 时间虚高好几倍。所以计时分两段：warmup（含编译）与 steady state，
   成本对比只用后者。不分段的话，"加了正则慢 30%"这种结论可能纯粹是两个 run 的编译
   时间差。

3. **per-iter 与 per-optimizer-step 差 grad_accum 倍。** LitGPT 的 iter 是 micro-batch，
   本项目的 `gradient_accumulation_iters` 约 125（取决于 global_batch_size），两套口径
   混在一份日志里必然读错。这里一律按 **per optimizer step** 报，并把运行时的真实
   grad_accum 一起打出来让读者能换算——不写死数字，因为它是配置的函数。

   这个数对本方法是重大利好，所以更要写明：惩罚每个 optimizer step 只注入一次，而 SFT
   那边 accum=8，同一份惩罚开销在 CPT 下被摊薄约 16 倍。成本表不写 accum 值，"开销只占
   x%"这个结论既无法复现，也很容易被质疑是挑了有利配置。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

import torch

BYTES_PER_GIB = 1024**3


def _is_cuda(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"


def normalize_devices(devices) -> list[torch.device]:
    """去重但保持调用方给的顺序。"""
    unique: OrderedDict[str, torch.device] = OrderedDict()
    for device in devices:
        resolved = torch.device(device)
        unique.setdefault(str(resolved), resolved)
    return list(unique.values())


def reset_peak_memory(devices) -> None:
    """开一个新的峰值统计窗口。

    要在 warmup 结束后调一次：编译期的临时显存峰值可能比 steady state 还高，不重置的话
    成本表报的是编译峰值，而那个数与训练本身无关。
    """
    for device in normalize_devices(devices):
        if _is_cuda(device):
            torch.cuda.reset_peak_memory_stats(device)


def peak_memory_stats(devices) -> dict[str, Any]:
    """上次重置以来的显存峰值，单位 GiB。

    `allocated` 是张量实际持有的，`reserved` 是缓存分配器从驱动拿走的——问"能不能塞进
    这张卡"时看的是后者。

    跨设备取 **max 而不是 sum**，理由见模块 docstring 第 1 条。
    """
    per_device: dict[str, dict[str, float]] = {}
    for device in normalize_devices(devices):
        if not _is_cuda(device):
            continue
        per_device[str(device)] = {
            "allocated_gb": torch.cuda.max_memory_allocated(device) / BYTES_PER_GIB,
            "reserved_gb": torch.cuda.max_memory_reserved(device) / BYTES_PER_GIB,
        }

    if not per_device:
        return {}

    stats: dict[str, Any] = {
        "peak_memory_allocated_gb": max(v["allocated_gb"] for v in per_device.values()),
        "peak_memory_reserved_gb": max(v["reserved_gb"] for v in per_device.values()),
    }
    if len(per_device) > 1:
        stats["peak_memory_by_device"] = per_device
    return stats


def current_memory_gb(device) -> float:
    """当前已 reserve 的显存，用于逐步日志。"""
    if not _is_cuda(device):
        return 0.0
    return torch.cuda.memory_reserved(device) / BYTES_PER_GIB


class PhaseTimer:
    """按阶段累计 wall time，并把 warmup 与 steady state 分开。

    `enabled=False` 时 `track()` 是空的上下文管理器，被测循环全速运行、计时键干脆不出现
    在记录里——不会留下一堆 0 让人误以为"这个阶段不花时间"。

    用法：每个 optimizer step 结束时调一次 `mark_step()`。前 `warmup_steps` 个 step 的
    耗时进 warmup 桶，之后进 steady 桶。per-step 的分母用 steady step 数，所以
    `warmup_steps` 设小了不会算错，只会把编译时间摊进结果里。
    """

    def __init__(
        self,
        device: torch.device | str,
        enabled: bool = False,
        warmup_steps: int = 20,
    ) -> None:
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self.warmup_steps = max(0, int(warmup_steps))
        # 只有 CUDA 需要在读时钟之前显式排空流水线
        self._sync = self.enabled and _is_cuda(self.device)
        self.reset()

    def reset(self) -> None:
        self.totals: dict[str, float] = {}
        self.warmup_totals: dict[str, float] = {}
        self.steps_seen = 0

    @property
    def in_warmup(self) -> bool:
        return self.steps_seen < self.warmup_steps

    @property
    def steady_steps(self) -> int:
        return max(0, self.steps_seen - self.warmup_steps)

    @contextmanager
    def track(self, name: str):
        if not self.enabled:
            yield
            return
        if self._sync:
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._sync:
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - start
            bucket = self.warmup_totals if self.in_warmup else self.totals
            bucket[name] = bucket.get(name, 0.0) + elapsed

    def mark_step(self) -> None:
        """每个 optimizer step 结束时调一次。"""
        self.steps_seen += 1

    def as_record(self, prefix: str = "time_") -> dict[str, float]:
        """steady state 的阶段耗时，总秒数 + 每 step 毫秒数。

        两个都给：总秒数用来算占比，每 step 毫秒数是跨 run 唯一可比的量（两个 run 的
        step 数往往不同）。
        """
        if not self.enabled:
            return {}
        record: dict[str, float] = {}
        steps = self.steady_steps
        for name, value in sorted(self.totals.items()):
            record[f"{prefix}{name}_sec"] = value
            if steps:
                record[f"{prefix}{name}_ms_per_step"] = value / steps * 1000.0
        record[f"{prefix}steady_steps"] = float(steps)
        for name, value in sorted(self.warmup_totals.items()):
            record[f"{prefix}warmup_{name}_sec"] = value
        return record


def cost_record(
    devices,
    grad_accum: int,
    train_sec: float | None = None,
    steps: int | None = None,
    tokens: int | None = None,
    timer: PhaseTimer | None = None,
    covariance_bytes: int = 0,
    reference_bytes: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把成本表需要的数汇总成一条记录。

    `grad_accum` 是必填而不是可选：per-step 的每一个数都要靠它才能换算回 per-iter，
    而惩罚开销的占比更是直接由它决定（见模块 docstring 第 3 条）。
    """
    record: dict[str, Any] = {"grad_accum": int(grad_accum)}
    if train_sec is not None:
        record["train_sec"] = train_sec
    if steps:
        record["steps"] = steps
        if train_sec:
            record["sec_per_step"] = train_sec / steps
    if tokens is not None:
        record["tokens"] = tokens
        if train_sec:
            record["tokens_per_sec"] = tokens / train_sec
    if covariance_bytes:
        record["covariance_memory_gb"] = covariance_bytes / BYTES_PER_GIB
    if reference_bytes:
        record["reference_memory_gb"] = reference_bytes / BYTES_PER_GIB
    record.update(peak_memory_stats(devices))
    if timer is not None:
        record.update(timer.as_record())
    if extra:
        record.update(extra)
    return record


def format_cost_summary(record: dict[str, Any]) -> str:
    """把一条成本记录渲染成可读的块。

    一律 per optimizer step，并且把 grad_accum 打在第一行——这份输出会被直接抄进论文的
    成本表，缺了 accum 值那张表就不可复现。
    """
    lines = ["---- cost ----"]
    lines.append(f"grad_accum           : {record.get('grad_accum', '?')}"
                 "   （以下 per-step 均指 per optimizer step）")

    train_sec = record.get("train_sec")
    if train_sec is not None:
        lines.append(f"train wall time      : {train_sec:.1f} s")
    if record.get("steps") is not None:
        lines.append(f"optimizer steps      : {record['steps']}")
    if record.get("sec_per_step") is not None:
        lines.append(f"per-step time        : {record['sec_per_step'] * 1000:.1f} ms")
    # CPT 里样本都是定长块，samples/s 没有可比性，只报 tokens/s
    if record.get("tokens_per_sec") is not None:
        lines.append(f"throughput           : {record['tokens_per_sec']:.0f} tokens/s")

    if record.get("peak_memory_allocated_gb") is not None:
        lines.append(
            f"peak GPU memory      : {record['peak_memory_allocated_gb']:.2f} GiB allocated / "
            f"{record['peak_memory_reserved_gb']:.2f} GiB reserved（单卡峰值，跨卡取 max）"
        )
    if record.get("covariance_memory_gb"):
        lines.append(f"  of which C         : {record['covariance_memory_gb']:.3f} GiB")
    if record.get("reference_memory_gb"):
        lines.append(f"  of which W₀        : {record['reference_memory_gb']:.3f} GiB")

    steady = record.get("time_steady_steps")
    if steady:
        lines.append(f"phase timing         : steady state {int(steady)} step"
                     "（warmup 与编译已排除）")
    for key in sorted(record):
        if key.startswith("time_") and key.endswith("_ms_per_step"):
            phase = key[len("time_") : -len("_ms_per_step")]
            share = ""
            total_key = f"time_{phase}_sec"
            if train_sec and record.get(total_key):
                share = f"  ({100.0 * record[total_key] / train_sec:.1f}% of train)"
            lines.append(f"  {phase:<18}: {record[key]:.2f} ms/step{share}")
    return "\n".join(lines)
