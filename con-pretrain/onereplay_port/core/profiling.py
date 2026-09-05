"""Wall-time and GPU-memory accounting for the training loop.

Two kinds of measurement live here, with very different costs:

Memory peaks come from the CUDA allocator's own counters, so reading them is
free and they are always collected.

Per-phase timing is not free: a GPU launch is asynchronous, so timing a phase
requires torch.cuda.synchronize(), which drains the pipeline and slows training
down. Phase timers therefore only run when profiling is explicitly enabled.
Epoch-level wall time is always collected because one synchronize per epoch is
negligible.
"""

# =============================================================================
# PORT NOTE 文件状态：整体搬运，这是 LitGPT 给不了的东西
#
# onereplay/core/profiling.py 的逐字节副本，只加注释、未改代码行。校验：
#     diff onereplay/core/profiling.py con-pretrain/onereplay_port/core/profiling.py
#
# LitGPT 自带的 throughput 日志（pretrain.py L365-390）只报 tokens/s、flops 和
# 一个 memory 项，没有 peak allocated 的 C / W0 分解，也没有分阶段计时。而成本对比正是
# 这次工作的核心指标，所以这份东西必须搬。
#
# 两处口径问题要在多卡下重新定义，见下面 peak_memory_stats 和 format_cost_summary 的批注。
# =============================================================================

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
    """De-duplicate a device list while keeping the caller's order."""

    unique: OrderedDict[str, torch.device] = OrderedDict()
    for device in devices:
        resolved = torch.device(device)
        unique.setdefault(str(resolved), resolved)
    return list(unique.values())


def reset_peak_memory(devices) -> None:
    """Start a fresh peak-memory window on every CUDA device we track."""

    for device in normalize_devices(devices):
        if _is_cuda(device):
            torch.cuda.reset_peak_memory_stats(device)


# PORT NOTE [MODIFY] 单卡原样正确；FSDP 多卡下 total_* 这个"求和"口径是错的。
#
# L69-70 把各卡峰值**相加**。单进程多卡时这个和还有点意义，但 FSDP 是每卡一个进程、每个进程
# 只看到自己那张卡，求和得到的数既不是"单卡能不能装下"（那要看 max），也不是全局占用
# （因为每个进程各自打印一遍）。
#
# 建议改成报两个数：per-card peak（各 rank 的 max，用 fabric.all_reduce(..., reduce_op="max")
# 聚合）和 rank0 的明细。成本表对外只报 per-card peak——"能不能塞进一张卡"才是审稿人关心的量。
# 另外记得只在 fabric.global_rank == 0 上打印，否则 8 卡会刷 8 份。
def peak_memory_stats(devices) -> dict[str, Any]:
    """Peak allocated/reserved memory since the last reset, in GiB.

    allocated is what tensors actually hold; reserved is what the caching
    allocator took from the driver and is the number that matters when asking
    "does this fit on the card".
    """

    per_device: dict[str, dict[str, float]] = {}
    total_allocated = 0.0
    total_reserved = 0.0
    for device in normalize_devices(devices):
        if not _is_cuda(device):
            continue
        allocated = torch.cuda.max_memory_allocated(device) / BYTES_PER_GIB
        reserved = torch.cuda.max_memory_reserved(device) / BYTES_PER_GIB
        per_device[str(device)] = {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
        }
        total_allocated += allocated
        total_reserved += reserved

    if not per_device:
        return {}

    stats: dict[str, Any] = {
        "peak_memory_allocated_gb": total_allocated,
        "peak_memory_reserved_gb": total_reserved,
    }
    if len(per_device) > 1:
        stats["peak_memory_by_device"] = per_device
    return stats


def current_memory_gb(device) -> float:
    """Currently reserved memory on one device, for inline step logging."""

    if not _is_cuda(device):
        return 0.0
    return torch.cuda.memory_reserved(device) / BYTES_PER_GIB


# PORT NOTE [KEEP，但注意与 torch.compile 的相互作用] 代码原样可用。
#
# track() 里的 torch.cuda.synchronize 本身没问题，但 LitGPT 默认 torch.compile
# （pretrain.py L213），前若干步是编译时间，会把 time_forward 之类的均值污染得很厉害。
# 计时必须跳过 warmup：前 ~20 step 只跑不记，或者干脆分两段报（compile+warmup / steady state）。
# 成本对比只应该用 steady state 的数。
#
# 另外 LitGPT 自己在 L366 有一次 running_loss.compute().item()，注释明说是昂贵的
# device-to-host 同步，且每 log_iter_interval 步一次。它会和这里的 synchronize 叠加。
# 做正式测速时把 log_iter_interval 调大，或者两边的同步点都算进同一份预算里。
class PhaseTimer:
    """Accumulate wall time per named training phase.

    When enabled is False every track() call is a no-op context manager, so the
    instrumented loop runs at full speed and the timing keys are simply absent
    from the reported record.
    """

    def __init__(self, device: torch.device | str, enabled: bool = False) -> None:
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        # Only CUDA needs an explicit drain before reading the clock.
        self._sync = self.enabled and _is_cuda(self.device)
        self.totals: dict[str, float] = {}

    def reset(self) -> None:
        self.totals = {}

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
            self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - start)

    def as_record(self, prefix: str = "time_") -> dict[str, float]:
        """Phase totals in seconds, ready to merge into a metrics record."""

        return {f"{prefix}{name}_sec": value for name, value in sorted(self.totals.items())}


# PORT NOTE [MODIFY] 要和 LitGPT 自己的 throughput 日志并存，得先决定对外报哪一套。
#
# LitGPT 在 pretrain.py L375-390 已经报了 loss / iter / step / iter_time / tokens /
# total_tokens / throughput，口径是 per-iter（micro-batch），而这份 summary 是 per-epoch +
# per-step（optimizer step）。CPT 里 gradient_accumulation_iters 默认是
# 512/(devices*nodes)/4，单卡即 **128**（args.py L56-64 + pretrain.py L59-60），
# 两套口径差 128 倍，混在一份日志里必然读错。
#
# 两件事：
#   1. 明确按 **per optimizer step** 报，并把 accumulation_iters 一起打出来，让读者能换算。
#   2. 这个 accum=128 对本方法是重大利好：惩罚每个 optimizer step 只注入一次，
#      而 SFT 时 accum=8，也就是同一份惩罚开销现在被摊薄 16 倍。成本表里要明确写出
#      accum 值，否则"开销只占 x%"这个结论无法复现，也容易被质疑是挑了有利配置。
#      建议扫一下 accum（比如 8/32/128）报一条开销随 accum 变化的曲线，主动把这点讲清楚。
#
# CPT 里 samples/s 意义不大（样本都是定长块），tokens/s 才是可比的量；epoch 概念也基本消失
# （预训练按 max_tokens/max_iters 走），train_sec 的分母要跟着改。
def format_cost_summary(record: dict[str, Any]) -> str:
    """Render one epoch's cost numbers as a short human-readable block."""

    lines = ["---- cost ----"]
    train_sec = record.get("train_sec")
    if train_sec is not None:
        lines.append(f"train wall time      : {train_sec:.1f} s")
    if record.get("eval_sec") is not None:
        lines.append(f"eval  wall time      : {record['eval_sec']:.1f} s")
    if record.get("samples_per_sec") is not None:
        lines.append(f"throughput           : {record['samples_per_sec']:.2f} samples/s")
    if record.get("tokens_per_sec") is not None:
        lines.append(f"                       {record['tokens_per_sec']:.0f} tokens/s")
    if record.get("sec_per_step") is not None:
        lines.append(f"per-step time        : {record['sec_per_step'] * 1000:.1f} ms")
    if record.get("peak_memory_allocated_gb") is not None:
        lines.append(
            f"peak GPU memory      : {record['peak_memory_allocated_gb']:.2f} GiB allocated / "
            f"{record['peak_memory_reserved_gb']:.2f} GiB reserved"
        )
    if record.get("covariance_memory_gb"):
        lines.append(f"  of which C matrices: {record['covariance_memory_gb']:.3f} GiB")
    if record.get("reference_memory_gb"):
        lines.append(f"  of which W0 snapshot: {record['reference_memory_gb']:.3f} GiB")
    for key in sorted(record):
        if key.startswith("time_") and key.endswith("_sec"):
            phase = key[len("time_") : -len("_sec")]
            share = ""
            if train_sec:
                share = f"  ({100.0 * record[key] / train_sec:.1f}% of train)"
            lines.append(f"  {phase:<18}: {record[key]:.1f} s{share}")
    return "\n".join(lines)
