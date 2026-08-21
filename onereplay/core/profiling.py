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
