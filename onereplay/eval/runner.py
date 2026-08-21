"""Evaluation runner: load a model once and run multiple metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import torch

from onereplay.core.modeling import (
    attach_adapter,
    is_lora_adapter_dir,
    load_causal_lm_and_tokenizer,
    set_seed,
)


class Metric(Protocol):
    name: str

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        ...


def load_eval_model(
    model_dir: str,
    model_name: str,
    use_bf16: int = 1,
    adapter_path: str = "",
    extra_config: Any | None = None,
):
    """Load the base model, a LoRA adapter on top of it, or a full checkpoint.

    adapter_path is really "the training output directory". Full fine-tuning
    writes complete weights there instead of an adapter, in which case the
    directory is loaded as a model in its own right, tokenizer included.
    """

    if adapter_path and not is_lora_adapter_dir(adapter_path):
        checkpoint = Path(adapter_path)
        return load_causal_lm_and_tokenizer(
            str(checkpoint.parent), checkpoint.name, use_bf16, extra_config
        )

    model, tokenizer = load_causal_lm_and_tokenizer(
        model_dir, model_name, use_bf16, extra_config
    )
    if adapter_path:
        model = attach_adapter(model, adapter_path)
    return model, tokenizer


def get_metric(name: str) -> Metric:
    """Instantiate a metric by short name."""

    registry = {
        "ifeval": "onereplay.eval.metrics.ifeval:IFEvalMetric",
        "multiif": "onereplay.eval.metrics.multiif:MultiIFMetric",
        "commonsense": "onereplay.eval.metrics.commonsense:CommonsenseLossMetric",
        "gsm8k": "onereplay.eval.metrics.gsm8k:GSM8KMetric",
        "aime": "onereplay.eval.metrics.aime:AIMEMetric",
        "humaneval": "onereplay.eval.metrics.humaneval:HumanEvalMetric",
        "mbpp": "onereplay.eval.metrics.mbpp:MBPPMetric",
        "safety": "onereplay.eval.metrics.safety:SafetyMetric",
        "math500": "onereplay.eval.metrics.math500:MATH500Metric",
        "amc": "onereplay.eval.metrics.math500:AMCMetric",
    }
    if name not in registry:
        raise ValueError(f"Unknown metric {name!r}. Choose from: {sorted(registry)}")
    module_path, class_name = registry[name].split(":")
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)()


def run_evaluation(
    model_dir: str,
    model_name: str,
    metric_names: list[str],
    out_dir: str,
    adapter_path: str = "",
    run_name: str = "",
    use_bf16: int = 1,
    seed: int = 1,
    gpu: int = 0,
    metric_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the model once and run each requested metric."""

    set_seed(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    name = run_name or (Path(adapter_path).name if adapter_path else "base")
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_eval_model(
        model_dir, model_name, use_bf16=use_bf16, adapter_path=adapter_path
    )
    model.to(device)
    model.eval()

    cfg = dict(metric_cfg or {})
    cfg.setdefault("run_name", name)
    cfg.setdefault("adapter_path", adapter_path)
    cfg.setdefault("seed", seed)
    cfg.setdefault("output_root", str(output_root))

    summary: dict[str, Any] = {
        "run_name": name,
        "adapter_path": adapter_path,
        "metrics": {},
    }
    for metric_name in metric_names:
        metric = get_metric(metric_name)
        metric_out = output_root / metric_name / name
        metric_out.mkdir(parents=True, exist_ok=True)
        local_cfg = dict(cfg)
        local_cfg["output_dir"] = str(metric_out)
        result = metric.run(model, tokenizer, device, local_cfg)
        summary["metrics"][metric_name] = result
        print(json.dumps({"metric": metric_name, **result}, ensure_ascii=False, indent=2))

    summary_path = output_root / f"{name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
