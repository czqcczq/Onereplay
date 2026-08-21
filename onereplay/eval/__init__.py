"""Evaluation package: shared runner and pluggable metrics."""

from onereplay.eval.runner import get_metric, load_eval_model, run_evaluation

__all__ = ["get_metric", "load_eval_model", "run_evaluation"]
