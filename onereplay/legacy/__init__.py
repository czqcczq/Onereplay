"""Archived scripts kept for reference and for the refactor parity check.

Nothing here is on the active path. The live entry points are
onereplay/scripts/{collect_cov,train,evaluate}.py, which use onereplay.core,
onereplay.data, onereplay.trainers, and onereplay.eval.

train_commonsense_lora.py is still imported by scripts/verify_refactor.py to
prove the new SFTTrainer reproduces the original loop's numbers.
"""
