"""Domain-specialist SFT data preparation (Math / Medical / Finance).

Three specialists are trained from the same Qwen3-4B-Base checkpoint, one per
domain, and only later combined into continual-learning sequences. For the
comparison between them to mean anything, the only thing allowed to differ is
the domain corpus itself: same base model, same training code, same optimizer,
same budget, same serialization, same length rule.

This package owns the "same serialization, same length rule" half. Each
prepare_*.py turns one raw corpus into the shared record schema, and common.py
holds everything that must not be reimplemented per domain -- above all the
token count, which decides whether a row survives the 4096 cutoff.
"""
