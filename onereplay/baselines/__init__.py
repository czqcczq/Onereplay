"""Glue onto the vendored comparison baselines. Nothing here is OneReplay.

Every module in this package exists to run somebody else's published method
against the same data, the same loop and the same metrics schema as our own
arms. The methods themselves are not reimplemented here: each upstream project
is kept as an unmodified clone under ``baseline/`` at a pinned commit, and the
file below it only adapts our inputs to their entry points. Each module's
header names its upstream URL, commit and license, and lists every place where
our setting forced a departure from theirs.

The split from ``onereplay/core`` is deliberate. ``core`` holds the parts a
result of ours depends on -- the covariance estimator, the penalty, the Fisher
and the model plumbing -- and a reader auditing our claims should not have to
tell those apart from a third party's algorithm sitting in the same directory.
EWC and replay stay in ``core`` despite also being comparison arms, because
they are expressed through the same ``regularizer.py`` interface our own
penalty uses and separating them would mean duplicating that interface.

No module here is imported at package scope: they pull in peft, upstream source
trees and other heavy optional dependencies, and a lightweight caller importing
``onereplay.baselines`` should not pay for a baseline it is not running. Import
the submodule you need.
"""
