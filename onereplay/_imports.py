"""Centralize path setup so every module can import onereplay consistently.

Call ensure_project_root() once before importing sibling packages when the
script is run as a file rather than as a package.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> Path:
    """Insert the repository root on sys.path if it is not already present.

    Returns the project root Path (the parent of the onereplay package).
    """

    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root
