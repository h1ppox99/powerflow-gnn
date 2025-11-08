"""Pytest configuration for ensuring local package imports resolve."""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Insert repository root into sys.path so tests can import `src`."""

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
