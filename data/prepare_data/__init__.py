"""Utilities for working with the PowerGraph dataset."""

from __future__ import annotations

from pathlib import Path

from .powergrid import PowerGrid
from .synthetic_data import SyntheticPowerGrid

__all__ = ["PowerGrid", "SyntheticPowerGrid", "discover_powergraph_root"]


def discover_powergraph_root(base_path: str | Path | None = None) -> Path:
    """Locate the directory holding the PowerGraph dataset folders.

    The function walks the immediate children of the provided ``base_path``—or
    the repository's ``data`` directory by default—and returns the first path
    where subdirectories follow the ``<dataset_name>/raw`` convention shipped
    with the Figshare archive.
    """

    base = Path(base_path) if base_path is not None else Path(__file__).resolve().parent.parent

    candidates = [base] + [p for p in base.iterdir() if p.is_dir()]


    for candidate in candidates:
        try:
            subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        except PermissionError:
            continue
        if any((subdir /subdir.name/ "raw").is_dir() for subdir in subdirs):
            return candidate

    raise FileNotFoundError(
        "Could not locate a dataset root with the expected <dataset>/raw structure "
        f"within {base} or its immediate subdirectories."
    )
