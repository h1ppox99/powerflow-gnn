"""Helpers for loading PowerGraph data within the repository."""

from __future__ import annotations

from pathlib import Path

from .prepare_data import PowerGrid, discover_powergraph_root


def list_available_powergraph_datasets(root: str | Path | None = None) -> list[str]:
    """Return the dataset folders found under the resolved PowerGraph root."""
    root_path = Path(root) if root is not None else discover_powergraph_root()
    return sorted(
        entry.name
        for entry in root_path.iterdir()
        if entry.is_dir() and (entry /entry.name/ "raw").is_dir()
    )


def load_powergraph_dataset(
    name: str,
    datatype: str = "node",
    root: str | Path | None = None,
    *,
    transform=None,
    pre_transform=None,
    pre_filter=None,
) -> PowerGrid:
    """Instantiate the PowerGraph ``InMemoryDataset`` with minimal configuration."""

    resolved_root = Path(root) if root is not None else discover_powergraph_root()
    return PowerGrid(
        root=resolved_root,
        name=name,
        datatype=datatype,
        transform=transform,
        pre_transform=pre_transform,
        pre_filter=pre_filter,
    )
