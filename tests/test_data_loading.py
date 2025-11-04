"""Sanity checks for PowerGraph data loaders."""

from __future__ import annotations
import pytest
import torch
from torch_geometric.loader import DataLoader

from data.utils_data import (
    list_available_powergraph_datasets,
    load_powergraph_dataset,
)


@pytest.mark.skipif(DataLoader is None, reason="torch_geometric is not available")
@pytest.mark.skipif(torch is None, reason="torch is not available")
def test_powergraph_dataset_batches() -> None:
    print(" Listing available PowerGraph datasets...")
    datasets = list_available_powergraph_datasets()
    print(f"Found datasets: {datasets}")
    assert datasets, " No datasets found under data/raw."

    name = datasets[0]
    print(f" Loading dataset: {name}")
    dataset = load_powergraph_dataset(name=name, datatype="node")
    print(f" Dataset loaded: {len(dataset)} samples")

    assert len(dataset) > 0, "❌ Dataset is empty after processing."

    sample = dataset[0]
    print(f" Sample features: x={sample.x.shape}, edge_index={sample.edge_index.shape}")

    assert sample.x is not None and sample.x.numel() > 0
    assert sample.edge_index is not None and sample.edge_index.numel() > 0

    batch_size = 4
    print(f" Creating DataLoader with batch_size={batch_size}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f" DataLoader created (num_samples={len(dataset)})")

    batch = next(iter(loader))
    print(
        f" Batch loaded: x={batch.x.shape}, "
        f"edge_index={batch.edge_index.shape}, "
        f"batch_tensor_shape={batch.batch.shape}"
    )

    assert batch.x.size(0) > 0
    assert batch.edge_index.size(1) > 0
    assert isinstance(batch.batch, torch.Tensor)

    print("All sanity checks passed successfully!")
