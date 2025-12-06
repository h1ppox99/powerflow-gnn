"""Sanity checks for PowerGraph data loaders."""

from __future__ import annotations
import pytest
import torch
from torch_geometric.loader import DataLoader #type: ignore[import]

from data.utils_data import (
    list_available_powergraph_datasets,
    load_powergraph_dataset,
)


@pytest.mark.skipif(DataLoader is None, reason="torch_geometric is not available")
@pytest.mark.skipif(torch is None, reason="torch is not available")
def test_powergraph_dataset_batches() -> None:
    print("Listing available PowerGraph datasets...")
    datasets = list_available_powergraph_datasets()
    print(f"Found datasets: {datasets}")
    assert datasets, "No datasets found under data/raw."

    name = datasets[0]
    print(f"Loading dataset: {name}")
    dataset = load_powergraph_dataset(name=name, datatype="nodeopf")
    print(f"Dataset loaded: {len(dataset)} samples")

    assert len(dataset) > 0, "Dataset is empty after processing."

    sample = dataset[0]


    # ------------------------------------------------------------------
    # Pseudo node-type recovery (from x[:, 2])
    # ------------------------------------------------------------------
    assert sample.x.size(1) >= 3, "Expected NodeType to be stored in x[:, 2]"

    # Heuristic de dé-normalisation :
    # NodeType ∈ {1,2,3} → après normalisation valeurs ~ {1/max,2/max,3/max}
    nt_feat = sample.x[:, 2]

    # Recover discrete node types robustly
    nt_recovered = torch.round(
        nt_feat / nt_feat.unique().max() * 3
    ).long()

    uniq_nt = torch.unique(nt_recovered)

    print(
        f"  pseudo node_type (from x[:,2]) : unique={uniq_nt.tolist()}, "
        f"num={len(uniq_nt)}"
    )

    assert torch.all((uniq_nt >= 1) & (uniq_nt <= 3)), \
        f"Unexpected node types recovered: {uniq_nt.tolist()}"

    if len(uniq_nt) < 3:
        print(
            "  NOTE: Not all {PQ, PV, Slack} present in this graph "
            "(normal for small networks)"
        )

    # ------------------------------------------------------------------
    # Edge type check (expected absent)
    # ------------------------------------------------------------------
    if hasattr(sample, "edge_type") and sample.edge_type is not None:
        unique_edges = torch.unique(sample.edge_type)
        print(f"  edge_type      : {sample.edge_type.shape}, unique={unique_edges.tolist()}")
    else:
        print("  edge_type      : not present (expected for OPF baseline)")

    assert sample.x is not None and sample.x.numel() > 0
    assert sample.edge_index is not None and sample.edge_index.numel() > 0

    # ------------------------------------------------------------------
    # Batch test
    # ------------------------------------------------------------------
    batch_size = 4
    print(f"Creating DataLoader with batch_size={batch_size}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    batch = next(iter(loader))
    print("Batch loaded:")
    print(
        f"  x={batch.x.shape}, "
        f"edge_index={batch.edge_index.shape}, "
        f"batch_tensor={batch.batch.shape}"
    )

    # Batch-level pseudo node types
    nt_feat_b = batch.x[:, 2]
    nt_rec_b = torch.round(
        nt_feat_b / nt_feat_b.unique().max() * 3
    ).long()
    uniq_b = torch.unique(nt_rec_b)

    print(
        f"  batch pseudo node_type unique={uniq_b.tolist()} "
        f"(num={len(uniq_b)})"
    )

    assert torch.all((uniq_b >= 1) & (uniq_b <= 3))

    assert batch.x.size(0) > 0
    assert batch.edge_index.size(1) > 0
    assert isinstance(batch.batch, torch.Tensor)

    print("All sanity checks passed successfully.")
