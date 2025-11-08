# Models Guide

This folder gathers every learnable architecture used on top of the `PowerGrid`
[`InMemoryDataset`](../../data/prepare_data/powergrid.py). The goal of this
guide is to capture the data contract exposed by the loader and outline the
expected shapes, helper utilities, and coding patterns so you can add new
models quickly without re-discovering how samples are structured.

```
src/models/
├── __init__.py          # Export shortcuts, keep this up to date
├── layers.py            # Shared building blocks (convs, pooling, norm)
├── graphsage_pi.py      # Baseline physics-informed GraphSAGE
├── pna_pi.py            # Planned PNA-based variant
└── README.md            # ← you are here
```

## Recap: What the PowerGrid dataset provides

Loaders live under `data.prepare_data` and are re-exported via
`data.utils_data`. Typical usage:

```python
from data.utils_data import load_powergraph_dataset
from torch_geometric.loader import DataLoader

dataset = load_powergraph_dataset(name="ieee118", datatype="node")
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
```

Every sample is a `torch_geometric.data.Data` object already normalized and
stored on `cuda` if available at processing time (see `_DEVICE` in
`powergrid.py`). When batching, still call `batch = batch.to(device)` so that
newly created tensors (e.g. masks) sit on the same device as the model.

### Node-level tasks (`datatype="node"` or `"nodeopf"`)

| Attribute | Type / shape | Notes for model authors |
|-----------|--------------|-------------------------|
| `x` | `FloatTensor [num_nodes, num_features]` | Raw per-node quantities from `X.mat` (or a subset of columns for `nodeopf`), divided by `max(|X|)` over the whole split. |
| `edge_index` | `LongTensor [2, num_edges]` | COO indices derived from `edge_index.mat`, duplicated to make undirected edges. |
| `edge_attr` | `FloatTensor [num_edges, feat_dim]` | Edge features from `edge_attr.mat`, L2-normalized along each column. Use inside edge-aware convolutions (e.g. PNA, GAT). |
| `y` | `FloatTensor [num_nodes, target_dim]` | Targets from `Y_polar.mat` (`node`) or `Y_polar_opf.mat` (`nodeopf`), normalized by `max(|Y|)` per dimension. Models should output in the same scale. |
| `mask` | `BoolTensor [num_nodes, target_dim]` | Indicates which node/target entries are valid (zero rows come from padding in the Matlab export). Apply it before computing losses or metrics. |
| `maxs` | `FloatTensor [target_dim]` | Max-abs factors used during preprocessing. Multiply predictions by `maxs` to recover physical units. |

### Graph-level tasks (`datatype` in `{binary, regression, multiclass}`)

| Attribute | Type / shape | Notes |
|-----------|--------------|-------|
| `x` | `FloatTensor [num_nodes, 3]` | Features from `B_f_tot`. Already on `_DEVICE`. |
| `edge_index` | `LongTensor [2, num_edges]` | Built from `blist.mat`, pruned to remove zero-flow lines, then symmetrised. |
| `edge_attr` | `FloatTensor [num_edges, 4]` | Aggregated edge descriptors `E_f_post`, concatenated for both directions. |
| `y` | `FloatTensor [1, target_dim]` | Graph-level label: binary outage flag, regression MW target, or multiclass category (stored as argmax). |
| `edge_mask` | `FloatTensor [num_edges, 1]` | Optional explanation mask (`exp.mat`), duplicated for both directions. Hand it to attention/interpretability heads if needed. |
| `idx` | `int` | Sample index inside the MatLab archive. Useful for debugging or reproducibility. |

### General observations

- Every processed tensor already lives on `_DEVICE = torch.device("cuda" if
  available else "cpu")` when the dataset was created. Models should still
  support re-homing tensors via `data = data.to(device)` because downstream
  scripts may move batches explicitly.
- No global node IDs are stored; batching uses PyG's standard `batch` vector.
  Pooling layers (`global_mean_pool`, etc.) should consume `batch`.
- The loader never adds positional encodings or self-loops. Add them in
  `DataLoader` transforms if your architecture needs them.
- Normalization factors (`maxs`) live per-sample but are constant across the
  dataset split because they were computed globally before `torch.save`.

## Building models that plug into the dataset

1. **Forward signature.** Prefer either:
   ```python
   def forward(self, data: Data) -> torch.Tensor:
       x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, "edge_attr", None)
       return self._forward_impl(x, edge_index, edge_attr=edge_attr)
   ```
   or
   ```python
   def forward(self, x, edge_index, edge_attr=None, batch=None):
       ...
   ```
   The first style keeps compatibility with `torch_geometric.loader.DataLoader`
   batches and lets you access auxiliary fields (`mask`, `maxs`, `edge_mask`).

2. **Match targets.**
   - Node-level models must output `[num_nodes, target_dim]`. Before computing
     the loss, slice with the boolean mask:
     ```python
     preds = model(batch)
     loss = mse_loss(preds[batch.mask], batch.y[batch.mask])
     ```
   - Graph-level models should pool over `batch.batch` and output `[batch_size,
     target_dim]`. For binary/multiclass targets use BCE/CE; for regression use
     MSE or MAE.

3. **Edge features.** `edge_attr` is always available and already normalized,
   so expose keyword arguments that allow `None` for architectures that do not
   consume them.

4. **Rescaling.** Multiply predictions by `batch.maxs` only when you need values
   in physical units (e.g. to feed the penalties in `src/losses/physical_loss.py`).
   Store both normalized and denormalized outputs if you log metrics at multiple
   scales.

5. **Physical constraints.** The (placeholder) routines in `src/losses` expect
   per-node injections and per-edge flows. When designing a model, keep
   intermediate tensors (edge messages, voltage magnitude/angle estimates)
   accessible so you can route them into the physics-informed losses.

6. **Device handling.** Because `PowerGrid` may serialize CUDA tensors, always
  ensure modules call `.to(x.device)` on parameters/buffers created on the fly.
  Example: `self.register_buffer("angle_scale", torch.tensor(1.0))` in
  `GraphSAGE_PI` automatically migrates when `model.to(device)` is invoked.

7. **PNA setup.** `PNA_PI` requires two runtime values at instantiation time:
  `deg_histogram = compute_degree_histogram(dataset)` and
  `edge_dim = dataset[0].edge_attr.size(-1)`. Compute the histogram once, right
  after loading the dataset, and reuse it for every split so PNA sees consistent
  neighborhood statistics.

## Example template

```python
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

class PowerGridGraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        if batch is None:
            raise ValueError("Batched Data objects must include `batch`.")
        graph_repr = global_mean_pool(x, batch)
        return self.head(graph_repr)
```

Swap in `edge_attr`-aware layers (e.g. `torch_geometric.nn.PNAConv`) or custom
blocks defined in `layers.py` for node-level regressors. Reuse the same outline
for `pna_pi.py` to keep file structure consistent: model definition, helper
builders, and optional factory functions.

## Workflow for adding a new model

1. Create `<name>_pi.py` under `src/models/` and implement your `nn.Module`.
2. Export it from `src/models/__init__.py` so experiment scripts can import it.
3. If you introduce reusable layers, house them in `layers.py` with docstrings
   explaining expected tensor shapes.
4. Provide a minimal usage example (similar to the template above) either at
   the bottom of the file guarded by `if __name__ == "__main__":` or inside the
   docstring.
5. Test the integration by running `uv run pytest tests/test_data_loading.py`
   (ensures the dataset is accessible) and your training script with a small
   subset to validate the forward pass.

Following this checklist keeps every model immediately compatible with the
`PowerGrid` `InMemoryDataset`, enforces consistent tensor scaling, and shortens
bring-up time for new physics-informed architectures.
