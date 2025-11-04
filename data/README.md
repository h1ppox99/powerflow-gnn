# Data Directory

This folder centralises everything related to the PowerGraph benchmark used in
the project. The expected structure is:

```
data/
├── README.md                 # This guide
├── raw/                      # Original Figshare archive contents
│   ├── <dataset>/raw/        # e.g. ieee118/raw, uk/raw, texas/raw, ...
│   └── README.md             # Source reference
├── prepare_data/               # PyTorch Geometric dataset implementation
│   └── powergrid.py          # PowerGrid InMemoryDataset loader
└── utils_data.py             # Convenience helpers for loading datasets
```

### Raw data layout

Each dataset extracted from the Figshare archive must live under `data/raw` and
keep its original folder structure. For example, the IEEE 118-node benchmark
should be located at `data/raw/ieee118/raw/` and contain files such as
`edge_index.mat`, `edge_attr.mat`, `X.mat`, `Y_polar.mat`, etc. No absolute
paths are needed; the helper utilities automatically discover the correct root.

### Helper scripts

Two utility scripts are available under `tests/` to quickly validate the data:

- `tests/inspect_features.py` lists the available datasets, reports the basic
  node/edge statistics for a chosen sample, and confirms that the loader works.
- `tests/visualize_dataset.py` renders a graph sample via Matplotlib (optionally
  saving the figure) so you can verify the topology and feature distributions.

Use them with `uv run python <script> ...` to ensure they run inside the project
environment that bundles `mat73`, PyTorch, and Torch Geometric.
