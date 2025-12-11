### PowerGraph Raw Data

The PowerGraph benchmark is installed locally from the Figshare archive:
https://figshare.com/articles/dataset/PowerGraph/22820534?file=50081700

Each dataset keeps the original `<dataset_name>/raw` layout so that the
`PowerGrid` loader in `data/prepare_data/powergrid.py` can process the files
directly. Use the helpers in `data/utils_data.py` to list available datasets or
instantiate the loader without referencing any hard-coded paths.

# Data Directory

This folder centralises everything related to the PowerGraph benchmark used in
the project. The expected structure is:

```
data/
├── README.md                 # This guide
├── raw/                      # Original Figshare archive contents
│   ├── <dataset>/<dataset>/raw/        # e.g. ieee118/ieee118/raw, uk/uk/raw, texas/texas/raw, ...
│   └── README.md             # Source reference
├── prepare_data/               # PyTorch Geometric dataset implementation
│   └── powergrid.py          # PowerGrid InMemoryDataset loader
└── utils_data.py             # Convenience helpers for loading datasets
```

### Raw data layout

Each dataset extracted from the Figshare archive must live under `data/raw` and
keep its original folder structure. For example, the IEEE 118-node benchmark
should be located at `data/raw/ieee118/ieee118/raw/` and contain files such as
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
