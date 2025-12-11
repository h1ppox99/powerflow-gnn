"""Generate a stylized IEEE-118 heterogeneous graph visual.

Run:
    python plot_ieee118_visual.py
Outputs:
    output/ieee118_hhmpnn_visual.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

# Configure numerical libs before they initialize threads.
import os


def _discover_root() -> Path:
    """Return repository root whether this file is in repo/ or repo/scripts/."""
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent]
    for cand in candidates + list(here.parents):
        if (cand / "data" / "raw").is_dir():
            return cand
    return here.parent  # fallback; will fail later if data missing


REPO_ROOT = _discover_root()
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "output" / "mpl-cache"))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch


RAW_DIR = REPO_ROOT / "data" / "raw" / "ieee118" / "ieee118" / "raw"
OUTPUT_PATH = REPO_ROOT / "output" / "ieee118_hhmpnn_visual.png"


def load_ieee118_sample(raw_dir: Path = RAW_DIR) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one IEEE-118 sample (node attrs + edges) without torch dependencies."""
    # Node features (P, Q, _, NodeType) are stored as an HDF5 cell array.
    with h5py.File(raw_dir / "Xopf.mat", "r") as f:
        x_ref = f["X"][0, 0]  # first scenario
        x = f[x_ref][...]    # shape (4, num_nodes)
    node_types = x[3].astype(int)
    p = x[0]
    q = x[1]
    pq_magnitude = np.sqrt(p ** 2 + q ** 2)

    # Edges come from a MATLAB v5 file.
    edge_mat = scipy.io.loadmat(raw_dir / "edge_index_opf.mat")
    edge_index = edge_mat.get("edge_index_opf", edge_mat.get("edge_index"))
    edge_attr = scipy.io.loadmat(raw_dir / "edge_attr_opf.mat")["edge_attr"]
    return node_types, pq_magnitude, edge_index, edge_attr


def build_graph(
    node_types: np.ndarray,
    pq_magnitude: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Deduplicate edges, drop self-loops, and attach attributes."""
    g = nx.Graph()
    num_nodes = node_types.shape[0]
    g.add_nodes_from(range(num_nodes))

    # Prepare node metadata for styling.
    load_range = np.ptp(pq_magnitude)
    load_norm = (pq_magnitude - pq_magnitude.min()) / (load_range + 1e-6)
    for i in range(num_nodes):
        g.nodes[i]["type"] = int(node_types[i])
        g.nodes[i]["load_norm"] = float(load_norm[i])

    # Deduplicate edges by averaging attributes of symmetric duplicates.
    aggregated: Dict[Tuple[int, int], list[np.ndarray]] = {}
    for (u, v), attr in zip(edge_index.astype(int), edge_attr):
        u -= 1  # MATLAB -> 0-based
        v -= 1
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        aggregated.setdefault(key, []).append(attr)

    edges: list[Tuple[int, int]] = []
    attrs: list[np.ndarray] = []
    for key, vals in aggregated.items():
        edges.append(key)
        attrs.append(np.mean(np.vstack(vals), axis=0))

    # Attach weights for layout variation.
    for (u, v), attr in zip(edges, attrs):
        strength = float(np.linalg.norm(attr))
        g.add_edge(u, v, strength=strength)

    return g, np.array(edges, dtype=int), np.vstack(attrs)


def draw_graph_panel(
    ax,
    g: nx.Graph,
    edges: np.ndarray,
    edge_strength: np.ndarray,
) -> None:
    """Render the IEEE-118 graph with a single edge style and typed nodes."""
    bg = "#0b1021"
    ax.set_facecolor(bg)

    # Layout favors readability over geographic accuracy.
    pos = nx.spring_layout(
        g,
        weight="strength",
        seed=7,
        k=0.18,
        iterations=200,
    )

    # Single edge style, width scaled by strength for visual variation.
    widths = 1.8 + 1.0 * (edge_strength / edge_strength.max())
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=[tuple(e) for e in edges],
        width=widths,
        edge_color="#7E8AA2",
        alpha=0.85,
        ax=ax,
    )

    node_palette = {1: "#8BC34A", 2: "#FF7043", 3: "#FFD54F"}
    node_shapes = {1: "o", 2: "s", 3: "*"}
    node_labels = {1: "PQ load", 2: "PV generator", 3: "Slack"}

    # Draw nodes by type with load-dependent sizing.
    for ntype in (1, 2, 3):
        nodes = [n for n, d in g.nodes(data=True) if d["type"] == ntype]
        if not nodes:
            continue
        sizes = [130 + 170 * g.nodes[n]["load_norm"] for n in nodes]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=nodes,
            node_shape=node_shapes[ntype],
            node_size=sizes,
            node_color=node_palette[ntype],
            edgecolors="#f8fafc",
            linewidths=0.6,
            alpha=0.95,
            ax=ax,
        )

    ax.set_title("IEEE-118 typed grid (loads, generators, slack)", color="#e2e8f0", fontsize=13, pad=12)
    ax.axis("off")

    # Build combined legend.
    node_handles = [
        Line2D([0], [0], marker=node_shapes[t], color="none", label=node_labels[t],
               markerfacecolor=node_palette[t], markeredgecolor="#f8fafc", markersize=16)
        for t in (1, 2, 3)
    ]
    ax.legend(
        handles=node_handles,
        loc="lower left",
        frameon=False,
        ncol=1,
        labelcolor="#e2e8f0",
        fontsize=22,
        borderpad=0.5,
        labelspacing=0.8,
        handletextpad=0.8,
    )


def main() -> None:
    node_types, pq_magnitude, edge_index, edge_attr = load_ieee118_sample()
    print(f"[ieee118] edge_attr loaded: shape={edge_attr.shape}, "
          f"first_row={edge_attr[0].tolist() if edge_attr.size > 0 else 'n/a'}")
    g, edges, edge_attr = build_graph(node_types, pq_magnitude, edge_index, edge_attr)
    edge_strength = np.linalg.norm(edge_attr, axis=1)

    fig = plt.figure(figsize=(10, 9), dpi=240, facecolor="#0b1021")
    ax_graph = fig.add_subplot(111)

    draw_graph_panel(ax_graph, g, edges, edge_strength)

    fig.suptitle("IEEE-118 typed grid (loads, generators, slack)", color="#e2e8f0", fontsize=15, y=0.98)
    fig.tight_layout(pad=1.0)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, facecolor=fig.get_facecolor(), dpi=300)
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
