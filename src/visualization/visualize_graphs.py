"""Visual debugging helpers for PowerGrid-style graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def plot_graph_sample(
    data: Data,
    values: Tensor | Sequence[float] | None = None,
    *,
    default_channel: int = 2,
    layout: str = "spring",
    cmap: str = "viridis",
    node_size: int = 140,
    with_labels: bool = False,
    colorbar_label: str = "Value",
    seed: int = 7,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str | None = None,
) -> Path | None:
    """Plot a single graph with nodes colored by ``values`` (defaults to |V|)."""

    graph, pos = _graph_and_layout(data, layout=layout, seed=seed)
    color_values = _resolve_values(data, values, default_channel)

    fig, ax = plt.subplots(figsize=(6, 5))
    nodes = _draw_graph(
        ax,
        graph,
        pos,
        color_values,
        cmap=cmap,
        node_size=node_size,
        with_labels=with_labels,
    )
    fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.03, label=colorbar_label)
    ax.set_axis_off()
    ax.set_title(title or "Graph sample")

    output_path = None
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_predictions_vs_targets(
    data: Data,
    predictions: Tensor,
    targets: Tensor,
    *,
    component: int = 2,
    layout: str = "spring",
    cmap: str = "viridis",
    node_size: int = 140,
    seed: int = 7,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str | None = None,
) -> Path | None:
    """Side-by-side node color map comparing predictions with ground truth."""

    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must share the same shape.")
    if component >= predictions.size(-1):
        raise ValueError(f"Component {component} is out of bounds for tensors with {predictions.size(-1)} columns.")

    graph, pos = _graph_and_layout(data, layout=layout, seed=seed)
    preds = _to_1d(predictions[..., component])
    trues = _to_1d(targets[..., component])
    vmin = min(min(preds), min(trues))
    vmax = max(max(preds), max(trues))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    _add_panel(
        fig,
        axes[0],
        graph,
        pos,
        preds,
        cmap=cmap,
        node_size=node_size,
        title="Prediction",
        vmin=vmin,
        vmax=vmax,
    )
    _add_panel(
        fig,
        axes[1],
        graph,
        pos,
        trues,
        cmap=cmap,
        node_size=node_size,
        title="Target",
        vmin=vmin,
        vmax=vmax,
    )
    fig.suptitle(title or f"Component {component}")

    output_path = None
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def _graph_and_layout(data: Data, layout: str, seed: int):
    graph = to_networkx(data, to_undirected=True, remove_self_loops=True)
    if layout == "spring":
        pos = nx.spring_layout(graph, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        raise ValueError(
            f"Unknown layout {layout!r}. Supported options: spring, kamada_kawai, circular."
        )
    return graph, pos


def _resolve_values(
    data: Data,
    values: Tensor | Sequence[float] | None,
    default_channel: int,
) -> list[float]:
    if values is None:
        if getattr(data, "y", None) is None:
            raise ValueError("No node values provided and `data` does not expose `y`.")
        if default_channel >= data.y.size(-1):
            raise ValueError("Default channel is out of bounds for data.y.")
        resolved = data.y[:, default_channel]
    else:
        resolved = values
    values_list = _to_1d(resolved)
    if len(values_list) != data.num_nodes:
        raise ValueError(
            f"Expected {data.num_nodes} node values, received {len(values_list)}."
        )
    return values_list


def _to_1d(values: Tensor | Sequence[float]) -> list[float]:
    if isinstance(values, Tensor):
        tensor = values.detach().float().view(-1)
        return tensor.cpu().tolist()
    out = [float(v) for v in values]
    if not out:
        raise ValueError("Received an empty sequence of node values.")
    return out


def _draw_graph(
    ax,
    graph,
    pos,
    values: Sequence[float],
    *,
    cmap: str,
    node_size: int,
    with_labels: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
):
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=values,
        node_size=node_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
    )
    nx.draw_networkx_edges(graph, pos, ax=ax, width=0.8, alpha=0.7)
    if with_labels:
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    return nodes


def _add_panel(fig, ax, graph, pos, values, **kwargs):
    nodes = _draw_graph(ax, graph, pos, values, **kwargs)
    ax.set_axis_off()
    ax.set_title(kwargs.get("title", ""))
    fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.03)


__all__ = ["plot_graph_sample", "plot_predictions_vs_targets"]
