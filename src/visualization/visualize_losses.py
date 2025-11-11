"""Helpers to turn training logs into publication-ready loss curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import matplotlib.pyplot as plt


HistoryRow = Mapping[str, float | int]


def load_history(log_path: str | Path) -> list[MutableMapping[str, float]]:
    """Load a list of epoch dictionaries from ``.json``, ``.jsonl`` or ``.csv`` files."""

    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Training log {path} was not found.")

    if path.suffix.lower() in {".json", ".jsonl"}:
        rows = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return [_ensure_numeric(row) for row in rows]

    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "," if path.suffix.lower() == ".csv" else "\t"
        with path.open() as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [_ensure_numeric(row) for row in reader]

    raise ValueError(
        f"Unsupported log format {path.suffix!r}. Expected .json, .jsonl, .csv or .tsv."
    )


def plot_training_curves(
    history: Sequence[HistoryRow] | Iterable[HistoryRow] | str | Path,
    *,
    metrics: Sequence[str] | None = None,
    smoothing: int = 1,
    title: str | None = None,
    xlabel: str = "Epoch",
    ylabel: str = "Value",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot selected metrics from a history list or from a file path."""

    entries = _standardize_history(history)
    if not entries:
        raise ValueError("Received an empty training history.")

    metric_names = (
        list(metrics)
        if metrics is not None
        else [
            key
            for key in entries[0].keys()
            if key != "epoch" and isinstance(entries[0][key], (int, float))
        ]
    )
    if not metric_names:
        raise ValueError("No numeric metrics found to plot.")

    epochs = [
        float(entry["epoch"]) if "epoch" in entry else float(idx + 1)
        for idx, entry in enumerate(entries)
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in metric_names:
        values = [float(entry[name]) for entry in entries]
        ax.plot(epochs, _moving_average(values, smoothing), label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Training curves")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    output_path = None
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def _ensure_numeric(row: Mapping[str, object]) -> MutableMapping[str, float]:
    parsed: MutableMapping[str, float] = {}
    for key, value in row.items():
        if value is None or value == "":
            continue
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _standardize_history(history: Sequence[HistoryRow] | Iterable[HistoryRow] | str | Path):
    if isinstance(history, (str, Path)):
        return load_history(history)
    return list(history)


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    averaged: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        averaged.append(sum(window_values) / len(window_values))
    return averaged


__all__ = ["load_history", "plot_training_curves"]
