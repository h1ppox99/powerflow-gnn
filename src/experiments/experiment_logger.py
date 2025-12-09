"""Utilities to record every experiment run into a shared CSV log."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Mapping, MutableMapping, Optional

DEFAULT_LOG_PATH = Path("experiment_runs.csv")

_FIELDNAMES = [
    "timestamp_utc",
    "config_path",
    "config_name",
    "config_hash",
    "model_name",
    "data_backend",
    "data_grid",
    "data_task",
    "train_epochs",
    "train_lr",
    "train_batch_size",
    "train_loss",
    "train_scheduler",
    "test_loss",
    "total_mse",
    "test_mse_num",
    "test_rmse_num",
    "test_rmse_theta",
    "test_mse_p",
    "test_mse_q",
    "test_mse_v",
    "test_mse_theta",
    "test_mean_kcl",
    "config_yaml",
]


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_config(cfg: Mapping[str, object]) -> MutableMapping[str, object]:
    """Extracts a consistent summary of the configuration."""
    model = cfg.get("model", {}) if isinstance(cfg, Mapping) else {}
    train = cfg.get("train", {}) if isinstance(cfg, Mapping) else {}
    data = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}

    return {
        "model_name": model.get("name"),
        "data_backend": data.get("backend"),
        "data_grid": data.get("grid"),
        "data_task": data.get("task"),
        "train_epochs": train.get("epochs"),
        "train_lr": train.get("lr"),
        "train_batch_size": train.get("batch_size"),
        "train_loss": train.get("loss"),
        "train_scheduler": (train.get("scheduler") or {}).get("type") if isinstance(train.get("scheduler"), Mapping) else None,
    }


def _config_hash(cfg_text: str) -> str:
    return hashlib.sha1(cfg_text.encode("utf-8")).hexdigest()


def _read_config_text(config_path: Path) -> Optional[str]:
    try:
        return config_path.read_text()
    except Exception:
        return None


def _serialize_config(cfg: Mapping[str, object]) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def _ensure_log_header(log_path: Path) -> bool:
    """Ensure CSV header matches current schema; rewrite if columns changed."""
    if not log_path.exists():
        return False
    try:
        with log_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == _FIELDNAMES:
                return True
            rows = list(reader)
        with log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                normalized = {key: row.get(key, "") for key in _FIELDNAMES}
                writer.writerow(normalized)
        return True
    except Exception:
        return False


def log_experiment_run(
    config_path: str | Path,
    cfg: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> Path:
    """Append a single experiment run (config + results) to the shared CSV log.

    Args:
        config_path: Path to the YAML/JSON config used.
        cfg: Parsed configuration dictionary.
        metrics: Final metrics returned by the training loop (expects test metrics).
        log_path: CSV file to append to (default: ``experiment_runs.csv`` at repo root).
    """
    config_path = Path(config_path)
    # Store config as a single-line JSON string to avoid CSV row breaks.
    config_snapshot = _serialize_config(cfg)
    config_summary = _normalize_config(cfg)

    mse_num = metrics.get("mse_num")
    if mse_num is None and metrics.get("rmse_num") is not None:
        try:
            mse_num = float(metrics["rmse_num"]) ** 2
        except Exception:
            mse_num = None

    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config_path": str(config_path),
        "config_name": config_path.name,
        "config_hash": _config_hash(config_snapshot),
        **config_summary,
        "test_loss": _safe_float(metrics.get("loss")),
        "total_mse": _safe_float(
            metrics.get("total_mse", metrics.get("loss_mse", metrics.get("mse")))
        ),
        "test_mse_num": _safe_float(mse_num),
        "test_rmse_num": _safe_float(metrics.get("rmse_num")),
        "test_rmse_theta": _safe_float(metrics.get("rmse_theta")),
        "test_mse_p": _safe_float(metrics.get("mse_p")),
        "test_mse_q": _safe_float(metrics.get("mse_q")),
        "test_mse_v": _safe_float(metrics.get("mse_v")),
        "test_mse_theta": _safe_float(metrics.get("mse_theta")),
        "test_mean_kcl": _safe_float(metrics.get("mean_kcl_residual")),
        "config_yaml": config_snapshot,
    }

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = _ensure_log_header(log_path)
    clean_row = {key: ("" if row.get(key) is None else row.get(key, "")) for key in _FIELDNAMES}
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(clean_row)

    return log_path


__all__ = ["log_experiment_run", "DEFAULT_LOG_PATH"]
