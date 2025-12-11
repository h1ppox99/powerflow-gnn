#!/usr/bin/env bash
set -euo pipefail

# Train the TransformerConv baseline on PowerGraph OPF (IEEE118).

UV_CACHE_DIR=${UV_CACHE_DIR:-./.uv_cache}
export UV_CACHE_DIR
export PYTHONPATH=.

uv run src/experiments/run_experiment.py --config src/config/hh_one_attention.yaml
