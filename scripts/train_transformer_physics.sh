#!/usr/bin/env bash
# Run the TransformerConv baseline with a global virtual node.
set -euo pipefail

UV_CACHE_DIR=${UV_CACHE_DIR:-./.uv_cache}
export UV_CACHE_DIR
export PYTHONPATH=.

CFG="src/config/transformer_physics.yaml"


uv run src/experiments/run_experiment.py --config "$CFG"
