#!/usr/bin/env bash
set -euo pipefail

# Train the HeGGA no-attention ablation on PowerGraph OPF (IEEE118).

UV_CACHE_DIR=${UV_CACHE_DIR:-./.uv_cache}
export UV_CACHE_DIR
export PYTHONPATH=.

uv run src/experiments/run_experiment.py --config src/config/hegga_noattn_lappe.yaml
