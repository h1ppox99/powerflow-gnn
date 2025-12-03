# CLI run for reproducibility

#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
uv run python -m src.experiments.run_experiment --config src/config/graphsage.yaml