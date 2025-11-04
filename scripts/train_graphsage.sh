# CLI run for reproducibility

#!/usr/bin/env bash
set -euo pipefail
uv run PYTHONPATH=src python -m src.experiments.run_experiment --config src/config/default.yaml