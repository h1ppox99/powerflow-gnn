#!/usr/bin/env bash
# Run the Transformer baseline across the four PowerGraph grids.
set -euo pipefail

UV_CACHE_DIR=${UV_CACHE_DIR:-./.uv_cache}
export UV_CACHE_DIR
export PYTHONPATH=.

BASE_CFG="src/config/transformer_baseline.yaml"
TASKS=("IEEE39" "IEEE118" "UK")

mkdir -p output/tmp_configs

for GRID in "${TASKS[@]}"; do
  TMP_CFG="output/tmp_configs/transformer_${GRID}.yaml"
  # Clone the baseline config and swap in the requested grid (plus a grid-specific output dir).
  python - "$BASE_CFG" "$GRID" "$TMP_CFG" <<'PY'
import pathlib, sys, yaml

base, grid, out = sys.argv[1:]
with open(base) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("data", {})["grid"] = grid
logging_cfg = cfg.setdefault("logging", {})
logging_cfg.setdefault("output_dir", f"output/{grid.lower()}")

pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    yaml.safe_dump(cfg, f)
print(f"[config] wrote {out} for grid={grid}")
PY

  echo "=== Running grid ${GRID} ==="
  uv run src/experiments/run_experiment.py --config "$TMP_CFG"
  echo
done
