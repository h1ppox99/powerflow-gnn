#!/usr/bin/env bash
# Run the TransformerConv baseline with a global virtual node.
set -euo pipefail

UV_CACHE_DIR=${UV_CACHE_DIR:-./.uv_cache}
export UV_CACHE_DIR
export PYTHONPATH=.

BASE_CFG=${1:-src/config/transformer_baseline.yaml}
TMP_CFG="output/tmp_configs/transformer_vn.yaml"

mkdir -p output/tmp_configs

# Clone the baseline config but switch the model to transformer_vn so we reuse the pipeline as-is.
python - "$BASE_CFG" "$TMP_CFG" <<'PY'
import sys, yaml, pathlib
src, dst = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("model", {})["name"] = "transformer_vn"
cfg.setdefault("logging", {}).setdefault("output_dir", "output/transformer_vn")
pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f)
print(f"[config] wrote {dst} (model=transformer_vn)")
PY

uv run src/experiments/run_experiment.py --config "$TMP_CFG"
