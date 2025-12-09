#!/usr/bin/env bash
# Run the physics_weight sweep (10^-5 to 10^3) using the generic sweeper.
set -euo pipefail

UV_CACHE_DIR=${UV_CACHE_DIR:-./.uv_cache}
export UV_CACHE_DIR
export PYTHONPATH=.

# --- Configuration ---
CFG="src/config/transformer_physics.yaml"
SCRIPT="src/experiments/run_sweep.py"

# --- Sweep Parameters ---
# We want 10^-5 to 10^3. 
# Since we use --log, START/STOP are powers of 10.
PARAM="train.physics_weight"
START="-5"  
STOP="3"    
STEPS="9"   # 9 steps: 1e-5, 1e-4, ..., 1e3
FLAGS="--log"

echo "========================================================"
echo "Starting Sweep: $PARAM"
echo "Range: 10^$START to 10^$STOP ($STEPS steps)"
echo "Config: $CFG"
echo "========================================================"

uv run "$SCRIPT" \
    --config "$CFG" \
    --param "$PARAM" \
    --start "$START" \
    --stop "$STOP" \
    --steps "$STEPS" \
    $FLAGS