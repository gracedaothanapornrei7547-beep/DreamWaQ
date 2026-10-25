#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 14: Auto-detect and Export Unified Model to ONNX/JIT
# ==============================================================================

WORKSPACE_DIR="/home/rum/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
EXP_DIR=$(find /home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_unified_stage14/ -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)

if [ -z "$EXP_DIR" ]; then
    echo "[ERROR] No Stage 14 experiment directory found!"
    exit 1
fi

LATEST_PT=$(find "$EXP_DIR" -maxdepth 1 -name "model_*.pt" | sort -V | tail -n 1)

if [ -z "$LATEST_PT" ]; then
    echo "[ERROR] No model_*.pt found in $EXP_DIR"
    exit 1
fi

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Exporting Stage 14 Unified Model"
echo "  Experiment Dir: $EXP_DIR"
echo "  Checkpoint    : $LATEST_PT"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/play_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Unified-Stage14-Play \
    --headless \
    --checkpoint "$LATEST_PT" \
    --max_steps 1

echo "[SUCCESS] Stage 14 exported to $EXP_DIR/exported"

