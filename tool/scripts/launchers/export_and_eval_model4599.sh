#!/bin/bash
set -e

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
EXP_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-11_20-55_waxberry_flat_fast_straight_heading_hover_finetune_300iter"

echo "[1/2] Exporting model_4598.pt to ONNX and JIT deploy bundle..."
$PYTHON_EXEC scripts/rsl_rl/dreamwaq/play_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Flat-Play \
    --headless \
    --checkpoint "$EXP_DIR/model_4598.pt" \
    --max_steps 1

echo "[2/2] Extracting foot trajectory and body vibration quantitative metrics..."
$PYTHON_EXEC tool/scripts/launchers/extract_flat_gait_metrics.py "$EXP_DIR/exported"
