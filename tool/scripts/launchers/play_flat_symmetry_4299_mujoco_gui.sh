#!/bin/bash
set -e

# ==============================================================================
# One-click Launcher: Play 4299-iter Flat Symmetry Model in MuJoCo GUI Viewer
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
MODEL_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-11_19-35_waxberry_flat_straight_symmetry_finetune_300iter/exported"
ASSETS_DIR="/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"

export DISPLAY=${DISPLAY:-:0}

SPEED=${1:-1.0}
echo "=========================================================="
echo "  Playing Unitree Go2 DreamWaQ 4299-iter Flat Symmetry Model"
echo "  Speed    : $SPEED m/s"
echo "  Model Dir: $MODEL_DIR"
echo "  Scene    : Flat ground"
echo "  Display  : $DISPLAY"
echo "=========================================================="

$PYTHON_EXEC deploy/sim2sim/sim2sim_mujoco.py \
    --model_dir "$MODEL_DIR" \
    --assets_dir "$ASSETS_DIR" \
    --scene flat \
    --vx "$SPEED" \
    --vy 0.0 \
    --wz 0.0 \
    --duration 0 \
    --heading_mode
