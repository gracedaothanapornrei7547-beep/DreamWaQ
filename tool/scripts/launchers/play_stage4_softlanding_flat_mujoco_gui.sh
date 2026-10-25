#!/bin/bash
set -e

# ==============================================================================
# One-click Launcher: Play Stage 4 Soft Landing Model (6194 iter) in MuJoCo Flat Ground
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
MODEL_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_softlanding_stage4/2026-10-12_15-25_waxberry_stage4_softlanding_from_mixed5895_300iter/exported"
ASSETS_DIR="/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"

export DISPLAY=${DISPLAY:-:0}

SPEED=${1:-1.0}
echo "=========================================================="
echo "  Playing Unitree Go2 Stage 4 Soft Landing Model (6194 iter)"
echo "  Speed    : $SPEED m/s"
echo "  Model Dir: $MODEL_DIR"
echo "  Scene    : Flat ground (Z_ptp = 5.34mm Golden Standard)"
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
    --heading_mode \
    "${@:2}"

