#!/bin/bash
set -e

# ==============================================================================
# One-click Launcher: Play Unitree Go2 with QWEASD Keyboard Teleoperation in MuJoCo
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
DEFAULT_MODEL_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_stairs_straight_corridor/2026-10-12_13-36_waxberry_stage2_stairs_from_flat5098_finetune_500iter/exported"
MODEL_DIR="${1:-$DEFAULT_MODEL_DIR}"
SCENE="${2:-stairs_5steps}"
VX="${3:-0.45}"
ASSETS_DIR="/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"

export DISPLAY=${DISPLAY:-:0}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: QWEASD Interactive Teleoperation"
echo "  Model Dir: $MODEL_DIR"
echo "  Scene    : $SCENE"
echo "  Init Vx  : $VX m/s"
echo "  Display  : $DISPLAY"
echo "=========================================================="

$PYTHON_EXEC deploy/sim2sim/sim2sim_mujoco.py \
    --model_dir "$MODEL_DIR" \
    --assets_dir "$ASSETS_DIR" \
    --scene "$SCENE" \
    --vx "$VX" \
    --vy 0.0 \
    --wz 0.0 \
    --duration 0 \
    --teleop \
    --heading_mode

