#!/bin/bash
set -e

# ==============================================================================
# One-click Launcher: Play Stage 4 Soft Landing Model with QWEASD Keyboard Teleoperation
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
MODEL_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_softlanding_stage4/2026-10-12_15-25_waxberry_stage4_softlanding_from_mixed5895_300iter/exported"
ASSETS_DIR="/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"

export DISPLAY=${DISPLAY:-:0}

SCENE="${1:-stairs_5steps}"
VX="${2:-0.5}"
echo "=========================================================="
echo "  Unitree Go2 Stage 4 Soft Landing Teleoperation (QWEASD Keypad)"
echo "  Controls : W/S: Vx | A/D: Vy | Q/E: Wz | Space: Brake | R: Reset"
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
    --heading_mode \
    "${@:3}"

