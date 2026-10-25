#!/bin/bash
set -e

# ==============================================================================
# One-click Launcher: Play Stage 6 Surgery Model with QWEASD Keyboard Teleoperation
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
# Auto-detect latest Stage 6 checkpoint export directory
MODEL_DIR=$(find /home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_surgery_stage6 -type d -name "exported" 2>/dev/null | sort | tail -n 1)

if [ -z "$MODEL_DIR" ]; then
    MODEL_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_surgery_stage6/latest/exported"
fi

ASSETS_DIR="/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"
export DISPLAY=${DISPLAY:-:0}

SCENE="${1:-stairs_5steps}"
VX="${2:-0.5}"

echo "=========================================================="
echo "  Unitree Go2 Stage 6 Surgery Model Teleoperation"
echo "  Features : Bilateral Symmetry Lock + Front Hip Neutral Lock (No Inward Drift!)"
echo "  Controls : W/S: Vx | A/D: Vy | Q/E: Wz | Space: Brake | R: Reset"
echo "  Camera   : C/V: Toggle Free/Tracking Cam | Mouse: 360° Rotate & Zoom"
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

