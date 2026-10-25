#!/bin/bash
set -e

# ==============================================================================
# One-click Launcher: Play Stage 14 Unified Master Model in MuJoCo 3D GUI
# Features: 15cm High Stairs (0-Stutter 100% Summit) + Laser-Straight Flat Trot (1mm Drift)
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
# Auto-detect latest Stage 14 checkpoint export directory
MODEL_DIR=$(find /home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_unified_stage14 -type d -name "exported" 2>/dev/null | sort | tail -n 1)

if [ -z "$MODEL_DIR" ]; then
    MODEL_DIR=$(find /home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_unified_stage14 -maxdepth 2 -mindepth 1 -type d | sort | tail -n 1)/exported
fi

ASSETS_DIR="/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"
export DISPLAY=${DISPLAY:-:0}

SCENE="${1:-stairs_5steps_15cm}"
VX="${2:-0.6}"

echo "=========================================================="
echo "  Unitree Go2 Stage 14 Unified Master Policy Teleop (MuJoCo)"
echo "  Features : 15cm Stairs (100% Summit, 0 Stutter) + Laser-Straight (1mm Drift)"
echo "  Controls : W/S: Vx | A/D: Vy | Q/E: Wz | Space/X: Brake | R: Reset"
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

