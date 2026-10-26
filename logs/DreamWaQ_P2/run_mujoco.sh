#!/bin/bash
set -e

# ==============================================================================
# DreamWaQ Phase 2 Golden Milestone: Stage 14 Unified Master Model
# Features: 15cm Stairs (100% Summit, 0 Stutter) + Laser-Straight Flat Trot (1mm Drift)
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
MODEL_DIR="/home/rum/dreamwaq_lab/logs/DreamWaQ_P2/exported"
ASSETS_DIR="/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"
export DISPLAY=${DISPLAY:-:0}

SCENE="${1:-stairs_5steps_15cm}"
VX="${2:-0.6}"

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ Phase 2 (P2) Golden Master Baseline"
echo "  Model Dir: $MODEL_DIR"
echo "  Scene    : $SCENE"
echo "  Init Vx  : $VX m/s"
echo "  Features : 15cm Stairs (0 Stutter) + Laser-Straight (1mm Drift)"
echo "  Controls : W/S: Vx | A/D: Vy | Q/E: Wz | Space: Brake | R: Reset"
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

