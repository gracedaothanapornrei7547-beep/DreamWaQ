#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ MuJoCo 3D Interactive Sim2Sim Launcher
# ==============================================================================

# 定位 dreamwaq_lab 根目录 (tool/scripts/launchers/ -> ../../../)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
cd "$REPO_ROOT"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
MODEL_DIR="${1:-/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_stairs_straight_corridor/2026-10-12_13-36_waxberry_stage2_stairs_from_flat5098_finetune_500iter/exported}"
SCENE="${2:-stairs_5steps}"
VX="${3:-0.6}"

export DISPLAY="${DISPLAY:-:0}"

echo "=========================================================="
echo "  Launching Unitree Go2 DreamWaQ MuJoCo 3D Interactive Sim2Sim"
echo "  Repository: $REPO_ROOT"
echo "  Model Dir : $MODEL_DIR"
echo "  Scene     : $SCENE"
echo "  Command Vx: $VX m/s"
echo "  Display   : $DISPLAY"
echo "=========================================================="

$PYTHON_EXEC deploy/sim2sim/sim2sim_mujoco.py \
    --model_dir "$MODEL_DIR" \
    --scene "$SCENE" \
    --vx "$VX" \
    --heading_mode \
    --duration 0 \
    "${@:4}"

