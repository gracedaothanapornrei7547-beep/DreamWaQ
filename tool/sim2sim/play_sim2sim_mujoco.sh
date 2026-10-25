#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
MODEL_DIR="${1:-$SCRIPT_DIR/logs/DreamQaW_P2/1_straight/exported}"
SCENE="${2:-flat}"
VX="${3:-0.8}"

export DISPLAY="${DISPLAY:-:0}"

echo "=========================================================="
echo "  Launching Unitree Go2 DreamWaQ MuJoCo 3D Interactive Sim2Sim"
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

