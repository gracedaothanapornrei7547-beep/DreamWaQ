#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ IsaacLab 3D GUI Live Visualizer (Stairs/Steps)
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
cd "$REPO_ROOT"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
DEFAULT_CKPT="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_stairs_straight_corridor/2026-10-11_16-23_waxberry_stairs_clearance_calf_symmetry_finetune_1000iter/model_8593.pt"
CKPT="${1:-$DEFAULT_CKPT}"
NUM_ENVS=${2:-4}

export DISPLAY="${DISPLAY:-:0}"

echo "=========================================================="
echo "  Launching IsaacLab 3D GUI Live Visualizer"
echo "  Workspace : $REPO_ROOT"
echo "  Task      : Unitree-Go2-DreamWaQ-Stairs-Play"
echo "  Checkpoint: $CKPT"
echo "  Envs      : $NUM_ENVS"
echo "  Display   : $DISPLAY"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/play_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Stairs-Play \
    --num_envs "$NUM_ENVS" \
    --real-time \
    --checkpoint "$CKPT"

