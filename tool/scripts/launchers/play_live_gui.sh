#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
CKPT="${1:-$SCRIPT_DIR/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-04_14-48_trot_level_positive_stride/model_999.pt}"

export DISPLAY="${DISPLAY:-:0}"

echo "=========================================================="
echo "  Launching Isaac Sim 3D GUI Live Playback"
echo "  Checkpoint: $CKPT"
echo "  Display:    $DISPLAY"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/play_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Rough-Play \
    --num_envs 1 \
    --real-time \
    --checkpoint "$CKPT"
