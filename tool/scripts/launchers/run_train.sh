#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"

NUM_ENVS=${1:-1024}
MAX_ITERS=${2:-1000}
RUN_NAME=${3:-trot_sagittal_strict_pitch_damp}

echo "=========================================================="
echo "  Starting Unitree Go2 DreamWaQ Training"
echo "  Workspace:    $SCRIPT_DIR"
echo "  Python:       $PYTHON_EXEC"
echo "  Environments: $NUM_ENVS"
echo "  Iterations:   $MAX_ITERS"
echo "  Run Name:     $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Rough \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --run_name "$RUN_NAME"
