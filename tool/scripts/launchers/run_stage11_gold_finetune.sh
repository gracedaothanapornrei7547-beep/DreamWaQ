#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 11: Gold Baseline Surgical Alignment (500 iterations)
# 方案：根治台阶落脚惩罚死锁 (Landing Gate) + 消除训练航向重置撕扯 (Laser-Straight Heading Lock)
# 从 Stage 10 model_6990.pt 加训 500 轮 (6990 -> 7490)
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_optimal_stage10/2026-10-12_21-33_waxberry_stage10_optimal_zero_arch_from_equivariant6841_150iter"
BASE_CHECKPOINT="model_6990.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-500}
RUN_NAME=${3:-waxberry_stage11_gold_from_optimal6990_500iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 11 Gold Baseline Surgical Training"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Gold-Stage11"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6990 -> 7490)"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Gold-Stage11 \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"

