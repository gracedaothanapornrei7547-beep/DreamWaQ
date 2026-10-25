#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 12 (Route B): Flexible Compliance Loosening & 0.3s Lookahead (300 iterations)
# 方案：
# 1. 0.3s 前瞻感知强制激活前腿高提膝 (alpha_ahead -> 1.0, 目标抬腿 16cm, 豁免平地超额惩罚)
# 2. 刚性约束全面松绑 80% (降至 -0.05 极弱引导)，归还 Trot 顺应性，消除向右偏摆
# 3. 从 Stage 10 黄金检查点 model_6990.pt 加训 300 轮 (6990 -> 7290)
# ==============================================================================

WORKSPACE_DIR="/home/rum/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_optimal_stage10/2026-10-12_21-33_waxberry_stage10_optimal_zero_arch_from_equivariant6841_150iter"
BASE_CHECKPOINT="model_6990.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-300}
RUN_NAME=${3:-waxberry_stage12_flex_loose_from_optimal6990_300iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 12 Route B Flexible Loosening Training"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Gold-Stage11"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6990 -> 7290)"
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

