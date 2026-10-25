#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 13 (Route B Purify): Pure Flat Equivariant Symmetry Wash (200 iterations)
# 方案：彻底切断楼梯地形，切换为纯平地走廊，施加对角对称 + 双向镜像等变 + 四肢功耗均等约束
# 从 Stage 10 黄金基准 model_6990.pt 快训 200 轮 (6990 -> 7190)，彻底洗净非对称神经突触
# ==============================================================================

WORKSPACE_DIR="/home/rum/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_optimal_stage10/2026-10-12_21-33_waxberry_stage10_optimal_zero_arch_from_equivariant6841_150iter"
BASE_CHECKPOINT="model_6990.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-200}
RUN_NAME=${3:-waxberry_stage13_flat_purify_from_optimal6990_200iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 13 Pure Flat Symmetry Wash"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-FlatPurify-Stage13"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6990 -> 7190)"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-FlatPurify-Stage13 \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"

