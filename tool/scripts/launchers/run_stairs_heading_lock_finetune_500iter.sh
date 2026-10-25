#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ Heading Lock Stairs Corridor Targeted Finetuning (500 iters)
# ==============================================================================
# 核心技术设计：
# 1. 接力模型：基于已完全具备 15cm 跨阶爬楼动态平衡肌肉记忆的 model_6097.pt；
# 2. 闭环纠偏：注入 heading_command = True, ranges.heading = (0.0, 0.0), stiffness = 1.0；
# 3. 动态纠偏力矩：ranges.ang_vel_z = (-1.0, 1.0)，在遭遇非对称碰撞时主动反向打轮拉回中轴线；
# 4. 梯队展开：max_init_terrain_level = 5 (~9.7cm)，兼顾中高阶梯，杜绝低阶磨洋工；
# 5. 攻坚轮次：500 轮（6097 -> 6597 轮），耗时约 20 ~ 25 分钟。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_stairs_straight_corridor/2026-10-10_21-38_waxberry_stairs_corridor_physical_curriculum_1000iter"
BASE_CHECKPOINT="model_6097.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-500}
RUN_NAME=${3:-waxberry_stairs_heading_lock_finetune_500iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Heading Lock Stairs Corridor (500 iters)"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Stairs"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6097 -> 6597)"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Stairs \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"

