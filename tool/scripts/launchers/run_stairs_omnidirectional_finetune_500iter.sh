#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ Omnidirectional Stair Domain Targeted Finetuning (500 iters)
# ==============================================================================
# 核心技术设计：
# 1. 接力模型：基于已具备 15cm 爬楼与 Heading Lock 闭环的成熟检查点 model_6596.pt；
# 2. WaxberryS 全向台阶指令域激活：
#    - ranges.lin_vel_x = (0.40, 0.70) m/s（正向攻坚）
#    - ranges.lin_vel_y = (-0.25, 0.25) m/s（主动覆盖斜向爬阶与侧向跨阶）
#    - ranges.ang_vel_z = (-0.50, 0.50) rad/s（主动覆盖爬阶原地回旋与自转）
#    - rel_heading_envs = 0.50（50% 环境带航向锁保证长跑，50% 环境完全自由采样旋转）
# 3. 动力学机制威慑：全向指令的随机对称压力，瞬间摧毁右后腿单侧外甩偷分，逼出绝对对称紧凑步态；
# 4. 攻坚轮次：500 轮（6596 -> 7096 轮），预计耗时约 20 ~ 25 分钟。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_stairs_straight_corridor/2026-10-11_13-21_waxberry_stairs_heading_lock_finetune_500iter"
BASE_CHECKPOINT="model_6596.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-500}
RUN_NAME=${3:-waxberry_stairs_omnidirectional_finetune_500iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Omnidirectional Stairs Corridor (500 iters)"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Stairs"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6596 -> 7096)"
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
