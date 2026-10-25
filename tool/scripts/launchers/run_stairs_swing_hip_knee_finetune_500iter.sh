#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ Targeted Swing Hip & Knee Flexion Finetuning (500 iters)
# ==============================================================================
# 核心技术设计：
# 1. 接力模型：基于已具备 15cm 全向阶梯通过能力的最新成熟检查点 model_7095.pt；
# 2. 运动学三维正交解耦与摆动相靶向锁：
#    - 放宽小腿关节软限位至 0.96 (soft_joint_pos_limit_factor = 0.96)，解锁后腿正向深屈膝空间；
#    - 激活摆动相专属髋关节外展重罚 (swing_hip_abduction_l2 = -0.50)，只惩罚腾空画弧外甩；
#    - 激活双后肢对称外展约束 (rear_hip_swing_symmetry_l2 = -0.10)，消除右后腿单侧独立外甩；
#    - 智能动态门控 (|vy| < 0.08 & |wz| < 0.15)：直行爬阶严管外甩，斜爬与自转时约束瞬间清零；
# 3. 全向台阶指令域持续激活：
#    - ranges.lin_vel_x = (0.40, 0.70) m/s（正向攻坚）
#    - ranges.lin_vel_y = (-0.25, 0.25) m/s（斜向爬阶与侧向跨阶）
#    - ranges.ang_vel_z = (-0.50, 0.50) rad/s（原地回旋与自转）
#    - rel_heading_envs = 0.50（50% 航向锁长跑，50% 自由旋转）
# 4. 攻坚轮次：500 轮（7095 -> 7595 轮），预计耗时约 20 ~ 25 分钟。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_stairs_straight_corridor/2026-10-11_14-22_waxberry_stairs_omnidirectional_finetune_500iter"
BASE_CHECKPOINT="model_7095.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-500}
RUN_NAME=${3:-waxberry_stairs_swing_hip_knee_finetune_500iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Targeted Swing Hip & Knee Finetuning (500 iters)"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Stairs"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (7095 -> 7595)"
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

