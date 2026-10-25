#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 9: Smooth Stairs & Straight Drift Erasure Fine-tuning
# 方案 A: 单模型终极平滑手术 (150 轮, 6841 -> 6991)
# ==============================================================================
# 核心技术目标：
# 1. 继承底模：Stage 7 镜像等变终模 (model_6841.pt，左右等变、四腿平衡、偏航误差仅 0.0414 rad/s)；
# 2. 彻底解决爬楼梯卡顿/卡壳问题：
#    - undesired_contacts: weight 从 -0.40 降至 -0.02 (消除二值硬惩罚)；
#    - soft_contact_velocity_damping: weight = -0.05 (允许低速顺滑贴阶滑行，仅重惩高速粗暴硬撞，消灭碰阶回缩假死与卡顿)；
# 3. 彻底解决走直线向左偏摆问题：
#    - straight_ang_vel_z: weight = -0.25 (直行指令下严惩偏航角速度 |w_z|^2)；
#    - flat_heading_error_l2: weight = -0.40 (平地绝对航向角死锁)；
#    - rear_hip_neutral_pos: weight = -0.50 (后腿髋关节双向中立零位锁，消灭 RL/RR 偏推)；
# 4. 全量维持 Stage 7/8 动力学铁笼：
#    - front_hip_neutral_pos_l2 = -0.80, bilateral_hip_symmetry_equivariance_l2 = -0.50, power_distribution_variance_l2 = -1.0e-4
#    - dual_bounded_pitch_adaptive_foot_clearance_l1 = -0.25 (15cm 极限高台阶提膝 + 平地 22mm 磁悬浮滑行)
# 5. 训练规模：4096 并行环境，150 轮加训（约 8 分钟），单模型实现平地磁悬浮激光级绝对直线 + 15cm 极限高台阶丝滑无顿挫大一统！
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_equivariant_stage7/2026-10-12_19-45_waxberry_stage7_equivariant_allinone_from_surgery6642_200iter"
BASE_CHECKPOINT="model_6841.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-150}
RUN_NAME=${3:-waxberry_stage9_smooth_straight_from_equivariant6841_150iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 9 Smooth Stairs & Straight Drift Erasure"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Smooth-Stage9"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6841 -> 6991)"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Smooth-Stage9 \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"

