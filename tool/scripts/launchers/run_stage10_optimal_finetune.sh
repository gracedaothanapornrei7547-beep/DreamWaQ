#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 10: Optimal Zero-Architecture-Change Equivariant & Smooth Alignment
# 方案：0 网络修改 · 纯物理与等变 Loss 级最优解 (150 轮, 6841 -> 6991)
# ==============================================================================
# 核心技术目标：
# 1. 0 架构变动与无损继承：
#    - 严格保持 5 帧 225 维本体感知输入与 12 维动作输出，继承 Stage 7 镜像等变终模 (model_6841.pt)；
# 2. 彻底消灭直行向左偏摆问题：
#    - CENet 强制物理镜像等变自监督损失：\hat{v}_y(flip(h_t)) = -\hat{v}_y(h_t)，从网络源头彻底铲除幽灵侧向速度偏置；
#    - straight_ang_vel_z = -0.30 (平地直行偏航角速度刚性锁强化)；
#    - flat_heading_error_l2 = -0.40 (平地绝对航向刚性死锁)；
#    - rear_hip_neutral_pos = -0.50 (后腿双髋对称中立锁)；
# 3. 彻底消灭台阶卡顿与卡壳问题：
#    - undesired_contacts: weight = 0.0 (彻底清零小腿/大腿碰阶惩罚，消灭避险回缩假死 Retraction Reflex)；
#    - soft_contact_velocity_damping: weight = 0.0 (彻底清零碰阶速度阻尼，杜绝碰阶降速与停滞)；
#    - feet_contact_velocity_z: weight = 0.0 (彻底清零足端触地垂直速度阻尼，释放落脚动量，消除蹬阶迟疑)；
# 4. 全量维持 Stage 7/8/9 动力学铁笼：
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
RUN_NAME=${3:-waxberry_stage10_optimal_zero_arch_from_equivariant6841_150iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 10 Optimal Zero-Arch Alignment"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Optimal-Stage10"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6841 -> 6991)"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Optimal-Stage10 \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"
