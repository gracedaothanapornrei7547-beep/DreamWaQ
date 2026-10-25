#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ: Fast Straight [0.4, 1.2]m/s Heading Lock & Hovering Finetuning (300 iters)
# ==============================================================================
# 核心技术设计：
# 1. 接力底模：基于刚收敛的 4299 轮黄金对称底模 model_4299.pt；
# 2. 训练端 Yaw 自抗扰强化闭环体系：
#    - 闭环航向指令激活：heading_command=True, stiffness=1.0, ranges.heading=(0.0, 0.0), rel_heading_envs=1.0；
#    - 绝对偏航角二次方严惩：flat_heading_error_l2 = -0.50，对任何偏航施加绝对惩罚，逼迫策略反向打轮回正；
#    - 偏航角速度严格追踪：track_ang_vel_z_exp = 1.0 (std=0.40)；
#    - 正交横向漂移严惩加码：lin_vel_xy_orthogonal_l2 = -0.30；
# 3. 极致悬浮平顺滑行步态（消除颠簸）：
#    - 目标站高压低：base_height_relative_l2 = -2.0, target_height = 0.30m (降低重心，贴地巡航)；
#    - 垂直蹦跳严厉阻尼：lin_vel_z_l2 = -3.0 (抑制上下颠簸振动)；
#    - 躯干平稳刚性锁：flat_orientation_l2 = -0.40, ang_vel_xy_l2 = -0.10 (抑制 Roll/Pitch 摇晃)；
# 4. 极致收敛外摆（纯正矢状面屈膝，告别外八字）：
#    - 摆动相外展严惩：swing_hip_abduction_l2 = -0.80；
#    - 髋关节对称硬约束：rear_hip_swing_symmetry_l2 = -0.30, hip_pos_deviation_l1 = -0.15；
# 5. 指令域拓宽：vx ∈ [0.4, 1.2] m/s 高速宽温区；
# 6. 耗时预估：300 轮 (4299 -> 4599 轮)，耗时约 10 ~ 12 分钟。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-11_19-35_waxberry_flat_straight_symmetry_finetune_300iter"
BASE_CHECKPOINT="model_4299.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-300}
RUN_NAME=${3:-waxberry_flat_fast_straight_heading_hover_finetune_300iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Fast Straight Heading & Hovering Finetuning"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Flat"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (4299 -> 4599)"
echo "  Speed Range    : [0.4, 1.2] m/s"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Flat \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME" 2>&1 | tee "tool/scripts/launchers/train_flat_fast_straight_heading_hover_300iter.log"
