#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ: Flat Omnidirectional Translation & In-Place Turn (500 iters)
# ==============================================================================
# 核心技术设计：
# 1. 接力底模：基于刚收敛的 4598 轮黄金终模 model_4598.pt (继承悬浮滑行与极佳躯干稳定性)；
# 2. 运动模态一：平地全向平移且 Yaw 锁定（给 XY 速度走，Yaw 朝向严格不变，螃蟹步/斜行）：
#    - rel_heading_envs = 0.50，50% 环境强制激活 heading_command=True, ranges.heading=(0.0, 0.0)；
#    - vx ∈ [-0.6, 1.0] m/s (覆盖倒退与快跑), vy ∈ [-0.5, 0.5] m/s (覆盖全向横向滑行)；
#    - flat_heading_error_l2 强力锁定偏航角偏差，死死咬住 0.0 朝向；
# 3. 运动模态二：平地原地自转（In-place Zero-Radius Spinning）：
#    - 50% 环境直接采样自由角速度 wz ∈ [-1.0, 1.0] rad/s；
#    - flat_heading_error_l2、swing_hip_abduction_l2 与对称惩罚全部具备 |wz| >= 0.15 动态门控；
#    - 自转时门控瞬时清零，彻底释放四肢对向切向迈步自由度，爆发最大旋转力矩；
# 4. 攻坚轮次：500 轮 (4598 -> 5098 轮)，预计耗时约 15 ~ 18 分钟。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-11_20-55_waxberry_flat_fast_straight_heading_hover_finetune_300iter"
BASE_CHECKPOINT="model_4598.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-500}
RUN_NAME=${3:-waxberry_flat_omni_translate_and_spin_finetune_500iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Flat Omnidirectional & In-Place Turn"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Flat"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (4598 -> 5098)"
echo "  Speed Ranges   : vx=[-0.6, 1.0], vy=[-0.5, 0.5], wz=[-1.0, 1.0]"
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
    --run_name "$RUN_NAME" 2>&1 | tee "tool/scripts/launchers/train_flat_omni_turn_500iter.log"
