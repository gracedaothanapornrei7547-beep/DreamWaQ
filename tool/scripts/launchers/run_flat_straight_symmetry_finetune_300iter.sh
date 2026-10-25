#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ Flat Straight Symmetry Targeted Finetuning (300 iters)
# ==============================================================================
# 核心技术设计：
# 1. 接力底模：基于 4000 轮原模 model_4000.pt（近 4 亿步强化学习收敛的稳健 CENet 速度感知）；
# 2. 根治走直线后腿外甩（RL: +19.7° vs RR: -13.2°，偏差 6.5°）：
#    - 地形：纯平地（Plane，彻底剔除台阶几何高度突变对步态对称性的多任务干扰）；
#    - 指令：锁定纯直行区间 vx ∈ [0.4, 0.8] m/s, vy = 0.0, wz = 0.0；
#    - 门控：动态门控 (|vy| < 0.08 & |wz| < 0.15) 100% 恒开，全时段全样本严厉施加惩罚；
#    - 摆动相外展严惩：swing_hip_abduction_l2 = -0.50（杜绝画弧外甩）；
#    - 四肢全镜像对称硬约束：rear_hip_swing_symmetry_l2 = -0.20（强制两后腿 1:1 对称）；
#    - 降温学习率：lr = 2.0e-4，平稳吸收残余不对称误差，绝不破坏 CENet 稳态。
# 3. 攻坚轮次：300 轮（4000 -> 4300 轮），RTX 4060 预计耗时约 12 ~ 15 分钟。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-09_22-32_waxberry_benchmark_rough_continue_4000iter"
BASE_CHECKPOINT="model_4000.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-300}
RUN_NAME=${3:-waxberry_flat_straight_symmetry_finetune_300iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Flat Straight Symmetry Finetuning"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Flat"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (4000 -> 4300)"
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
    --run_name "$RUN_NAME"
