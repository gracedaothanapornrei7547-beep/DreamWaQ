#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 7: CENet Bilateral Equivariance & Power Balance All-in-One Training
# 方案 B：全地形大一统单模型终极对齐 (200 轮, 6642 -> 6842)
# ==============================================================================
# 核心技术目标：
# 1. 继承底模：Stage 6 镜像手术终模 (model_6642.pt，内偏已拉直)；
# 2. CENet 编码器源头注入【左右镜像等变损失】：
#    || z(flip(h_t)) - flip(z(h_t)) ||^2 + || v(flip(h_t)) - flip(v(h_t)) ||^2
#    彻底铲除隐空间单侧阻抗极化偏置，强制左右潜变量对偶等变！
# 3. 注入几何与动力学铁笼约束：
#    - power_distribution_variance_l2 = -1.0e-4 (强化 10 倍，强制四腿功率均摊，消灭单侧主导)
#    - hip_pos_deviation_l1 = -0.50 (强化总髋位姿偏离约束，锁死四腿髋角绝对对中)
#    - front_hip_neutral_pos_l2 = -0.80 (持续锁死左前腿拉直)
#    - bilateral_hip_symmetry_equivariance_l2 = -0.50 (左右前腿与后腿严格镜像)
# 4. 训练规模：4096 并行环境，200 轮高速对齐，单模型平地直行与台阶越障兼备！
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_surgery_stage6/2026-10-12_18-53_waxberry_stage6_surgery_straight_symmetry_from_decoupled6493_150iter"
BASE_CHECKPOINT="model_6642.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-200}
RUN_NAME=${3:-waxberry_stage7_equivariant_allinone_from_surgery6642_200iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 7 Bilateral Equivariance All-in-One"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Equivariant"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6642 -> 6842)"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Equivariant \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"

