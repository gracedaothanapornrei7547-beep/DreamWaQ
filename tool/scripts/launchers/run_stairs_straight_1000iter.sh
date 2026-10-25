#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ Phase 2: Forward Stairs Fine-tuning (1000 iterations)
# ==============================================================================
# 核心路线：
# 1. 彻底回退至第一阶段平地直行黄金成熟底模 (model_5098.pt)，杜绝侧跨动力学污染；
# 2. 80% 开阔单向正向上楼梯（阶高 3cm~15cm）；
# 3. 重生姿态硬锁正对台阶法向 (yaw in [-0.05, 0.05] rad, 误差 < 2.8°)；
# 4. 指令硬锁纯前向 (vx in [0.45, 0.75], vy=0, wz=0)；
# 5. 正交侧漂强镇压 (-0.15) + 髋关节 0° 锁 (-0.15) + 对角对称软正则 (-0.02) + 抬腿净空 (+0.09m, -0.03)。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/isaac/dreamwaq_lab/logs/DreamQaW_P2/1_straight"
BASE_CHECKPOINT="model_5098.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-1000}
RUN_NAME=${3:-waxberry_stairs_straight_1000iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Forward Stairs Fine-tuning (1000 iters)"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Stairs"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (5098 -> 6098)"
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

