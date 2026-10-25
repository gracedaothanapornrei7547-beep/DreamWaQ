#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 3: Transfer 5596-iter Stairs Policy into Mixed Terrain Fine-tuning
# ==============================================================================
# 核心技术设计：
# 1. 接力底模：5596 轮单向阶梯长廊攻坚终模 (model_5596.pt)；
# 2. 地形配比：40% 纯平地长廊 (MeshPlaneTerrainCfg) + 60% 单向阶梯长廊 (MeshStraightAscendingStairsTerrainCfg)；
# 3. 俯仰自适应单边净空 (Pitch-Adaptive Clearance)：
#    - 平地 (|pitch| < 3°): 净空门限自动回缩至 0.04m，步态轻盈紧凑，根治高抬腿猛跺地地面反力冲击；
#    - 台阶 (pitch > 6°): 自适应扩大至 0.16m，强力引导深屈膝高提膝翻越 15cm 大台阶；
# 4. 姿态精准解耦 (roll_orientation_l2 = -0.25)：锁定横滚 Roll，释放俯仰 Pitch；
# 5. 平衡垂直速度阻尼 (lin_vel_z_l2 = -0.30)：平地抑制起伏颠簸，爬楼梯保障向上攀爬动能；
# 6. 消除潜变量混淆：CENet 16 维隐空间 z 在同一批次内同时学习平地光滑流形与台阶接触阻抗。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_stairs_straight_corridor/2026-10-12_13-36_waxberry_stage2_stairs_from_flat5098_finetune_500iter"
BASE_CHECKPOINT="model_5596.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-300}
RUN_NAME=${3:-waxberry_stage3_mixed_fusion_from_stairs5596_300iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 3 Mixed Fusion Fine-tuning"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Mixed"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Mixed \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"
