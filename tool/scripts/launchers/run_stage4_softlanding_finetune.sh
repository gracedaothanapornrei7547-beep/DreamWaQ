#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 4: Soft Landing Damping & Contact Force Cushioning Fine-tuning
# ==============================================================================
# 核心技术目标：
# 1. 继承底模：Stage 3 混合融合微调终模 (model_5895.pt)；
# 2. 注入软着陆阻尼：feet_contact_velocity_z (-0.06)，迫使足端触地瞬间主动缓速，消除跺地撞击；
# 3. 接触力超限软化：feet_contact_force_excess (-0.0001)，抑制膝关节瞬态硬碰撞力矩；
# 4. 动作二阶平滑：action_smoothness_2nd (-0.005)，根除 50Hz 高频抽搐振荡；
# 5. 强化垂向速度阻尼：lin_vel_z_l2 (-0.75)，平地机身垂直抖动彻底压入 < 10.0mm 黄金线；
# 6. 保留 100% 越障：保持 stairs_target_clearance=0.16m 与台阶长廊课程，确保 5 级台阶翻越零退化。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_mixed_stage3/2026-10-12_14-42_waxberry_stage3_mixed_fusion_from_stairs5596_300iter"
BASE_CHECKPOINT="model_5895.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-300}
RUN_NAME=${3:-waxberry_stage4_softlanding_from_mixed5895_300iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 4 Soft Landing Fine-tuning"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-SoftLanding"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-SoftLanding \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"

