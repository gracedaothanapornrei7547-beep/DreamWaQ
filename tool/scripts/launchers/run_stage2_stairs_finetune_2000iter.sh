#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ Phase 2 Stage 2: Stairs-Focused Curriculum Fine-tuning
# ==============================================================================
# 核心路线：以平地成熟直行底模 (model_5098.pt) 为先验，80% 连续台阶攻坚微调 2000 轮
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/isaac/dreamwaq_lab/logs/DreamQaW_P2/1_straight"
BASE_CHECKPOINT="model_5098.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-2000}
RUN_NAME=${3:-waxberry_stage2_stairs_finetune_2000iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ Stage 2: Stairs Fine-tuning (2000 iters)"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Stairs"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS"
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
