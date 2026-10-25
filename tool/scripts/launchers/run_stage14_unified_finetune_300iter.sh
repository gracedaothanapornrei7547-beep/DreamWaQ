#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 14: Unified Flat-Stair Synergy Training (300 iterations)
# 方案：以 Stage 13 纯平地洗直检查点 model_7189.pt 为基石，返回台阶走廊地形
# 融合 0.3s 前瞻感知门控 (clearance=-0.12) 与黄金等变对称力场 (bilateral_hip=-0.20, power_var=-5e-5)
# 从 7189 加训 300 轮 (7189 -> 7489)，彻底实现平地激光直行与 15cm 台阶攀爬完美大一统！
# ==============================================================================

WORKSPACE_DIR="/home/rum/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat_purify_stage13/2026-10-14_13-06_waxberry_stage13_flat_purify_from_optimal6990_200iter"
BASE_CHECKPOINT="model_7189.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-300}
RUN_NAME=${3:-waxberry_stage14_unified_from_flatpurify7189_300iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 14 Unified Flat-Stair Synergy"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Unified-Stage14"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (7189 -> 7489)"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Unified-Stage14 \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"

