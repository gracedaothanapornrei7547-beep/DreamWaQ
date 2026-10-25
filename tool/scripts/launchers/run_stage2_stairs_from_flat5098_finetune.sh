#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 2: Transfer 5098-iter Flat Golden Policy into Stairs Fine-tuning
# ==============================================================================
# 核心技术设计：
# 1. 接力底模：5098 轮平地悬浮滑行 + 全向平移锁航向 + 零半径自转黄金终模 (model_5097.pt)；
# 2. 彻底拆除姿态锁红线 (W4)：
#    - flat_orientation_l2 = 0.0 (彻底释放 Pitch 仰角，允许顺应 15°~25° 楼梯坡度自然上扬)；
#    - lin_vel_z_l2 = -0.10 (放宽垂直上升速度阻尼，保障爬升动力)；
#    - base_height_relative_l2 = -0.20 (软化恒高约束，适应台阶踏面落差)；
# 3. 动态跨阶安全净空：
#    - terrain_relative_foot_clearance_l1 目标净空 0.16m，权重 -0.15；
#    - undesired_contacts = -0.20 (平稳引导抬腿提膝，杜绝遇阶畏缩停步)；
# 4. 渐进式阶梯自适应课程：
#    - 起步梯队 max_init_terrain_level = 2 (从 3~5.6cm 低阶平滑起跑，随 mdp.terrain_levels_vel 自动向 15cm 突破)；
# 5. 全向指令域覆盖：
#    - lin_vel_x in [0.4, 0.7] m/s, lin_vel_y in [-0.25, 0.25] m/s, ang_vel_z in [-0.5, 0.5] rad/s；
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-11_21-18_waxberry_flat_omni_translate_and_spin_finetune_500iter"
BASE_CHECKPOINT="model_5097.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-500}
RUN_NAME=${3:-waxberry_stage2_stairs_from_flat5098_finetune_500iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: WaxberryS Stage 2 Stairs Fine-tuning"
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
