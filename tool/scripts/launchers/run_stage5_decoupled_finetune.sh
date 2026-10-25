#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 5: Decoupled Flat Pitch Lock & Dual-Bounded Clearance Fine-tuning
# ==============================================================================
# 核心技术目标：
# 1. 继承底模：Stage 4 软着陆阻尼终模 (model_6194.pt)；
# 2. 注入解耦平地俯仰刚性锁：terrain_adaptive_pitch_orientation_l2 (-0.40)
#    - 平地巡航时将 Pitch 晃动强行死锁在 <= 1.5° (消除跷跷板点头颠簸)；
#    - 台阶大仰角时 gate 平滑降为 0，彻底解脱俯仰锁，释放爬楼自由度 (遵循红线 W4)；
# 3. 注入双向限幅足端净空惩罚：dual_bounded_pitch_adaptive_foot_clearance_l1 (-0.20)
#    - 平地设定 flat_max_clearance = 0.028m，超额抬腿处以 3.0x 重罚，死死压在 10~25mm 悬浮滑行；
#    - 台阶上超额惩罚归零，激励 0.16m 深度提膝，保持 100% 翻越 5 级台阶；
# 4. 垂向蹦跳严厉阻尼加倍：lin_vel_z_l2 (-1.50) 强力压平机身起伏；
# 5. 角速度与恒高刚性加固：ang_vel_xy_l2 (-0.10), base_height_relative_l2 (-1.00, target=0.30m)；
# 6. 触地软着陆与动作二阶平滑再强化：feet_contact_velocity_z (-0.08), action_smoothness_2nd (-0.008)。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_softlanding_stage4/2026-10-12_15-25_waxberry_stage4_softlanding_from_mixed5895_300iter"
BASE_CHECKPOINT="model_6194.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-300}
RUN_NAME=${3:-waxberry_stage5_decoupled_flat_pitch_lock_from_softlanding6194_300iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 5 Decoupled Fine-tuning"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Decoupled"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6194 -> 6494)"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Decoupled \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"

