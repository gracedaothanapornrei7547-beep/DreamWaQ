#!/bin/bash
set -e

# ==============================================================================
# Unitree Go2 DreamWaQ 16cm Clearance, Calf Contact & Full Symmetry (1000 iters)
# ==============================================================================
# 核心技术设计：
# 1. 接力底模：基于已完全具备 15cm 爬楼与深屈膝肌肉记忆的成熟检查点 model_7594.pt；
# 2. 根治前腿小腿杆触地与全腿画弧外甩：
#    - 足端抬腿清除度 target_clearance 提升至 0.16m（完全高过 15cm 阶高），权重加强至 -0.15；
#    - 小腿杆非预期碰撞 undesired_contacts 惩罚加码至 -0.50，根除小腿支撑/挂阶偷分；
#    - 四肢前后全对称镜像约束 full_hip_swing_symmetry_l2 加码至 -0.15，彻底封死前/后单侧独立外甩；
#    - 摆动相外展重罚 swing_hip_abduction_l2 = -0.50 配合全向动态门控 (|vy| < 0.08 & |wz| < 0.15)；
# 3. 全向台阶指令域持续激活：
#    - ranges.lin_vel_x = (0.40, 0.70) m/s（正向攻坚）
#    - ranges.lin_vel_y = (-0.25, 0.25) m/s（斜向爬阶与侧向跨阶）
#    - ranges.ang_vel_z = (-0.50, 0.50) rad/s（原地回旋与自转）
#    - rel_heading_envs = 0.50（50% 航向锁长跑，50% 自由旋转）
# 4. 攻坚轮次：1000 轮（7594 -> 8594 轮），预计耗时约 45 ~ 50 分钟。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_stairs_straight_corridor/2026-10-11_15-32_waxberry_stairs_swing_hip_knee_finetune_500iter"
BASE_CHECKPOINT="model_7594.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-1000}
RUN_NAME=${3:-waxberry_stairs_clearance_calf_symmetry_finetune_1000iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: 16cm Clearance & Calf Protection (1000 iters)"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Stairs"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (7594 -> 8594)"
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
