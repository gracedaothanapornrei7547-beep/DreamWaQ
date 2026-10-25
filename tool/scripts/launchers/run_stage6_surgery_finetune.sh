#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 6: Unified Symmetry Surgery & Equivariance Lock Fine-tuning
# ==============================================================================
# 核心技术目标：
# 1. 继承底模：Stage 5 解耦俯仰与双限幅净空终模 (model_6493.pt)；
# 2. 注入全域等变镜像对称硬锁：bilateral_hip_symmetry_equivariance_l2 (-0.50)
#    - 彻底剔除 min=0.0 单边截断后门，无论是外甩还是内扣，左右不对称一律严惩；
# 3. 注入前腿髋关节双向中立零位锁：front_hip_neutral_pos_l2 (-0.80)
#    - 强制将左前髋 FL 偏离角从 -9.3° (内偏 15°) 强行拉直回 0° ~ 5.7° 黄金支撑角度；
# 4. 激活四肢功率均衡方差惩罚：power_distribution_variance_l2 (-1.0e-5)
#    - 对标 agent.md 与 DreamWaQ.md 规范，瓦解单肢内收借力与非对称负荷偷懒；
# 5. 保留 Stage 5 全部成果：
#    - 自适应平地俯仰刚性锁 (-0.40) + 双向限幅足端净空 (平地 <=28mm 严惩，台阶 0.16m 释放)；
#    - 足端软着陆阻尼与动作二阶平滑；
#    - 40% 平地长廊 + 60% 直线阶梯自适应混合地形。
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_decoupled_stage5/2026-10-12_16-58_waxberry_stage5_decoupled_flat_pitch_lock_from_softlanding6194_300iter"
BASE_CHECKPOINT="model_6493.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-150}
RUN_NAME=${3:-waxberry_stage6_surgery_straight_symmetry_from_decoupled6493_150iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 6 Symmetry Surgery Fine-tuning"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Surgery"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6493 -> 6643)"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Surgery \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"

