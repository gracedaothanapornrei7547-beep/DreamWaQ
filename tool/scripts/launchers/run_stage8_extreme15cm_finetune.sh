#!/bin/bash
set -e

# ==============================================================================
# WaxberryS Stage 8: Extreme 15cm Stairs Knee-Lift Fine-tuning & All-in-One
# 方案：15cm 极限阶高专项提膝加训 (150 轮, 6841 -> 6991)
# ==============================================================================
# 核心技术目标：
# 1. 继承底模：Stage 7 镜像等变终模 (model_6841.pt，左右等变、四腿平衡)；
# 2. 阶梯课程锁定在 10cm ~ 16cm 极限高度区间：
#    MeshStraightAscendingStairsTerrainCfg(step_height_range=(0.10, 0.16))
# 3. 越障足端提膝拔高与平地磁悬浮解耦：
#    - 引入高程扫描仪局部落差前瞻感知 (span > 4cm 即提前激活越障提膝态)
#    - 越障态足端净空目标拔高至 0.15m (悬空整整 15cm，大开大合跨步)
#    - 平地超额严惩维持不变，平地保持 2.2cm 磁悬浮滑行
# 4. 严惩磕阶：undesired_contacts 加码至 -0.40 (严惩小腿与台阶蹭碰)
# 5. 训练规模：4096 并行环境，150 轮高速对齐，实现单模型平地滑行+15cm 极限高台阶翻越！
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
BASE_CHECKPOINT_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_equivariant_stage7/2026-10-12_19-45_waxberry_stage7_equivariant_allinone_from_surgery6642_200iter"
BASE_CHECKPOINT="model_6841.pt"
NUM_ENVS=${1:-4096}
MAX_ITERS=${2:-150}
RUN_NAME=${3:-waxberry_stage8_extreme15cm_kneelift_from_equivariant6841_150iter}

echo "=========================================================="
echo "  Unitree Go2 DreamWaQ: Stage 8 Extreme 15cm Stairs Knee-Lift"
echo "  Workspace      : $WORKSPACE_DIR"
echo "  Task           : Unitree-Go2-DreamWaQ-Extreme15cm"
echo "  Base Checkpoint: $BASE_CHECKPOINT_DIR/$BASE_CHECKPOINT"
echo "  Environments   : $NUM_ENVS"
echo "  Iterations     : $MAX_ITERS (6841 -> 6991)"
echo "  Run Name       : $RUN_NAME"
echo "=========================================================="

$PYTHON_EXEC scripts/rsl_rl/dreamwaq/train_dreamwaq.py \
    --task Unitree-Go2-DreamWaQ-Extreme15cm \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --resume \
    --load_run "$BASE_CHECKPOINT_DIR" \
    --checkpoint "$BASE_CHECKPOINT" \
    --run_name "$RUN_NAME"

