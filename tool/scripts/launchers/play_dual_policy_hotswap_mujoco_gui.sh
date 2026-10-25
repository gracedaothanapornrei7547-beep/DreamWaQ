#!/bin/bash
set -e

# ==============================================================================
# One-click Launcher: Dual-Policy Hot-Swap Deployment (4598 Flat + 6642 Stairs)
# 工业级双策略无缝热插拔：平地磁悬浮 Trot + 台阶极速越障
# ==============================================================================

WORKSPACE_DIR="/home/rum/isaac/dreamwaq_lab"
cd "$WORKSPACE_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"

# 1. Flat Policy (4598 平地黄金底模 - 左右对称差 0.2°, 磁悬浮级极稳)
FLAT_MODEL_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-11_19-35_waxberry_flat_straight_symmetry_finetune_300iter/exported"

# 2. Stairs Policy (6642 镜像手术终模 - 彻底拉直内偏, 100% 翻越台阶)
STAIRS_MODEL_DIR="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_surgery_stage6/2026-10-12_18-53_waxberry_stage6_surgery_straight_symmetry_from_decoupled6493_150iter/exported"

ASSETS_DIR="/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"
export DISPLAY=${DISPLAY:-:0}

SCENE="${1:-stairs_5steps}"
VX="${2:-0.5}"

# 根据场景智能设定初始模式（台阶场景默认 [2] 越障攻坚；平地场景默认 [1] 磁悬浮极稳；均可随时按 1/2/Tab 热插拔）
if [[ "$SCENE" =~ "stair" ]]; then
    INIT_POLICY=2
    INIT_DESC="[2] 6642 台阶攻坚 (越障模式, 随时按 1 切平地)"
else
    INIT_POLICY=1
    INIT_DESC="[1] 4598 平地悬浮 (磁悬浮模式, 随时按 2 切台阶)"
fi

echo "=================================================================="
echo "  🚀 Unitree Go2 工业多策略无缝热插拔系统 (Dual-Policy Hot-Swap)"
echo "=================================================================="
echo "  [1] 平地极稳模式 : 4598 黄金底模 (左右对称差 0.2°, 优雅磁悬浮 Trot)"
echo "  [2] 台阶越障模式 : 6642 手术终模 (直腿对中, 100% 连续翻越 5 级台阶)"
echo "  当前默认激活     : $INIT_DESC"
echo "------------------------------------------------------------------"
echo "  🎮 键盘遥控快捷键:"
echo "    [ 1 ]      : 瞬间热切至 [4598 平地悬浮模式]"
echo "    [ 2 ]      : 瞬间热切至 [6642 台阶攻坚模式]"
echo "    [ Tab / T ]: 循环轮换策略 (无缝平滑过渡，电机无冲击)"
echo "    [ W / S ]  : 前进加速 (+0.15 m/s) / 后退减速 (-0.15 m/s)"
echo "    [ A / D ]  : 左横移 (+0.10 m/s) / 右横移 (-0.10 m/s)"
echo "    [ Q / E ]  : 原地左自转 (+0.20 rad/s) / 右自转 (-0.20 rad/s)"
echo "    [Space / X]: 急停刹车 (速度归零)"
echo "    [   R   ]  : 复位机器狗回到原点"
echo "    [ C / V ]  : 切换视角 (自由漫游视角 / 自动跟随视角)"
echo "    [ 鼠标 ]   : 左键 360° 旋转轨道, 右键/滚轮无级缩放"
echo "------------------------------------------------------------------"
echo "  场景地图 : $SCENE"
echo "  初始航速 : $VX m/s"
echo "  显示输出 : $DISPLAY"
echo "=================================================================="

$PYTHON_EXEC deploy/sim2sim/sim2sim_mujoco.py \
    --flat_model_dir "$FLAT_MODEL_DIR" \
    --stairs_model_dir "$STAIRS_MODEL_DIR" \
    --init_policy "$INIT_POLICY" \
    --assets_dir "$ASSETS_DIR" \
    --scene "$SCENE" \
    --vx "$VX" \
    --vy 0.0 \
    --wz 0.0 \
    --duration 0 \
    --teleop \
    --heading_mode \
    "${@:3}"
