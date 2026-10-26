# Unitree Go2 DreamWaQ Phase 2 (P2) 黄金基准存档文档

> **存档绝对路径**: `/home/rum/isaac/dreamwaq_lab/logs/DreamWaQ_P2`  
> **更新时间**: 2026-10-14  
> **状态**: 黄金基准已固化（已彻底替换淘汰早期存在偏航与卡顿的废弃版本）

---

## 1. 核心目录拓扑

```text
logs/DreamWaQ_P2/
├── 1_flat_purify/          # [Stage 13] 纯平地镜像等变洗髓模型 (model_7189.pt)
│   ├── model_7189.pt       # 偏航角偏差骤降 72.2% (20.35° -> 5.66°) 的对称洗直底模
│   └── exported/           # 导出的 TorchScript JIT、ONNX 及部署包
├── 2_unified_stairs/       # [Stage 14] 大一统黄金基准模型 (model_7488.pt)
│   ├── model_7488.pt       # 15cm 极限台阶 100% 登顶、登顶侧偏仅 1mm (+0.001m)、0 卡顿
│   └── exported/           # 最终部署端完整 ONNX / JIT 与 deploy.yaml
├── exported -> 2_unified_stairs/exported # 快捷软链接至最新大一统导出资产
├── 1_straight -> 1_flat_purify          # 兼容软链接
├── 2_stair -> 2_unified_stairs          # 兼容软链接
├── run_mujoco.sh           # 一键拉起 MuJoCo 3D 渲染与键盘遥控
└── README.md               # 本存档说明文档
```

---

## 2. 核心战果与关键性能指标

| 评估维度 | 早期废弃基线 (旧 P2) | **本次全新同步归档 (新 DreamWaQ_P2)** | 改善幅度与物理学结论 |
| :--- | :--- | :--- | :--- |
| **直行偏航漂移** | 频繁向左/右摆动偏航 ($> 0.8\text{m}$) | **闭环直行漂移仅 $-0.159\text{m}$，平均偏航仅 $0.95^\circ$** | **漂移缩减 80%+**，激光级笔直 |
| **15cm 台阶攀爬** | 第一级台阶频繁卡死、停顿颠簸 | **0.3s 前瞻门控提膝，100% 登顶 5 阶 15cm 楼梯** | **彻底粉碎第一阶卡顿** |
| **台阶登顶侧偏** | 登顶侧偏 $> 0.5\text{m}$ (极易侧翻) | **登顶侧偏仅 $+0.001\text{m}$（绝对居中 1 毫米）** | **侧漂近乎清零** |
| **极限巡航速度** | 超过 $0.4\text{ m/s}$ 容易失稳 | **$1.20\text{ m/s}$ 满油门极速冲穿 4.5 米天台** | 超高力矩爆发与动态鲁棒性 |
| **网络架构状态** | 225 维时序本体感知 | **225 维时序本体感知（0 架构变动）** | 100% 纯净 LibTorch/ONNX 部署 |

---

## 3. 一键可视化与遥测指令

### ① 15cm 台阶场景交互遥控 (MuJoCo 3D)
```bash
/home/rum/dreamwaq_lab/logs/DreamWaQ_P2/run_mujoco.sh stairs_5steps_15cm 0.6
```

### ② 纯平地直线高速跑道遥测
```bash
/home/rum/dreamwaq_lab/logs/DreamWaQ_P2/run_mujoco.sh flat 0.8
```

