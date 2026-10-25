# DreamWaQ Lab 自研工具箱与历史资产库 (Toolbox)

本目录汇总了本项目历次研发、消融、度量与验证过程中编写的非官方工具脚本与历史日志，保持仓库根目录与官方 `scripts/` 的极简整洁。

---

## 目录结构索引

```
tool/
├── logs/                 # 历史各阶段训练、微调与推演日志 (*.log)
├── scripts/
│   ├── benchmarks/       # 速度扫频、抗漂移横向评测脚本 (evaluate_*.py, test_*.py)
│   ├── captures/         # 各阶段 3D 视口自动截图工具 (capture_stage*.py)
│   ├── diagnostics/      # 动力学探针、小跑对称性分析、功耗度量工具 (diagnose_*.py, deep_*.py)
│   ├── gui_players/      # 各阶段历史可视化推演脚本 (play_*_gui.py)
│   └── launchers/        # 训练启动与 TensorBoard 自动化脚本 (*.sh)
└── sim2sim/              # 原生 C++ MuJoCo Sim2Sim 动力学跨引擎评测套件
```

---

## 一、 快速使用指南

### 1. 运行原生 C++ Sim2Sim 跨仿真器验证
```bash
/home/rum/dreamwaq_lab/tool/sim2sim/sim2sim_go2_cpp \
  --model /home/rum/isaac/dreamwaq_lab/logs/DreamQaW_P2/1_straight/exported/policy.onnx \
  --scene /home/rum/himloco_assets/mujoco_menagerie/unitree_go2/scene.xml \
  --duration 10.0 --vx 0.8
```

### 2. 运行 Python 版 MuJoCo 实况视窗
```bash
bash /home/rum/isaac/dreamwaq_lab/tool/scripts/launchers/play_sim2sim_mujoco.sh \
  /home/rum/isaac/dreamwaq_lab/logs/DreamQaW_P2/1_straight/exported flat
```

### 3. 查看历史阶段训练日志
```bash
tail -f /home/rum/dreamwaq_lab/tool/logs/train_waxberry_flat_finetune_antidrift_500iter.log
```

