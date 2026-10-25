#!/usr/bin/env python3
"""
量化提取四足机器人四腿运动学与动力学非对称指标 (Gait & Leg Asymmetry Analyzer)
对比：
1. Stage 6 镜像手术模型 (6642)
2. 平地黄金底模 (4598)
3. 原始爬楼梯基线 (5596)
"""

import os
import sys
import numpy as np
import mujoco
import onnxruntime as ort

ASSETS_DIR = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"
SCENE_FLAT = os.path.join(ASSETS_DIR, "scene.xml")
SCENE_STAIRS = os.path.join(ASSETS_DIR, "scene_stairs_5steps.xml")
SCENE_STAIRS_15CM = os.path.join(ASSETS_DIR, "scene_stairs_5steps_15cm.xml")

MODELS = {
    "4598_flat_gold": "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-11_19-35_waxberry_flat_straight_symmetry_finetune_300iter/exported",
    "6842_stage7_equivariant": "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_equivariant_stage7/2026-10-12_19-45_waxberry_stage7_equivariant_allinone_from_surgery6642_200iter/exported",
    "6990_stage8_extreme15cm": "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_extreme15cm_stage8/2026-10-12_20-18_waxberry_stage8_extreme15cm_kneelift_from_equivariant6841_150iter/exported",
    "6990_stage9_smooth": "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_smooth_stage9/2026-10-12_20-45_waxberry_stage9_smooth_straight_from_equivariant6841_150iter/exported",
    "6990_stage10_optimal": "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_optimal_stage10/2026-10-12_21-33_waxberry_stage10_optimal_zero_arch_from_equivariant6841_150iter/exported",
}

# 引入 sim2sim 基础类
sys.path.insert(0, "/home/rum/dreamwaq_lab")
from deploy.sim2sim.sim2sim_mujoco import DreamWaQMuJoCoSim2Sim

def evaluate_gait(model_dir, scene_path, vx=0.5, num_steps=250, init_pos=None):
    if not os.path.exists(model_dir):
        return None
    env = DreamWaQMuJoCoSim2Sim(
        scene_path=scene_path,
        model_dir=model_dir,
        kp=28.0,
        kd=0.75,
        filter_alpha=0.6,
    )
    if init_pos is not None:
        env.reset(init_pos=init_pos)

    cmd = np.array([vx, 0.0, 0.0], dtype=np.float32)

    # 关节历史：[step, 12] (Isaac 拓扑: FL, FR, RL, RR 各 3 个关节: hip, thigh, calf)
    joint_pos_hist = []
    torque_hist = []
    contacts_hist = []
    pos_hist = []

    # 预热 20 步
    for _ in range(20):
        env.step(cmd)

    for _ in range(num_steps):
        info = env.step(cmd)
        q_isaac = info["actual_pos_isaac"]
        joint_pos_hist.append(q_isaac)
        pos_hist.append(env.data.qpos[0:3].copy())
        
        # contacts: FL, FR, RL, RR
        c = [1.0 if info["contacts"][k] else 0.0 for k in ["FL", "FR", "RL", "RR"]]
        contacts_hist.append(c)

        # torques
        tau = env.data.ctrl.copy()
        tau_isaac = tau[env.mujoco_to_isaac]
        torque_hist.append(tau_isaac)

    joint_pos_hist = np.array(joint_pos_hist) # [N, 12]
    contacts_hist = np.array(contacts_hist)   # [N, 4]
    torque_hist = np.array(torque_hist)       # [N, 12]

    # 关节索引映射:
    # FL: 0:hip, 1:thigh, 2:calf
    # FR: 3:hip, 4:thigh, 5:calf
    # RL: 6:hip, 7:thigh, 8:calf
    # RR: 9:hip, 10:thigh, 11:calf

    # 1. 均值关节位置 (度)
    deg = np.rad2deg(joint_pos_hist)
    mean_deg = np.mean(deg, axis=0)
    std_deg = np.std(deg, axis=0)

    # 2. 左右镜像对称差 (绝对偏差)
    # Hip: 理想左右对称应为 FL_hip = - FR_hip (即 FL_hip + FR_hip = 0)
    # Thigh: 理想左右对称应为 FL_thigh = FR_thigh (即 FL_thigh - FR_thigh = 0)
    # Calf: 理想左右对称应为 FL_calf = FR_calf (即 FL_calf - FR_calf = 0)
    front_hip_asymm = np.abs(np.mean(deg[:, 0] + deg[:, 3]))
    front_thigh_asymm = np.abs(np.mean(deg[:, 1] - deg[:, 4]))
    front_calf_asymm = np.abs(np.mean(deg[:, 2] - deg[:, 5]))

    rear_hip_asymm = np.abs(np.mean(deg[:, 6] + deg[:, 9]))
    rear_thigh_asymm = np.abs(np.mean(deg[:, 7] - deg[:, 10]))
    rear_calf_asymm = np.abs(np.mean(deg[:, 8] - deg[:, 11]))

    # 3. 前后肢异构差 (Front vs Rear)
    front_rear_thigh_diff = np.abs(np.mean(deg[:, [1, 4]]) - np.mean(deg[:, [7, 10]]))
    front_rear_calf_diff = np.abs(np.mean(deg[:, [2, 5]]) - np.mean(deg[:, [8, 11]]))

    # 4. 触地占空比 (Duty cycle) & 对角协调度
    # FL, FR, RL, RR
    duty = np.mean(contacts_hist, axis=0)
    # 对角线步态协调: FL+RR (对角对1) vs FR+RL (对角对2)
    diag1 = (contacts_hist[:, 0] == contacts_hist[:, 3]).mean() # FL 与 RR 同相度
    diag2 = (contacts_hist[:, 1] == contacts_hist[:, 2]).mean() # FR 与 RL 同相度

    # 5. 各腿平均功耗与力矩 (RMS Torque)
    rms_tau = np.sqrt(np.mean(torque_hist**2, axis=0))
    # 每腿功耗 (hip+thigh+calf)
    leg_tau = [
        rms_tau[0:3].mean(), # FL
        rms_tau[3:6].mean(), # FR
        rms_tau[6:9].mean(), # RL
        rms_tau[9:12].mean(), # RR
    ]

    return {
        "mean_deg": mean_deg,
        "std_deg": std_deg,
        "front_hip_asymm": front_hip_asymm,
        "front_thigh_asymm": front_thigh_asymm,
        "front_calf_asymm": front_calf_asymm,
        "rear_hip_asymm": rear_hip_asymm,
        "rear_thigh_asymm": rear_thigh_asymm,
        "rear_calf_asymm": rear_calf_asymm,
        "front_rear_thigh_diff": front_rear_thigh_diff,
        "front_rear_calf_diff": front_rear_calf_diff,
        "duty": duty,
        "diag1": diag1,
        "diag2": diag2,
        "leg_tau": leg_tau,
        "final_pos": pos_hist[-1] if len(pos_hist) > 0 else np.zeros(3),
    }

print("="*75)
print("     四足机器人四腿对称性与动力学深度量化采样")
print("="*75)

for name, m_dir in MODELS.items():
    if not os.path.exists(m_dir):
        continue
    res_flat = evaluate_gait(m_dir, SCENE_FLAT, vx=0.5, num_steps=250, init_pos=(0.0, 0.0, 0.35))
    res_stairs = evaluate_gait(m_dir, SCENE_STAIRS, vx=0.6, num_steps=350, init_pos=(1.0, 0.0, 0.35))
    res_stairs_15cm = evaluate_gait(m_dir, SCENE_STAIRS_15CM, vx=0.6, num_steps=400, init_pos=(1.0, 0.0, 0.35))
    
    print(f"\n>>> 模型: [{name}]")
    if res_flat:
        print("  【平地工况 Flat 0.5m/s (x=0.0 -> 平地 250步)】:")
        print(f"    - 前肢左右对称差: 髋={res_flat['front_hip_asymm']:.2f}°, 大腿={res_flat['front_thigh_asymm']:.2f}°, 小腿={res_flat['front_calf_asymm']:.2f}°")
        print(f"    - 后肢左右对称差: 髋={res_flat['rear_hip_asymm']:.2f}°, 大腿={res_flat['rear_thigh_asymm']:.2f}°, 小腿={res_flat['rear_calf_asymm']:.2f}°")
        print(f"    - 前后肢异构差: 大腿={res_flat['front_rear_thigh_diff']:.2f}°, 小腿={res_flat['front_rear_calf_diff']:.2f}°")
        print(f"    - 四腿触地占空比 [FL, FR, RL, RR]: [{res_flat['duty'][0]:.2f}, {res_flat['duty'][1]:.2f}, {res_flat['duty'][2]:.2f}, {res_flat['duty'][3]:.2f}]")
        print(f"    - 对角 Trot 步态同相度: 对角1(FL+RR)={res_flat['diag1']*100:.1f}%, 对角2(FR+RL)={res_flat['diag2']*100:.1f}%")
        print(f"    - 各腿平均力矩 (Nm) [FL, FR, RL, RR]: [{res_flat['leg_tau'][0]:.2f}, {res_flat['leg_tau'][1]:.2f}, {res_flat['leg_tau'][2]:.2f}, {res_flat['leg_tau'][3]:.2f}]")
        print(f"    - 各关节均值(deg): FL_hip={res_flat['mean_deg'][0]:.2f}°, FR_hip={res_flat['mean_deg'][3]:.2f}°, RL_hip={res_flat['mean_deg'][6]:.2f}°, RR_hip={res_flat['mean_deg'][9]:.2f}°")
        print(f"    - 平地行进位移: x={res_flat['final_pos'][0]:.2f}m, y_drift={res_flat['final_pos'][1]:.3f}m, z={res_flat['final_pos'][2]:.3f}m")

    if res_stairs:
        stairs_passed = "✅ 翻越成功 (登顶高台)" if res_stairs['final_pos'][0] > 3.0 and res_stairs['final_pos'][2] > 0.70 else ("⚠️ 翻越至台阶中段" if res_stairs['final_pos'][0] > 1.8 else "❌ 翻越失败/卡死")
        print(f"  【10cm 标准台阶 0.6m/s (x=1.0 -> 翻越5级台阶 350步)】: {stairs_passed}")
        print(f"    - 终点坐标: x={res_stairs['final_pos'][0]:.2f}m, y_drift={res_stairs['final_pos'][1]:.3f}m, z={res_stairs['final_pos'][2]:.3f}m")
        print(f"    - 前肢左右对称差: 髋={res_stairs['front_hip_asymm']:.2f}°, 大腿={res_stairs['front_thigh_asymm']:.2f}°, 小腿={res_stairs['front_calf_asymm']:.2f}°")
        print(f"    - 各腿平均力矩 (Nm) [FL, FR, RL, RR]: [{res_stairs['leg_tau'][0]:.2f}, {res_stairs['leg_tau'][1]:.2f}, {res_stairs['leg_tau'][2]:.2f}, {res_stairs['leg_tau'][3]:.2f}]")

    if res_stairs_15cm:
        stairs15_passed = "✅ 翻越成功 (登顶0.75m高台)" if res_stairs_15cm['final_pos'][0] > 3.0 and res_stairs_15cm['final_pos'][2] > 0.90 else ("⚠️ 翻越至台阶中段" if res_stairs_15cm['final_pos'][0] > 1.8 else "❌ 翻越失败/卡死")
        print(f"  【15cm 极限台阶 0.6m/s (x=1.0 -> 翻越5级高台阶 400步)】: {stairs15_passed}")
        print(f"    - 终点坐标: x={res_stairs_15cm['final_pos'][0]:.2f}m, y_drift={res_stairs_15cm['final_pos'][1]:.3f}m, z={res_stairs_15cm['final_pos'][2]:.3f}m")
        print(f"    - 前肢左右对称差: 髋={res_stairs_15cm['front_hip_asymm']:.2f}°, 大腿={res_stairs_15cm['front_thigh_asymm']:.2f}°, 小腿={res_stairs_15cm['front_calf_asymm']:.2f}°")
        print(f"    - 后肢左右对称差: 髋={res_stairs_15cm['rear_hip_asymm']:.2f}°, 大腿={res_stairs_15cm['rear_thigh_asymm']:.2f}°, 小腿={res_stairs_15cm['rear_calf_asymm']:.2f}°")
        print(f"    - 各腿平均力矩 (Nm) [FL, FR, RL, RR]: [{res_stairs_15cm['leg_tau'][0]:.2f}, {res_stairs_15cm['leg_tau'][1]:.2f}, {res_stairs_15cm['leg_tau'][2]:.2f}, {res_stairs_15cm['leg_tau'][3]:.2f}]")

