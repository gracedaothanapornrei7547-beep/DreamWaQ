import os
import sys
import time
from pathlib import Path
import numpy as np

# 导入 sim2sim 模块
repo_root = Path("/home/rum/dreamwaq_lab")
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "deploy" / "sim2sim"))

from sim2sim_mujoco import DreamWaQMuJoCoSim2Sim, compute_adaptive_heading_cmd, quat_to_euler_rpy

STAGE11_DIR = "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_gold_stage11/2026-10-13_16-43_waxberry_stage11_gold_from_optimal6990_500iter/exported"
STAGE12_DIR = "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_gold_stage11/2026-10-14_12-06_waxberry_stage12_flex_loose_from_optimal6990_300iter/exported"
STAGE13_DIR = "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat_purify_stage13/2026-10-14_13-06_waxberry_stage13_flat_purify_from_optimal6990_200iter/exported"
STAGE14_DIR = "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_unified_stage14/2026-10-14_13-19_waxberry_stage14_unified_from_flatpurify7189_300iter/exported"
ASSETS_DIR = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"

scene_map = {
    "flat": "scene.xml",
    "stairs": "scene_stairs_5steps.xml",
    "stairs_10cm": "scene_stairs_10cm.xml",
    "stairs_5steps": "scene_stairs_5steps.xml",
    "stairs_5steps_5cm": "scene_stairs_5steps_5cm.xml",
    "stairs_5steps_15cm": "scene_stairs_5steps_15cm.xml",
}

def evaluate_policy(model_dir, scene="flat", vx=0.6, duration=10.0, use_heading_control=False):
    scene_file = scene_map.get(scene, "scene.xml")
    scene_path = os.path.join(ASSETS_DIR, scene_file)
    
    env = DreamWaQMuJoCoSim2Sim(
        scene_path=scene_path,
        model_dir=model_dir,
        kp=28.0,
        kd=0.75,
        sim_dt=0.002,
        decimation=10,
        filter_alpha=0.6,
    )
    
    num_steps = int(duration / env.policy_dt)
    target_yaw_rad = 0.0
    cmd = np.array([vx, 0.0, 0.0], dtype=np.float32)
    
    xs, ys, zs, yaws = [], [], [], []
    hip_devs = [] # 记录四个髋关节的角度
    
    for step_idx in range(num_steps):
        if use_heading_control:
            cmd[2] = compute_adaptive_heading_cmd(
                env,
                target_yaw_rad=target_yaw_rad,
                kp=2.0,
                kd=0.12,
                track_k=0.0,
                scene=scene,
            )
        else:
            cmd[2] = 0.0
            
        info = env.step(cmd)
        
        pos = info["pos"]
        rpy = info["rpy"]
        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])
        yaws.append(rpy[2])
        # 记录 4 个髋关节位置
        qpos_hip = env.data.qpos[7:19:3]
        hip_devs.append(qpos_hip.copy())
        
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    yaws = np.array(yaws)
    hip_devs = np.array(hip_devs)
    
    return {
        "final_pos": (xs[-1], ys[-1], zs[-1]),
        "max_abs_y": np.max(np.abs(ys)),
        "final_y": ys[-1],
        "mean_abs_yaw": np.mean(np.abs(yaws)),
        "max_abs_yaw": np.max(np.abs(yaws)),
        "max_z": np.max(zs),
        "hip_dev_mean": np.mean(np.abs(hip_devs), axis=0),
        "total_distance": xs[-1] - xs[0],
    }

def main():
    print("================================================================================")
    print("  Stage 11 (刚性锁死) vs Stage 12 (柔性松绑) vs Stage 13 (纯平地洗髓200轮)")
    print("================================================================================\n")
    
    models = [
        ("Stage 11 (原刚性锁死)", STAGE11_DIR),
        ("Stage 12 (路线 B 柔性松绑流)", STAGE12_DIR),
        ("Stage 13 (纯平地洗髓200轮)", STAGE13_DIR),
        ("Stage 14 (大一统融合再适应300轮)", STAGE14_DIR),
    ]
    
    for name, m_dir in models:
        print(f"--- 正在评测: {name} ---")
        
        # 1. 平地原生直行 (无外挂航向锁定，纯测模型自发偏摆趋势)
        flat_res = evaluate_policy(m_dir, scene="flat", vx=0.6, duration=10.0, use_heading_control=False)
        print(f"  [平地原生直行 10s]")
        print(f"    - 净前进距离: {flat_res['total_distance']:.2f} m")
        print(f"    - 终点横向偏差 Y_final: {flat_res['final_y']:+.3f} m (偏摆特征: {'明显向右偏摆' if flat_res['final_y'] < -0.08 else '明显向左偏摆' if flat_res['final_y'] > 0.08 else '激光级平直居中'})")
        print(f"    - 最大横向摆幅 |Y|_max: {flat_res['max_abs_y']:.3f} m")
        print(f"    - 平均偏航偏差 |Yaw|_mean: {flat_res['mean_abs_yaw']:.2f}° (最大偏差: {flat_res['max_abs_yaw']:.2f}°)")
        print(f"    - 四髋角活动均值 [FL, FR, RL, RR]: {np.round(flat_res['hip_dev_mean'], 3)} rad")
        
        # 2. 15cm 台阶攀爬 (无外挂辅助)
        stair_res = evaluate_policy(m_dir, scene="stairs_5steps_15cm", vx=0.6, duration=15.0, use_heading_control=False)
        print(f"  [15cm 极限台阶攀爬 15s]")
        print(f"    - 最终高度 Z_final: {stair_res['final_pos'][2]:.2f} m / 最大到达高度 Z_max: {stair_res['max_z']:.2f} m")
        print(f"    - 攀爬结果: {'登顶成功 (越过 5 阶 15cm 台阶)' if stair_res['max_z'] > 0.80 else '第一阶严重卡死/停顿'}")
        print()

if __name__ == "__main__":
    main()

