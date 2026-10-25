import os, sys, time, torch
from pathlib import Path
import numpy as np

source_root = Path('/home/rum/dreamwaq_lab/source/dreamwaq_lab')
rsl_rl_root = Path('/home/rum/dreamwaq_lab/scripts/rsl_rl')
for p in [str(source_root), str(rsl_rl_root), str(rsl_rl_root / 'dreamwaq')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dreamwaq_actor_critic import register_dreamwaq_actor_critic
from dreamwaq_ppo import register_dreamwaq_ppo
register_dreamwaq_actor_critic()
register_dreamwaq_ppo()

import dreamwaq_lab.tasks
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({'headless': True, 'device': 'cuda:0'})
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.dreamwaq_env_cfg import UnitreeGo2DreamWaQRoughEnvCfg_PLAY
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.agents.rsl_rl_dreamwaq_cfg import UnitreeGo2DreamWaQRoughRunnerCfg

def quat_to_rpy_deg(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll * 180.0 / np.pi, pitch * 180.0 / np.pi, yaw * 180.0 / np.pi

def evaluate_checkpoint(ckpt_path, name="model_4200"):
    print("\n" + "="*80)
    print(f"EVALUATING CHECKPOINT: {name} -> {ckpt_path}")
    print("="*80)
    
    cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    # 严格固定命令为纯前向直线走 0.8m/s，0 侧滑，0 偏航
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    
    env = gym.make('Unitree-Go2-DreamWaQ-Rough-Play', cfg=cfg)
    env = RslRlVecEnvWrapper(env)
    
    runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
    agent_cfg_dict = runner_cfg.to_dict()
    agent_cfg_dict['device'] = 'cuda:0'
    
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
    runner.load(ckpt_path)
    actor_critic = runner.alg.policy
    
    obs, _ = env.reset()
    
    v_true_list, v_hat_list = [], []
    roll_list, pitch_list, yaw_list = [], [], []
    pos_x_list, pos_y_list, pos_z_list = [], [], []
    contact_list = []
    joint_pos_list = []
    action_list = []
    latent_z_list = []
    
    # 运行 300 步 (6 秒物理推演)
    for step in range(300):
        with torch.inference_mode():
            obs_dict = env.unwrapped.observation_manager.compute()
            policy_obs = obs_dict['dreamwaq_policy']
            history_obs = obs_dict['dreamwaq_policy_history']
            
            if hasattr(actor_critic, 'cenet') and actor_critic.cenet is not None:
                cenet_out = actor_critic.cenet(history_obs, sample_latent=False)
                v_hat = cenet_out.estimated_velocity[0].cpu().numpy()
                z = cenet_out.latent[0].cpu().numpy()
            else:
                v_hat = np.zeros(3)
                z = np.zeros(16)
            
            actions = actor_critic.act_inference(obs)
            obs, _, dones, _ = env.step(actions)
            actor_critic.reset(dones)
            
        robot = env.unwrapped.scene['robot']
        v_true = robot.data.root_lin_vel_b[0].cpu().numpy()
        quat = robot.data.root_quat_w[0].cpu().numpy()
        pos = robot.data.root_pos_w[0].cpu().numpy()
        r, p, y = quat_to_rpy_deg(quat)
        j_pos = robot.data.joint_pos[0].cpu().numpy()
        
        c_forces = env.unwrapped.scene.sensors['contact_forces'].data.net_forces_w[0, [4, 8, 12, 16]].cpu().numpy()
        c_norm = np.linalg.norm(c_forces, axis=-1)
        
        v_true_list.append(v_true)
        v_hat_list.append(v_hat)
        roll_list.append(r)
        pitch_list.append(p)
        yaw_list.append(y)
        pos_x_list.append(pos[0])
        pos_y_list.append(pos[1])
        pos_z_list.append(pos[2])
        contact_list.append(c_norm > 5.0)
        joint_pos_list.append(j_pos)
        action_list.append(actions[0].cpu().numpy())
        latent_z_list.append(z)
        
    env.close()
    
    v_true_arr = np.array(v_true_list)
    v_hat_arr = np.array(v_hat_list)
    roll_arr = np.array(roll_list)
    pitch_arr = np.array(pitch_list)
    yaw_arr = np.array(yaw_list)
    pos_x_arr = np.array(pos_x_list)
    pos_y_arr = np.array(pos_y_list)
    contact_arr = np.array(contact_list)
    j_pos_arr = np.array(joint_pos_list)
    act_arr = np.array(action_list)
    latent_z_arr = np.array(latent_z_list)
    
    # 统计数据
    print(f"--- 1. 速度追踪与估计评估 (Command: vx=0.8, vy=0.0, wz=0.0) ---")
    print(f"   物理真值 v_true: vx={np.mean(v_true_arr[:, 0]):.3f}m/s | vy={np.mean(v_true_arr[:, 1]):.3f}m/s | vz={np.mean(v_true_arr[:, 2]):.3f}m/s")
    print(f"   估计值   v_hat : vx={np.mean(v_hat_arr[:, 0]):.3f}m/s | vy={np.mean(v_hat_arr[:, 1]):.3f}m/s | vz={np.mean(v_hat_arr[:, 2]):.3f}m/s")
    v_err = v_hat_arr - v_true_arr
    print(f"   估计残差 v_err : vx_bias={np.mean(v_err[:, 0]):.3f}m/s (RMSE={np.sqrt(np.mean(v_err[:, 0]**2)):.3f}) | vy_bias={np.mean(v_err[:, 1]):.3f}m/s | vz_bias={np.mean(v_err[:, 2]):.3f}m/s")
    
    print(f"\n--- 2. 空间轨迹偏航与横向偏移 (Drift Metrics) ---")
    start_pos = np.array([pos_x_arr[0], pos_y_arr[0]])
    end_pos = np.array([pos_x_arr[-1], pos_y_arr[-1]])
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    print(f"   起点 -> 终点: dx={dx:+.3f}m, dy={dy:+.3f}m (横向净漂移量!)")
    drift_angle = np.arctan2(dy, max(dx, 1e-5)) * 180 / np.pi
    print(f"   空间轨迹偏移角: {drift_angle:+.2f}° (直线走航偏斜角)")
    print(f"   Yaw 航向角变化: 起始={yaw_arr[0]:.2f}° -> 终点={yaw_arr[-1]:.2f}° | 净偏转={yaw_arr[-1]-yaw_arr[0]:+.2f}° | 最大绝对值={np.max(np.abs(yaw_arr)):.2f}°")
    print(f"   Roll 横滚振荡 : Mean={np.mean(roll_arr):.2f}° | Std={np.std(roll_arr):.2f}° | P2P={np.ptp(roll_arr):.2f}°")
    print(f"   Pitch 俯仰振荡: Mean={np.mean(pitch_arr):.2f}° | Std={np.std(pitch_arr):.2f}° | P2P={np.ptp(pitch_arr):.2f}°")

    print(f"\n--- 3. 四肢步态接触与相位 (Gait Phase & Symmetry) ---")
    duty = np.mean(contact_arr, axis=0)
    print(f"   触地占空比: FL={duty[0]*100:.1f}%, FR={duty[1]*100:.1f}%, RL={duty[2]*100:.1f}%, RR={duty[3]*100:.1f}%")
    diag1 = (duty[0] + duty[3]) / 2  # FL + RR
    diag2 = (duty[1] + duty[2]) / 2  # FR + RL
    print(f"   对角组 1 (FL+RR): {diag1*100:.1f}% | 对角组 2 (FR+RL): {diag2*100:.1f}% | 对角对称差: {abs(diag1-diag2)*100:.1f}%")
    print(f"   左右前肢差: {abs(duty[0]-duty[1])*100:.1f}% | 左右后肢差: {abs(duty[2]-duty[3])*100:.1f}%")

    print(f"\n--- 4. 关节平均角度与动作输出 (左右对称性核查) ---")
    hip_names = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip']
    hip_idx = [0, 3, 6, 9]
    for n, i in zip(hip_names, hip_idx):
        print(f"   {n:8s}: Pos={np.mean(j_pos_arr[:, i])*180/np.pi:+.2f}° | Act_mean={np.mean(act_arr[:, i]):+.3f} | Act_std={np.std(act_arr[:, i]):.3f}")

    print(f"\n--- 5. CENet 隐空间 16维激活能量 ---")
    z_std = np.std(latent_z_arr, axis=0)
    print(f"   z 各维标准差: {[round(float(x), 3) for x in z_std]}")
    print(f"   z 平均绝对值: {float(np.mean(np.abs(latent_z_arr))):.3f}")

if __name__ == '__main__':
    ckpt_4200 = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-09_18-21_finetune_rough_800iter_velocity_patch/model_4200.pt'
    ckpt_3800 = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-09_18-21_finetune_rough_800iter_velocity_patch/model_3800.pt'
    evaluate_checkpoint(ckpt_4200, "model_4200 (最新1200轮)")
    evaluate_checkpoint(ckpt_3800, "model_3800 (目标800轮)")
    simulation_app.close()
