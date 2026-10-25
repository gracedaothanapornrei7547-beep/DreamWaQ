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

cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
cfg.scene.num_envs = 1
cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

env = gym.make('Unitree-Go2-DreamWaQ-Rough-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

checkpoint_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-09_12-13_finetune_300iter_antidrift_stabilized/model_3298.pt'
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(checkpoint_path)
actor_critic = runner.alg.policy

obs, _ = env.reset()

v_true_list = []
v_hat_list = []
roll_list, pitch_list, yaw_list = [], [], []
contact_list = []
joint_pos_list = []
action_list = []
latent_z_list = []

for step in range(200):
    with torch.inference_mode():
        # 提取 policy 和 history 观测
        obs_dict = env.unwrapped.observation_manager.compute()
        policy_obs = obs_dict['dreamwaq_policy']
        history_obs = obs_dict['dreamwaq_policy_history']
        
        # 通过 CENet 计算 v_hat 和 z
        if hasattr(actor_critic, 'cenet') and actor_critic.cenet is not None:
            cenet_out = actor_critic.cenet(history_obs, sample_latent=False)
            v_hat = cenet_out.estimated_velocity[0].cpu().numpy()
            z = cenet_out.latent[0].cpu().numpy()
        else:
            v_hat = np.zeros(3)
            z = np.zeros(16)
        
        # 执行动作
        actions = actor_critic.act_inference(obs)
        obs, _, dones, _ = env.step(actions)
        actor_critic.reset(dones)

    robot = env.unwrapped.scene['robot']
    v_true = robot.data.root_lin_vel_b[0].cpu().numpy()
    quat = robot.data.root_quat_w[0].cpu().numpy()
    r, p, y = quat_to_rpy_deg(quat)
    j_pos = robot.data.joint_pos[0].cpu().numpy()
    
    # 足端接触力
    c_forces = env.unwrapped.scene.sensors['contact_forces'].data.net_forces_w[0, [4, 8, 12, 16]].cpu().numpy()
    c_norm = np.linalg.norm(c_forces, axis=-1)
    
    v_true_list.append(v_true)
    v_hat_list.append(v_hat)
    roll_list.append(r)
    pitch_list.append(p)
    yaw_list.append(y)
    contact_list.append(c_norm > 5.0)
    joint_pos_list.append(j_pos)
    action_list.append(actions[0].cpu().numpy())
    latent_z_list.append(z)

v_true_arr = np.array(v_true_list)
v_hat_arr = np.array(v_hat_list)
roll_arr = np.array(roll_list)
pitch_arr = np.array(pitch_list)
yaw_arr = np.array(yaw_list)
contact_arr = np.array(contact_list)
j_pos_arr = np.array(joint_pos_list)
act_arr = np.array(action_list)
latent_z_arr = np.array(latent_z_list)

print("="*80)
print("=== DWAQ CENET & DYNAMICS DEEP DIAGNOSTIC REPORT ===")
print("="*80)
print(f"1. 速度估计误差 (CENet v_hat vs v_true):")
v_err = v_hat_arr - v_true_arr
print(f"   v_x: True={np.mean(v_true_arr[:, 0]):.3f}m/s | Hat={np.mean(v_hat_arr[:, 0]):.3f}m/s | Bias={np.mean(v_err[:, 0]):.3f}m/s | RMSE={np.sqrt(np.mean(v_err[:, 0]**2)):.3f}")
print(f"   v_y: True={np.mean(v_true_arr[:, 1]):.3f}m/s | Hat={np.mean(v_hat_arr[:, 1]):.3f}m/s | Bias={np.mean(v_err[:, 1]):.3f}m/s | RMSE={np.sqrt(np.mean(v_err[:, 1]**2)):.3f}")
print(f"   v_z: True={np.mean(v_true_arr[:, 2]):.3f}m/s | Hat={np.mean(v_hat_arr[:, 2]):.3f}m/s | Bias={np.mean(v_err[:, 2]):.3f}m/s | RMSE={np.sqrt(np.mean(v_err[:, 2]**2)):.3f}")

print(f"\n2. 姿态振荡特性 (Posture Oscillations):")
print(f"   Roll  (横滚): Mean={np.mean(roll_arr):.2f}° | Std={np.std(roll_arr):.2f}° | P2P={np.ptp(roll_arr):.2f}° | Min={np.min(roll_arr):.2f}°, Max={np.max(roll_arr):.2f}°")
print(f"   Pitch (俯仰): Mean={np.mean(pitch_arr):.2f}° | Std={np.std(pitch_arr):.2f}° | P2P={np.ptp(pitch_arr):.2f}° | Min={np.min(pitch_arr):.2f}°, Max={np.max(pitch_arr):.2f}°")
print(f"   Yaw   (偏航): Start={yaw_arr[0]:.2f}° -> End={yaw_arr[-1]:.2f}° | Net Drift={yaw_arr[-1]-yaw_arr[0]:+.2f}° | Max Absolute={np.max(np.abs(yaw_arr)):.2f}°")

print(f"\n3. 步态接触对称性 (Gait Duty Cycle & Contact Time):")
duty = np.mean(contact_arr, axis=0)
print(f"   FL (前左): {duty[0]*100:.1f}% | FR (前右): {duty[1]*100:.1f}%")
print(f"   RL (后左): {duty[2]*100:.1f}% | RR (后右): {duty[3]*100:.1f}%")
print(f"   前肢左右差: {abs(duty[0]-duty[1])*100:.1f}% | 后肢左右差: {abs(duty[2]-duty[3])*100:.1f}%")

print(f"\n4. 关节动作输出统计 (Joint Actions Mean & Std):")
for i, name in enumerate(['FL_hip', 'FL_thigh', 'FL_calf', 'FR_hip', 'FR_thigh', 'FR_calf', 'RL_hip', 'RL_thigh', 'RL_calf', 'RR_hip', 'RR_thigh', 'RR_calf']):
    print(f"   {name:10s}: Pos={np.mean(j_pos_arr[:, i])*180/np.pi:+.2f}° | Act_mean={np.mean(act_arr[:, i]):+.3f} | Act_std={np.std(act_arr[:, i]):.3f}")

print(f"\n5. CENet 隐变量 z (16维) 激活方差:")
z_std = np.std(latent_z_arr, axis=0)
print(f"   z 各维标准差: {[round(x, 3) for x in z_std]}")
print(f"   z 平均绝对值: {np.mean(np.abs(latent_z_arr)):.3f}")

simulation_app.close()
