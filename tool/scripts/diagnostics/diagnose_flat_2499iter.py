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
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.dreamwaq_env_cfg import UnitreeGo2DreamWaQFlatEnvCfg_PLAY
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.agents.rsl_rl_dreamwaq_cfg import UnitreeGo2DreamWaQFlatRunnerCfg

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

cfg = UnitreeGo2DreamWaQFlatEnvCfg_PLAY()
cfg.scene.num_envs = 1
cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

env = gym.make('Unitree-Go2-DreamWaQ-Flat-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

checkpoint_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-09_13-06_pure_cold_start_flat_2500iter/model_2499.pt'
runner_cfg = UnitreeGo2DreamWaQFlatRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(checkpoint_path)
actor_critic = runner.alg.policy
actor_critic.eval()

obs, _ = env.reset()

v_true_list = []
v_est_list = []
yaw_list = []
roll_list = []
pitch_list = []
ang_vel_roll_list = []
z_latent_list = []

print("\n>>> 开始采集 200 拍 (4.0s) 物理探针数据...")
with torch.no_grad():
    for step in range(200):
        # 1. 抽取 CENet 估计值
        obs_dict = env.unwrapped.observation_manager.compute()
        history_obs = obs_dict['dreamwaq_policy_history']
        
        if hasattr(actor_critic, 'cenet') and actor_critic.cenet is not None:
            cenet_out = actor_critic.cenet(history_obs, sample_latent=False)
            v_est = cenet_out.estimated_velocity[0].cpu().numpy()
            z_latent = cenet_out.latent[0].cpu().numpy()
        else:
            v_est = np.zeros(3)
            z_latent = np.zeros(16)
        
        # 2. 抽取真实物理状态
        unwrapped_env = env.unwrapped
        robot = unwrapped_env.scene['robot']
        v_true = robot.data.root_lin_vel_b.cpu().numpy()[0]
        quat = robot.data.root_quat_w.cpu().numpy()[0]
        ang_vel = robot.data.root_ang_vel_b.cpu().numpy()[0]
        
        r, p, y = quat_to_rpy_deg(quat)
        
        v_true_list.append(v_true)
        v_est_list.append(v_est)
        yaw_list.append(y)
        roll_list.append(r)
        pitch_list.append(p)
        ang_vel_roll_list.append(ang_vel[0])
        z_latent_list.append(z_latent)
        
        # 前向步进
        actions = actor_critic.act_inference(obs)
        obs, _, _, _ = env.step(actions)

v_true_arr = np.array(v_true_list)
v_est_arr = np.array(v_est_list)
roll_arr = np.array(roll_list)
pitch_arr = np.array(pitch_list)
yaw_arr = np.array(yaw_list)
ang_vel_roll_arr = np.array(ang_vel_roll_list)
z_latent_arr = np.array(z_latent_list)

print("\n================== 纯平地 2500 轮冷启动物理验收探针报告 ==================")
print(f"1. 前向速度 Vx 对比 (目标 0.80 m/s):")
print(f"   真实 Vx 均值: {np.mean(v_true_arr[:, 0]):.4f} m/s (标准差: {np.std(v_true_arr[:, 0]):.4f})")
print(f"   估计 Vx 均值: {np.mean(v_est_arr[:, 0]):.4f} m/s (标准差: {np.std(v_est_arr[:, 0]):.4f})")
print(f"   速度回归绝对误差 |Vx - V_est|: {np.mean(np.abs(v_true_arr[:, 0] - v_est_arr[:, 0])):.4f} m/s")
print(f"   速度符号是否对齐 (方向一致性): {'【完美同向】' if np.mean(v_est_arr[:, 0]) > 0 else '【存在倒挂】'}")

print(f"\n2. 侧向漂移 Vy 与偏航 Yaw 对比 (目标 0.00):")
print(f"   真实 Vy 均值: {np.mean(v_true_arr[:, 1]):.4f} m/s (峰值: {np.max(np.abs(v_true_arr[:, 1])):.4f} m/s)")
print(f"   估计 Vy 均值: {np.mean(v_est_arr[:, 1]):.4f} m/s")
print(f"   偏航角 Yaw 漂移范围: [{np.min(yaw_arr):.2f}°, {np.max(yaw_arr):.2f}°], 全程漂移量: {np.ptp(yaw_arr):.2f}°")

print(f"\n3. 姿态稳定性与晃动指标 (Roll 振荡):")
print(f"   横滚角 Roll 均值: {np.mean(roll_arr):.3f}°, 标准差: {np.std(roll_arr):.3f}°")
print(f"   俯仰角 Pitch 均值: {np.mean(pitch_arr):.3f}°, 标准差: {np.std(pitch_arr):.3f}°")
print(f"   横滚角速度 w_roll RMS: {np.sqrt(np.mean(ang_vel_roll_arr**2)):.4f} rad/s")

print(f"\n4. 变分自编码隐变量 z (16 维) 动力学特征:")
print(f"   隐变量范数 ||z|| 均值: {np.mean(np.linalg.norm(z_latent_arr, axis=1)):.4f}")
print(f"   隐变量各维方差均值: {np.mean(np.var(z_latent_arr, axis=0)):.6f}")
print("=========================================================================\n")

simulation_app.close()
