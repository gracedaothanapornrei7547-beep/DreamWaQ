import argparse
import sys
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "source" / "dreamwaq_lab"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rsl_rl"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rsl_rl" / "dreamwaq"))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from dreamwaq_actor_critic import register_dreamwaq_actor_critic
from dreamwaq_ppo import register_dreamwaq_ppo
import dreamwaq_lab.tasks
from isaaclab_tasks.utils import parse_env_cfg

register_dreamwaq_actor_critic()
register_dreamwaq_ppo()

env_cfg = parse_env_cfg("Unitree-Go2-DreamWaQ-Rough-Play", device="cuda:0", num_envs=4)
# set terrain level 2 (stairs)
env_cfg.scene.terrain.max_init_terrain_level = 2
env = gym.make("Unitree-Go2-DreamWaQ-Rough-Play", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.agents.rsl_rl_dreamwaq_cfg import UnitreeGo2DreamWaQRoughRunnerCfg
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict["device"] = "cuda:0"

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device="cuda:0")
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device="cuda:0")

obs = env.get_observations()
robot = env.unwrapped.scene["robot"]
contact_sensor = env.unwrapped.scene["contact_forces"]

torques_list = []
base_lin_vel_b_list = []
base_ang_vel_b_list = []
euler_angles_list = []
root_pos_w_list = []

from isaaclab.utils.math import euler_xyz_from_quat

print("Collecting 200 simulation steps on stairs...")
for step in range(200):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)

    # Robot physics data (env 0)
    torques = robot.data.applied_torque[0].cpu().numpy()
    lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
    ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy()
    quat = robot.data.root_quat_w[0]
    roll, pitch, yaw = euler_xyz_from_quat(quat.unsqueeze(0))
    pos = robot.data.root_pos_w[0].cpu().numpy()

    torques_list.append(torques)
    base_lin_vel_b_list.append(lin_vel)
    base_ang_vel_b_list.append(ang_vel)
    euler_angles_list.append([roll.item(), pitch.item(), yaw.item()])
    root_pos_w_list.append(pos)

torques = np.array(torques_list)
base_lin_vel = np.array(base_lin_vel_b_list)
base_ang_vel = np.array(base_ang_vel_b_list)
euler_angles = np.rad2deg(np.array(euler_angles_list))
root_pos = np.array(root_pos_w_list)

joint_names = robot.data.joint_names
print("\n" + "="*60)
print("ISAACLAB 底层物理遥测数据诊断报告 (Stairs Level 2)")
print("="*60)

print("\n1. 【关节电机输出力矩 (Joint Applied Torques)】(单位: N·m, 峰值限制 ~24 N·m):")
for i, name in enumerate(joint_names):
    mean_tau = np.mean(np.abs(torques[:, i]))
    max_tau = np.max(np.abs(torques[:, i]))
    print(f"  {name:16s}: 平均绝对力矩 = {mean_tau:5.2f} N·m, 峰值力矩 = {max_tau:5.2f} N·m")

rear_torque = np.mean(np.abs(torques[:, [3,4,5, 9,10,11]]))
front_torque = np.mean(np.abs(torques[:, [0,1,2, 6,7,8]]))
print(f"  --> 前腿平均力矩: {front_torque:.2f} N·m, 后腿平均力矩: {rear_torque:.2f} N·m")

print("\n2. 【机身运动线速度与侧向溜车】(Robot Base Frame, m/s):")
print(f"  前向速度 vx: 平均 = {np.mean(base_lin_vel[:, 0]):.3f} m/s, 最小值 = {np.min(base_lin_vel[:, 0]):.3f} m/s, 最大值 = {np.max(base_lin_vel[:, 0]):.3f} m/s")
print(f"  横向漂移 vy: 平均绝对值 = {np.mean(np.abs(base_lin_vel[:, 1])):.3f} m/s, 最大侧滑 = {np.max(np.abs(base_lin_vel[:, 1])):.3f} m/s")
print(f"  垂直颠簸 vz: 平均绝对值 = {np.mean(np.abs(base_lin_vel[:, 2])):.3f} m/s")

print("\n3. 【机身姿态角与偏航漂移】(单位: 度 deg):")
print(f"  Roll (横滚): 平均 = {np.mean(euler_angles[:, 0]):.2f}°, 范围 = [{np.min(euler_angles[:, 0]):.2f}°, {np.max(euler_angles[:, 0]):.2f}°]")
print(f"  Pitch (俯仰/抬头): 平均 = {np.mean(euler_angles[:, 1]):.2f}°, 范围 = [{np.min(euler_angles[:, 1]):.2f}°, {np.max(euler_angles[:, 1]):.2f}°]")
yaw_drift = euler_angles[-1, 2] - euler_angles[0, 2]
print(f"  Yaw 累计偏航漂移: {yaw_drift:.2f}° (偏航角速度 wz RMS = {np.rad2deg(np.sqrt(np.mean(base_ang_vel[:, 2]**2))):.2f}°/s)")

print("\n4. 【前进位移与爬升表现】:")
print(f"  前向 X 总位移: {root_pos[-1, 0] - root_pos[0, 0]:.2f} m")
print(f"  垂直 Z 爬升高度: {root_pos[-1, 2] - root_pos[0, 2]:.2f} m")

simulation_app.close()
