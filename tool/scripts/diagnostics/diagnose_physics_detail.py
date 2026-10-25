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

all_pos = []
all_vel = []
all_ang_vel = []
all_quat = []
all_torques = []

from isaaclab.utils.math import euler_xyz_from_quat

for step in range(300):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)

    pos = robot.data.root_pos_w.cpu().numpy() # [num_envs, 3]
    vel = robot.data.root_lin_vel_b.cpu().numpy() # [num_envs, 3]
    ang_vel = robot.data.root_ang_vel_b.cpu().numpy() # [num_envs, 3]
    quat = robot.data.root_quat_w # [num_envs, 4]
    r, p, y = euler_xyz_from_quat(quat)
    rpy = torch.stack([r, p, y], dim=-1).cpu().numpy() # [num_envs, 3]
    tau = robot.data.applied_torque.cpu().numpy() # [num_envs, 12]

    all_pos.append(pos)
    all_vel.append(vel)
    all_ang_vel.append(ang_vel)
    all_quat.append(rpy)
    all_torques.append(tau)

all_pos = np.array(all_pos) # [300, 4, 3]
all_vel = np.array(all_vel) # [300, 4, 3]
all_ang_vel = np.array(all_ang_vel) # [300, 4, 3]
all_quat = np.rad2deg(np.array(all_quat)) # [300, 4, 3]
all_torques = np.array(all_torques) # [300, 4, 12]

print("\n" + "="*70)
print("4个并行环境详细遥测对比 (Env 0 ~ Env 3, 300 steps = 6.0秒)")
print("="*70)

for e in range(4):
    dx = all_pos[-1, e, 0] - all_pos[0, e, 0]
    dy = all_pos[-1, e, 1] - all_pos[0, e, 1]
    dz = all_pos[-1, e, 2] - all_pos[0, e, 2]
    dist = np.sqrt(dx**2 + dy**2)
    mean_vx = np.mean(all_vel[:, e, 0])
    mean_vy = np.mean(np.abs(all_vel[:, e, 1]))
    mean_pitch = np.mean(all_quat[:, e, 1])
    max_pitch = np.max(all_quat[:, e, 1])
    yaw_drift = all_quat[-1, e, 2] - all_quat[0, e, 2]
    max_yaw_drift = np.max(np.abs(all_quat[:, e, 2] - all_quat[0, e, 2]))
    avg_tau = np.mean(np.abs(all_torques[:, e, :]))
    max_tau = np.max(np.abs(all_torques[:, e, :]))
    # rear vs front tau
    front_tau = np.mean(np.abs(all_torques[:, e, [0,1,2, 6,7,8]]))
    rear_tau = np.mean(np.abs(all_torques[:, e, [3,4,5, 9,10,11]]))

    print(f"\n--- 环境 Env {e} ---")
    print(f"  X轴推进位移: {dx:+.2f} m | Y轴侧向漂移: {dy:+.2f} m | Z轴升降高度: {dz:+.2f} m")
    print(f"  平均前向速度 vx: {mean_vx:.2f} m/s | 侧向漂移速度 |vy|: {mean_vy:.3f} m/s")
    print(f"  机身俯仰角 Pitch: 平均={mean_pitch:+.1f}°, 峰值={max_pitch:+.1f}°")
    print(f"  偏航角 Yaw 漂移: 最终={yaw_drift:+.1f}°, 最大绝对漂移={max_yaw_drift:.1f}°")
    print(f"  电机输出力矩: 平均={avg_tau:.2f} N·m, 峰值={max_tau:.2f} N·m (前腿={front_tau:.2f} N·m, 后腿={rear_tau:.2f} N·m)")

simulation_app.close()
