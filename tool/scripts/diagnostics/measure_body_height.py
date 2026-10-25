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
height_scanner = env.unwrapped.scene.sensors["height_scanner"]

height_records = []
thigh_angles = []
calf_angles = []

print("Running 150 steps to measure base height...")
for step in range(150):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)

    # Base z in world
    base_z = robot.data.root_pos_w[:, 2] # [4]
    
    # Scanner ray hits
    ray_hits_z = height_scanner.data.ray_hits_w[..., 2] # [4, 187]
    finite_mask = torch.isfinite(ray_hits_z)
    all_valid_count = finite_mask.sum(dim=1).clamp(min=1)
    terrain_z_mean = torch.where(finite_mask, ray_hits_z, torch.zeros_like(ray_hits_z)).sum(dim=1) / all_valid_count
    
    # Relative height = base_z - terrain_z_mean
    rel_height = (base_z - terrain_z_mean).cpu().numpy()
    height_records.append(rel_height)
    
    # Joint positions (FL_thigh: 1, FR_thigh: 4, RL_thigh: 7, RR_thigh: 10)
    # Joint names in order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
    j_pos = robot.data.joint_pos.cpu().numpy() # [4, 12]
    # Check joint names
    thigh_idx = [1, 4, 7, 10]
    calf_idx = [2, 5, 8, 11]
    thigh_angles.append(j_pos[:, thigh_idx])
    calf_angles.append(j_pos[:, calf_idx])

height_records = np.array(height_records) # [150, 4]
thigh_angles = np.array(thigh_angles) # [150, 4, 4]
calf_angles = np.array(calf_angles) # [150, 4, 4]

print("\n" + "="*70)
print("ISAACLAB 实测机身离地高度与关节弯曲数据分析")
print("="*70)

for e in range(4):
    h = height_records[:, e]
    print(f"\n【环境 Env {e}】")
    print(f"  相对局部地形高度: 平均 = {np.mean(h):.3f} m, 最小 = {np.min(h):.3f} m, 最大 = {np.max(h):.3f} m")
    print(f"  各关节平均角度 (rad):")
    print(f"    Thigh (大腿): FL={np.mean(thigh_angles[:, e, 0]):.2f}, FR={np.mean(thigh_angles[:, e, 1]):.2f}, RL={np.mean(thigh_angles[:, e, 2]):.2f}, RR={np.mean(thigh_angles[:, e, 3]):.2f}")
    print(f"    Calf  (小腿): FL={np.mean(calf_angles[:, e, 0]):.2f}, FR={np.mean(calf_angles[:, e, 1]):.2f}, RL={np.mean(calf_angles[:, e, 2]):.2f}, RR={np.mean(calf_angles[:, e, 3]):.2f}")

print("\n【4个环境总体机身离地高度统计】:")
print(f"  总体平均相对高度: {np.mean(height_records):.3f} m (标准设计参考: 0.30m ~ 0.32m)")
print(f"  总体最低高度:     {np.min(height_records):.3f} m")

simulation_app.close()
