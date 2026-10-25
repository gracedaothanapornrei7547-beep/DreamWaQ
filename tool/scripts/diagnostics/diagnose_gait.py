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
parser.add_argument("--task", type=str, default="Unitree-Go2-DreamWaQ-Flat-Play")
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

env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.agents.rsl_rl_dreamwaq_cfg import UnitreeGo2DreamWaQRoughRunnerCfg

runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict["device"] = "cuda:0"

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device="cuda:0")
runner.load(args_cli.checkpoint)

policy = runner.get_inference_policy(device="cuda:0")

obs, _ = env.reset()
cmd = env.unwrapped.command_manager.get_command("base_velocity")
print(f"Target command velocity: vx={cmd[0, 0].item():.2f} m/s, vy={cmd[0, 1].item():.2f} m/s, wz={cmd[0, 2].item():.2f} rad/s")

robot = env.unwrapped.scene["robot"]
contact_sensor = env.unwrapped.scene["contact_forces"]

# Record data
contacts = []
joint_pos_list = []
vx_list = []

print("Running 150 diagnostic steps...")
for step in range(150):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)
    vx_list.append(robot.data.root_lin_vel_b[0, 0].item())


    c_forces = contact_sensor.data.net_forces_w[0] # [num_bodies, 3]
    foot_ids, _ = contact_sensor.find_bodies(".*foot")
    forces_mag = torch.norm(c_forces[foot_ids], dim=-1).cpu().numpy()
    contacts.append(forces_mag)

    j_pos = robot.data.joint_pos[0].cpu().numpy().copy()
    joint_pos_list.append(j_pos)

contacts = np.array(contacts) # [150, 4]
joint_pos_list = np.array(joint_pos_list) # [150, 12]

foot_names = [contact_sensor.body_names[i] for i in foot_ids]

vx_mean = np.mean(vx_list)
base_z = robot.data.root_pos_w[0, 2].item()
print("\n================ KINEMATICS TELEMETRY ================")
print(f"Base linear velocity (vx mean): {vx_mean:+.3f} m/s (min={np.min(vx_list):+.3f}, max={np.max(vx_list):+.3f})")
print(f"Base height (z):                {base_z:.3f} m")


print("\n================ FOOT CONTACT FORCES (N) ================")
for i, name in enumerate(foot_names):
    print(f"{name:10s}: mean={contacts[:, i].mean():.1f}N, max={contacts[:, i].max():.1f}N, min={contacts[:, i].min():.1f}N")

print("\n================ FOOT DUTY CYCLE (% time in contact) ================")
for thresh in [1.0, 5.0, 10.0, 20.0]:
    print(f"--- Threshold > {thresh:.1f} N ---")
    bin_contacts = (contacts > thresh).astype(float)
    for i, name in enumerate(foot_names):
        print(f"  {name:10s}: {np.mean(bin_contacts[:, i])*100:.1f}%")

bin_5n = (contacts > 5.0).astype(float)
print("\n================ CONTACT CORRELATION MATRIX (Threshold > 5.0 N) ================")
corr = np.corrcoef(bin_5n.T)
header = " " * 12 + " ".join([f"{name:>10s}" for name in foot_names])
print(header)
for i, name1 in enumerate(foot_names):
    row_str = " ".join([f"{corr[i, j]:+10.2f}" for j in range(len(foot_names))])
    print(f"{name1:10s} : {row_str}")

joint_names = robot.data.joint_names
print("\n================ JOINT MEAN, MIN, MAX (deg) ================")
for i, name in enumerate(joint_names):
    mean_deg = np.rad2deg(np.mean(joint_pos_list[:, i]))
    min_deg = np.rad2deg(np.min(joint_pos_list[:, i]))
    max_deg = np.rad2deg(np.max(joint_pos_list[:, i]))
    print(f"{name:16s}: mean={mean_deg:+6.1f}°  range=[{min_deg:+6.1f}°, {max_deg:+6.1f}°]")


simulation_app.close()
