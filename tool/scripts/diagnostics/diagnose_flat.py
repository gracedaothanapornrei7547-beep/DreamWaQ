import os, sys, torch, json
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

cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
cfg.scene.num_envs = 1
cfg.scene.terrain.terrain_type = 'plane'
cfg.scene.terrain.terrain_generator = None
cfg.curriculum.terrain_levels = None
cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

env = gym.make('Unitree-Go2-DreamWaQ-Rough-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

resume_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-07_19-52_stage8_rough_pure_waxberry_continue/model_6749.pt'
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')

obs = env.get_observations()

contacts_list = []
forces_list = []
base_pos_list = []
base_rpy_list = []
base_vel_list = []
joint_pos_list = []
joint_vel_list = []
foot_pos_b_list = []

for step in range(200):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)

    unwrapped = env.unwrapped
    robot = unwrapped.scene['robot']
    c_forces = unwrapped.scene.sensors['contact_forces'].data.net_forces_w[0, [4, 8, 12, 16]].cpu().numpy()
    contact_norm = np.linalg.norm(c_forces, axis=-1)
    in_contact = (contact_norm > 5.0).astype(int)
    
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    root_lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
    j_pos = robot.data.joint_pos[0].cpu().numpy()
    j_vel = robot.data.joint_vel[0].cpu().numpy()
    b_pos_w = robot.data.body_pos_w[0, [4, 8, 12, 16]].cpu().numpy()
    
    w, x, y, z = root_quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    contacts_list.append(in_contact)
    forces_list.append(contact_norm)
    base_pos_list.append(root_pos)
    base_rpy_list.append([roll, pitch])
    base_vel_list.append(root_lin_vel)
    joint_pos_list.append(j_pos)
    joint_vel_list.append(j_vel)
    foot_pos_b_list.append(b_pos_w - root_pos)

contacts = np.array(contacts_list[30:])
forces = np.array(forces_list[30:])
base_pos = np.array(base_pos_list[30:])
base_rpy = np.array(base_rpy_list[30:])
base_vel = np.array(base_vel_list[30:])
joint_pos = np.array(joint_pos_list[30:])
foot_pos_b = np.array(foot_pos_b_list[30:])

duty_cycle = np.mean(contacts, axis=0) * 100.0
trot_diag1 = float(np.corrcoef(contacts[:, 0], contacts[:, 3])[0, 1]) if np.std(contacts[:, 0]) > 0 and np.std(contacts[:, 3]) > 0 else 0.0
trot_diag2 = float(np.corrcoef(contacts[:, 1], contacts[:, 2])[0, 1]) if np.std(contacts[:, 1]) > 0 and np.std(contacts[:, 2]) > 0 else 0.0
trot_ipsi  = float(np.corrcoef(contacts[:, 0], contacts[:, 1])[0, 1]) if np.std(contacts[:, 0]) > 0 and np.std(contacts[:, 1]) > 0 else 0.0

switches = np.sum(np.abs(np.diff(contacts, axis=0)), axis=0) / 2.0
total_time = len(contacts) * 0.02
cadence = switches / total_time

res = {
    'terrain': 'flat',
    'mean_vx': float(np.mean(base_vel[:, 0])),
    'duty_cycle_FL_FR_RL_RR': [round(float(x), 1) for x in duty_cycle],
    'trot_diag_sync_FL_RR': round(trot_diag1, 3),
    'trot_diag_sync_FR_RL': round(trot_diag2, 3),
    'trot_ipsi_sync_FL_FR': round(trot_ipsi, 3),
    'step_frequency_Hz': [round(float(x), 2) for x in cadence],
    'mean_pitch_deg': round(float(np.degrees(np.mean(base_rpy[:, 1]))), 2),
    'std_pitch_deg': round(float(np.degrees(np.std(base_rpy[:, 1]))), 2),
    'mean_roll_deg': round(float(np.degrees(np.mean(base_rpy[:, 0]))), 2),
    'std_roll_deg': round(float(np.degrees(np.std(base_rpy[:, 0]))), 2),
    'mean_base_height_z': round(float(np.mean(base_pos[:, 2])), 3),
    'std_base_height_z': round(float(np.std(base_pos[:, 2])), 3),
    'swing_foot_lift_FL_m': round(float(np.max(foot_pos_b[:, 0, 2]) - np.min(foot_pos_b[:, 0, 2])), 3),
    'swing_foot_lift_RL_m': round(float(np.max(foot_pos_b[:, 2, 2]) - np.min(foot_pos_b[:, 2, 2])), 3),
    'rom_FL_thigh_deg': round(float(np.degrees(np.max(joint_pos[:, 1]) - np.min(joint_pos[:, 1]))), 1),
    'rom_FL_calf_deg': round(float(np.degrees(np.max(joint_pos[:, 2]) - np.min(joint_pos[:, 2]))), 1),
    'rom_RL_thigh_deg': round(float(np.degrees(np.max(joint_pos[:, 7]) - np.min(joint_pos[:, 7]))), 1),
    'rom_RL_calf_deg': round(float(np.degrees(np.max(joint_pos[:, 8]) - np.min(joint_pos[:, 8]))), 1),
    'mean_joint_angles': {
        'FL_thigh': round(float(np.mean(joint_pos[:, 1])), 2),
        'FL_calf': round(float(np.mean(joint_pos[:, 2])), 2),
        'RL_thigh': round(float(np.mean(joint_pos[:, 7])), 2),
        'RL_calf': round(float(np.mean(joint_pos[:, 8])), 2),
    }
}

print("\n--- FLAT GAIT DIAGNOSIS RESULT ---", flush=True)
print(json.dumps(res, indent=2), flush=True)
simulation_app.close()
