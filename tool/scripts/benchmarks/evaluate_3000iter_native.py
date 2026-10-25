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
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

checkpoint_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-08_22-15_qinzx03_native_golden_standard_3000iter/model_2999.pt'

# 1. 评估金字塔台阶 (12cm 连续台阶)
print('>>> [1/2] 正在评估 12cm 金字塔台阶表现...')
cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
cfg.scene.num_envs = 1
stair_gen = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        'pyramid_stairs_inv': terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.10, 0.12),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        )
    }
)
cfg.scene.terrain.terrain_generator = stair_gen
cfg.commands.base_velocity.ranges.lin_vel_x = (0.7, 0.7)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

env = gym.make('Unitree-Go2-DreamWaQ-Rough-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_dict = runner_cfg.to_dict()
agent_dict['device'] = 'cuda:0'
runner = OnPolicyRunner(env, agent_dict, log_dir=None, device='cuda:0')
runner.load(checkpoint_path)
policy = runner.get_inference_policy(device='cuda:0')

obs = env.get_observations()

contacts_list = []
forces_list = []
base_pos_list = []
base_rpy_list = []
base_vel_list = []
joint_pos_list = []
joint_vel_list = []
joint_torques_list = []
foot_pos_b_list = []

for step in range(300):
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
    j_torques = robot.data.applied_torque[0].cpu().numpy()
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
    joint_torques_list.append(j_torques)
    foot_pos_b_list.append(b_pos_w - root_pos)

contacts = np.array(contacts_list[50:])
base_pos = np.array(base_pos_list[50:])
base_rpy = np.array(base_rpy_list[50:])
base_vel = np.array(base_vel_list[50:])
joint_vel = np.array(joint_vel_list[50:])
joint_torques = np.array(joint_torques_list[50:])
foot_pos_b = np.array(foot_pos_b_list[50:])

duty_cycle = np.mean(contacts, axis=0) * 100.0
trot_diag1 = float(np.corrcoef(contacts[:, 0], contacts[:, 3])[0, 1]) if np.std(contacts[:, 0]) > 0 and np.std(contacts[:, 3]) > 0 else 0.0
trot_diag2 = float(np.corrcoef(contacts[:, 1], contacts[:, 2])[0, 1]) if np.std(contacts[:, 1]) > 0 and np.std(contacts[:, 2]) > 0 else 0.0
trot_ipsi  = float(np.corrcoef(contacts[:, 0], contacts[:, 1])[0, 1]) if np.std(contacts[:, 0]) > 0 and np.std(contacts[:, 1]) > 0 else 0.0

# 功率计算 (P = |tau * omega|)
joint_power = np.abs(joint_torques * joint_vel)
leg_power = {
    'FL': float(np.mean(np.sum(joint_power[:, 0:3], axis=-1))),
    'FR': float(np.mean(np.sum(joint_power[:, 3:6], axis=-1))),
    'RL': float(np.mean(np.sum(joint_power[:, 6:9], axis=-1))),
    'RR': float(np.mean(np.sum(joint_power[:, 9:12], axis=-1))),
}

stair_res = {
    'checkpoint': 'model_2999.pt',
    'distance_forward_x_m': round(float(base_pos[-1, 0] - base_pos[0, 0]), 2),
    'climb_height_delta_z_m': round(float(base_pos[-1, 2] - base_pos[0, 2]), 2),
    'mean_vx': round(float(np.mean(base_vel[:, 0])), 3),
    'mean_roll_deg': round(float(np.degrees(np.mean(base_rpy[:, 0]))), 2),
    'std_roll_deg': round(float(np.degrees(np.std(base_rpy[:, 0]))), 2),
    'mean_pitch_deg': round(float(np.degrees(np.mean(base_rpy[:, 1]))), 2),
    'duty_cycle_FL_FR_RL_RR': [round(float(x), 1) for x in duty_cycle],
    'trot_diag_sync_FL_RR': round(trot_diag1, 3),
    'trot_diag_sync_FR_RL': round(trot_diag2, 3),
    'leg_power_W': {k: round(v, 1) for k, v in leg_power.items()},
    'front_power_ratio_FR_FL': round(leg_power['FR'] / max(leg_power['FL'], 1e-3), 2),
    'rear_power_ratio_RR_RL': round(leg_power['RR'] / max(leg_power['RL'], 1e-3), 2),
    'swing_clearance_max_FL_m': round(float(np.max(foot_pos_b[:, 0, 2]) - np.min(foot_pos_b[:, 0, 2])), 3),
    'swing_clearance_max_FR_m': round(float(np.max(foot_pos_b[:, 1, 2]) - np.min(foot_pos_b[:, 1, 2])), 3),
}

print('\n================ STAIR DIAGNOSIS RESULTS ================')
print(json.dumps(stair_res, indent=2))
print('=========================================================\n')

simulation_app.close()

