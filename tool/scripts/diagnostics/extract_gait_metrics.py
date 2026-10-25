import os, sys, argparse, json, torch
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

parser = argparse.ArgumentParser()
parser.add_argument('--terrain', type=str, default='flat', choices=['flat', 'stairs'])
parser.add_argument('--steps', type=int, default=150)
parser.add_argument('--speed', type=float, default=0.8)
parser.add_argument('--checkpoint', type=str, default='/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-08_12-07_stage9_flat_stair_coexistence_continue/model_7399.pt')
args = parser.parse_args()

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

cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
cfg.scene.num_envs = 1
cfg.commands.base_velocity.ranges.lin_vel_x = (args.speed, args.speed)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

if args.terrain == 'stairs':
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
    cfg.scene.terrain.max_init_terrain_level = 2
else:
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.terrain_levels = None

env = gym.make('Unitree-Go2-DreamWaQ-Rough-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

resume_path = args.checkpoint
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')

sensor = env.unwrapped.scene.sensors['contact_forces']
robot = env.unwrapped.scene['robot']

feet_sensor_idx, _ = sensor.find_bodies('.*_foot')
feet_body_idx, _ = robot.find_bodies('.*_foot')

obs = env.get_observations()

contacts_list = []
forces_list = []
base_pos_list = []
base_rpy_list = []
base_vel_list = []
joint_pos_list = []
joint_vel_list = []
feet_pos_b_list = []
feet_vel_w_list = []

for step in range(args.steps):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)

    c_forces = sensor.data.net_forces_w[0, feet_sensor_idx].cpu().numpy()
    c_norm = np.linalg.norm(c_forces, axis=-1)
    in_contact = (c_norm > 5.0).astype(int)

    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    root_lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
    j_pos = robot.data.joint_pos[0].cpu().numpy()
    j_vel = robot.data.joint_vel[0].cpu().numpy()
    
    # feet world position & velocity
    b_pos_w = robot.data.body_pos_w[0, feet_body_idx].cpu().numpy()
    b_vel_w = robot.data.body_lin_vel_w[0, feet_body_idx].cpu().numpy()
    
    # euler angles
    w, x, y, z = root_quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    contacts_list.append(in_contact)
    forces_list.append(c_norm)
    base_pos_list.append(root_pos)
    base_rpy_list.append([roll, pitch])
    base_vel_list.append(root_lin_vel)
    joint_pos_list.append(j_pos)
    joint_vel_list.append(j_vel)
    feet_pos_b_list.append(b_pos_w - root_pos)
    feet_vel_w_list.append(b_vel_w)

# Discard warm-up steps
warmup = 25
contacts = np.array(contacts_list[warmup:])
forces = np.array(forces_list[warmup:])
base_pos = np.array(base_pos_list[warmup:])
base_rpy = np.array(base_rpy_list[warmup:])
base_vel = np.array(base_vel_list[warmup:])
joint_pos = np.array(joint_pos_list[warmup:])
feet_pos_b = np.array(feet_pos_b_list[warmup:])
feet_vel_w = np.array(feet_vel_w_list[warmup:])

duty_cycle = np.mean(contacts, axis=0) * 100.0

# Diagonal correlations
def safe_corr(a, b):
    if np.std(a) < 1e-5 or np.std(b) < 1e-5:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

trot_diag1 = safe_corr(contacts[:, 0], contacts[:, 3]) # FL & RR
trot_diag2 = safe_corr(contacts[:, 1], contacts[:, 2]) # FR & RL
trot_ipsi  = safe_corr(contacts[:, 0], contacts[:, 1]) # FL & FR (should be negative)

switches = np.sum(np.abs(np.diff(contacts, axis=0)), axis=0) / 2.0
total_time = len(contacts) * 0.02
cadence = switches / total_time

# Stance slip velocity
slip_vels = []
for i in range(4):
    stance_mask = contacts[:, i] == 1
    if np.any(stance_mask):
        slip_xy = np.linalg.norm(feet_vel_w[stance_mask, i, :2], axis=-1)
        slip_vels.append(float(np.mean(slip_xy)))
    else:
        slip_vels.append(0.0)

# Swing foot lift (机身坐标系下足端 Z 方向振幅)
swing_lifts = [float(np.max(feet_pos_b[:, i, 2]) - np.min(feet_pos_b[:, i, 2])) for i in range(4)]

res = {
    'terrain': args.terrain,
    'cmd_speed': args.speed,
    'actual_mean_vx': round(float(np.mean(base_vel[:, 0])), 3),
    'duty_cycle_pct': {
        'FL': round(float(duty_cycle[0]), 1),
        'FR': round(float(duty_cycle[1]), 1),
        'RL': round(float(duty_cycle[2]), 1),
        'RR': round(float(duty_cycle[3]), 1),
    },
    'trot_coherence': {
        'diag_FL_RR': round(trot_diag1, 3),
        'diag_FR_RL': round(trot_diag2, 3),
        'ipsi_FL_FR_antiphase': round(trot_ipsi, 3),
    },
    'cadence_Hz': {
        'FL': round(float(cadence[0]), 2),
        'FR': round(float(cadence[1]), 2),
        'RL': round(float(cadence[2]), 2),
        'RR': round(float(cadence[3]), 2),
    },
    'base_orientation': {
        'mean_pitch_deg': round(float(np.degrees(np.mean(base_rpy[:, 1]))), 2),
        'std_pitch_deg': round(float(np.degrees(np.std(base_rpy[:, 1]))), 2),
        'mean_roll_deg': round(float(np.degrees(np.mean(base_rpy[:, 0]))), 2),
        'std_roll_deg': round(float(np.degrees(np.std(base_rpy[:, 0]))), 2),
    },
    'base_height': {
        'mean_z': round(float(np.mean(base_pos[:, 2])), 3),
        'std_z': round(float(np.std(base_pos[:, 2])), 3),
    },
    'foot_slip_velocity_m_per_s': {
        'FL': round(slip_vels[0], 3),
        'FR': round(slip_vels[1], 3),
        'RL': round(slip_vels[2], 3),
        'RR': round(slip_vels[3], 3),
    },
    'swing_foot_lift_m': {
        'FL': round(swing_lifts[0], 3),
        'FR': round(swing_lifts[1], 3),
        'RL': round(swing_lifts[2], 3),
        'RR': round(swing_lifts[3], 3),
    },
    'joint_rom_deg': {
        'FL_thigh': round(float(np.degrees(np.max(joint_pos[:, 4]) - np.min(joint_pos[:, 4]))), 1),
        'FL_calf': round(float(np.degrees(np.max(joint_pos[:, 8]) - np.min(joint_pos[:, 8]))), 1),
        'RL_thigh': round(float(np.degrees(np.max(joint_pos[:, 6]) - np.min(joint_pos[:, 6]))), 1),
        'RL_calf': round(float(np.degrees(np.max(joint_pos[:, 10]) - np.min(joint_pos[:, 10]))), 1),
    },
    'mean_joint_angles_rad': {
        'FL_thigh': round(float(np.mean(joint_pos[:, 4])), 2),
        'FL_calf': round(float(np.mean(joint_pos[:, 8])), 2),
        'RL_thigh': round(float(np.mean(joint_pos[:, 6])), 2),
        'RL_calf': round(float(np.mean(joint_pos[:, 10])), 2),
    }
}

print(f"\n=================== TELEMETRY RESULT ({args.terrain.upper()}) ===================", flush=True)
print(json.dumps(res, indent=2), flush=True)
print("==========================================================================", flush=True)

simulation_app.close()
os._exit(0)

