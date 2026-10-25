import os, sys, time, torch
from pathlib import Path
import numpy as np
import json

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

cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
cfg.scene.num_envs = 1
cfg.commands.base_velocity.ranges.lin_vel_x = (0.7, 0.7)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

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

env = gym.make('Unitree-Go2-DreamWaQ-Rough-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

default_ckpt = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-08_17-07_stage12_symmetric_power_balance_stairs_1000iter/model_9296.pt'
resume_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.pt') else default_ckpt
print(f"[INFO] Running diagnostic on checkpoint: {resume_path}", flush=True)
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')

robot = env.unwrapped.scene['robot']
sensor = env.unwrapped.scene.sensors['contact_forces']

foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
feet_sensor_idx, _ = sensor.find_bodies(foot_names)
feet_body_idx, _ = robot.find_bodies(foot_names)
joint_names = robot.data.joint_names

obs = env.get_observations()

# 记录数据容器
data = {
    'steps': 0,
    'contacts': [],
    'foot_pos_z_b': [], # 足端相对 base 的 z 轴高度
    'foot_vel_xy_w': [],
    'joint_pos': [],
    'joint_vel': [],
    'joint_tau': [],
    'root_pos_w': [],
    'root_lin_vel_b': [],
    'root_ang_vel_b': [],
    'root_rpy': [],
    'cenet_pred_vx': [],
    'cenet_latent_norm': [],
}

print("[INFO] Collecting Deep Stair Physics Telemetry for 180 steps...", flush=True)

for step in range(180):
    with torch.inference_mode():
        policy_module = runner.alg.policy
        cenet_out = policy_module.compute_cenet_outputs(obs, sample_latent=False)
        pred_vel = cenet_out.estimated_velocity
        pred_latent = cenet_out.latent
        
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)

    # 忽略前 30 步平地启动，记录 30~170 步攀爬过程
    if step >= 30:
        c_forces = sensor.data.net_forces_w[0, feet_sensor_idx].cpu().numpy()
        c_norm = np.linalg.norm(c_forces, axis=-1)
        in_contact = (c_norm > 5.0).astype(int)

        r_pos = robot.data.root_pos_w[0].cpu().numpy()
        r_quat = robot.data.root_quat_w[0].cpu().numpy()
        r_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
        r_ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy()
        
        # euler
        w, x, y, z = r_quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        j_pos = robot.data.joint_pos[0].cpu().numpy()
        j_vel = robot.data.joint_vel[0].cpu().numpy()
        j_tau = robot.data.applied_torque[0].cpu().numpy()

        b_pos_w = robot.data.body_pos_w[0, feet_body_idx].cpu().numpy()
        b_vel_w = robot.data.body_lin_vel_w[0, feet_body_idx].cpu().numpy()
        foot_rel_z = b_pos_w[:, 2] - r_pos[2]
        foot_slip_xy = np.linalg.norm(b_vel_w[:, :2], axis=-1)

        data['contacts'].append(in_contact)
        data['foot_pos_z_b'].append(foot_rel_z)
        data['foot_vel_xy_w'].append(foot_slip_xy)
        data['joint_pos'].append(j_pos)
        data['joint_vel'].append(j_vel)
        data['joint_tau'].append(j_tau)
        data['root_pos_w'].append(r_pos)
        data['root_lin_vel_b'].append(r_vel)
        data['root_ang_vel_b'].append(r_ang_vel)
        data['root_rpy'].append([roll, pitch, yaw])
        data['cenet_pred_vx'].append(pred_vel[0, 0].item())
        data['cenet_latent_norm'].append(torch.norm(pred_latent[0]).item())

# 分析
contacts = np.array(data['contacts']) # [T, 4]
duty_factors = np.mean(contacts, axis=0) * 100.0

dt = env.unwrapped.step_dt
cadences = []
for leg in range(4):
    c = contacts[:, leg]
    strikes = np.diff((c > 0).astype(int)) == 1
    num_cycles = np.sum(strikes)
    total_time = len(c) * dt
    cadences.append(num_cycles / total_time if total_time > 0 else 0)

foot_rel_z = np.array(data['foot_pos_z_b']) # [T, 4]
swing_lifts = []
for leg in range(4):
    in_swing = contacts[:, leg] == 0
    if np.any(in_swing):
        swing_lifts.append(float(np.max(foot_rel_z[in_swing, leg]) - np.min(foot_rel_z[:, leg])))
    else:
        swing_lifts.append(0.0)

# 关节 ROM 与力矩
j_pos = np.array(data['joint_pos']) # [T, 12]
j_tau = np.array(data['joint_tau'])
j_vel = np.array(data['joint_vel'])
j_power = np.abs(j_tau * j_vel) # [T, 12]

rom_deg = np.ptp(j_pos, axis=0) * 180.0 / np.pi
mean_pos_rad = np.mean(j_pos, axis=0)
mean_tau = np.mean(np.abs(j_tau), axis=0)
mean_pow = np.mean(j_power, axis=0)

r_rpy = np.array(data['root_rpy']) * 180.0 / np.pi
r_vel = np.array(data['root_lin_vel_b'])
r_ang = np.array(data['root_ang_vel_b'])

# 组装四腿详细报告
report = {
    'gait_symmetry': {
        'duty_factor_pct': {name: round(float(duty_factors[i]), 1) for i, name in enumerate(foot_names)},
        'cadence_Hz': {name: round(float(cadences[i]), 2) for i, name in enumerate(foot_names)},
        'swing_lift_cm': {name: round(float(swing_lifts[i] * 100), 1) for i, name in enumerate(foot_names)},
        'diag_FL_RR_coherence': round(float(np.corrcoef(contacts[:, 0], contacts[:, 3])[0, 1]), 3),
        'diag_FR_RL_coherence': round(float(np.corrcoef(contacts[:, 1], contacts[:, 2])[0, 1]), 3),
        'ipsi_FL_FR_phase': round(float(np.corrcoef(contacts[:, 0], contacts[:, 1])[0, 1]), 3)
    },
    'power_distribution_watts': {
        'FL_total_power': round(float(np.sum(mean_pow[:3])), 2),
        'FR_total_power': round(float(np.sum(mean_pow[3:6])), 2),
        'RL_total_power': round(float(np.sum(mean_pow[6:9])), 2),
        'RR_total_power': round(float(np.sum(mean_pow[9:12])), 2),
        'left_total_power': round(float(np.sum(mean_pow[:3]) + np.sum(mean_pow[6:9])), 2),
        'right_total_power': round(float(np.sum(mean_pow[3:6]) + np.sum(mean_pow[9:12])), 2),
        'left_to_right_ratio': round(float((np.sum(mean_pow[:3]) + np.sum(mean_pow[6:9])) / max(np.sum(mean_pow[3:6]) + np.sum(mean_pow[9:12]), 1e-4)), 2),
        'front_total_power': round(float(np.sum(mean_pow[:6])), 2),
        'rear_total_power': round(float(np.sum(mean_pow[6:12])), 2),
        'rear_to_front_ratio': round(float(np.sum(mean_pow[6:12]) / max(np.sum(mean_pow[:6]), 1e-4)), 2)
    },
    'joint_kinematics_summary': {
        name: {
            'rom_deg': round(float(rom_deg[i]), 1),
            'mean_angle_deg': round(float(mean_pos_rad[i] * 180.0 / np.pi), 1),
            'mean_torque_Nm': round(float(mean_tau[i]), 2),
            'mean_power_W': round(float(mean_pow[i]), 2)
        } for i, name in enumerate(joint_names)
    },
    'body_state': {
        'mean_pitch_deg': round(float(np.mean(r_rpy[:, 1])), 2),
        'std_pitch_deg': round(float(np.std(r_rpy[:, 1])), 2),
        'mean_roll_deg': round(float(np.mean(r_rpy[:, 0])), 2),
        'std_roll_deg': round(float(np.std(r_rpy[:, 0])), 2),
        'mean_yaw_rate_rad_s': round(float(np.mean(np.abs(r_ang[:, 2]))), 3),
        'mean_vx_real': round(float(np.mean(r_vel[:, 0])), 3),
        'mean_vx_cenet_pred': round(float(np.mean(data['cenet_pred_vx'])), 3),
        'mean_cenet_latent_norm': round(float(np.mean(data['cenet_latent_norm'])), 2)
    }
}

print("==================== DEEP STAIR DIAGNOSTIC REPORT ====================")
print(json.dumps(report, indent=2))
print("=======================================================================")

simulation_app.close()

