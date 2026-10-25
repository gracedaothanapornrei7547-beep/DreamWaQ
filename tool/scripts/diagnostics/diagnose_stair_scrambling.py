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

import argparse

parser = argparse.ArgumentParser(description="Diagnose stair scrambling kinematics and action smoothness.")
parser.add_argument("--checkpoint", type=str, default="/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-08_15-35_stage11_smooth_trot_stair_refine/model_8297.pt", help="Checkpoint path")
parser.add_argument("--speed", type=float, default=0.85, help="Target forward speed")
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

resume_path = args.checkpoint
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')

robot = env.unwrapped.scene['robot']
contact_sensor = env.unwrapped.scene.sensors['contact_forces']

obs = env.get_observations()

# 记录指标
log_data = {
    'cmd_vx': [], 'real_vx': [], 'cenet_vx': [],
    'joint_vel_front': [], 'joint_vel_rear': [],
    'joint_torque_front': [], 'joint_torque_rear': [],
    'joint_power_front': [], 'joint_power_rear': [],
    'foot_contact': [], 'foot_air_time': [],
    'actions': [], 'action_diff': []
}

prev_actions = None
print("[INFO] Starting Stair Scrambling Physics Diagnosis (150 steps)...", flush=True)

for step in range(150):
    with torch.inference_mode():
        # 获取 CENet 内部估计
        policy_module = runner.alg.policy
        cenet_out = policy_module.compute_cenet_outputs(obs, sample_latent=False)
        pred_vel = cenet_out.estimated_velocity
        pred_latent = cenet_out.latent
        
        actions = policy(obs)
        if prev_actions is not None:
            action_diff = torch.norm(actions - prev_actions, dim=-1).item()
        else:
            action_diff = 0.0
        prev_actions = actions.clone()
        
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)

    # 提取第 80 到 140 步（纯在楼梯上攀爬阶段）
    if step >= 70:
        cmd_v = env.unwrapped.command_manager.get_command('base_velocity')[0, 0].item()
        real_v = robot.data.root_lin_vel_b[0, 0].item()
        pred_v = pred_vel[0, 0].item()
        
        # 关节速度与力矩: FL/FR (0..5), RL/RR (6..11)
        j_vel = robot.data.joint_vel[0].abs()
        j_tau = robot.data.applied_torque[0].abs()
        j_pow = j_vel * j_tau
        
        # 前肢关节 (FL/FR: indices [0,1,2, 3,4,5] or similar)
        # Go2 关节映射：
        # FL: hip, thigh, calf; FR: hip, thigh, calf; RL: ...; RR: ...
        front_power = j_pow[:6].sum().item()
        rear_power = j_pow[6:].sum().item()
        front_vel = j_vel[:6].mean().item()
        rear_vel = j_vel[6:].mean().item()
        front_tau = j_tau[:6].mean().item()
        rear_tau = j_tau[6:].mean().item()
        
        contact_force = contact_sensor.data.net_forces_w[0, :, 2] # 4 feet z-force
        contacts = (contact_force > 1.0).float().cpu().numpy()
        
        log_data['cmd_vx'].append(cmd_v)
        log_data['real_vx'].append(real_v)
        log_data['cenet_vx'].append(pred_v)
        log_data['joint_vel_front'].append(front_vel)
        log_data['joint_vel_rear'].append(rear_vel)
        log_data['joint_torque_front'].append(front_tau)
        log_data['joint_torque_rear'].append(rear_tau)
        log_data['joint_power_front'].append(front_power)
        log_data['joint_power_rear'].append(rear_power)
        log_data['action_diff'].append(action_diff)
        log_data['foot_contact'].append(contacts)

# 统计分析
import json
results = {
    'velocity_tracking': {
        'mean_cmd_vx': float(np.mean(log_data['cmd_vx'])),
        'mean_real_vx': float(np.mean(log_data['real_vx'])),
        'mean_cenet_pred_vx': float(np.mean(log_data['cenet_vx'])),
        'velocity_error': float(np.mean(np.array(log_data['cmd_vx']) - np.array(log_data['real_vx'])))
    },
    'joint_kinematics': {
        'front_joint_vel_mean': float(np.mean(log_data['joint_vel_front'])),
        'rear_joint_vel_mean': float(np.mean(log_data['joint_vel_rear'])),
        'front_vs_rear_vel_ratio': float(np.mean(log_data['joint_vel_front']) / max(np.mean(log_data['joint_vel_rear']), 1e-5)),
        'front_joint_torque_mean': float(np.mean(log_data['joint_torque_front'])),
        'rear_joint_torque_mean': float(np.mean(log_data['joint_torque_rear'])),
        'front_power_watts': float(np.mean(log_data['joint_power_front'])),
        'rear_power_watts': float(np.mean(log_data['joint_power_rear'])),
    },
    'action_smoothness': {
        'mean_action_step_diff': float(np.mean(log_data['action_diff'])),
        'max_action_step_diff': float(np.max(log_data['action_diff']))
    }
}

print("================ STAIR SCRAMBLING DIAGNOSTIC REPORT ================", flush=True)
print(json.dumps(results, indent=2), flush=True)
print("====================================================================", flush=True)

simulation_app.close()
