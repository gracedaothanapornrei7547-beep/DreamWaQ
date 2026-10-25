import os, sys, torch
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

def run_diagnostic(terrain_type='stairs', steps=200, target_speed=0.8):
    cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.commands.base_velocity.ranges.lin_vel_x = (target_speed, target_speed)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    if terrain_type == 'stairs':
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
    foot_pos_w_list = []
    foot_pos_b_list = []

    for step in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            runner.alg.policy.reset(dones)

        unwrapped = env.unwrapped
        robot = unwrapped.scene['robot']
        c_forces = unwrapped.scene.sensors['contact_forces'].data.net_forces_w[0, [4, 8, 12, 16]].cpu().numpy() # FL, FR, RL, RR
        contact_norm = np.linalg.norm(c_forces, axis=-1)
        in_contact = (contact_norm > 5.0).astype(int)
        
        root_pos = robot.data.root_pos_w[0].cpu().numpy()
        root_quat = robot.data.root_quat_w[0].cpu().numpy() # w, x, y, z
        root_lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
        j_pos = robot.data.joint_pos[0].cpu().numpy()
        j_vel = robot.data.joint_vel[0].cpu().numpy()
        b_pos_w = robot.data.body_pos_w[0, [4, 8, 12, 16]].cpu().numpy()
        
        # approximate pitch / roll
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
        foot_pos_w_list.append(b_pos_w)
        foot_pos_b_list.append(b_pos_w - root_pos)

    env.close()

    contacts = np.array(contacts_list[20:]) # discard first 20 steps
    forces = np.array(forces_list[20:])
    base_pos = np.array(base_pos_list[20:])
    base_rpy = np.array(base_rpy_list[20:])
    base_vel = np.array(base_vel_list[20:])
    joint_pos = np.array(joint_pos_list[20:])
    foot_pos_b = np.array(foot_pos_b_list[20:])

    # 1. Duty Cycle (占比: FL, FR, RL, RR)
    duty_cycle = np.mean(contacts, axis=0) * 100.0
    
    # 2. 对角同步度 Trot Coherence:
    # 理想 Trot: FL 与 RR 触地状态相同 (corr=1.0), FR 与 RL 相同; 且 FL 与 FR 反相 (corr=-1.0)
    trot_diag1 = np.corrcoef(contacts[:, 0], contacts[:, 3])[0, 1] if np.std(contacts[:, 0]) > 0 and np.std(contacts[:, 3]) > 0 else 0.0 # FL & RR
    trot_diag2 = np.corrcoef(contacts[:, 1], contacts[:, 2])[0, 1] if np.std(contacts[:, 1]) > 0 and np.std(contacts[:, 2]) > 0 else 0.0 # FR & RL
    trot_ipsi  = np.corrcoef(contacts[:, 0], contacts[:, 1])[0, 1] if np.std(contacts[:, 0]) > 0 and np.std(contacts[:, 1]) > 0 else 0.0 # FL & FR

    # 3. 步频估算 (触地状态跳变频率)
    switches = np.sum(np.abs(np.diff(contacts, axis=0)), axis=0) / 2.0
    total_time = len(contacts) * 0.02
    cadence = switches / total_time # Hz

    # 4. 机身俯仰角 Pitch & 侧倾角 Roll
    mean_pitch = np.degrees(np.mean(base_rpy[:, 1]))
    std_pitch = np.degrees(np.std(base_rpy[:, 1]))
    mean_roll = np.degrees(np.mean(base_rpy[:, 0]))
    std_roll = np.degrees(np.std(base_rpy[:, 0]))

    # 5. 机身平均高度与颠簸
    mean_z = np.mean(base_pos[:, 2])
    std_z = np.std(base_pos[:, 2])
    mean_vx = np.mean(base_vel[:, 0])

    # 6. 足端摆动抬腿高度 (机身坐标系下的 Z 方向振幅)
    foot_lift_fl = np.max(foot_pos_b[:, 0, 2]) - np.min(foot_pos_b[:, 0, 2])
    foot_lift_rl = np.max(foot_pos_b[:, 2, 2]) - np.min(foot_pos_b[:, 2, 2])

    # 7. 关节活动度 ROM (Range of Motion):
    # Joint index:
    # FL: [0, 1, 2] (hip, thigh, calf)
    # FR: [3, 4, 5]
    # RL: [6, 7, 8]
    # RR: [9, 10, 11]
    rom_fl_thigh = np.max(joint_pos[:, 1]) - np.min(joint_pos[:, 1])
    rom_fl_calf  = np.max(joint_pos[:, 2]) - np.min(joint_pos[:, 2])
    rom_rl_thigh = np.max(joint_pos[:, 7]) - np.min(joint_pos[:, 7])
    rom_rl_calf  = np.max(joint_pos[:, 8]) - np.min(joint_pos[:, 8])

    return {
        'terrain': terrain_type,
        'mean_vx': mean_vx,
        'duty_cycle': duty_cycle, # [FL, FR, RL, RR]
        'trot_diag1': trot_diag1,
        'trot_diag2': trot_diag2,
        'trot_ipsi': trot_ipsi,
        'cadence': cadence,
        'mean_pitch': mean_pitch,
        'std_pitch': std_pitch,
        'mean_roll': mean_roll,
        'std_roll': std_roll,
        'mean_z': mean_z,
        'std_z': std_z,
        'foot_lift_fl': foot_lift_fl,
        'foot_lift_rl': foot_lift_rl,
        'rom_fl_thigh': rom_fl_thigh,
        'rom_fl_calf': rom_fl_calf,
        'rom_rl_thigh': rom_rl_thigh,
        'rom_rl_calf': rom_rl_calf,
    }

print("=== 开始抽取平地步态特征数据 ===", flush=True)
res_flat = run_diagnostic('flat', steps=160, target_speed=0.8)

print("=== 开始抽取楼梯步态特征数据 ===", flush=True)
res_stair = run_diagnostic('stairs', steps=160, target_speed=0.8)

simulation_app.close()

import json
print("\n" + "="*80)
print("【ISAACLAB GAIT TELEMETRY EXTRACTED】")
print("="*80)
print("FLAT_DATA:", json.dumps(res_flat, indent=2, default=lambda x: round(float(x), 4) if isinstance(x, (np.floating, float)) else x.tolist() if isinstance(x, np.ndarray) else x))
print("STAIR_DATA:", json.dumps(res_stair, indent=2, default=lambda x: round(float(x), 4) if isinstance(x, (np.floating, float)) else x.tolist() if isinstance(x, np.ndarray) else x))
