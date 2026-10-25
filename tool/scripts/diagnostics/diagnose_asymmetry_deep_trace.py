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
cfg.commands.base_velocity.ranges.lin_vel_x = (0.65, 0.65)
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
default_ckpt = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-08_18-36_stage13_flat_symmetric_trot_reanchoring/model_9795.pt'
resume_path = default_ckpt
is_flat = False
for arg in sys.argv[1:]:
    if arg.endswith('.pt'):
        resume_path = arg
    elif arg == '--flat':
        is_flat = True

if is_flat:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    print(f"[INFO] Diagnostic mode: PURE FLAT GROUND | ckpt: {resume_path}", flush=True)
else:
    cfg.scene.terrain.terrain_generator = stair_gen
    cfg.scene.terrain.max_init_terrain_level = 2
    print(f"[INFO] Diagnostic mode: 12CM STAIRS | ckpt: {resume_path}", flush=True)

env = gym.make('Unitree-Go2-DreamWaQ-Rough-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

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

obs = env.get_observations()

# 收集攀爬台阶时刻的精确左右非对称指标
trace = {
    'roll_deg': [],
    'pitch_deg': [],
    'fl_power': [],
    'fr_power': [],
    'rl_power': [],
    'rr_power': [],
    'fl_z_rel': [],
    'fr_z_rel': [],
    'rl_z_rel': [],
    'rr_z_rel': [],
    'fl_contact': [],
    'fr_contact': [],
    'rl_contact': [],
    'rr_contact': [],
    'vx_real': [],
    'vx_hat': [],
}

print("[INFO] Collecting Asymmetry Trace across 200 steps...", flush=True)

for step in range(200):
    with torch.inference_mode():
        policy_module = runner.alg.policy
        cenet_out = policy_module.compute_cenet_outputs(obs, sample_latent=False)
        pred_vel = cenet_out.estimated_velocity
        
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)

    # 提取机器人本体坐标系状态
    r_quat = robot.data.root_quat_w[0].cpu().numpy()
    # quat to euler
    w, x, y, z = r_quat
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)) * 180.0 / np.pi
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0)) * 180.0 / np.pi
    
    # 精确获取四条腿的关节索引
    if not hasattr(robot, "_bilateral_joint_ids"):
        robot._fl_ids = [robot.find_joints("FL_hip_joint")[0][0], robot.find_joints("FL_thigh_joint")[0][0], robot.find_joints("FL_calf_joint")[0][0]]
        robot._fr_ids = [robot.find_joints("FR_hip_joint")[0][0], robot.find_joints("FR_thigh_joint")[0][0], robot.find_joints("FR_calf_joint")[0][0]]
        robot._rl_ids = [robot.find_joints("RL_hip_joint")[0][0], robot.find_joints("RL_thigh_joint")[0][0], robot.find_joints("RL_calf_joint")[0][0]]
        robot._rr_ids = [robot.find_joints("RR_hip_joint")[0][0], robot.find_joints("RR_thigh_joint")[0][0], robot.find_joints("RR_calf_joint")[0][0]]
        robot._bilateral_joint_ids = True

    # 提取关节功率
    tau = robot.data.applied_torque[0].cpu().numpy()
    dq = robot.data.joint_vel[0].cpu().numpy()
    p = np.abs(tau * dq)
    fl_p = np.sum(p[robot._fl_ids])
    fr_p = np.sum(p[robot._fr_ids])
    rl_p = np.sum(p[robot._rl_ids])
    rr_p = np.sum(p[robot._rr_ids])

    # 提取足端相对机身的高度（本体坐标系，消除俯仰角与侧倾角干扰）
    from isaaclab.utils.math import quat_apply_inverse
    foot_pos_rel_w = robot.data.body_pos_w[:, feet_body_idx, :] - robot.data.root_pos_w.unsqueeze(1)
    num_feet = foot_pos_rel_w.shape[1]
    root_quat_w = robot.data.root_quat_w.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_rel_w.reshape(-1, 3)).reshape_as(foot_pos_rel_w)[0].cpu().numpy()
    
    c_forces = sensor.data.net_forces_w[0, feet_sensor_idx].cpu().numpy()
    c_norm = np.linalg.norm(c_forces, axis=-1)

    trace['roll_deg'].append(roll)
    trace['pitch_deg'].append(pitch)
    trace['fl_power'].append(fl_p)
    trace['fr_power'].append(fr_p)
    trace['rl_power'].append(rl_p)
    trace['rr_power'].append(rr_p)
    trace['fl_z_rel'].append(foot_pos_b[0, 2])
    trace['fr_z_rel'].append(foot_pos_b[1, 2])
    trace['rl_z_rel'].append(foot_pos_b[2, 2])
    trace['rr_z_rel'].append(foot_pos_b[3, 2])
    trace['fl_contact'].append(int(c_norm[0] > 5.0))
    trace['fr_contact'].append(int(c_norm[1] > 5.0))
    trace['rl_contact'].append(int(c_norm[2] > 5.0))
    trace['rr_contact'].append(int(c_norm[3] > 5.0))
    trace['vx_real'].append(robot.data.root_lin_vel_b[0, 0].item())
    trace['vx_hat'].append(pred_vel[0, 0].item())

# 分析攀爬阶段 (step 60 到 180)
s_start, s_end = 60, 180
rolls = np.array(trace['roll_deg'][s_start:s_end])
fl_p = np.array(trace['fl_power'][s_start:s_end])
fr_p = np.array(trace['fr_power'][s_start:s_end])
rl_p = np.array(trace['rl_power'][s_start:s_end])
rr_p = np.array(trace['rr_power'][s_start:s_end])
fl_z = np.array(trace['fl_z_rel'][s_start:s_end])
fr_z = np.array(trace['fr_z_rel'][s_start:s_end])
rl_z = np.array(trace['rl_z_rel'][s_start:s_end])
rr_z = np.array(trace['rr_z_rel'][s_start:s_end])

print("\n================ ASYMMETRY DIAGNOSTIC SUMMARY ================")
print(f"Mean Roll: {np.mean(rolls):.2f}° | Max Roll: {np.max(rolls):.2f}° | Min Roll: {np.min(rolls):.2f}°")
print(f"FL Power: {np.mean(fl_p):.2f}W (Peak {np.max(fl_p):.2f}W)")
print(f"FR Power: {np.mean(fr_p):.2f}W (Peak {np.max(fr_p):.2f}W)")
print(f"RL Power: {np.mean(rl_p):.2f}W (Peak {np.max(rl_p):.2f}W)")
print(f"RR Power: {np.mean(rr_p):.2f}W (Peak {np.max(rr_p):.2f}W)")
print(f"Front Power Asymmetry Ratio (FR / FL): {np.mean(fr_p) / max(np.mean(fl_p), 1e-3):.2f}x")
print(f"Rear Power Asymmetry Ratio (RR / RL): {np.mean(rr_p) / max(np.mean(rl_p), 1e-3):.2f}x")
print(f"Total Right vs Left Ratio ((FR+RR)/(FL+RL)): {(np.mean(fr_p)+np.mean(rr_p)) / max(np.mean(fl_p)+np.mean(rl_p), 1e-3):.2f}x")
print(f"FR Swing Peak (Max Height rel to base): {np.max(fr_z):.3f}m | FL Swing Peak: {np.max(fl_z):.3f}m")
print(f"FR Swing Min (Lowest Stance rel to base): {np.min(fr_z):.3f}m | FL Swing Min: {np.min(fl_z):.3f}m")
print("===============================================================\n")

simulation_app.close()

