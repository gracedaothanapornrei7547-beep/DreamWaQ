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

cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
cfg.scene.num_envs = 1

# 100% 12cm continuous ascending stairs
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

resume_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-07_19-52_stage8_rough_pure_waxberry_continue/model_6749.pt'
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')

obs = env.get_observations()
print('Running 220 steps with model_6749.pt on 12cm stairs...', flush=True)

records = []
initial_x = None

for step in range(220):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)

    unwrapped = env.unwrapped
    robot = unwrapped.scene['robot']
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
    body_pos_w = robot.data.body_pos_w[0].cpu().numpy()
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    
    if initial_x is None:
        initial_x = root_pos[0]
    
    rel_x = root_pos[0] - initial_x
    feet_pos = body_pos_w[[4, 8, 12, 16]]
    
    scanner = unwrapped.scene.sensors['height_scanner']
    ray_hits_z = scanner.data.ray_hits_w[0, :, 2].cpu().numpy()
    finite_hits = ray_hits_z[np.isfinite(ray_hits_z)]
    local_ground_max = float(np.max(finite_hits)) if len(finite_hits) > 0 else 0.0

    fl_thigh = float(joint_pos[1])
    fl_calf = float(joint_pos[2])
    fr_thigh = float(joint_pos[4])
    fr_calf = float(joint_pos[5])
    rl_thigh = float(joint_pos[7])
    rl_calf = float(joint_pos[8])
    rr_thigh = float(joint_pos[10])
    rr_calf = float(joint_pos[11])

    r = {
        'step': step,
        't': step * 0.02,
        'rel_x': float(rel_x),
        'base_z': float(root_pos[2]),
        'vx': float(root_lin_vel[0]),
        'vy': float(root_lin_vel[1]),
        'g_max': local_ground_max,
        'fl_z': float(feet_pos[0, 2]),
        'fr_z': float(feet_pos[1, 2]),
        'rl_z': float(feet_pos[2, 2]),
        'rr_z': float(feet_pos[3, 2]),
        'fl_thigh': fl_thigh,
        'fl_calf': fl_calf,
        'rl_thigh': rl_thigh,
        'rl_calf': rl_calf,
    }
    records.append(r)
    if step % 10 == 0:
        print(f"step {step:3d} (t={r['t']:.2f}s) | dx={r['rel_x']:.3f}m | vx={r['vx']:.2f}m/s | vy={r['vy']:+.2f}m/s | base_z={r['base_z']:.3f}m | g_max={r['g_max']:.3f}m | FL_z={r['fl_z']:.3f} | RL_z={r['rl_z']:.3f}", flush=True)

simulation_app.close()

print('\n' + '='*100, flush=True)
print('【Stage 8 纯净架构 model_6749 (深度训练 1500 轮) 12cm 台阶遥测分析报告】', flush=True)
print('='*100, flush=True)
for r in records[::10]:
    print(f"| {r['t']:.2f}s | step {r['step']:3d} | 位移dx={r['rel_x']:.3f}m | Vx={r['vx']:.2f}m/s | Vy={r['vy']:+.2f}m/s | base_z={r['base_z']:.3f}m | g_max={r['g_max']:.3f}m | FL_z={r['fl_z']:.3f}m | RL_z={r['rl_z']:.3f}m |", flush=True)

# 汇总指标
vxs = [r['vx'] for r in records]
base_zs = [r['base_z'] for r in records]
print(f"\n[SUMMARY] 平均前进速度: {np.mean(vxs):.3f} m/s | 最终爬坡高度 base_z: {base_zs[-1]:.3f} m (初始 {base_zs[0]:.3f} m, 累计攀爬 +{base_zs[-1]-base_zs[0]:.3f} m)")
print(f"[SUMMARY] 总位移: {records[-1]['rel_x']:.3f} m / 4.40s")

