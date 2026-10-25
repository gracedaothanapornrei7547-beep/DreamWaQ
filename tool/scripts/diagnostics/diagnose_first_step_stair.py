import os
import sys
from pathlib import Path
import torch
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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
env = gym.make('Unitree-Go2-DreamWaQ-Rough-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

resume_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-07_15-22_stage6_agile_high_stepping_cadence/model_4946.pt'
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict["device"] = "cuda:0"

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device="cuda:0")
runner.load(resume_path)
policy = runner.get_inference_policy(device="cuda:0")

obs = env.get_observations()
print('Starting diagnostic rollout...', flush=True)

records = []
step_count = 100 # 2.0 seconds

for step in range(step_count):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)

    unwrapped = env.unwrapped
    robot = unwrapped.scene['robot']
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
    body_pos_w = robot.data.body_pos_w[0].cpu().numpy()
    body_lin_vel_w = robot.data.body_lin_vel_w[0].cpu().numpy()
    
    # 4 feet indices: FL: 4, FR: 8, RL: 12, RR: 16
    feet_pos = body_pos_w[[4, 8, 12, 16]]
    feet_vel = body_lin_vel_w[[4, 8, 12, 16]]
    
    scanner = unwrapped.scene.sensors['height_scanner']
    ray_hits_z = scanner.data.ray_hits_w[0, :, 2].cpu().numpy()
    finite_hits = ray_hits_z[np.isfinite(ray_hits_z)]
    local_ground_mean = float(np.mean(finite_hits)) if len(finite_hits) > 0 else 0.0
    local_ground_max = float(np.max(finite_hits)) if len(finite_hits) > 0 else 0.0

    r = {
        'step': step,
        't': step * 0.02,
        'x': float(root_pos[0]),
        'z': float(root_pos[2]),
        'vx': float(root_lin_vel[0]),
        'g_max': local_ground_max,
        'clearance_max': float(root_pos[2] - local_ground_max),
        'fl_z': float(feet_pos[0, 2]),
        'fr_z': float(feet_pos[1, 2]),
        'fl_vx': float(feet_vel[0, 0]),
        'fr_vx': float(feet_vel[1, 0]),
    }
    records.append(r)
    if step % 10 == 0:
        print(f"step {step:3d} (t={r['t']:.2f}s) | x={r['x']:.3f} | vx={r['vx']:.2f} | z={r['z']:.3f} | g_max={r['g_max']:.3f} | rel_h={r['clearance_max']:.3f} | FL_z={r['fl_z']:.3f} | FR_z={r['fr_z']:.3f}", flush=True)

print('\n' + '='*85, flush=True)
print('【IsaacLab 阶梯攀爬真实遥测诊断表】', flush=True)
print('='*85, flush=True)
print('| 时间(s) | 步数 | 机器狗X坐标 | 前向线速度Vx | 机身高度Z | 台阶最高面 | 相对净空(max) | FL足端Z | FR足端Z | FL足端Vx | FR足端Vx |', flush=True)
print('|---|---|---|---|---|---|---|---|---|---|---|', flush=True)
for r in records[::5]:
    print(f"| {r['t']:.2f}s | {r['step']:3d} | {r['x']:.3f}m | {r['vx']:.2f}m/s | {r['z']:.3f}m | {r['g_max']:.3f}m | {r['clearance_max']:.3f}m | {r['fl_z']:.3f}m | {r['fr_z']:.3f}m | {r['fl_vx']:.2f}m/s | {r['fr_vx']:.2f}m/s |", flush=True)

simulation_app.close()

