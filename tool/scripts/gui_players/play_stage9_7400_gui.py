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

os.environ['DISPLAY'] = ':0'
app_launcher = AppLauncher({'headless': False, 'device': 'cuda:0'})
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
cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.0)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

# 12cm 连续楼梯考验（中央 3m 平台平地起步 -> 连续踏入 12cm 金字塔台阶）
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

resume_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-08_12-07_stage9_flat_stair_coexistence_continue/model_7399.pt'
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')

print('[INFO] Successfully loaded model_7399.pt (7400 iterations). Initializing camera view...', flush=True)

# 初始视角：位于出生点后侧偏上，对准中央平地平台与台阶入口
origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
env.unwrapped.sim.set_camera_view(
    eye=origin + np.array([-1.8, -1.8, 1.2]),
    target=origin + np.array([1.2, 0.0, 0.4])
)

obs = env.get_observations()
step_count = 0
dt = env.unwrapped.step_dt

print('[INFO] Interactive 3D GUI is now LIVE on DISPLAY=:0 (Free Camera Navigation Unlocked)!', flush=True)

while simulation_app.is_running():
    start_t = time.time()
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)
    
    step_count += 1
    if step_count % 50 == 0:
        robot = env.unwrapped.scene['robot']
        rx = float(robot.data.root_pos_w[0, 0].cpu())
        rz = float(robot.data.root_pos_w[0, 2].cpu())
        vx = float(robot.data.root_lin_vel_b[0, 0].cpu())
        pitch = float(np.degrees(robot.data.root_rpy_b[0, 1].cpu().numpy() if hasattr(robot.data, 'root_rpy_b') else 0.0))
        print(f"[3D Play Step {step_count:4d}] x={rx:+.3f}m | z={rz:+.3f}m | vx={vx:+.2f}m/s", flush=True)

    elapsed = time.time() - start_t
    if dt > elapsed:
        time.sleep(dt - elapsed)

simulation_app.close()

