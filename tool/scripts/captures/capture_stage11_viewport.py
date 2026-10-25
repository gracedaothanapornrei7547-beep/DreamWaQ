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
import mss

cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
cfg.scene.num_envs = 1
cfg.commands.base_velocity.ranges.lin_vel_x = (0.7, 0.8)
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

resume_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-08_15-35_stage11_smooth_trot_stair_refine/model_8297.pt'
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')

print('[INFO] Loaded Stage 11 (8300 iterations) smooth model. Setting camera view...', flush=True)

origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
env.unwrapped.sim.set_camera_view(
    eye=origin + np.array([-1.8, -1.8, 1.2]),
    target=origin + np.array([1.2, 0.0, 0.4])
)

obs = env.get_observations()
dt = env.unwrapped.step_dt
out_path = '/home/rum/.gemini/antigravity/brain/7b59cd11-515a-4ded-ad5e-413fbabe33fe/isaac_sim_stage11_8300_climbing_live.png'

with mss.mss(display=':0') as sct:
    for step_count in range(1, 160):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            runner.alg.policy.reset(dones)
        
        if step_count == 135:
            sct.shot(output=out_path)
            print(f'[INFO] Captured live Stage 11 climbing screenshot at step {step_count}: {out_path}', flush=True)

print('[INFO] Completed Stage 11 capture run.', flush=True)
simulation_app.close()

