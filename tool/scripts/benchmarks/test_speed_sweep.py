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

for target_cmd in [0.8, 1.0, 1.2, 1.4]:
    # 强制改写速度指令
    env.unwrapped.command_manager.get_term('base_velocity').vel_command_b[:, 0] = target_cmd
    obs = env.get_observations()
    vxs = []
    for step in range(80):
        # 保持指令
        env.unwrapped.command_manager.get_term('base_velocity').vel_command_b[:, 0] = target_cmd
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            runner.alg.policy.reset(dones)
        robot = env.unwrapped.scene['robot']
        vx = float(robot.data.root_lin_vel_b[0, 0].cpu())
        vxs.append(vx)
    mean_vx = np.mean(vxs[20:])
    max_vx = np.max(vxs)
    print(f"[TEST SPEED] 指令 Cmd={target_cmd:.1f} m/s => 实测平均 Vx={mean_vx:.3f} m/s | 峰值 Vx={max_vx:.3f} m/s", flush=True)

simulation_app.close()

