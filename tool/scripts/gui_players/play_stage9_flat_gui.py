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
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.dreamwaq_env_cfg import UnitreeGo2DreamWaQFlatEnvCfg_PLAY
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.agents.rsl_rl_dreamwaq_cfg import UnitreeGo2DreamWaQRoughRunnerCfg

cfg = UnitreeGo2DreamWaQFlatEnvCfg_PLAY()
cfg.scene.num_envs = 1

env = gym.make('Unitree-Go2-DreamWaQ-Flat-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

resume_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-07_19-04_stage9_finetune_trot_800/model_2298.pt'
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')

print('[INFO] Successfully loaded model_2298.pt. Initializing camera view on robot...', flush=True)

origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
env.unwrapped.sim.set_camera_view(
    eye=origin + np.array([-1.8, -1.8, 0.9]),
    target=origin + np.array([0.8, 0.0, 0.3])
)

obs, _ = env.reset()
step_count = 0
dt = env.unwrapped.step_dt
captured = False

print('[INFO] Interactive 3D GUI is now LIVE on DISPLAY=:0 (Stage 9 Flat Trot)!', flush=True)

while simulation_app.is_running():
    start_t = time.time()
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        runner.alg.policy.reset(dones)
    
    step_count += 1
    if step_count % 30 == 0:
        robot = env.unwrapped.scene['robot']
        rx = float(robot.data.root_pos_w[0, 0].cpu())
        rz = float(robot.data.root_pos_w[0, 2].cpu())
        vx = float(robot.data.root_lin_vel_b[0, 0].cpu())
        print(f"[Stage 9 Trot Step {step_count:4d}] x={rx:+.3f}m | z={rz:+.3f}m | vx={vx:+.2f}m/s", flush=True)

    if step_count == 80 and not captured:
        captured = True
        try:
            import mss
            with mss.mss(display=':0') as sct:
                out = '/home/rum/.gemini/antigravity/brain/7b59cd11-515a-4ded-ad5e-413fbabe33fe/isaac_sim_stage9_flat_trot_unlocked.png'
                sct.shot(output=out)
                print(f'[INFO] Stage 9 snapshot saved to: {out}', flush=True)
        except Exception as e:
            print(f'[WARN] Failed to capture snapshot: {e}', flush=True)

    elapsed = time.time() - start_t
    if dt > elapsed:
        time.sleep(dt - elapsed)

simulation_app.close()

