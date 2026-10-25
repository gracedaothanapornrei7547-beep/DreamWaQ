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

# Headless run for fast accurate numerical profiling
app_launcher = AppLauncher({'headless': True, 'device': 'cuda:0'})
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.dreamwaq_env_cfg import UnitreeGo2DreamWaQRoughEnvCfg_PLAY
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.agents.rsl_rl_dreamwaq_cfg import UnitreeGo2DreamWaQRoughRunnerCfg
def quat_to_rpy_deg(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll * 180.0 / np.pi, pitch * 180.0 / np.pi, yaw * 180.0 / np.pi

def test_checkpoint(checkpoint_path, num_steps=200):
    cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    env = gym.make('Unitree-Go2-DreamWaQ-Rough-Play', cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
    agent_cfg_dict = runner_cfg.to_dict()
    agent_cfg_dict['device'] = 'cuda:0'

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device='cuda:0')

    obs, _ = env.reset()
    roll_list, pitch_list, yaw_list = [], [], []
    vy_list, vx_list = [], []
    rl_hip_list, rr_hip_list = [], []

    for _ in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            runner.alg.policy.reset(dones)

        robot = env.unwrapped.scene['robot']
        quat = robot.data.root_quat_w[0].cpu().numpy()
        r_deg, p_deg, y_deg = quat_to_rpy_deg(quat)
        roll_list.append(r_deg)
        pitch_list.append(p_deg)
        yaw_list.append(y_deg)

        vx = float(robot.data.root_lin_vel_b[0, 0].cpu())
        vy = float(robot.data.root_lin_vel_b[0, 1].cpu())
        vx_list.append(vx)
        vy_list.append(vy)

        joint_pos = robot.data.joint_pos[0].cpu().numpy()
        rl_hip_list.append(joint_pos[6] * 180.0 / np.pi)
        rr_hip_list.append(joint_pos[9] * 180.0 / np.pi)

    env.close()

    roll_arr = np.array(roll_list)
    yaw_arr = np.array(yaw_list)
    vy_arr = np.array(vy_list)
    rl_hip_arr = np.array(rl_hip_list)
    rr_hip_arr = np.array(rr_hip_list)

    roll_peak_to_peak = np.ptp(roll_arr)
    yaw_drift = abs(yaw_arr[-1] - yaw_arr[0])
    vy_mean = np.mean(vy_arr)
    rl_hip_mean = np.mean(rl_hip_arr)
    rr_hip_mean = np.mean(rr_hip_arr)

    return {
        'checkpoint': checkpoint_path,
        'roll_p2p': roll_peak_to_peak,
        'yaw_drift': yaw_drift,
        'vy_mean': vy_mean,
        'rl_hip_mean': rl_hip_mean,
        'rr_hip_mean': rr_hip_mean,
    }

if __name__ == '__main__':
    log_dir = Path('/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-09_12-13_finetune_300iter_antidrift_stabilized')
    checkpoints = sorted(log_dir.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]) if p.stem.split('_')[1].isdigit() else -1)
    if checkpoints:
        latest_ckpt = str(checkpoints[-1])
        print(f"Testing {latest_ckpt}...")
        res = test_checkpoint(latest_ckpt)
        print(f"RESULTS: {res}")
    simulation_app.close()
