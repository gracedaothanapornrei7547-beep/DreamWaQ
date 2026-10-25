from isaaclab.app import AppLauncher
app = AppLauncher({'headless': True, 'device': 'cuda:0'}).app

import os
import sys
import glob
from pathlib import Path
import numpy as np
import torch

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
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.dreamwaq_env_cfg import UnitreeGo2DreamWaQFlatEnvCfg_PLAY
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.agents.rsl_rl_dreamwaq_cfg import UnitreeGo2DreamWaQFlatRunnerCfg

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

def run_probe(checkpoint_path):
    print(f"[PROBE] Loading checkpoint: {checkpoint_path}")
    cfg = UnitreeGo2DreamWaQFlatEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.viewer.origin_type = "world"
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    env = gym.make('Unitree-Go2-DreamWaQ-Flat-Play', cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = UnitreeGo2DreamWaQFlatRunnerCfg()
    agent_cfg_dict = runner_cfg.to_dict()
    agent_cfg_dict['device'] = 'cuda:0'

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device='cuda:0')

    obs, _ = env.reset()

    pos_list, rpy_list, joints_list = [], [], []
    real_vx_list, est_vx_list = [], []

    for step in range(300):
        with torch.no_grad():
            actions = policy(obs)
            # 提取 CENet 显式估计速度 (前 3 维为 estimated lin_vel)
            try:
                cenet_out = runner.alg.policy.compute_cenet_outputs(obs, sample_latent=False)
                est_v = cenet_out.estimated_velocity[0].cpu().numpy()
                est_vx_list.append(est_v[0])
            except Exception as e:
                est_vx_list.append(0.0)

            obs, _, _, _ = env.step(actions)

        robot = env.unwrapped.scene['robot']
        pos = robot.data.root_pos_w[0].cpu().numpy()
        quat = robot.data.root_quat_w[0].cpu().numpy()
        ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy()
        lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
        real_vx_list.append(lin_vel[0])

        r, p, y = quat_to_rpy_deg(quat)
        j = robot.data.joint_pos[0].cpu().numpy()

        joints_list.append(j)
        pos_list.append(pos)
        rpy_list.append([r, p, y, ang_vel[0], ang_vel[1], ang_vel[2]])

    env.close()

    pos_arr = np.array(pos_list)
    rpy_arr = np.array(rpy_list)
    j_arr = np.array(joints_list)
    real_vx_arr = np.array(real_vx_list)
    est_vx_arr = np.array(est_vx_list)

    delta_x = pos_arr[-1, 0] - pos_arr[0, 0]
    delta_y = pos_arr[-1, 1] - pos_arr[0, 1]
    drift_rate = (delta_y / delta_x * 100) if abs(delta_x) > 1e-3 else 0.0

    fl_thigh = np.mean(j_arr[:, 4]) * 180 / np.pi
    fr_thigh = np.mean(j_arr[:, 5]) * 180 / np.pi
    rl_thigh = np.mean(j_arr[:, 6]) * 180 / np.pi
    rr_thigh = np.mean(j_arr[:, 7]) * 180 / np.pi

    fl_calf = np.mean(j_arr[:, 8]) * 180 / np.pi
    fr_calf = np.mean(j_arr[:, 9]) * 180 / np.pi
    rl_calf = np.mean(j_arr[:, 10]) * 180 / np.pi
    rr_calf = np.mean(j_arr[:, 11]) * 180 / np.pi

    roll_amp = np.ptp(rpy_arr[:, 0])
    pitch_amp = np.ptp(rpy_arr[:, 1])
    yaw_amp = np.ptp(rpy_arr[:, 2])

    print("\n================== 第 300 轮物理探针诊断报告 ==================")
    print(f"1. 航向位移: 前向 Delta_X = {delta_x:+.3f}m, 侧偏 Delta_Y = {delta_y:+.3f}m, 漂移率 = {drift_rate:+.2f}%")
    print(f"2. 真实前向速度: Mean v_x = {np.mean(real_vx_arr):+.3f} m/s (Min: {np.min(real_vx_arr):.2f}, Max: {np.max(real_vx_arr):.2f})")
    print(f"3. CENet 速度感知: Mean v_hat_x = {np.mean(est_vx_arr):+.3f} m/s | 符号一致性: {'同向正常 (无倒挂)' if np.mean(real_vx_arr)*np.mean(est_vx_arr)>0 else '倒挂异常'}")
    print(f"4. 大腿摆角对称性:")
    print(f"   - 前腿: FL = {fl_thigh:+.1f}°, FR = {fr_thigh:+.1f}° (偏差: {abs(fl_thigh - fr_thigh):.1f}°)")
    print(f"   - 后腿: RL = {rl_thigh:+.1f}°, RR = {rr_thigh:+.1f}° (偏差: {abs(rl_thigh - rr_thigh):.1f}°)")
    print(f"5. 小腿支撑角对称性:")
    print(f"   - 前腿: FL = {fl_calf:+.1f}°, FR = {fr_calf:+.1f}° (偏差: {abs(fl_calf - fr_calf):.1f}°)")
    print(f"   - 后腿: RL = {rl_calf:+.1f}°, RR = {rr_calf:+.1f}° (偏差: {abs(rl_calf - rr_calf):.1f}°)")
    print(f"6. 机身晃动峰峰值:")
    print(f"   - Roll 横滚晃动: {roll_amp:.2f}°")
    print(f"   - Pitch 俯仰起伏: {pitch_amp:.2f}°")
    print(f"   - Yaw 偏航摆动: {yaw_amp:.2f}°")
    print("============================================================\n")

if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-09_17-11_pure_academic_minimal_locks_2000iter/model_300.pt'
    run_probe(ckpt)

