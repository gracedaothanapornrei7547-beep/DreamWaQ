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

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({'headless': True, 'device': 'cuda:0'})
simulation_app = app_launcher.app

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

def run_diagnostic(ckpt_path):
    print("="*80)
    print(f"DIAGNOSING MILESTONE CHECKPOINT: {ckpt_path}")
    print("="*80)

    cfg = UnitreeGo2DreamWaQFlatEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.events.reset_base.params['pose_range'] = {'x': (0.0, 0.0), 'y': (0.0, 0.0), 'yaw': (0.0, 0.0)}

    env = gym.make('Unitree-Go2-DreamWaQ-Flat-Play', cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, UnitreeGo2DreamWaQFlatRunnerCfg().to_dict(), log_dir=None, device='cuda:0')
    runner.load(ckpt_path)
    policy = runner.alg.policy

    obs, _ = env.reset()
    robot = env.unwrapped.scene['robot']

    v_true_list, v_hat_list = [], []
    pos_list, rpy_list = [], []
    jpos_list = []
    contact_list = []

    for step in range(150):
        with torch.inference_mode():
            cenet_out = policy.compute_cenet_outputs(obs, sample_latent=False)
            v_hat = cenet_out.estimated_velocity[0].cpu().numpy()
            actions = policy.act_inference(obs)
            obs, _, _, _ = env.step(actions)

        v_true = robot.data.root_lin_vel_b[0].cpu().numpy()
        pos = robot.data.root_pos_w[0].cpu().numpy()
        quat = robot.data.root_quat_w[0].cpu().numpy()
        rpy = quat_to_rpy_deg(quat)
        jpos = robot.data.joint_pos[0].cpu().numpy()

        sensor = env.unwrapped.scene.sensors['contact_forces']
        if not hasattr(env.unwrapped, '_foot_body_ids'):
            env.unwrapped._foot_body_ids, _ = sensor.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
        c_forces = sensor.data.net_forces_w[0, env.unwrapped._foot_body_ids].cpu().numpy()
        c_norm = np.linalg.norm(c_forces, axis=-1)

        v_true_list.append(v_true)
        v_hat_list.append(v_hat)
        pos_list.append(pos)
        rpy_list.append(rpy)
        jpos_list.append(jpos)
        contact_list.append(c_norm > 5.0)

    v_true_arr = np.array(v_true_list)
    v_hat_arr = np.array(v_hat_list)
    pos_arr = np.array(pos_list)
    rpy_arr = np.array(rpy_list)
    jpos_arr = np.array(jpos_list)
    contacts = np.array(contact_list)

    print("\n--- 1. CENet 速度感知精度 ---")
    v_err = v_hat_arr - v_true_arr
    print(f"   v_x 物理真值: {np.mean(v_true_arr[:, 0]):.3f} m/s | CENet估计值: {np.mean(v_hat_arr[:, 0]):.3f} m/s | 偏差: {np.mean(v_err[:, 0]):+.3f} m/s (RMSE: {np.sqrt(np.mean(v_err[:, 0]**2)):.3f})")
    print(f"   v_y 物理真值: {np.mean(v_true_arr[:, 1]):.3f} m/s | CENet估计值: {np.mean(v_hat_arr[:, 1]):.3f} m/s | 偏差: {np.mean(v_err[:, 1]):+.3f} m/s")

    print("\n--- 2. 空间巡航笔直度与姿态平衡 ---")
    dx = pos_arr[-1, 0] - pos_arr[0, 0]
    dy = pos_arr[-1, 1] - pos_arr[0, 1]
    drift_deg = np.arctan2(dy, max(dx, 1e-5)) * 180 / np.pi
    print(f"   前向位移: {dx:+.3f}m | 横向侧漂: {dy:+.3f}m | 轨迹偏斜角: {drift_deg:+.2f}°")
    print(f"   Roll  横滚角: 均值={np.mean(rpy_arr[:, 0]):+.2f}° | 标准差={np.std(rpy_arr[:, 0]):.2f}° | 峰峰值={np.ptp(rpy_arr[:, 0]):.2f}°")
    print(f"   Pitch 俯仰角: 均值={np.mean(rpy_arr[:, 1]):+.2f}° | 标准差={np.std(rpy_arr[:, 1]):.2f}° | 峰峰值={np.ptp(rpy_arr[:, 1]):.2f}°")
    print(f"   Yaw   航向漂移: 起始={rpy_arr[0, 2]:+.2f}° -> 终点={rpy_arr[-1, 2]:+.2f}° (净转角={rpy_arr[-1, 2]-rpy_arr[0, 2]:+.2f}°)")

    print("\n--- 3. 步态类型判定与接触相位分解 ---")
    trot_cnt, bound_cnt, pace_cnt, other_cnt = 0, 0, 0, 0
    for s in range(len(contacts)):
        fl, fr, rl, rr = [int(x) for x in contacts[s]]
        if (fl and rr and not fr and not rl) or (fr and rl and not fl and not rr):
            trot_cnt += 1
        elif (fl and fr and not rl and not rr) or (rl and rr and not fl and not fr):
            bound_cnt += 1
        elif (fl and rl and not fr and not rr) or (fr and rr and not fl and not rl):
            pace_cnt += 1
        else:
            other_cnt += 1

    total_steps = len(contacts)
    print(f"   纯正 Trot (对角交替踏步): {trot_cnt}/{total_steps} ({trot_cnt/total_steps*100:.1f}%)")
    print(f"   Bound     (兔跳同迈并联): {bound_cnt}/{total_steps} ({bound_cnt/total_steps*100:.1f}%)")
    print(f"   Pace      (溜蹄侧向同迈): {pace_cnt}/{total_steps} ({pace_cnt/total_steps*100:.1f}%)")
    print(f"   其他/双支撑/全悬空相位  : {other_cnt}/{total_steps} ({other_cnt/total_steps*100:.1f}%)")

    print("\n--- 4. 四肢关节对称性检验 (L2 差值) ---")
    # 0: FL_hip, 1: FR_hip, 2: RL_hip, 3: RR_hip
    # 4: FL_thigh, 5: FR_thigh, 6: RL_thigh, 7: RR_thigh
    # 8: FL_calf, 9: FR_calf, 10: RL_calf, 11: RR_calf
    mean_jpos = np.mean(jpos_arr, axis=0)
    diff_thigh_diag1 = abs(mean_jpos[4] - mean_jpos[7]) * 180 / np.pi  # FL vs RR
    diff_thigh_diag2 = abs(mean_jpos[5] - mean_jpos[6]) * 180 / np.pi  # FR vs RL
    diff_hip_lr1 = abs(mean_jpos[0] + mean_jpos[1]) * 180 / np.pi      # FL + FR
    diff_hip_lr2 = abs(mean_jpos[2] + mean_jpos[3]) * 180 / np.pi      # RL + RR
    print(f"   对角大腿对称差 (FL vs RR): {diff_thigh_diag1:.2f}°")
    print(f"   对角大腿对称差 (FR vs RL): {diff_thigh_diag2:.2f}°")
    print(f"   前肢左右髋对称差 (|FL+FR|): {diff_hip_lr1:.2f}°")
    print(f"   后肢左右髋对称差 (|RL+RR|): {diff_hip_lr2:.2f}°")

if __name__ == '__main__':
    log_base = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat'
    runs = sorted([d for d in os.listdir(log_base) if 'waxberry_golden_stage1' in d])
    if not runs:
        print('No waxberry run directory found!')
        sys.exit(1)
    latest_run = os.path.join(log_base, runs[-1])
    ckpt = os.path.join(latest_run, 'model_300.pt')
    if not os.path.exists(ckpt):
        # 寻找目前已存的最大编号检查点
        ckpts = sorted([f for f in os.listdir(latest_run) if f.startswith('model_') and f.endswith('.pt')])
        if ckpts:
            ckpt = os.path.join(latest_run, ckpts[-1])
            print(f'model_300.pt not found yet, diagnosing latest: {ckpts[-1]}')
        else:
            print(f'No checkpoint found in {latest_run}')
            sys.exit(1)
    run_diagnostic(ckpt)
    simulation_app.close()

