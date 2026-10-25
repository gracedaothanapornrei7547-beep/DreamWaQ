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

# Headless mode for deep offline telemetry extraction
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

cfg = UnitreeGo2DreamWaQFlatEnvCfg_PLAY()
cfg.scene.num_envs = 1
cfg.scene.height_scanner.debug_vis = False
cfg.viewer.origin_type = "world"

cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

env = gym.make('Unitree-Go2-DreamWaQ-Flat-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

# Find latest checkpoint
log_base = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat'
runs = sorted([d for d in os.listdir(log_base) if 'waxberry_golden_stage1' in d])
latest_run = os.path.join(log_base, runs[-1])
resume_path = os.path.join(latest_run, 'model_850.pt')
if not os.path.exists(resume_path):
    resume_path = os.path.join(latest_run, 'model_800.pt')

runner_cfg = UnitreeGo2DreamWaQFlatRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')
policy_nn = runner.alg.policy

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']
sensor = env.unwrapped.scene.sensors['contact_forces']
foot_ids, _ = sensor.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])

history_data = {
    'time': [],
    'pos_x': [], 'pos_y': [], 'pos_z': [],
    'roll': [], 'pitch': [], 'yaw': [],
    'v_x_b': [], 'v_y_b': [], 'v_z_b': [],
    'omega_z_b': [],
    'v_x_hat': [], 'v_y_hat': [],
    'contacts': [],
    'joint_pos': [],
    'actions': [],
}

print(f"Starting 200-step trajectory recording with checkpoint: {resume_path}", flush=True)

for step in range(200):
    with torch.inference_mode():
        cenet_out = policy_nn.compute_cenet_outputs(obs, sample_latent=False)
        v_hat = cenet_out.estimated_velocity[0].cpu().numpy()
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        policy_nn.reset(dones)

    pos = robot.data.root_pos_w[0].cpu().numpy()
    quat = robot.data.root_quat_w[0].cpu().numpy()
    r, p, y = quat_to_rpy_deg(quat)
    vel_b = robot.data.root_lin_vel_b[0].cpu().numpy()
    ang_vel_b = robot.data.root_ang_vel_b[0].cpu().numpy()
    jpos = robot.data.joint_pos[0].cpu().numpy()
    
    c_forces = sensor.data.net_forces_w[0, foot_ids].cpu().numpy()
    c_norm = np.linalg.norm(c_forces, axis=-1)
    in_contact = c_norm > 5.0

    history_data['time'].append(step * 0.02)
    history_data['pos_x'].append(pos[0])
    history_data['pos_y'].append(pos[1])
    history_data['pos_z'].append(pos[2])
    history_data['roll'].append(r)
    history_data['pitch'].append(p)
    history_data['yaw'].append(y)
    history_data['v_x_b'].append(vel_b[0])
    history_data['v_y_b'].append(vel_b[1])
    history_data['v_z_b'].append(vel_b[2])
    history_data['omega_z_b'].append(ang_vel_b[2])
    history_data['v_x_hat'].append(v_hat[0])
    history_data['v_y_hat'].append(v_hat[1])
    history_data['contacts'].append(in_contact)
    history_data['joint_pos'].append(jpos)
    history_data['actions'].append(actions[0].cpu().numpy())

# Convert to arrays
for k in history_data:
    history_data[k] = np.array(history_data[k])

contacts = history_data['contacts']  # shape: (200, 4) - [FL, FR, RL, RR]
jpos = history_data['joint_pos']      # shape: (200, 12)

print("\n" + "="*80)
print("ISAAC LAB 深度物理诊断报告 (200 步时序提取)")
print("="*80)

print("\n1. 运动学与轨迹偏航分析:")
dx = history_data['pos_x'][-1] - history_data['pos_x'][0]
dy = history_data['pos_y'][-1] - history_data['pos_y'][0]
total_dist = np.sqrt(dx**2 + dy**2)
yaw_start = history_data['yaw'][0]
yaw_end = history_data['yaw'][-1]
yaw_drift = yaw_end - yaw_start
print(f"  前向净位移 dx: {dx:+.2f} m | 侧向净位移 dy: {dy:+.2f} m (净偏斜角: {np.arctan2(dy, dx)*180/np.pi:+.1f}°)")
print(f"  航向角变化: {yaw_start:+.1f}° -> {yaw_end:+.1f}° (净偏转: {yaw_drift:+.1f}°, 偏转角速度均值: {np.mean(history_data['omega_z_b'])*180/np.pi:+.2f}°/s)")
print(f"  平均前进速度 vx: {np.mean(history_data['v_x_b']):.2f} m/s (指令 0.80 m/s)")
print(f"  平均横向速度 vy: {np.mean(history_data['v_y_b']):.3f} m/s (指令 0.00 m/s)")
print(f"  平均机身高度 z:  {np.mean(history_data['pos_z']):.3f} m")

print("\n2. CENet 速度感知自监督误差:")
err_vx = history_data['v_x_hat'] - history_data['v_x_b']
err_vy = history_data['v_y_hat'] - history_data['v_y_b']
print(f"  v_x 真实值均值: {np.mean(history_data['v_x_b']):.3f} m/s | CENet估计: {np.mean(history_data['v_x_hat']):.3f} m/s | 偏差: {np.mean(err_vx):+.3f} m/s (RMSE: {np.sqrt(np.mean(err_vx**2)):.3f})")
print(f"  v_y 真实值均值: {np.mean(history_data['v_y_b']):.3f} m/s | CENet估计: {np.mean(history_data['v_y_hat']):.3f} m/s | 偏差: {np.mean(err_vy):+.3f} m/s")

print("\n3. 四足接触相位与步态分解:")
# contacts: FL, FR, RL, RR
fl_duty = np.mean(contacts[:, 0]) * 100
fr_duty = np.mean(contacts[:, 1]) * 100
rl_duty = np.mean(contacts[:, 2]) * 100
rr_duty = np.mean(contacts[:, 3]) * 100
print(f"  四腿触地占空比 (Duty Cycle): FL={fl_duty:.1f}% | FR={fr_duty:.1f}% | RL={rl_duty:.1f}% | RR={rr_duty:.1f}%")

# 相位分类
pure_trot_cnt = 0
bound_cnt = 0
pace_cnt = 0
all_ground_cnt = 0
all_air_cnt = 0
diag1_cnt = 0  # FL + RR
diag2_cnt = 0  # FR + RL
other_cnt = 0

for t in range(len(contacts)):
    fl, fr, rl, rr = contacts[t]
    n_contact = fl + fr + rl + rr
    if n_contact == 4:
        all_ground_cnt += 1
    elif n_contact == 0:
        all_air_cnt += 1
    elif (fl and rr) and (not fr and not rl):
        diag1_cnt += 1
        pure_trot_cnt += 1
    elif (fr and rl) and (not fl and not rr):
        diag2_cnt += 1
        pure_trot_cnt += 1
    elif (fl and fr) and (not rl and not rr):
        bound_cnt += 1
    elif (rl and rr) and (not fl and not fr):
        bound_cnt += 1
    elif (fl and rl) and (not fr and not rr):
        pace_cnt += 1
    elif (fr and rr) and (not fl and not rl):
        pace_cnt += 1
    else:
        other_cnt += 1

total_steps = len(contacts)
print(f"  纯对角 Trot 相位: {pure_trot_cnt}/{total_steps} ({pure_trot_cnt/total_steps*100:.1f}%) [FL+RR: {diag1_cnt}, FR+RL: {diag2_cnt}]")
print(f"  Bound 兔跳相位:   {bound_cnt}/{total_steps} ({bound_cnt/total_steps*100:.1f}%)")
print(f"  Pace 溜蹄同侧:    {pace_cnt}/{total_steps} ({pace_cnt/total_steps*100:.1f}%)")
print(f"  四足全触地相位:   {all_ground_cnt}/{total_steps} ({all_ground_cnt/total_steps*100:.1f}%)")
print(f"  四足全腾空相位:   {all_air_cnt}/{total_steps} ({all_air_cnt/total_steps*100:.1f}%)")
print(f"  其他三足/杂合相位: {other_cnt}/{total_steps} ({other_cnt/total_steps*100:.1f}%)")

print("\n4. 关节运动学统计与左右对称性 (单位: 度):")
# 关节:
# 0: FL_hip,   1: FR_hip,   2: RL_hip,   3: RR_hip
# 4: FL_thigh, 5: FR_thigh, 6: RL_thigh, 7: RR_thigh
# 8: FL_calf,  9: FR_calf, 10: RL_calf, 11: RR_calf
j_deg = jpos * 180.0 / np.pi
j_mean = np.mean(j_deg, axis=0)
j_std = np.std(j_deg, axis=0)

print("  关节平均角度 (Mean ± Std):")
print(f"    髋关节 (Hip Roll) : FL={j_mean[0]:+5.1f}±{j_std[0]:.1f}° | FR={j_mean[1]:+5.1f}±{j_std[1]:.1f}° | RL={j_mean[2]:+5.1f}±{j_std[2]:.1f}° | RR={j_mean[3]:+5.1f}±{j_std[3]:.1f}°")
print(f"    大腿   (Thigh Pitch): FL={j_mean[4]:+5.1f}±{j_std[4]:.1f}° | FR={j_mean[5]:+5.1f}±{j_std[5]:.1f}° | RL={j_mean[6]:+5.1f}±{j_std[6]:.1f}° | RR={j_mean[7]:+5.1f}±{j_std[7]:.1f}°")
print(f"    小腿   (Calf Pitch) : FL={j_mean[8]:+5.1f}±{j_std[8]:.1f}° | FR={j_mean[9]:+5.1f}±{j_std[9]:.1f}° | RL={j_mean[10]:+5.1f}±{j_std[10]:.1f}° | RR={j_mean[11]:+5.1f}±{j_std[11]:.1f}°")

print("\n5. 对称性失衡指标:")
print(f"    前肢大腿左右摆幅 (ROM): FL={np.ptp(j_deg[:, 4]):.1f}° vs FR={np.ptp(j_deg[:, 5]):.1f}°")
print(f"    后肢大腿左右摆幅 (ROM): RL={np.ptp(j_deg[:, 6]):.1f}° vs RR={np.ptp(j_deg[:, 7]):.1f}°")
print(f"    前肢左右髋对称和 (|FL_hip + FR_hip|): {abs(j_mean[0] + j_mean[1]):.2f}°")
print(f"    后肢左右髋对称和 (|RL_hip + RR_hip|): {abs(j_mean[2] + j_mean[3]):.2f}°")
print(f"    对角大腿平均差异: |FL_thigh - RR_thigh|={abs(j_mean[4] - j_mean[7]):.2f}° | |FR_thigh - RL_thigh|={abs(j_mean[5] - j_mean[6]):.2f}°")

print("\n" + "="*80)

# Save arrays to a npz file for detailed plotting / inspection
np.savez('/home/rum/dreamwaq_lab/scripts/gait_inspection_data.npz', **history_data)
print("Trajectory data saved to /home/rum/dreamwaq_lab/scripts/gait_inspection_data.npz", flush=True)

simulation_app.close()
os._exit(0)

