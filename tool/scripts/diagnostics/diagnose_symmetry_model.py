from isaaclab.app import AppLauncher
app = AppLauncher({'headless': True, 'device': 'cuda:0'}).app

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
cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

env = gym.make('Unitree-Go2-DreamWaQ-Flat-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

checkpoint_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-09_16-36_finetune_boost_thrust_300iter/model_1850.pt'
runner_cfg = UnitreeGo2DreamWaQFlatRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(checkpoint_path)
policy = runner.get_inference_policy(device='cuda:0')

obs, _ = env.reset()

pos_list, rpy_list, joints_list = [], [], []

for step in range(300):
    with torch.no_grad():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        
    robot = env.unwrapped.scene['robot']
    pos = robot.data.root_pos_w[0].cpu().numpy()
    quat = robot.data.root_quat_w[0].cpu().numpy()
    ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy()
    r, p, y = quat_to_rpy_deg(quat)
    
    j = robot.data.joint_pos[0].cpu().numpy()
    joints_list.append(j)
    pos_list.append(pos)
    rpy_list.append([r, p, y, ang_vel[0], ang_vel[1], ang_vel[2]])

pos_arr = np.array(pos_list)
rpy_arr = np.array(rpy_list)
j_arr = np.array(joints_list)

delta_x = pos_arr[-1, 0] - pos_arr[0, 0]
delta_y = pos_arr[-1, 1] - pos_arr[0, 1]
drift_rate = delta_y / delta_x * 100

fl_hip = np.mean(j_arr[:, 0])*180/np.pi
fr_hip = np.mean(j_arr[:, 1])*180/np.pi
rl_hip = np.mean(j_arr[:, 2])*180/np.pi
rr_hip = np.mean(j_arr[:, 3])*180/np.pi

fl_thigh = np.mean(j_arr[:, 4])*180/np.pi
fr_thigh = np.mean(j_arr[:, 5])*180/np.pi
rl_thigh = np.mean(j_arr[:, 6])*180/np.pi
rr_thigh = np.mean(j_arr[:, 7])*180/np.pi

fl_calf = np.mean(j_arr[:, 8])*180/np.pi
fr_calf = np.mean(j_arr[:, 9])*180/np.pi
rl_calf = np.mean(j_arr[:, 10])*180/np.pi
rr_calf = np.mean(j_arr[:, 11])*180/np.pi

print('\n================== 对称性微调后物理探针指标 ==================')
print(f'1. 前向位移: Delta_X = {delta_x:.3f}m, 侧偏: Delta_Y = {delta_y:.3f}m')
print(f'   横向漂移率: {drift_rate:+.2f}% (微调前为 +11.72%)')
print(f'   偏航角 Yaw: 起点={rpy_arr[0, 2]:.2f}°, 终点={rpy_arr[-1, 2]:.2f}°, 全程极差={np.ptp(rpy_arr[:, 2]):.2f}°')

print(f'\n2. 机身姿态晃动:')
print(f'   横滚 Roll: 均值={np.mean(rpy_arr[:, 0]):.2f}°, 峰峰值振幅={np.ptp(rpy_arr[:, 0]):.2f}°, 标准差={np.std(rpy_arr[:, 0]):.2f}° (微调前峰峰值 4.96°)')
print(f'   俯仰 Pitch: 均值={np.mean(rpy_arr[:, 1]):.2f}°, 峰峰值振幅={np.ptp(rpy_arr[:, 1]):.2f}°, 标准差={np.std(rpy_arr[:, 1]):.2f}°')
print(f'   横滚角速度 w_roll RMS: {np.sqrt(np.mean(rpy_arr[:, 3]**2))*180/np.pi:.2f} deg/s')

print(f'\n3. 12 关节左右对称度检验:')
print(f'   前髋 Hip Roll: FL={fl_hip:+.2f}° vs FR={fr_hip:+.2f}° | 偏差: {abs(fl_hip - (-fr_hip)):.2f}°')
print(f'   后髋 Hip Roll: RL={rl_hip:+.2f}° vs RR={rr_hip:+.2f}° | 偏差: {abs(rl_hip - (-rr_hip)):.2f}°')
print(f'   前大腿 Thigh:  FL={fl_thigh:+.2f}° vs FR={fr_thigh:+.2f}° | 偏差: {abs(fl_thigh - fr_thigh):.2f}° (微调前高达 38.60°)')
print(f'   后大腿 Thigh:  RL={rl_thigh:+.2f}° vs RR={rr_thigh:+.2f}° | 偏差: {abs(rl_thigh - rr_thigh):.2f}° (微调前高达 24.34°)')
print(f'   前小腿 Calf:   FL={fl_calf:+.2f}° vs FR={fr_calf:+.2f}° | 偏差: {abs(fl_calf - fr_calf):.2f}° (微调前为 15.06°)')
print(f'   后小腿 Calf:   RL={rl_calf:+.2f}° vs RR={rr_calf:+.2f}° | 偏差: {abs(rl_calf - rr_calf):.2f}°')
print('==============================================================\n')

app.close()

