from isaaclab.app import AppLauncher
import os, sys, time, torch
from pathlib import Path
import numpy as np
import threading

# 辅助线程：窗口启动后温和置顶一次
def activate_window_task():
    for _ in range(8):
        time.sleep(2.0)
        try:
            from Xlib import X, display, protocol
            d = display.Display(':0')
            root = d.screen().root
            _NET_CLIENT_LIST = d.intern_atom('_NET_CLIENT_LIST')
            _NET_ACTIVE_WINDOW = d.intern_atom('_NET_ACTIVE_WINDOW')
            _NET_WM_NAME = d.intern_atom('_NET_WM_NAME')
            WM_NAME = d.intern_atom('WM_NAME')
            client_list = root.get_full_property(_NET_CLIENT_LIST, X.AnyPropertyType)
            if client_list:
                for wid in client_list.value:
                    w = d.create_resource_object('window', wid)
                    name = ''
                    for atom in [_NET_WM_NAME, WM_NAME]:
                        prop = w.get_full_property(atom, 0)
                        if prop and prop.value:
                            name = prop.value.decode('utf-8', errors='ignore') if isinstance(prop.value, bytes) else str(prop.value)
                            break
                    if 'isaac sim' in name.lower():
                        event = protocol.event.ClientMessage(
                            window=w,
                            client_type=_NET_ACTIVE_WINDOW,
                            data=(32, [1, X.CurrentTime, 0, 0, 0])
                        )
                        root.send_event(event, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
                        w.configure(stack_mode=X.Above)
                        d.sync()
                        return
        except Exception:
            pass

threading.Thread(target=activate_window_task, daemon=True).start()

os.environ['DISPLAY'] = ':0'
app_launcher = AppLauncher({'headless': False, 'device': 'cuda:0'})
simulation_app = app_launcher.app

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
cfg.scene.height_scanner.debug_vis = False
cfg.viewer.origin_type = "world"

# 0.8 m/s 直线巡航
cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

env = gym.make('Unitree-Go2-DreamWaQ-Flat-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

if hasattr(env.unwrapped, "viewport_camera_controller") and env.unwrapped.viewport_camera_controller is not None:
    vcc = env.unwrapped.viewport_camera_controller
    if hasattr(vcc, "_viewport_camera_update_handle") and vcc._viewport_camera_update_handle is not None:
        vcc._viewport_camera_update_handle.unsubscribe()
        vcc._viewport_camera_update_handle = None
    vcc.cfg.origin_type = "world"

resume_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-09_16-36_finetune_boost_thrust_300iter/model_1850.pt'
print(f'[INFO] Loading latest thrust-boosted checkpoint: {resume_path}', flush=True)

runner_cfg = UnitreeGo2DreamWaQFlatRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')

print('[INFO] Successfully loaded policy into GPU.', flush=True)

origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
env.unwrapped.sim.set_camera_view(
    eye=origin + np.array([-2.4, -2.2, 1.1]),
    target=origin + np.array([1.2, 0.0, 0.35])
)

obs, _ = env.reset()
step_count = 0
dt = env.unwrapped.step_dt
captured = False

print('[INFO] Interactive 3D Boosted-Thrust GUI is LIVE on DISPLAY=:0 with anti-lag 50Hz frame regulation!', flush=True)

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
        ry = float(robot.data.root_pos_w[0, 1].cpu())
        rz = float(robot.data.root_pos_w[0, 2].cpu())
        vx = float(robot.data.root_lin_vel_b[0, 0].cpu())
        vy = float(robot.data.root_lin_vel_b[0, 1].cpu())
        
        quat = robot.data.root_quat_w[0].cpu().numpy()
        roll_deg, pitch_deg, yaw_deg = quat_to_rpy_deg(quat)
        
        print(f"[Boost Step {step_count:4d}] x={rx:+.2f}m y={ry:+.2f}m | Roll={roll_deg:+.2f}° Pitch={pitch_deg:+.2f}° Yaw={yaw_deg:+.2f}° | vx={vx:+.2f} vy={vy:+.3f}m/s", flush=True)

    if step_count == 100 and not captured:
        captured = True
        try:
            import mss
            with mss.MSS(display=':0') as sct:
                out_path = '/home/rum/.gemini/antigravity/brain/7b59cd11-515a-4ded-ad5e-413fbabe33fe/isaac_sim_thrust_boost_live.png'
                sct.shot(output=out_path)
                print(f'[INFO] Live snapshot saved to: {out_path}', flush=True)
        except Exception as e:
            print(f'[WARN] Failed to capture snapshot: {e}', flush=True)

    # 50Hz 控频
    elapsed = time.time() - start_t
    if dt > elapsed:
        time.sleep(dt - elapsed)
    else:
        time.sleep(0.002)

simulation_app.close()

