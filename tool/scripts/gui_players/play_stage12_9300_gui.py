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

import threading

def activate_window_task():
    for _ in range(10):
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

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.dreamwaq_env_cfg import UnitreeGo2DreamWaQRoughEnvCfg_PLAY
from dreamwaq_lab.tasks.manager_based.unitree_go2.dreamwaq.agents.rsl_rl_dreamwaq_cfg import UnitreeGo2DreamWaQRoughRunnerCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

cfg = UnitreeGo2DreamWaQRoughEnvCfg_PLAY()
cfg.scene.num_envs = 1
# 仿生巡航速度：0.65 m/s 从容稳健登梯
cfg.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.7)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

# 12cm 连续台阶
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
# 解除相机锁定：设为 world 模式，防止 ViewportCameraController 注册每帧强制复位回调
cfg.viewer.origin_type = "world"

env = gym.make('Unitree-Go2-DreamWaQ-Rough-Play', cfg=cfg)
env = RslRlVecEnvWrapper(env)

# 彻底注销任何可能存在的视窗相机锁定回调，100% 将运镜权交给老大鼠标和键盘
if hasattr(env.unwrapped, "viewport_camera_controller") and env.unwrapped.viewport_camera_controller is not None:
    vcc = env.unwrapped.viewport_camera_controller
    if hasattr(vcc, "_viewport_camera_update_handle") and vcc._viewport_camera_update_handle is not None:
        vcc._viewport_camera_update_handle.unsubscribe()
        vcc._viewport_camera_update_handle = None
    vcc.cfg.origin_type = "world"

resume_path = '/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough/2026-10-08_17-07_stage12_symmetric_power_balance_stairs_1000iter/model_9296.pt'
runner_cfg = UnitreeGo2DreamWaQRoughRunnerCfg()
agent_cfg_dict = runner_cfg.to_dict()
agent_cfg_dict['device'] = 'cuda:0'

runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device='cuda:0')
runner.load(resume_path)
policy = runner.get_inference_policy(device='cuda:0')

print('[INFO] Successfully loaded model_9296.pt. Setting initial framing once (Camera unlocked)...', flush=True)

# 仅在初始启动时做一次舒适取景，后续绝不干涉相机，完全交由老大自由运镜
origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
env.unwrapped.sim.set_camera_view(
    eye=origin + np.array([-1.8, -1.8, 1.2]),
    target=origin + np.array([1.2, 0.0, 0.4])
)

obs = env.get_observations()
step_count = 0
dt = env.unwrapped.step_dt

print('[INFO] Interactive 3D GUI is now LIVE on DISPLAY=:0 with 100% FREE camera navigation unlocked!', flush=True)

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
        roll = float(robot.data.root_quat_w[0].cpu().numpy()[0]) # Approximate or telemetry
        print(f"[3D Play Step {step_count:4d}] x={rx:+.3f}m | z={rz:+.3f}m | vx={vx:+.2f}m/s", flush=True)

    elapsed = time.time() - start_t
    if dt > elapsed:
        time.sleep(dt - elapsed)

simulation_app.close()

