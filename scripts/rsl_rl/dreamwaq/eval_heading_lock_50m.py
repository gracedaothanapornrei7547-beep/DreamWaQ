# Copyright (c) 2024 LunarQuadruped Project.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""50 米长程直行基准测试：开环速度模式 vs 航向闭环锁模式 对比评测脚本。"""

import argparse
import math
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RSL_RL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = RSL_RL_ROOT.parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "dreamwaq_lab"
for p in (SCRIPT_DIR, RSL_RL_ROOT, SOURCE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Evaluate 50m straight run with/without heading lock.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from dreamwaq_actor_critic import register_dreamwaq_actor_critic
from dreamwaq_ppo import register_dreamwaq_ppo
import dreamwaq_lab.tasks

register_dreamwaq_actor_critic()
register_dreamwaq_ppo()


def run_benchmark(env, runner, heading_lock: bool, target_dist: float = 50.0, cmd_vx: float = 0.8):
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd_term.cfg.heading_command = heading_lock
    if heading_lock:
        cmd_term.cfg.heading_control_stiffness = 1.0
        cmd_term.cfg.rel_heading_envs = 1.0
        cmd_term.cfg.ranges.heading = (0.0, 0.0)
        cmd_term.cfg.ranges.ang_vel_z = (-1.0, 1.0)
        cmd_term.heading_target.zero_()
        cmd_term.is_heading_env.fill_(True)
    else:
        cmd_term.cfg.ranges.ang_vel_z = (0.0, 0.0)
        cmd_term.is_heading_env.fill_(False)

    cmd_term.cfg.ranges.lin_vel_x = (cmd_vx, cmd_vx)
    cmd_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_term.cfg.rel_standing_envs = 0.0

    policy = runner.get_inference_policy(device="cuda:0")

    with torch.no_grad():
        obs_ret = env.reset()
        obs = obs_ret[0] if isinstance(obs_ret, tuple) else obs_ret

    robot = env.unwrapped.scene["robot"]

    x_list, y_list, yaw_list = [], [], []
    vx_list, vy_list, wz_list = [], [], []
    
    start_pos = robot.data.root_pos_w[0, :2].clone()
    curr_dist = 0.0
    step = 0
    max_steps = 4000  # 最多跑 80 秒

    with torch.no_grad():
        while curr_dist < target_dist and step < max_steps:
            act = policy(obs)
            obs, _, _, _ = env.step(act)

            pos_w = robot.data.root_pos_w[0]
            vel_b = robot.data.root_lin_vel_b[0]
            ang_b = robot.data.root_ang_vel_b[0]
            heading_w = robot.data.heading_w[0].item()

            dx = pos_w[0].item() - start_pos[0].item()
            dy = pos_w[1].item() - start_pos[1].item()
            curr_dist = math.sqrt(dx * dx + dy * dy)

            x_list.append(dx)
            y_list.append(dy)
            yaw_list.append(heading_w * 180.0 / math.pi)
            vx_list.append(vel_b[0].item())
            vy_list.append(vel_b[1].item())
            wz_list.append(ang_b[2].item())
            step += 1

            if step % 250 == 0 or curr_dist >= target_dist:
                mode_str = "HeadingLock" if heading_lock else "OpenLoop"
                print(f"[{mode_str}] Step {step} ({step*0.02:.1f}s): dist={curr_dist:.2f}m, dx={dx:.2f}m, dy={dy:.3f}m, yaw={heading_w*180.0/math.pi:.2f}°, vx={vel_b[0].item():.3f}m/s")

    x_arr = torch.tensor(x_list)
    y_arr = torch.tensor(y_list)
    yaw_arr = torch.tensor(yaw_list)
    vx_arr = torch.tensor(vx_list)
    vy_arr = torch.tensor(vy_list)
    wz_arr = torch.tensor(wz_list)

    total_time = step * 0.02
    final_dx = x_list[-1]
    final_dy = y_list[-1]
    final_yaw = yaw_list[-1]
    max_lateral_dev = torch.max(torch.abs(y_arr)).item()

    return {
        "heading_lock": heading_lock,
        "dist": curr_dist,
        "time": total_time,
        "steps": step,
        "final_dx": final_dx,
        "final_dy": final_dy,
        "max_lateral": max_lateral_dev,
        "final_yaw": final_yaw,
        "mean_yaw_err": torch.mean(torch.abs(yaw_arr)).item(),
        "max_yaw_err": torch.max(torch.abs(yaw_arr)).item(),
        "mean_vx": torch.mean(vx_arr).item(),
        "mean_vy": torch.mean(vy_arr).item(),
        "mean_wz": torch.mean(wz_arr).item(),
    }


@hydra_task_config("Unitree-Go2-DreamWaQ-Flat-Play", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    # 延长 episode 时间至 200 秒并关闭中途超时截断，保证 50 米连续直行不被重置
    env_cfg.episode_length_s = 200.0
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    if hasattr(env_cfg.events, "motor_zero_offset"):
        env_cfg.events.motor_zero_offset = None

    # 严格固定测试起点，排除初始朝向随机扰动
    if hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
        env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
        }

    env = gym.make("Unitree-Go2-DreamWaQ-Flat-Play", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")

    ckpt = "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-10_14-57_waxberry_flat_finetune_antidrift_500iter/model_5098.pt"
    print(f"[INFO] Loading fine-tuned checkpoint: {ckpt}")
    runner.load(ckpt)

    print("\n>>> 开始测试 1：【开环速度模式 (无闭环，纯角速度 wz=0)】走 50 米...")
    res_open = run_benchmark(env, runner, heading_lock=False, target_dist=50.0, cmd_vx=0.8)

    print("\n>>> 开始测试 2：【航向闭环锁模式 (Heading Lock，目标航向 0°)】走 50 米...")
    res_closed = run_benchmark(env, runner, heading_lock=True, target_dist=50.0, cmd_vx=0.8)

    print("\n" + "=" * 70)
    print("           宇树 Go2 (model_5098) 50米长程直行遥测基准对比报告")
    print("=" * 70)
    print(f"{'评测维度':<22} | {'开环速度模式 (无闭环)':<20} | {'航向闭环锁模式 (Heading Lock)':<20}")
    print("-" * 70)
    print(f"{'行走总直线距离 (m)':<20} | {res_open['dist']:<22.2f} | {res_closed['dist']:<22.2f}")
    print(f"{'用时 (秒)':<24} | {res_open['time']:<22.2f} | {res_closed['time']:<22.2f}")
    print(f"{'平均前向线速度 vx (m/s)':<18} | {res_open['mean_vx']:<22.3f} | {res_closed['mean_vx']:<22.3f}")
    print(f"{'终点横向侧移 Y (m)':<20} | {res_open['final_dy']:<22.3f} | {res_closed['final_dy']:<22.3f}")
    print(f"{'全程最大横向偏差 (m)':<18} | {res_open['max_lateral']:<22.3f} | {res_closed['max_lateral']:<22.3f}")
    print(f"{'终点偏航角 Yaw (度)':<20} | {res_open['final_yaw']:<22.2f} | {res_closed['final_yaw']:<22.2f}")
    print(f"{'全程平均航向误差 (度)':<18} | {res_open['mean_yaw_err']:<22.2f} | {res_closed['mean_yaw_err']:<22.2f}")
    print(f"{'全程最大航向误差 (度)':<18} | {res_open['max_yaw_err']:<22.2f} | {res_closed['max_yaw_err']:<22.2f}")
    print(f"{'平均横向速度 vy (m/s)':<18} | {res_open['mean_vy']:<22.4f} | {res_closed['mean_vy']:<22.4f}")
    print(f"{'平均角速度 wz (rad/s)':<19} | {res_open['mean_wz']:<22.4f} | {res_closed['mean_wz']:<22.4f}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
