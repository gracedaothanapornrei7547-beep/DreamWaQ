# Copyright (c) 2024 LunarQuadruped Project.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Play/export a DreamWaQ checkpoint with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RSL_RL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = RSL_RL_ROOT.parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "dreamwaq_lab"
for path in (SCRIPT_DIR, RSL_RL_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _sanitize_headless_display_env(args) -> None:
    """Avoid X11 probing crashes when Isaac Sim is launched in pure headless mode."""

    if getattr(args, "headless", False) and not getattr(args, "enable_cameras", False):
        for name in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY"):
            os.environ.pop(name, None)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from dreamwaq_actor_critic import register_dreamwaq_actor_critic  # isort: skip
from dreamwaq_exporter import export_dreamwaq_full_policy  # isort: skip
from dreamwaq_ppo import register_dreamwaq_ppo  # isort: skip

parser = argparse.ArgumentParser(description="Play/export a DreamWaQ checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-Go2-DreamWaQ-Rough-Play", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of play steps before exiting.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

_sanitize_headless_display_env(args_cli)

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import dreamwaq_lab.tasks  # noqa: F401
from dreamwaq_lab.deployment import export_deploy_cfg

# Play 也需要注册 DreamWaQ 类，否则 checkpoint 里的 class_name 找不到本地实现。
register_dreamwaq_actor_critic()
register_dreamwaq_ppo()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading DreamWaQ experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during DreamWaQ play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"DreamWaQ play expects OnPolicyRunner, got {agent_cfg.class_name}.")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO]: Loading DreamWaQ checkpoint from: {resume_path}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy
    obs = env.get_observations()

    agent_cfg_dict = agent_cfg.to_dict()
    export_deploy_cfg(
        env.unwrapped,
        log_dir,
        actor_groups=agent_cfg_dict["obs_groups"]["policy"],
        model_input_names=["current_obs", "history_obs"],
        robot_name="go2",
        model_name="policy.pt",
        history_is_term_major=agent_cfg_dict["policy"].get("history_is_term_major", True),
    )

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_dreamwaq_full_policy(policy_nn, path=export_model_dir, filename_prefix="policy", example_obs=obs)

    print(
        "[INFO] DreamWaQ export uses the full deployable path: "
        "current dreamwaq_policy + dreamwaq_policy_history -> CENet -> actor. "
        "Privileged critic observations are ignored by the exported inference model."
    )

    dt = env.unwrapped.step_dt
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # obs 里包含 dreamwaq_policy 和 dreamwaq_policy_history；critic privileged obs 不参与推理动作。
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        timestep += 1
        if args_cli.video:
            if timestep == args_cli.video_length:
                break
        if args_cli.max_steps is not None:
            if timestep >= args_cli.max_steps:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
