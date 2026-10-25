# Copyright (c) 2024 LunarQuadruped Project.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Train DreamWaQ with RSL-RL PPO."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time
from datetime import datetime
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
from dreamwaq_ppo import register_dreamwaq_ppo  # isort: skip

parser = argparse.ArgumentParser(description="Train DreamWaQ with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-Go2-DreamWaQ-Rough", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="DreamWaQ training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration."
)
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
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import dreamwaq_lab.tasks  # noqa: F401
from dreamwaq_lab.deployment import export_deploy_cfg

# RSL-RL 的 runner 通过字符串 class_name 查找类。这里把 DreamWaQ 专用类注册进去，
# 不需要修改 site-packages 里的 rsl_rl。
register_dreamwaq_actor_critic()
register_dreamwaq_ppo()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _last_dim(obs: torch.Tensor) -> int:
    return int(obs.shape[-1])


def _print_dreamwaq_dimensions(obs: TensorDict, num_actions: int) -> None:
    """训练启动前做一次强校验，尽早发现 observation group 或维度接错。"""

    required_groups = {
        "dreamwaq_policy",
        "dreamwaq_policy_history",
        "dreamwaq_reconstruction_target",
        "dreamwaq_critic",
    }
    missing_groups = sorted(required_groups.difference(obs.keys()))
    if missing_groups:
        raise RuntimeError(f"Missing DreamWaQ observation groups: {missing_groups}")

    proprio_dim = _last_dim(obs["dreamwaq_policy"])
    history_dim = _last_dim(obs["dreamwaq_policy_history"])
    reconstruction_target_dim = _last_dim(obs["dreamwaq_reconstruction_target"])
    critic_dim = _last_dim(obs["dreamwaq_critic"])
    history_length = history_dim // proprio_dim

    print("\n========== DreamWaQ runtime dimensions ==========")
    print(f"policy obs dim: {proprio_dim}")
    print(f"critic obs dim: {critic_dim}")
    print(f"proprio dim: {proprio_dim}")
    print(f"history dim: {history_dim}")
    print(f"history length: {history_length}")
    print("estimated velocity dim: 3")
    print("latent dim: 16")
    print(f"action dim: {num_actions}")
    print(f"reconstruction target dim: {reconstruction_target_dim}")
    print("================================================\n")

    if history_dim % proprio_dim != 0:
        raise RuntimeError(f"History dim {history_dim} is not divisible by proprio dim {proprio_dim}.")
    if history_length != 5:
        raise RuntimeError(f"Expected DreamWaQ history length 5, got {history_length}.")
    if reconstruction_target_dim != proprio_dim:
        raise RuntimeError(
            f"Expected reconstruction target dim {proprio_dim}, got {reconstruction_target_dim}."
        )
    if critic_dim <= proprio_dim:
        raise RuntimeError(f"Critic dim {critic_dim} does not contain privileged state beyond proprio dim {proprio_dim}.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported with CPU device.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume:
        if os.path.isabs(agent_cfg.load_run) or os.path.exists(agent_cfg.load_run):
            candidate_dir = agent_cfg.load_run
            resume_path = os.path.join(candidate_dir, agent_cfg.load_checkpoint)
        else:
            search_roots = [
                log_root_path,
                os.path.abspath(os.path.join("logs", "rsl_rl", "unitree", "unitree_go2_dreamwaq_flat")),
                os.path.abspath(os.path.join("logs", "rsl_rl", "unitree", "unitree_go2_dreamwaq_rough")),
            ]
            resume_path = None
            for s_root in search_roots:
                if os.path.exists(s_root):
                    try:
                        resume_path = get_checkpoint_path(s_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
                        break
                    except Exception:
                        pass
            if resume_path is None:
                raise ValueError(f"Could not find checkpoint for run '{agent_cfg.load_run}' in any of: {search_roots}")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    obs = env.get_observations()
    # 只打印一次，确认 actor/current、history、reconstruction target、critic 都已启用。
    _print_dreamwaq_dimensions(obs, env.num_actions)

    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"DreamWaQ training expects OnPolicyRunner, got {agent_cfg.class_name}.")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
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

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
