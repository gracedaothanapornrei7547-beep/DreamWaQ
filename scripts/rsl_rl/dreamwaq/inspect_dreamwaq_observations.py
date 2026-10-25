# Copyright (c) 2024 LunarQuadruped Project.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect DreamWaQ observation groups without starting a training run."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
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

parser = argparse.ArgumentParser(description="Inspect DreamWaQ observation dimensions.")
parser.add_argument("--task", type=str, default="Unitree-Go2-DreamWaQ-Rough", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to instantiate.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

_sanitize_headless_display_env(args_cli)

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import dreamwaq_lab.tasks  # noqa: F401


def _last_dim(obs: torch.Tensor) -> int:
    return int(obs.shape[-1])


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    # 这个脚本只检查 observation wiring，不创建 PPO runner，适合排查 group 是否启用和维度是否正确。
    print("\n========== DreamWaQ observation groups ==========")
    for name, value in obs.items():
        print(f"{name}: shape={tuple(value.shape)}")

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

    checks = {
        "proprio_dim_positive": proprio_dim > 0,
        "history_dim_multiple_of_proprio_dim": history_dim % proprio_dim == 0,
        "history_length_is_5": history_length == 5,
        "reconstruction_target_matches_proprio": reconstruction_target_dim == proprio_dim,
        "critic_has_privileged_extra": critic_dim > proprio_dim,
    }
    failed = [name for name, passed in checks.items() if not passed]

    print("\n========== DreamWaQ dimension summary ==========")
    print(f"proprio dim: {proprio_dim}")
    print(f"history dim: {history_dim}")
    print(f"history length: {history_length}")
    print(f"critic obs dim: {critic_dim}")
    print(f"reconstruction target dim: {reconstruction_target_dim}")

    print("\n========== DreamWaQ dimension checks ==========")
    for name, passed in checks.items():
        print(f"{name}: {'OK' if passed else 'FAILED'}")

    env.close()

    if failed:
        raise RuntimeError(f"DreamWaQ observation checks failed: {failed}")


if __name__ == "__main__":
    main()
    simulation_app.close()
