# Copyright (c) 2024 LunarQuadruped Project.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a DreamWaQ checkpoint on FIXED terrain + FIXED command.

Unlike ``play_dreamwaq.py`` (which keeps the -Play env's default mixed 4-cell
terrain at difficulty 0.6 and a command *range*), this script pins both so the
resulting ``error_vel_xy`` / ``error_vel_yaw`` are directly comparable to the
HIMLoco numbers (flat 0.058 / terrain-2.4 0.19 / yaw 0.164), which use the
identical isaaclab ``UniformVelocityCommand`` metric and the same episode
(20 s) + resampling (10 s) + decimation (4) conventions.

Two terrain modes:
  --terrain rough --difficulty 0.0   -> ~3 cm stairs + 2 cm rough/boxes (align HIMLoco 2.4)
  --terrain flat                     -> all sub-terrain heights zeroed (control floor, align 0.058)

Command is pinned to a straight line: lin_vel_x=(cmd_vx, cmd_vx), lin_vel_y=0, ang_vel_z=0.
"""

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
    if getattr(args, "headless", False) and not getattr(args, "enable_cameras", False):
        for name in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY"):
            os.environ.pop(name, None)


from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate DreamWaQ checkpoint on fixed terrain + fixed command.")
parser.add_argument("--task", type=str, default="Unitree-Go2-DreamWaQ-Rough-Play", help="Play/eval task id.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments.")
parser.add_argument("--max_steps", type=int, default=1000, help="Eval steps (1000 == one 20 s episode).")
parser.add_argument("--cmd_vx", type=float, default=1.0, help="Pinned forward command (m/s).")
parser.add_argument("--terrain", type=str, default="rough", choices={"rough", "flat"})
parser.add_argument("--difficulty", type=float, default=0.0, help="Terrain difficulty for --terrain rough.")
parser.add_argument("--warmup_steps", type=int, default=100, help="Steps to skip for the steady-state mean.")
parser.add_argument("--probe_cenet", action="store_true", default=False, help="Also measure CENet velocity-estimate error vs ground truth.")
parser.add_argument("--true_velocity", action="store_true", default=False, help="Ablation: feed ground-truth base_lin_vel to the actor instead of CENet's estimate.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

_sanitize_headless_display_env(args_cli)

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import cli_args
import dreamwaq_lab.tasks  # noqa: F401
from dreamwaq_actor_critic import register_dreamwaq_actor_critic
from dreamwaq_ppo import register_dreamwaq_ppo
from dreamwaq_lab.terrains import apply_play_terrain

register_dreamwaq_actor_critic()
register_dreamwaq_ppo()


def _configure_terrain(env_cfg, mode: str, difficulty: float) -> None:
    """Pin the play terrain to a fixed difficulty, and (optionally) flatten it."""
    seed = getattr(env_cfg, "terrain_seed", 1)
    apply_play_terrain(env_cfg, difficulty=difficulty, seed=seed)
    if mode == "flat":
        tg = env_cfg.scene.terrain.terrain_generator
        for sub in tg.sub_terrains.values():
            for attr in ("noise_range", "grid_height_range", "slope_range", "step_height_range"):
                if hasattr(sub, attr):
                    setattr(sub, attr, (0.0, 0.0))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    # --- fixed terrain ---
    _configure_terrain(env_cfg, args_cli.terrain, args_cli.difficulty)

    # --- fixed straight-line command ---
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    print(f"[INFO] terrain mode={args_cli.terrain} difficulty={args_cli.difficulty} "
          f"cmd_vx={args_cli.cmd_vx} num_envs={args_cli.num_envs}")

    # --- checkpoint ---
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        raise SystemExit("--checkpoint is required (absolute path to model_*.pt).")
    print(f"[INFO] Loading DreamWaQ checkpoint from: {resume_path}")

    # --- env ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    robot = unwrapped.scene["robot"]
    max_command_step = cmd_term.cfg.resampling_time_range[1] / unwrapped.step_dt
    print(f"[INFO] step_dt={unwrapped.step_dt:.4f} max_command_step={max_command_step:.1f}")

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=unwrapped.device)
    policy_nn = runner.alg.policy
    obs = env.get_observations()

    # --- metric accumulation (mirrors HIMLoco measure_steady_error.py) ---
    # Raw instantaneous error ||cmd_xy - vel_xy|| (m/s, body frame), one entry per step.
    err_steps = []
    err_yaw_steps = []
    vx_steps = []
    est_err_steps = []       # CENet velocity-estimate error ||est_xy - true_xy|| (m/s)
    est_vx_bias_steps = []   # CENet vx estimate bias (est_vx - true_vx)
    step = 0

    while simulation_app.is_running() and step < args_cli.max_steps:
        with torch.inference_mode():
            if args_cli.true_velocity:
                # Ablation: bypass CENet's velocity head, feed ground-truth body-frame
                # base_lin_vel (robot.data.root_lin_vel_b) into the actor while keeping the
                # CENet latent + current proprio. Isolates whether the ~0.08 m/s estimate
                # noise (not the gait) caps the flat-floor error.
                current_proprio = policy_nn.get_current_proprio_obs(obs)
                cenet_out = policy_nn.compute_cenet_outputs(obs, sample_latent=False)
                true_vel = robot.data.root_lin_vel_b
                actor_input = torch.cat((true_vel, cenet_out.latent_mu, current_proprio), dim=-1)
                actor_input = policy_nn.actor_obs_normalizer(actor_input)  # nn.Identity here
                if policy_nn.state_dependent_std:
                    actions = policy_nn.actor(actor_input)[..., 0, :]
                else:
                    actions = policy_nn.actor(actor_input)
            else:
                actions = policy(obs)
            if args_cli.probe_cenet:
                # estimated_velocity is CENet's supervised estimate of body-frame base_lin_vel,
                # produced from the history at the current timestep t (before env.step).
                est_vel = policy_nn.last_cenet_outputs.estimated_velocity   # (N,3)
                true_vel_t = robot.data.root_lin_vel_b                      # (N,3) ground truth at t
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        vel_b = robot.data.root_lin_vel_b
        ang_b = robot.data.root_ang_vel_b
        cmd_b = cmd_term.vel_command_b

        err_xy = torch.norm(cmd_b[:, :2] - vel_b[:, :2], dim=-1)
        err_yaw = torch.abs(cmd_b[:, 2] - ang_b[:, 2])
        err_steps.append(err_xy)
        err_yaw_steps.append(err_yaw)
        vx_steps.append(vel_b[:, 0])
        if args_cli.probe_cenet:
            est_err_steps.append(torch.norm(est_vel[:, :2] - true_vel_t[:, :2], dim=-1))
            est_vx_bias_steps.append(est_vel[:, 0] - true_vel_t[:, 0])
        step += 1

    err = torch.stack(err_steps, dim=0)   # (num_steps, num_envs)
    err_yaw = torch.stack(err_yaw_steps, dim=0)
    vx = torch.stack(vx_steps, dim=0)
    if args_cli.probe_cenet:
        est_err = torch.stack(est_err_steps, dim=0)
        est_vx_bias = torch.stack(est_vx_bias_steps, dim=0)

    warmup = args_cli.warmup_steps
    err_ss = err[warmup:]
    err_yaw_ss = err_yaw[warmup:]
    vx_ss = vx[warmup:]

    def _stats(t: torch.Tensor, name: str):
        flat = t.flatten().float()
        print(f"[{name}] mean={flat.mean().item():.4f}  std={flat.std().item():.4f}  "
              f"median={flat.median().item():.4f}  p90={torch.quantile(flat, 0.90).item():.4f}  "
              f"max={flat.max().item():.4f}")

    n = args_cli.max_steps
    print("\n================ FIXED-TERRAIN EVAL RESULT ================")
    print(f"terrain mode        : {args_cli.terrain} (difficulty {args_cli.difficulty})")
    print(f"velocity input      : {'ground-truth (ablation)' if args_cli.true_velocity else 'CENet estimate (baseline)'}")
    print(f"command             : straight {args_cli.cmd_vx} m/s (vy=0, yaw=0)")
    print(f"steps               : {n}  ({n * unwrapped.step_dt:.0f} s)  warmup skipped: {warmup} steps")
    _stats(vx, "achieved vx (full)")
    _stats(vx_ss, "achieved vx (steady)")
    _stats(err, "error_vel_xy RAW ||cmd-vel||_xy (full)")
    _stats(err_ss, "error_vel_xy RAW ||cmd-vel||_xy (steady)  <-- compare HIMLoco flat median 0.057")
    _stats(err_yaw_ss, "error_vel_yaw RAW |cmd-w|-yaw (steady)")

    # isaaclab "log" convention (sum of err/max_command_step over the episode) for the
    # training-metric comparison (HIMLoco terrain-2.4 = 0.19 is this same logged value).
    full_log_xy = (err.mean(dim=0).mean() * n / max_command_step).item()
    ss_log_xy = (err_ss.mean(dim=0).mean() * (n - warmup) / max_command_step).item()
    print(f"error_vel_xy (isaaclab log, full-episode) : {full_log_xy:.4f}   <-- compare HIMLoco terrain2.4 0.19")
    print(f"error_vel_xy (isaaclab log, steady-state) : {ss_log_xy:.4f}")

    if args_cli.probe_cenet:
        print("\n--- CENet velocity-estimate probe (steady-state) ---")
        _stats(est_err[warmup:], "CENet est error ||est_xy - true_xy|| (steady)")
        _stats(est_vx_bias[warmup:], "CENet vx bias (est_vx - true_vx) (steady)")
        true_vx_std = max(vx_ss.flatten().float().std().item(), 1e-9)
        print(f"  true vx std (steady) : {true_vx_std:.4f} m/s")
        print(f"  CENet est_err mean / true_vx_std = "
              f"{(est_err[warmup:].flatten().float().mean().item() / true_vx_std):.3f}")
    print("==========================================================")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
