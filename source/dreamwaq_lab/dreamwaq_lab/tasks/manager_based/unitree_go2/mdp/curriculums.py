# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_command_range_limits(
    cmd_term,
    max_lin_vel_x: float | None,
    max_lin_vel_y: float | None,
    max_ang_vel_z: float | None,
    min_lin_vel_x: float | None = None,
    min_lin_vel_y: float | None = None,
    min_ang_vel_z: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return asymmetric command limits for x/y/yaw velocity ranges."""

    limit_ranges = getattr(cmd_term.cfg, "limit_ranges", None)
    if limit_ranges is not None:
        return (
            (float(limit_ranges.lin_vel_x[0]), float(limit_ranges.lin_vel_x[1])),
            (float(limit_ranges.lin_vel_y[0]), float(limit_ranges.lin_vel_y[1])),
            (float(limit_ranges.ang_vel_z[0]), float(limit_ranges.ang_vel_z[1])),
        )

    if max_lin_vel_x is None or max_lin_vel_y is None or max_ang_vel_z is None:
        raise ValueError(
            "Command curriculum requires either command cfg limit_ranges or explicit "
            "max_lin_vel_x/max_lin_vel_y/max_ang_vel_z."
        )

    return (
        (float(-max_lin_vel_x if min_lin_vel_x is None else min_lin_vel_x), float(max_lin_vel_x)),
        (float(-max_lin_vel_y if min_lin_vel_y is None else min_lin_vel_y), float(max_lin_vel_y)),
        (float(-max_ang_vel_z if min_ang_vel_z is None else min_ang_vel_z), float(max_ang_vel_z)),
    )


def _expand_range_towards_limit(
    current_range: tuple[float, float],
    limit_range: tuple[float, float],
    step_size: float,
) -> tuple[float, float]:
    """Expand lower and upper bounds independently while respecting asymmetric limits."""

    lower = max(float(current_range[0]) - step_size, limit_range[0])
    upper = min(float(current_range[1]) + step_size, limit_range[1])
    return (lower, upper)


def _range_extent(value_range: Sequence[float]) -> float:
    return max(abs(float(value_range[0])), abs(float(value_range[1])))


def _range_progress(current_range: tuple[float, float], limit_range: tuple[float, float]) -> float:
    current_extent = _range_extent(current_range)
    limit_extent = max(_range_extent(limit_range), 1.0e-6)
    return min(current_extent / limit_extent, 1.0)


def _active_command_range_progress(
    current_ranges: Sequence[Sequence[float]],
    limit_ranges: Sequence[Sequence[float]],
    disabled_limit_eps: float = 1.0e-6,
) -> float:
    """Average command progress over axes that are actually enabled."""

    progresses: list[float] = []
    for current_range, limit_range in zip(current_ranges, limit_ranges):
        limit_extent = _range_extent(limit_range)
        if limit_extent <= disabled_limit_eps:      #如果某个命令维度的 limit_range 几乎为 0，那么这个维度不参与课程进度计算。
            continue
        progresses.append(min(_range_extent(current_range) / limit_extent, 1.0))

    # If all command axes are disabled, do not let the terrain gate deadlock.
    return sum(progresses) / len(progresses) if progresses else 1.0


# 地形课程定义
def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.
    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.
    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.
    Returns:
        The mean terrain level for the given environment ids.
    """
    # 提取必要组件 extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]              # 机器人实体（asset）
    terrain: TerrainImporter = env.scene.terrain                 # 地形管理器（terrain）
    command = env.command_manager.get_command("base_velocity")   # 强化学习给出的目标速度指令（command）
    # 计算实际行走距离 compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # 判断是否“升级” robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # 判断是否“降级” robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # 更新 update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_vel_new(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    min_command_speed: float = 0.1,                  # 当前速度指令太小时，不参与地形升降级判断。
    min_command_progress_for_terrain: float = 0.9,   # 只有速度命令课程进度达到一定比例后，才允许地形课程开始升降级。
    include_ang_vel_progress: bool = True,           # 地形升级 是否把yaw命令范围进度也算进去。
) -> torch.Tensor:
    """Curriculum based on progress ratio relative to commanded speed.

    机器人实际走的距离会和“按当前速度指令理论上应该走的距离”比较：
    三阶段逻辑：
    - Stage 1：命令范围进度 < min_command_progress_for_terrain 时，不升降级，只保持初始低等级地形。
      若 include_ang_vel_progress=False，则该进度只由线速度命令范围决定。
    - Stage 2：命令范围进度达到阈值后，允许地形课程开始升降级。
    - Stage 3：命令范围到最大后，地形课程完整运行。

    地形升降级条件：
    - cmd_speed <= min_command_speed 时，不参与地形课程。
    - distance > terrain_length * 0.5 时，升级到更难地形。
    - progress_ratio < 0.3 时，降级到更简单地形。
    """

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command(command_name)
    cmd_term = env.command_manager.get_term(command_name)

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    command_progress = torch.tensor(1.0, device=env.device)
    limit_ranges = getattr(cmd_term.cfg, "limit_ranges", None)
    if limit_ranges is not None:
        current_progress_ranges = [
            cmd_term.cfg.ranges.lin_vel_x,
            cmd_term.cfg.ranges.lin_vel_y,
        ]
        limit_progress_ranges = [
            limit_ranges.lin_vel_x,
            limit_ranges.lin_vel_y,
        ]
        if include_ang_vel_progress:
            current_progress_ranges.append(cmd_term.cfg.ranges.ang_vel_z)
            limit_progress_ranges.append(limit_ranges.ang_vel_z)
        command_progress = torch.tensor(
            _active_command_range_progress(
                current_progress_ranges,
                limit_progress_ranges,
            ),
            device=env.device,
        )

    if float(command_progress.item()) < min_command_progress_for_terrain:
        no_change = torch.zeros_like(env_ids, dtype=torch.bool, device=env.device)
        terrain.update_env_origins(env_ids, no_change, no_change)
        return torch.mean(terrain.terrain_levels.float())

    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    cmd_speed = torch.norm(command[env_ids, :2], dim=1)
    required_distance = cmd_speed * env.max_episode_length_s
    progress_ratio = distance / (required_distance + 1.0e-6)

    terrain_half_length = terrain.cfg.terrain_generator.size[0] * 0.5

    moving = cmd_speed > min_command_speed
    move_up = (distance > terrain_half_length) & moving
    move_down = (progress_ratio < 0.3) & moving
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


# 速度课程
def modify_command_ranges_on_survival(
    env,
    env_ids,
    term_name: str,
    step_size: float,
    success_ratio_threshold: float,
    sample_window: int,
    max_lin_vel_x: float | None = None,
    max_lin_vel_y: float | None = None,
    max_ang_vel_z: float | None = None,
    min_lin_vel_x: float | None = None,
    min_lin_vel_y: float | None = None,
    min_ang_vel_z: float | None = None,
):
    asset = env.scene["robot"]
    cmd_term = env.command_manager.get_term(term_name)
    x_limits, y_limits, yaw_limits = _resolve_command_range_limits(
        cmd_term,
        max_lin_vel_x=max_lin_vel_x,
        max_lin_vel_y=max_lin_vel_y,
        max_ang_vel_z=max_ang_vel_z,
        min_lin_vel_x=min_lin_vel_x,
        min_lin_vel_y=min_lin_vel_y,
        min_ang_vel_z=min_ang_vel_z,
    )
    # 第一次调用时初始化状态
    if not hasattr(env, "_command_curriculum_state"):
        init_x = max(abs(cmd_term.cfg.ranges.lin_vel_x[0]), abs(cmd_term.cfg.ranges.lin_vel_x[1]))
        init_y = max(abs(cmd_term.cfg.ranges.lin_vel_y[0]), abs(cmd_term.cfg.ranges.lin_vel_y[1]))
        init_yaw = max(abs(cmd_term.cfg.ranges.ang_vel_z[0]), abs(cmd_term.cfg.ranges.ang_vel_z[1]))

        env._command_curriculum_state = {
            "init_lin_vel_x": float(init_x),
            "init_lin_vel_y": float(init_y),
            "init_ang_vel_z": float(init_yaw),
            "num_resets": 0,
            "num_successes": 0,
        }

    state = env._command_curriculum_state

    if env_ids is not None and len(env_ids) > 0:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

        command = env.command_manager.get_command(term_name)
        cmd_speed = torch.norm(command[env_ids, :2], dim=1)

        distance = torch.norm(
            asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
            dim=1,
        )

        # 命令越大，要求走得越远
        required_distance = cmd_speed * env.max_episode_length_s * 0.3
        # 成功条件：活到 time_out 且走出了足够距离
        success = env.termination_manager.time_outs[env_ids] & (distance > required_distance)

        state["num_resets"] += int(env_ids.numel())
        state["num_successes"] += int(success.sum().item())

    if state["num_resets"] >= sample_window:
        success_ratio = state["num_successes"] / max(state["num_resets"], 1)

        if success_ratio >= success_ratio_threshold:
            cmd_term.cfg.ranges.lin_vel_x = _expand_range_towards_limit(
                cmd_term.cfg.ranges.lin_vel_x, x_limits, step_size
            )
            cmd_term.cfg.ranges.lin_vel_y = _expand_range_towards_limit(
                cmd_term.cfg.ranges.lin_vel_y, y_limits, step_size
            )
            cmd_term.cfg.ranges.ang_vel_z = _expand_range_towards_limit(
                cmd_term.cfg.ranges.ang_vel_z, yaw_limits, step_size
            )

        state["num_resets"] = 0
        state["num_successes"] = 0

    command_progress = _active_command_range_progress(
        (
            cmd_term.cfg.ranges.lin_vel_x,
            cmd_term.cfg.ranges.lin_vel_y,
            cmd_term.cfg.ranges.ang_vel_z,
        ),
        (x_limits, y_limits, yaw_limits),
    )
    return torch.tensor(command_progress, device=env.device)


def modify_command_ranges_on_tracking_reward(
    env,
    env_ids,
    term_name: str,
    step_size: float,
    reward_threshold: float,
    sample_window: int,
    lin_vel_reward_term_name: str = "track_lin_vel_xy_exp",
    ang_vel_reward_term_name: str = "track_ang_vel_z_exp",
    ang_reward_threshold: float | None = None,
    include_ang_vel: bool = True,
    min_command_speed: float = 0.1,
    max_lin_vel_x: float | None = None,
    max_lin_vel_y: float | None = None,
    max_ang_vel_z: float | None = None,
    min_lin_vel_x: float | None = None,
    min_lin_vel_y: float | None = None,
    min_ang_vel_z: float | None = None,
):
    """Expand command ranges when velocity-tracking rewards are high.

    升级条件：
    1. 只统计 cmd_speed > min_command_speed 的 reset env，站立/近零速度命令不参与。
    2. 累积统计的 moving env 数量达到 sample_window 后，计算 episode 平均 tracking reward。
    3. 若 avg_lin_reward > track_lin_vel_xy_exp.weight * reward_threshold，
       则 x/y 线速度命令范围各扩大 step_size，直到 limit_ranges。
    4. 若 include_ang_vel=True 且 avg_ang_reward > track_ang_vel_z_exp.weight * yaw_threshold，
       则 yaw 角速度命令范围扩大 step_size，直到 limit_ranges。yaw_threshold 默认等于 reward_threshold，
       也可以通过 ang_reward_threshold 单独设置。
    该速度课程只扩大命令范围，不做降级。
    """

    cmd_term = env.command_manager.get_term(term_name)
    x_limits, y_limits, yaw_limits = _resolve_command_range_limits(
        cmd_term,
        max_lin_vel_x=max_lin_vel_x,
        max_lin_vel_y=max_lin_vel_y,
        max_ang_vel_z=max_ang_vel_z,
        min_lin_vel_x=min_lin_vel_x,
        min_lin_vel_y=min_lin_vel_y,
        min_ang_vel_z=min_ang_vel_z,
    )

    if not hasattr(env, "_tracking_command_curriculum_state"):
        env._tracking_command_curriculum_state = {
            "num_resets": 0,
            "lin_reward_sum": 0.0,
            "ang_reward_sum": 0.0,
        }

    state = env._tracking_command_curriculum_state
    moving_env_ids = None

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    elif env_ids is not None:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if env_ids is not None and env_ids.numel() > 0:
        command = env.command_manager.get_command(term_name)
        cmd_speed = torch.norm(command[env_ids, :2], dim=1)
        moving_env_ids = env_ids[cmd_speed > min_command_speed]

    if moving_env_ids is not None and moving_env_ids.numel() > 0:
        lin_reward = (
            env.reward_manager._episode_sums[lin_vel_reward_term_name][moving_env_ids] / env.max_episode_length_s
        )
        state["lin_reward_sum"] += float(torch.sum(lin_reward).item())

        if include_ang_vel:
            ang_reward = (
                env.reward_manager._episode_sums[ang_vel_reward_term_name][moving_env_ids] / env.max_episode_length_s
            )
            state["ang_reward_sum"] += float(torch.sum(ang_reward).item())

        state["num_resets"] += int(moving_env_ids.numel())

    if state["num_resets"] >= sample_window:
        num_resets = max(state["num_resets"], 1)
        avg_lin_reward = state["lin_reward_sum"] / num_resets
        avg_ang_reward = state["ang_reward_sum"] / num_resets

        lin_reward_term = env.reward_manager.get_term_cfg(lin_vel_reward_term_name)
        if avg_lin_reward > lin_reward_term.weight * reward_threshold:
            cmd_term.cfg.ranges.lin_vel_x = _expand_range_towards_limit(
                cmd_term.cfg.ranges.lin_vel_x, x_limits, step_size
            )
            cmd_term.cfg.ranges.lin_vel_y = _expand_range_towards_limit(
                cmd_term.cfg.ranges.lin_vel_y, y_limits, step_size
            )

        if include_ang_vel:
            ang_reward_term = env.reward_manager.get_term_cfg(ang_vel_reward_term_name)
            yaw_threshold = reward_threshold if ang_reward_threshold is None else float(ang_reward_threshold)
            if avg_ang_reward > ang_reward_term.weight * yaw_threshold:
                cmd_term.cfg.ranges.ang_vel_z = _expand_range_towards_limit(
                    cmd_term.cfg.ranges.ang_vel_z, yaw_limits, step_size
                )

        state["num_resets"] = 0
        state["lin_reward_sum"] = 0.0
        state["ang_reward_sum"] = 0.0

    command_progress = _active_command_range_progress(
        (
            cmd_term.cfg.ranges.lin_vel_x,
            cmd_term.cfg.ranges.lin_vel_y,
            cmd_term.cfg.ranges.ang_vel_z,
        ),
        (x_limits, y_limits, yaw_limits),
    )
    return torch.tensor(command_progress, device=env.device)
