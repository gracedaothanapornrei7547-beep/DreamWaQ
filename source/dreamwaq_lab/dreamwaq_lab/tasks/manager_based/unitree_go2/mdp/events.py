# Copyright (c) 2024 LunarQuadruped Project.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Domain-randomization events shared by Unitree Go2 tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_motor_strength(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    strength_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Scale each environment's Unitree torque-speed limits.

    A single scale is sampled per environment and applied to every actuator
    group. This models system-wide torque-capacity changes such as battery
    voltage, current limiting, and thermal derating without changing the PD
    gains.
    """

    strength_min, strength_max = (float(value) for value in strength_range)
    if strength_min <= 0.0 or strength_min > strength_max:
        raise ValueError(f"Invalid motor strength range: {strength_range}.")

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    robot = env.scene[asset_cfg.name]
    strength = torch.empty(env_ids.numel(), 1, device=env.device).uniform_(strength_min, strength_max)

    if not hasattr(env, "unitree_motor_strength"):
        env.unitree_motor_strength = torch.ones(env.num_envs, robot.num_joints, device=env.device)
    env.unitree_motor_strength[env_ids] = strength

    if not hasattr(env, "_unitree_motor_strength_defaults"):
        defaults: dict[tuple[str, str], torch.Tensor] = {}
        for actuator_name, actuator in robot.actuators.items():
            for attribute in ("_effort_y1", "_effort_y2"):
                value = getattr(actuator, attribute, None)
                if isinstance(value, torch.Tensor):
                    defaults[(actuator_name, attribute)] = value.clone()
        env._unitree_motor_strength_defaults = defaults

    if not env._unitree_motor_strength_defaults:
        raise RuntimeError(
            "Motor-strength randomization requires a Unitree actuator exposing "
            "torque-speed limits (_effort_y1/_effort_y2)."
        )

    for (actuator_name, attribute), default in env._unitree_motor_strength_defaults.items():
        value = getattr(robot.actuators[actuator_name], attribute)
        value[env_ids] = default[env_ids] * strength


def randomize_action_joint_pos_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    action_term_name: str,
    offset_range: tuple[float, float],
) -> None:
    """Randomize the motor zero offset of a joint-position action term."""

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    action_term = env.action_manager.get_term(action_term_name)
    if not hasattr(action_term, "_offset") or not isinstance(action_term._offset, torch.Tensor):
        raise TypeError(
            f"Action term '{action_term_name}' does not expose a tensor '_offset', "
            "so it cannot be used for motor zero-offset randomization."
        )

    cache_name = f"_default_action_offset_{action_term_name}"
    default_offset = getattr(env, cache_name, None)
    if default_offset is None or default_offset.shape != action_term._offset.shape:
        default_offset = action_term._offset.clone()
        setattr(env, cache_name, default_offset)

    offset_noise = math_utils.sample_uniform(
        offset_range[0],
        offset_range[1],
        (env_ids.numel(), action_term._offset.shape[1]),
        device=env.device,
    )
    action_term._offset[env_ids] = default_offset[env_ids] + offset_noise


__all__ = ["randomize_action_joint_pos_offset", "randomize_motor_strength"]
