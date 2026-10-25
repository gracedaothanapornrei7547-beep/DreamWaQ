# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def bad_roll_pitch_orientation(
    env: ManagerBasedRLEnv,
    limit_roll: float,
    limit_pitch: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when roll or pitch exceeds its own angle limit."""

    asset: RigidObject = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.logical_or(torch.abs(roll) > limit_roll, torch.abs(pitch) > limit_pitch)


def low_base_height(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.22,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base height relative to terrain falls below threshold (prevent lying flat)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if hasattr(env.scene, "sensors") and "height_scanner" in env.scene.sensors:
        height_sensor = env.scene.sensors["height_scanner"]
        ray_hits_z = height_sensor.data.ray_hits_w[..., 2]
        valid_hits = torch.isfinite(ray_hits_z)
        valid_counts = valid_hits.sum(dim=-1).clamp(min=1)
        ground_z = torch.where(valid_hits, ray_hits_z, torch.zeros_like(ray_hits_z)).sum(dim=-1) / valid_counts
        rel_height = asset.data.root_pos_w[:, 2] - ground_z
        return torch.where(valid_hits.any(dim=-1), rel_height < minimum_height, asset.data.root_pos_w[:, 2] < minimum_height)

    return asset.data.root_pos_w[:, 2] < minimum_height

