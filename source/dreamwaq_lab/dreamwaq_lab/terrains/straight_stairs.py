"""Custom Straight Ascending Stairs terrain generator for Unitree Go2."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def straight_ascending_stairs_terrain(
    difficulty: float,
    cfg: MeshStraightAscendingStairsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a unidirectional straight ascending staircase corridor.

    Layout along x-axis:
    - [0, platform_start]: Flat start platform at z = 0.0 (robot spawns here facing +x)
    - [platform_start, platform_start + N * step_width]: Continuous ascending steps (+x is up!)
    - [platform_start + N * step_width, L]: Flat top platform at z = N * step_height

    Width along y-axis:
    - [0, W]: Flat across full width (no cliffs, no drop-offs, no lateral pitfalls)

    Args:
        difficulty: Float in [0.0, 1.0].
        cfg: Configuration parameters.

    Returns:
        tuple[list[trimesh.Trimesh], np.ndarray]: List of trimesh boxes and the origin vector.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    L, W = cfg.size
    platform_start = cfg.platform_start
    platform_end = cfg.platform_end
    step_width = cfg.step_width

    L_steps = max(0.0, L - platform_start - platform_end)
    num_steps = int(L_steps // step_width)
    base_pad = 0.5  # Foundation depth below z=0 to prevent physics ray penetrations

    meshes: list[trimesh.Trimesh] = []

    # 1. Flat start platform
    b_start_dims = (platform_start, W, base_pad)
    b_start_pos = (platform_start * 0.5, W * 0.5, -base_pad * 0.5)
    meshes.append(trimesh.creation.box(b_start_dims, trimesh.transformations.translation_matrix(b_start_pos)))

    # 2. Continuous ascending steps
    for k in range(num_steps):
        top_z = (k + 1) * step_height
        box_h = top_z + base_pad
        box_dims = (step_width, W, box_h)
        x_center = platform_start + (k + 0.5) * step_width
        z_center = top_z - box_h * 0.5
        box_pos = (x_center, W * 0.5, z_center)
        meshes.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    # 3. Flat top platform
    actual_steps_len = num_steps * step_width
    top_len = max(0.1, L - (platform_start + actual_steps_len))
    top_z = num_steps * step_height
    box_h = top_z + base_pad
    box_dims = (top_len, W, box_h)
    x_center = platform_start + actual_steps_len + top_len * 0.5
    z_center = top_z - box_h * 0.5
    box_pos = (x_center, W * 0.5, z_center)
    meshes.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    # 4. Robot spawn origin: Center of start platform at z = 0.0
    origin = np.array([platform_start * 0.5, W * 0.5, 0.0], dtype=np.float64)

    return meshes, origin


@configclass
class MeshStraightAscendingStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a unidirectional straight ascending stairs terrain."""

    function = straight_ascending_stairs_terrain

    step_height_range: tuple[float, float] = (0.03, 0.15)
    """Minimum and maximum height of steps (in m)."""

    step_width: float = 0.3
    """Width of each step along the x-axis (in m)."""

    platform_start: float = 2.5
    """Length of the initial flat ground before stairs begin (in m)."""

    platform_end: float = 2.5
    """Length of the final flat platform at the top of the stairs (in m)."""
