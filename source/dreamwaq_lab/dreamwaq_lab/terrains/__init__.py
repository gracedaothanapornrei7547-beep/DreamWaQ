"""Terrain configurations used by the Go2 DreamWaQ task."""

from .config import GO2_EXTENDED_ROUGH_TERRAINS_CFG
from .play_terrain_cfg import apply_play_terrain
from .straight_stairs import MeshStraightAscendingStairsTerrainCfg, straight_ascending_stairs_terrain

__all__ = [
    "GO2_EXTENDED_ROUGH_TERRAINS_CFG",
    "apply_play_terrain",
    "MeshStraightAscendingStairsTerrainCfg",
    "straight_ascending_stairs_terrain",
]
