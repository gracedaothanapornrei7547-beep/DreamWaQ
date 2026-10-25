"""Compact terrain selection helpers for DreamWaQ evaluation."""

from __future__ import annotations


_FIXED_FOUR_ORDER = (
    "random_rough",
    "boxes",
    "hf_pyramid_slope",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope_inv",
)

_FIXED_FOUR_PROPORTIONS = {
    "random_rough": 0.25,
    "boxes": 0.25,
    "hf_pyramid_slope": 0.25,
    "pyramid_stairs": 0.0,
    "pyramid_stairs_inv": 0.25,
    "hf_pyramid_slope_inv": 0.0,
}

_FIXED_STAIRS_ORDER = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "random_rough",
    "hf_pyramid_slope",
)

_FIXED_STAIRS_PROPORTIONS = {
    "pyramid_stairs": 0.40,
    "pyramid_stairs_inv": 0.40,
    "random_rough": 0.10,
    "hf_pyramid_slope": 0.10,
}


def apply_play_terrain(
    env_cfg,
    profile_name: str = "fixed_four",
    difficulty: float = 0.6,
    seed: int | None = 1,
) -> None:
    """Configure the deterministic terrain used during playback."""

    normalized_name = str(profile_name).strip().lower()
    if normalized_name in {"fixed_stairs", "stairs", "play_stairs"}:
        order = _FIXED_STAIRS_ORDER
        proportions = _FIXED_STAIRS_PROPORTIONS
    elif normalized_name in {"fixed_four", "play_four", "dreamwaq_play_four"}:
        order = _FIXED_FOUR_ORDER
        proportions = _FIXED_FOUR_PROPORTIONS
    else:
        raise ValueError(
            f"Unsupported DreamWaQ play terrain profile {profile_name!r}; "
            "expected 'fixed_four' or 'fixed_stairs'."
        )

    terrain = getattr(env_cfg.scene, "terrain", None)
    terrain_gen = getattr(terrain, "terrain_generator", None)
    if terrain_gen is None:
        return

    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum.terrain_levels = None

    difficulty = max(0.0, min(1.0, float(difficulty)))
    terrain.max_init_terrain_level = 0
    terrain_gen.num_rows = 1
    terrain_gen.num_cols = 4
    terrain_gen.curriculum = True
    terrain_gen.difficulty_range = (difficulty, difficulty)

    if seed is not None:
        env_cfg.seed = int(seed)
        terrain_gen.seed = int(seed)

    sub_terrains = terrain_gen.sub_terrains
    terrain_gen.sub_terrains = {
        name: sub_terrains[name] for name in order if name in sub_terrains
    }
    for name, sub_terrain_cfg in terrain_gen.sub_terrains.items():
        sub_terrain_cfg.proportion = proportions.get(name, 0.0)

