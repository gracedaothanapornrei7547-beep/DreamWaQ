# Copyright (c) 2024 LunarQuadruped Project.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Extended rough terrain configuration for Unitree Go2 training tasks."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


GO2_EXTENDED_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.03, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.03, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.0,
            grid_width=0.45,
            grid_height_range=(0.02, 0.07),
            platform_width=2.0,
            holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.0,
            noise_range=(0.02, 0.07),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.0,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.0,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.25,
        ),

        # 目前第一圈都是2x2m，也就是距离中心1m。
        # 如果有第二圈，第二圈是5.6x5.6m，也就是距离中心2.8m。（前提：platform_width=2.0）
        # rail_2_ratio = 0.6
        # rail_2_inner = platform_width + (size - platform_width) * 0.6
        # rail_2_inner = 2.0 + (8.0 - 2.0) * 0.6 = 5.6m
        "rails": terrain_gen.MeshRailsTerrainCfg(  # 两圈围栏
            proportion=0.0,
            rail_thickness_range=(0.04, 0.10),   # 围栏厚度
            rail_height_range=(0.03, 0.12),      # 围栏高度
            platform_width=2.0,     # 中心平台大小
        ),
        "pit": terrain_gen.MeshPitTerrainCfg(    # 盆地
            proportion=0.0,
            pit_depth_range=(0.03, 0.25),  # 高度
            platform_width=2.0,          
            double_pit=True,              # 是否生成两层
        ),
        "box": terrain_gen.MeshBoxTerrainCfg(    # 凸台
            proportion=0.0,
            box_height_range=(0.03, 0.25), # 高度
            platform_width=2.0,           
            double_box=True,              # 是否生成两层
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(    # 沟壑
            proportion=0.0,
            gap_width_range=(0.03, 0.30),  # 沟壑跳跃宽度
            platform_width=2.0,            
        ),
        "floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(  # 悬浮环
            proportion=0.0,
            ring_width_range=(0.20, 0.50),  # 悬浮环向外扩展的宽度
            ring_height_range=(0.25, 0.45), # 悬浮环离地高度（这里底层逻辑是反着插值，从高往低）
            ring_thickness=0.10,            # 悬浮环在z方向的厚度
            platform_width=2.0,
        ),
    },
)
"""Extended rough terrains configuration for Unitree Go2 tasks."""
