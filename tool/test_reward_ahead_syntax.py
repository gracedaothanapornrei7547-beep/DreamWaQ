import torch

def test_ahead_clearance_computation():
    num_envs = 8
    num_rays = 187
    num_feet = 4

    # 1. 模拟射线与足端
    ray_xy_sensor = torch.randn(num_envs, num_rays, 2)
    # 让前两只前足放在 x=0.2, y=+-0.15 附近
    foot_xy_sensor = torch.tensor([
        [0.20, 0.15],
        [0.20, -0.15],
        [-0.20, 0.15],
        [-0.20, -0.15],
    ]).unsqueeze(0).expand(num_envs, -1, -1)

    foot_heights = torch.full((num_envs, num_feet), 0.10)
    ray_hit_heights = torch.zeros(num_envs, num_rays)
    # 模拟前方 0.3m 处有 15cm 台阶 (x in [0.35, 0.60])
    is_step = (ray_xy_sensor[:, :, 0] > 0.30) & (ray_xy_sensor[:, :, 0] < 0.60)
    ray_hit_heights[is_step] = 0.15

    valid_rays = torch.ones((num_envs, num_rays), dtype=torch.bool)
    expanded_heights = ray_hit_heights.unsqueeze(1).expand(-1, num_feet, -1)

    has_valid_local = torch.ones((num_envs, num_feet), dtype=torch.bool)
    nearby_terrain_height = torch.zeros((num_envs, num_feet))
    flat_target_clearance = 0.022
    stairs_target_clearance = 0.16
    flat_max_clearance = 0.028
    excess_penalty_scale = 3.0

    cur_ground_height = torch.where(
        has_valid_local,
        nearby_terrain_height,
        foot_heights - flat_target_clearance,
    )

    pitch_up = torch.zeros(num_envs) # 平地水平
    alpha_pitch = torch.clamp((pitch_up - 0.03) / 0.12, min=0.0, max=1.0).unsqueeze(1)
    span_terrain = torch.zeros(num_envs)
    alpha_terrain = torch.clamp((span_terrain - 0.04) / 0.06, min=0.0, max=1.0).unsqueeze(1)

    # 0.3s 前瞻感知
    ray_x = ray_xy_sensor[:, :, 0].unsqueeze(1)
    ray_y = ray_xy_sensor[:, :, 1].unsqueeze(1)
    foot_x = foot_xy_sensor[:, :, 0].unsqueeze(2)
    foot_y = foot_xy_sensor[:, :, 1].unsqueeze(2)

    dx_ahead = ray_x - foot_x
    dy_ahead = ray_y - foot_y

    ahead_mask = (
        (dx_ahead >= 0.05)
        & (dx_ahead <= 0.40)
        & (torch.abs(dy_ahead) <= 0.15)
        & valid_rays.unsqueeze(1)
    )

    ahead_heights_safe = torch.where(
        ahead_mask,
        expanded_heights,
        torch.full_like(expanded_heights, -1e4),
    )
    ahead_max_height = torch.max(ahead_heights_safe, dim=-1).values
    has_valid_ahead = ahead_mask.any(dim=-1)

    delta_z_ahead = torch.where(
        has_valid_ahead,
        ahead_max_height - cur_ground_height,
        torch.zeros_like(cur_ground_height),
    ).clamp(min=0.0)

    alpha_ahead = torch.clamp((delta_z_ahead - 0.04) / 0.06, min=0.0, max=1.0)
    alpha = torch.maximum(torch.maximum(alpha_pitch, alpha_terrain), alpha_ahead)
    target_clearance = flat_target_clearance + alpha * (stairs_target_clearance - flat_target_clearance)

    clearance = foot_heights - nearby_terrain_height
    foot_vel_z = torch.full((num_envs, num_feet), 0.20)
    landing_gate = torch.clamp((foot_vel_z + 0.15) / 0.20, min=0.0, max=1.0)
    effective_landing_gate = (1.0 - alpha) * 1.0 + alpha * landing_gate

    clearance_shortfall = effective_landing_gate * torch.clamp(
        target_clearance - clearance,
        min=0.0,
    )
    flat_excess = torch.clamp(clearance - float(flat_max_clearance), min=0.0)
    flat_excess_penalty = (1.0 - alpha) * flat_excess * float(excess_penalty_scale)
    clearance_error = torch.clamp(clearance_shortfall + flat_excess_penalty, max=float(stairs_target_clearance))

    print("Success! Tensor shapes:")
    print(f"ahead_mask shape: {ahead_mask.shape}")
    print(f"alpha shape: {alpha.shape}, sample: {alpha[0]}")
    print(f"target_clearance shape: {target_clearance.shape}, sample: {target_clearance[0]}")
    print(f"clearance_error shape: {clearance_error.shape}")

if __name__ == "__main__":
    test_ahead_clearance_computation()
