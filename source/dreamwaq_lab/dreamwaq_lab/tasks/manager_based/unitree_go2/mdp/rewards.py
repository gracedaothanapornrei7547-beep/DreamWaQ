# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, quat_apply_yaw, quat_conjugate, quat_from_euler_xyz, yaw_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def base_height_relative_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    footprint_size: tuple[float, float] = (0.55, 0.35),
    reduction: str = "mean",
) -> torch.Tensor:
    """惩罚机身高度偏离“局部地形高度 + 目标高度”。

    原版世界坐标机身高度奖励在没有传感器时直接使用世界坐标系下的
    ``target_height``，在楼梯、方块、斜坡这类复杂地形上会把“爬上台阶后的正常高度”
    也当成过高来惩罚。有 height scanner 时，先估计 base 附近的局部地形高度，
    再计算 base 相对局部地形的高度误差：

    ``(base_z - terrain_z_local - target_height)^2``

    ``footprint_size`` 表示以 base/height scanner 为中心参与平均的局部区域尺寸，
    默认 0.55m x 0.35m，约覆盖 Go2 机身下方和附近支撑区域。
    ``reduction`` 可选 ``"mean"``、``"max"``、``"min"``：训练时默认用均值更平滑；
    如果希望台阶边缘更保守，可以在配置里改成 ``"max"``。
    没有传入 ``sensor_cfg`` 时，退化为平地高度误差，兼容 Flat 任务关闭 height scanner 的配置。
    """

    asset = env.scene[asset_cfg.name]
    if sensor_cfg is None:
        return torch.square(asset.data.root_pos_w[:, 2] - float(target_height))

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    ray_xy = sensor.ray_starts[..., :2]
    if ray_xy.dim() == 2:
        ray_xy = ray_xy.unsqueeze(0).expand(env.num_envs, -1, -1)

    half_x = float(footprint_size[0]) * 0.5
    half_y = float(footprint_size[1]) * 0.5
    local_mask = (torch.abs(ray_xy[..., 0]) <= half_x) & (torch.abs(ray_xy[..., 1]) <= half_y)

    ray_hit_z = sensor.data.ray_hits_w[..., 2]
    finite_mask = torch.isfinite(ray_hit_z)
    local_valid = local_mask & finite_mask

    all_valid_count = finite_mask.sum(dim=1).clamp(min=1)
    all_valid_mean = torch.where(finite_mask, ray_hit_z, torch.zeros_like(ray_hit_z)).sum(dim=1) / all_valid_count
    fallback_terrain_z = torch.where(
        finite_mask.any(dim=1),
        all_valid_mean,
        torch.zeros(env.num_envs, device=ray_hit_z.device, dtype=ray_hit_z.dtype),
    )

    if reduction == "mean":
        local_count = local_valid.sum(dim=1).clamp(min=1)
        local_terrain_z = torch.where(local_valid, ray_hit_z, torch.zeros_like(ray_hit_z)).sum(dim=1) / local_count
    elif reduction == "max":
        local_terrain_z = torch.where(local_valid, ray_hit_z, torch.full_like(ray_hit_z, -1.0e6)).max(dim=1).values
    elif reduction == "min":
        local_terrain_z = torch.where(local_valid, ray_hit_z, torch.full_like(ray_hit_z, 1.0e6)).min(dim=1).values
    else:
        raise ValueError(f"Unsupported reduction for base_height_relative_l2: {reduction!r}")

    local_terrain_z = torch.where(local_valid.any(dim=1), local_terrain_z, fallback_terrain_z)
    base_height_above_local_terrain = asset.data.root_pos_w[:, 2] - local_terrain_z
    return torch.square(base_height_above_local_terrain - float(target_height))


def feet_regulation(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    footprint_size: tuple[float, float] = (0.4, 0.3),
) -> torch.Tensor:
    """Penalize fast horizontal foot motion close to the local terrain.

    This follows the reference project's feet-regulation term.  When using the
    existing large height scanner, only its central ``footprint_size`` region is
    used to reproduce the reference 0.4 m x 0.3 m small scanner.
    """

    asset = env.scene[asset_cfg.name]
    base_pos_w = asset.data.root_pos_w
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_xy_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    if sensor_cfg is None:
        base_height = base_pos_w[:, 2]
    else:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        ray_xy = sensor.ray_starts[..., :2]
        if ray_xy.dim() == 2:
            ray_xy = ray_xy.unsqueeze(0).expand(env.num_envs, -1, -1)

        half_x = float(footprint_size[0]) * 0.5
        half_y = float(footprint_size[1]) * 0.5
        local_mask = (torch.abs(ray_xy[..., 0]) <= half_x) & (torch.abs(ray_xy[..., 1]) <= half_y)

        ray_hit_z = sensor.data.ray_hits_w[..., 2]
        local_valid = local_mask & torch.isfinite(ray_hit_z)
        valid_count = local_valid.sum(dim=1).clamp(min=1)
        estimated_ground_z = torch.where(
            local_valid,
            ray_hit_z,
            torch.zeros_like(ray_hit_z),
        ).sum(dim=1) / valid_count
        fallback_ground_z = base_pos_w[:, 2] - float(base_height_target)
        estimated_ground_z = torch.where(
            local_valid.any(dim=1),
            estimated_ground_z,
            fallback_ground_z,
        )
        base_height = base_pos_w[:, 2] - estimated_ground_z

    gravity_w = torch.as_tensor(env.sim.cfg.gravity, device=env.device, dtype=feet_pos_w.dtype)
    down_w = gravity_w / torch.linalg.vector_norm(gravity_w).clamp(min=1.0e-6)
    feet_to_base_height = torch.sum(
        (feet_pos_w - base_pos_w.unsqueeze(1)) * down_w.view(1, 1, 3),
        dim=-1,
    )
    feet_height = torch.clamp(base_height.unsqueeze(1) - feet_to_base_height, min=0.0)

    decay_height = max(0.025 * float(base_height_target), 1.0e-6)
    return torch.sum(
        torch.sum(torch.square(feet_xy_vel_w), dim=-1) * torch.exp(-feet_height / decay_height),
        dim=-1,
    )


# 关节功率：sum(|tau_i| * |qdot_i|)，适用于位置控制的策略。
def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint mechanical power using Unitree RL Lab's energy form.
    Computes ``sum_i |tau_i| * |qdot_i|`` from the actual applied joint torque
    and joint velocity. This is suitable for position-controlled policies where
    the action itself is not a torque command.
    """
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_vel) * torch.abs(applied_torque), dim=-1)



# # 站立策略用：惩罚机身 xy 平移速度。
# def base_lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     asset = env.scene[asset_cfg.name]
#     return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


# # 站立策略用：惩罚机身 yaw 自转角速度。
# def base_ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     asset = env.scene[asset_cfg.name]
#     return torch.square(asset.data.root_ang_vel_b[:, 2])

# 宇树的关节功率惩罚和论文里的关节加速度惩罚都是为了鼓励更平滑的动作，减少高频抖动。两者都可以用来实现这个目的，但原理不同：
# def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize the energy used by the robot's joints."""
#     asset: Articulation = env.scene[asset_cfg.name]

#     qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
#     qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
#     return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


#def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#    """Penalize joint position deviation from a target value."""
#    # extract the used quantities (to enable type-hinting)
#    asset: Articulation = env.scene[asset_cfg.name]
#    # wrap the joint positions to (-pi, pi)
#    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
#    # compute the reward
#    return torch.sum(torch.square(joint_pos - target), dim=1)

# 抬脚奖励
def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

# 惩罚某只足端一直悬空 超过max_air_time秒
def feet_air_time_limit(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    max_air_time: float = 1.0,
) -> torch.Tensor:
    """Penalize any foot staying in the air for too long."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    # 超过 max_air_time 的部分才罚
    penalty = torch.sum((air_time - max_air_time).clip(min=0.0), dim=1)

    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) > 0.1

    return penalty * moving


# 双足机器人的单脚支撑/迈步奖励（四足不用）
def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

# 脚在接触地面时打滑惩罚
def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def foot_swing_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.055,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize swing feet that stay below a target height on flat terrain."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    contact_force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = contact_force_history.norm(dim=-1).max(dim=1)[0] > contact_threshold
    swing_feet = ~in_contact

    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_shortfall = torch.clamp(target_height - foot_heights, min=0.0)
    swing_count = swing_feet.sum(dim=1).clamp(min=1)
    penalty = torch.sum(torch.square(height_shortfall) * swing_feet, dim=1) / swing_count

    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return penalty * moving


def paper_foot_clearance_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.04,
    nominal_base_height: float = 0.265,
) -> torch.Tensor:
    """Paper-style DreamWaQ foot-clearance penalty without height-scan input.

    Implements the Table-I term ``(p_f,z,des - p_f,z)^2 * v_f,xy``.
    The foot position and velocity are expressed in the robot base frame. Since
    this robot's feet are below the base at stance, ``target_height`` is treated
    as clearance above the nominal ground plane and converted to a desired
    base-frame foot z by subtracting ``nominal_base_height``.
    """

    asset = env.scene[asset_cfg.name]
    foot_pos_rel_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    num_feet = foot_pos_rel_w.shape[1]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_rel_w.reshape(-1, 3)).reshape_as(foot_pos_rel_w)
    foot_vel_b = quat_apply_inverse(root_quat_w, foot_vel_w.reshape(-1, 3)).reshape_as(foot_vel_w)

    desired_foot_z_b = float(target_height) - float(nominal_base_height)
    foot_height_error = desired_foot_z_b - foot_pos_b[..., 2]
    foot_lateral_speed = torch.norm(foot_vel_b[..., :2], dim=-1)
    return torch.sum(torch.square(foot_height_error) * foot_lateral_speed, dim=1)

    
def paper_foot_clearance_l2_new(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.04,
    nominal_base_height: float = 0.265,
) -> torch.Tensor:
    """DreamWaQ足端高度单边惩罚，不使用height scan。

    公式形式仍然参考论文里的 ``(p_f,z,des - p_f,z)^2 * ||v_f,xy||``，
    但这里改成单边版本：

    - 足端低于期望高度时，惩罚 ``max(p_f,z,des - p_f,z, 0)^2``；
    - 足端高于期望高度时，不再惩罚，避免高台阶/楼梯地形中“必须抬得更高”反而被固定目标高度惩罚。

    足端位置和速度都转换到机身坐标系下计算。由于四足支撑时足端通常位于base下方，
    ``target_height`` 表示相对名义地面的期望离地高度，函数内部会转换成base系下的期望足端z坐标。
    """

    asset = env.scene[asset_cfg.name]
    foot_pos_rel_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    num_feet = foot_pos_rel_w.shape[1]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_rel_w.reshape(-1, 3)).reshape_as(foot_pos_rel_w)
    foot_vel_b = quat_apply_inverse(root_quat_w, foot_vel_w.reshape(-1, 3)).reshape_as(foot_vel_w)

    # 期望足端高度：target_height 是相对地面的抬脚高度，转换到base坐标系后为 target_height - nominal_base_height。
    desired_foot_z_b = float(target_height) - float(nominal_base_height)
    # 单边惩罚：只惩罚“抬脚不够”，超过目标高度不惩罚。
    foot_height_shortfall = torch.clamp(desired_foot_z_b - foot_pos_b[..., 2], min=0.0)
    foot_lateral_speed = torch.norm(foot_vel_b[..., :2], dim=-1)
    return torch.sum(torch.square(foot_height_shortfall) * foot_lateral_speed, dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    target_clearance: float = 0.04,     # 足端超过障碍物高度超过4cm就满分
    contact_threshold: float = 1.0,
    top_k: int = 8,
) -> torch.Tensor:
    """足端越障奖励：鼓励摆动腿抬高到高于该足端附近的局部地形。

    相比之前“从整张 height scan 里做全局 top-k”的近似版本，这一版更接近论文里
    按每只脚分别计算局部 clearance 的思路：

    - 通过接触力识别哪些脚当前处于摆动相；
    - 将每只脚的位置变换到 height scanner 的 yaw 对齐局部坐标系；
    - 对每只脚分别寻找其附近最近的若干个地形采样点；
    - 用这些局部采样点估计该脚附近的地形高度；
    - 计算摆动脚相对局部地形的 clearance，并给予奖励。

    这样得到的是“以每只脚为中心”的局部越障奖励，而不是“以 base 为中心”的全局高度奖励。
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # 根据接触力识别当前处于摆动相的脚。
    contact_force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = contact_force_history.norm(dim=-1).max(dim=1)[0] > contact_threshold
    swing_feet = ~in_contact

    # 获取四个足端在世界坐标系下的当前高度。
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # 将足端位置变换到 height scanner 的 yaw 对齐局部坐标系。
    # 由于扫描器使用 ray_alignment="yaw"，因此在这个局部 xy 平面里比较最合适。
    sensor_pos_w = height_sensor.data.pos_w
    sensor_yaw_quat_w = yaw_quat(height_sensor.data.quat_w)
    foot_pos_rel_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - sensor_pos_w.unsqueeze(1)
    foot_pos_rel_sensor = quat_apply_inverse(
        sensor_yaw_quat_w.unsqueeze(1).expand(-1, foot_pos_rel_w.shape[1], -1).reshape(-1, 4),
        foot_pos_rel_w.reshape(-1, 3),
    ).reshape_as(foot_pos_rel_w)
    foot_xy_sensor = foot_pos_rel_sensor[..., :2]

    # height scanner 中每条射线在传感器局部坐标系下的起点 xy。
    # 这些点已经包含了传感器自身的 offset。
    ray_xy_sensor = height_sensor.ray_starts[..., :2]
    if ray_xy_sensor.dim() == 2:
        ray_xy_sensor = ray_xy_sensor.unsqueeze(0).expand(env.num_envs, -1, -1)

    # 对每只脚分别寻找最近的局部地形采样点。
    # 张量形状：[num_envs, num_feet, num_rays]
    foot_to_ray_dist = torch.cdist(foot_xy_sensor, ray_xy_sensor)
    num_rays = foot_to_ray_dist.shape[-1]
    top_k = min(top_k, num_rays)
    nearest_ray_ids = torch.topk(foot_to_ray_dist, k=top_k, dim=-1, largest=False).indices

    ray_hit_heights = height_sensor.data.ray_hits_w[..., 2]
    expanded_hits = ray_hit_heights.unsqueeze(1).expand(-1, foot_heights.shape[1], -1)
    local_hit_heights = torch.gather(expanded_hits, dim=2, index=nearest_ray_ids)

    # 对每只脚，使用附近采样点里最高的那个点作为保守的局部障碍高度估计。
    local_hit_heights = torch.where(
        torch.isfinite(local_hit_heights),
        local_hit_heights,
        torch.full_like(local_hit_heights, -1.0e6),
    )
    nearby_terrain_height = torch.max(local_hit_heights, dim=2).values

    # 计算间隙分数 (Clearance Score)
    # 间隙 = 足部高度 - 该足端附近地形高度
    clearance = foot_heights - nearby_terrain_height
    clearance_score = torch.clamp(clearance / target_clearance, min=0.0, max=1.0)

    swing_count = swing_feet.sum(dim=1).clamp(min=1)
    reward = torch.sum(clearance_score * swing_feet, dim=1) / swing_count

    # 当速度命令几乎为零时，不给这项奖励。
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= moving
    return reward


def _terrain_relative_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    target_clearance: float = 0.04,
    contact_threshold: float = 2.0,
    top_k: int = 8,
    speed_scale: float = 2.0,
    error_order: int = 2,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute a terrain-relative swing-foot clearance penalty."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # 使用当前接触力，避免历史窗口导致摆动相识别延迟。
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(contact_forces, dim=-1) > contact_threshold
    swing_feet = ~in_contact

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_heights = foot_pos_w[..., 2]

    # 足端转换到height scanner的yaw对齐局部坐标系。
    sensor_pos_w = height_sensor.data.pos_w
    sensor_yaw_quat_w = yaw_quat(height_sensor.data.quat_w)

    foot_pos_rel_w = foot_pos_w - sensor_pos_w.unsqueeze(1)
    foot_pos_sensor = quat_apply_inverse(
        sensor_yaw_quat_w.unsqueeze(1)
        .expand(-1, foot_pos_rel_w.shape[1], -1)
        .reshape(-1, 4),
        foot_pos_rel_w.reshape(-1, 3),
    ).reshape_as(foot_pos_rel_w)

    foot_xy_sensor = foot_pos_sensor[..., :2]

    # 射线局部起点已经包含RayCasterCfg中的offset。
    ray_xy_sensor = height_sensor.ray_starts[..., :2]
    if ray_xy_sensor.dim() == 2:
        ray_xy_sensor = ray_xy_sensor.unsqueeze(0).expand(
            env.num_envs, -1, -1
        )

    ray_hit_heights = height_sensor.data.ray_hits_w[..., 2]
    valid_rays = torch.isfinite(ray_hit_heights)

    # 计算足端到地形采样点的距离。
    foot_to_ray_dist = torch.cdist(foot_xy_sensor, ray_xy_sensor)

    # 无效射线不参与最近邻选择。
    foot_to_ray_dist = torch.where(
        valid_rays.unsqueeze(1),
        foot_to_ray_dist,
        torch.full_like(foot_to_ray_dist, float("inf")),
    )

    top_k = min(top_k, foot_to_ray_dist.shape[-1])
    nearest_ids = torch.topk(
        foot_to_ray_dist,
        k=top_k,
        dim=-1,
        largest=False,
    ).indices
    expanded_heights = ray_hit_heights.unsqueeze(1).expand(
        -1, foot_heights.shape[1], -1
    )
    local_heights = torch.gather(
        expanded_heights,
        dim=2,
        index=nearest_ids,
    )
    valid_local = torch.isfinite(local_heights)
    local_heights_safe = torch.where(
        valid_local,
        local_heights,
        torch.full_like(local_heights, -float("inf")),
    )
    nearby_terrain_height = torch.max(
        local_heights_safe,
        dim=2,
    ).values
    has_valid_local = valid_local.any(dim=2)
    # 没有有效地形点时关闭该脚的clearance惩罚。
    nearby_terrain_height = torch.where(
        has_valid_local,
        nearby_terrain_height,
        foot_heights - target_clearance,
    )

    clearance = foot_heights - nearby_terrain_height
    # 只惩罚离地间隙不足。
    clearance_shortfall = torch.clamp(
        target_clearance - clearance,
        min=0.0,
    )
    if normalize:
        # 归一化后误差范围为[0, 1]，使奖励尺度不再随target_clearance的单位和取值变化。
        clearance_error = torch.clamp(
            clearance_shortfall / max(float(target_clearance), 1.0e-6),
            max=1.0,
        )
    elif error_order == 1:
        # 未归一化的L1版本限制在[0, target_clearance]。
        clearance_error = torch.clamp(clearance_shortfall, max=float(target_clearance))
    else:
        clearance_error = clearance_shortfall

    if error_order == 1:
        shaped_clearance_error = clearance_error
    elif error_order == 2:
        shaped_clearance_error = torch.square(clearance_error)
    else:
        raise ValueError(f"Unsupported terrain-relative foot-clearance error order: {error_order}")

    # 运动较快的摆动足受到更明显的约束。
    foot_speed_xy = torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2],
        dim=-1,
    )
    speed_weight = torch.tanh(speed_scale * foot_speed_xy)
    swing_count = swing_feet.sum(dim=1).clamp(min=1)
    penalty = torch.sum(
        shaped_clearance_error
        * speed_weight
        * swing_feet,
        dim=1,
    ) / swing_count
    command = env.command_manager.get_command(command_name)
    # 同时考虑平移和原地转向。
    moving = (
        (torch.norm(command[:, :2], dim=1) > 0.1)
        | (torch.abs(command[:, 2]) > 0.1)
    )

    return penalty * moving


def terrain_relative_foot_clearance_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    target_clearance: float = 0.06,
    contact_threshold: float = 1.0,
    top_k: int = 8,
    speed_scale: float = 2.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Penalize terrain-relative swing-foot clearance shortfall with an L1 kernel.

    By default, the shortfall is divided by ``target_clearance`` and clamped to
    ``[0, 1]``.  Set ``normalize=False`` to return a raw L1 clearance penalty
    whose per-foot height-error factor lies in ``[0, target_clearance]``.
    
    只处理摆动足；
    用局部地形高度而不是世界坐标高度；
    只惩罚抬脚不足，不惩罚抬脚过高；
    结合足端水平速度；
    静止指令下关闭奖励。

    """

    return _terrain_relative_foot_clearance(
        env=env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        height_sensor_cfg=height_sensor_cfg,
        target_clearance=target_clearance,
        contact_threshold=contact_threshold,
        top_k=top_k,
        speed_scale=speed_scale,
        error_order=1,
        normalize=normalize,
    )


def terrain_relative_foot_clearance_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    target_clearance: float = 0.04,
    contact_threshold: float = 1.0,
    top_k: int = 8,
    speed_scale: float = 2.0,
) -> torch.Tensor:
    """Penalize terrain-relative swing-foot clearance shortfall with a raw L2 kernel."""

    return _terrain_relative_foot_clearance(
        env=env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        height_sensor_cfg=height_sensor_cfg,
        target_clearance=target_clearance,
        contact_threshold=contact_threshold,
        top_k=top_k,
        speed_scale=speed_scale,
        error_order=2,
        normalize=False,
    )



def _contact_states(
    contact_sensor: ContactSensor,
    body_ids: list[int] | slice,
    contact_threshold: float,
) -> torch.Tensor:
    """Return a boolean contact state for every requested body.

    The reward functions below intentionally use the same maximum-over-history
    convention as Isaac Lab's ``undesired_contacts`` term.  This makes a
    contact diagnostic and its corresponding reward directly comparable.
    """

    force_history = contact_sensor.data.net_forces_w_history[:, :, body_ids, :]
    return force_history.norm(dim=-1).max(dim=1)[0] > float(contact_threshold)


def _terrain_height_span(height_sensor: RayCaster) -> torch.Tensor:
    """Return the finite local terrain height span for each environment.

    A gap or a box produces a large span in the privileged height scanner,
    whereas level ground is close to zero.  Invalid ray hits deliberately do
    not make an environment look flat.
    """

    heights = height_sensor.data.ray_hits_w[..., 2]
    finite = torch.isfinite(heights)
    min_height = torch.where(finite, heights, torch.full_like(heights, float("inf"))).min(dim=1).values
    max_height = torch.where(finite, heights, torch.full_like(heights, -float("inf"))).max(dim=1).values
    span = max_height - min_height
    return torch.where(finite.any(dim=1), span, torch.full_like(span, float("inf")))


def trot_contact_phase_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    flatness_height_span: float = 0.04,
    min_forward_command: float = 0.20,
    max_forward_command: float = 0.80,
    max_abs_yaw_command: float = 0.15,
) -> torch.Tensor:
    """Reward a diagonal trot only on flat, straight, forward locomotion.

    The term is deliberately gated by the privileged height scanner.  It is
    exactly zero near gaps, pits, boxes, rough relief, standing commands, and
    turning commands, so it cannot directly oppose a jump/bound maneuver.
    Foot order must be ``FL, FR, RL, RR``.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
    contacts = _contact_states(contact_sensor, sensor_cfg.body_ids, contact_threshold)
    if contacts.shape[1] != 4:
        raise ValueError(
            "trot_contact_phase_reward requires exactly four feet ordered as FL, FR, RL, RR; "
            f"received {contacts.shape[1]} bodies."
        )
    fl, fr, rl, rr = contacts.unbind(dim=1)

    # Diagonal legs should share a contact state, and the two diagonal pairs
    # should be in opposite phases.  The two-contact mask excludes standing,
    # pronking, and four-foot support from receiving the gait bonus.
    diagonal_sync = (fl == rr) & (fr == rl)
    diagonal_opposition = (fl != fr) & (rr != rl)
    two_contacts = contacts.sum(dim=1) == 2
    trot_pattern = (diagonal_sync & diagonal_opposition & two_contacts).to(dtype=torch.float32)

    command = env.command_manager.get_command(command_name)
    flat = _terrain_height_span(height_sensor) <= float(flatness_height_span)
    straight_forward = (
        (command[:, 0] >= float(min_forward_command))
        & (command[:, 0] <= float(max_forward_command))
        & (torch.abs(command[:, 1]) <= 1.0e-6)
        & (torch.abs(command[:, 2]) <= float(max_abs_yaw_command))
    )
    return trot_pattern * (flat & straight_forward).to(dtype=torch.float32)


def rear_foot_obstacle_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    target_clearance: float = 0.08,
    contact_threshold: float = 1.0,
    obstacle_height_span: float = 0.08,
    top_k: int = 4,
) -> torch.Tensor:
    """Reward swing clearance of the two rear feet near non-flat terrain.

    This is a targeted alternative to a global foot-clearance reward.  It is
    zero on level terrain and only considers the rear feet (RL and RR), which
    are the feet observed to catch the far edge of a gap.  The reward remains
    valid for pits and boxes because it is based on the local scanned terrain,
    not a terrain-type label exposed to the actor.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 4 or len(sensor_cfg.body_ids) != 4:
        raise ValueError("rear_foot_obstacle_clearance_reward requires FL, FR, RL, RR foot body selections.")

    rear_ids = [2, 3]
    swing_rear_feet = ~_contact_states(contact_sensor, sensor_cfg.body_ids, contact_threshold)[:, rear_ids]
    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids[rear_ids], :]

    sensor_pos_w = height_sensor.data.pos_w
    sensor_yaw_quat_w = yaw_quat(height_sensor.data.quat_w)
    foot_rel_w = foot_positions - sensor_pos_w.unsqueeze(1)
    foot_rel_sensor = quat_apply_inverse(
        sensor_yaw_quat_w.unsqueeze(1).expand(-1, len(rear_ids), -1).reshape(-1, 4),
        foot_rel_w.reshape(-1, 3),
    ).reshape_as(foot_rel_w)
    ray_xy = height_sensor.ray_starts[..., :2]
    if ray_xy.dim() == 2:
        ray_xy = ray_xy.unsqueeze(0).expand(env.num_envs, -1, -1)
    nearest = torch.topk(
        torch.cdist(foot_rel_sensor[..., :2], ray_xy),
        k=min(int(top_k), ray_xy.shape[1]),
        dim=-1,
        largest=False,
    ).indices
    ray_heights = height_sensor.data.ray_hits_w[..., 2]
    local_heights = torch.gather(ray_heights.unsqueeze(1).expand(-1, len(rear_ids), -1), 2, nearest)
    local_heights = torch.where(torch.isfinite(local_heights), local_heights, torch.full_like(local_heights, -1.0e6))
    local_terrain = local_heights.max(dim=2).values
    clearance = foot_positions[..., 2] - local_terrain
    clearance_score = torch.clamp(clearance / float(target_clearance), min=0.0, max=1.0)
    swing_count = swing_rear_feet.sum(dim=1).clamp(min=1)
    reward = torch.sum(clearance_score * swing_rear_feet, dim=1) / swing_count

    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :2], dim=1) > 0.1
    near_obstacle = _terrain_height_span(height_sensor) >= float(obstacle_height_span)
    return reward * (moving & near_obstacle).to(dtype=torch.float32)


# 在 yaw 对齐后的机器人坐标系中，奖励 xy 线速度跟踪
def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

# 在世界坐标系中，奖励 yaw 角速度跟踪。
def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def _wtw_command_term(env: ManagerBasedRLEnv, command_name: str):
    """Return the WTW command term and fail loudly if the wrong command is wired."""

    command_term = env.command_manager.get_term(command_name)
    required_attrs = ("clock_inputs", "desired_contact_states", "foot_indices")
    missing = [attr for attr in required_attrs if not hasattr(command_term, attr)]
    if missing:
        raise AttributeError(f"Command term {command_name!r} is not a WalkTheseWaysCommand; missing {missing}.")
    return command_term


def wtw_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """WTW xy velocity tracking term ``exp(-|v_xy - v_cmd|^2 / sigma)``."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / sigma)


def wtw_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """WTW yaw-rate tracking term ``exp(-(w_z - w_cmd)^2 / sigma)``."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / sigma)


def wtw_tracking_contacts_shaped_force(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    sigma: float = 100.0,
) -> torch.Tensor:
    """Penalize contact force during commanded swing phase.

    This mirrors the official CoRL reward: the function itself returns a
    negative penalty and is normally used with a positive weight.
    """

    command_term = _wtw_command_term(env, command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    desired_contact = command_term.desired_contact_states
    reward = -(1.0 - desired_contact) * (1.0 - torch.exp(-torch.square(foot_forces) / sigma))
    return torch.mean(reward, dim=1)


def wtw_tracking_contacts_shaped_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    sigma: float = 10.0,
) -> torch.Tensor:
    """Penalize foot motion during commanded stance phase."""

    command_term = _wtw_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    foot_velocities = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)
    desired_contact = command_term.desired_contact_states
    reward = -desired_contact * (1.0 - torch.exp(-torch.square(foot_velocities) / sigma))
    return torch.mean(reward, dim=1)


def wtw_body_height_cmd_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    nominal_base_height: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Negative squared tracking error for commanded body height."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_height = nominal_base_height + command[:, 3]
    return -torch.square(asset.data.root_pos_w[:, 2] - target_height)


def wtw_orientation_control_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded body pitch/roll through projected gravity error."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    pitch_cmd = command[:, 10]
    roll_cmd = command[:, 11]
    zeros = torch.zeros_like(pitch_cmd)
    desired_base_quat = quat_from_euler_xyz(-roll_cmd, -pitch_cmd, zeros)
    gravity_vec = torch.zeros(env.num_envs, 3, device=command.device, dtype=command.dtype)
    gravity_vec[:, 2] = -1.0
    desired_projected_gravity = quat_apply_inverse(desired_base_quat, gravity_vec)
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2] - desired_projected_gravity[:, :2]), dim=1)


def wtw_feet_clearance_cmd_linear(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    foot_radius: float = 0.02,
) -> torch.Tensor:
    """WTW footswing-height tracking penalty for swing feet."""

    command_term = _wtw_command_term(env, command_name)
    command = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    phases = 1.0 - torch.abs(1.0 - torch.clamp(command_term.foot_indices * 2.0 - 1.0, 0.0, 1.0) * 2.0)
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    target_height = command[:, 9].unsqueeze(1) * phases + foot_radius
    clearance_error = torch.square(target_height - foot_height) * (1.0 - command_term.desired_contact_states)
    return torch.sum(clearance_error, dim=1)


def wtw_raibert_heuristic(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Raibert-style stance-width/length foot placement penalty."""

    command_term = _wtw_command_term(env, command_name)
    command = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]

    foot_pos_rel_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    root_quat_inv = quat_conjugate(asset.data.root_quat_w)
    root_quat_inv = root_quat_inv.unsqueeze(1).expand(-1, foot_pos_rel_w.shape[1], -1).reshape(-1, 4)
    footsteps_in_body_frame = quat_apply_yaw(root_quat_inv, foot_pos_rel_w.reshape(-1, 3)).reshape_as(foot_pos_rel_w)

    desired_stance_width = command[:, 12:13]
    desired_stance_length = command[:, 13:14]
    desired_ys_nom = torch.cat(
        (
            desired_stance_width / 2.0,
            -desired_stance_width / 2.0,
            desired_stance_width / 2.0,
            -desired_stance_width / 2.0,
        ),
        dim=1,
    )
    desired_xs_nom = torch.cat(
        (
            desired_stance_length / 2.0,
            desired_stance_length / 2.0,
            -desired_stance_length / 2.0,
            -desired_stance_length / 2.0,
        ),
        dim=1,
    )

    phases = torch.abs(1.0 - command_term.foot_indices * 2.0) - 0.5
    frequencies = command[:, 4].unsqueeze(1).clamp(min=1.0e-6)
    x_vel_des = command[:, 0:1]
    yaw_vel_des = command[:, 2:3]
    y_vel_des = yaw_vel_des * desired_stance_length / 2.0
    desired_ys_offset = phases * y_vel_des * (0.5 / frequencies)
    desired_ys_offset[:, 2:4] *= -1.0
    desired_xs_offset = phases * x_vel_des * (0.5 / frequencies)

    desired_ys_nom = desired_ys_nom + desired_ys_offset
    desired_xs_nom = desired_xs_nom + desired_xs_offset
    desired_footsteps_body_frame = torch.stack((desired_xs_nom, desired_ys_nom), dim=-1)
    err_raibert = desired_footsteps_body_frame - footsteps_in_body_frame[:, :, :2]
    return torch.sum(torch.square(err_raibert), dim=(1, 2))


def wtw_feet_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize horizontal foot speed while feet are in contact."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    contacts = contacts > contact_threshold
    foot_vel_xy = torch.square(torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1))
    return torch.sum(contacts * foot_vel_xy, dim=1)


def wtw_action_smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
    """First-order action smoothness penalty used by WTW."""

    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    return torch.sum(torch.square(action - prev_action), dim=1)


def wtw_action_smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Second-order action smoothness penalty used by WTW."""

    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    if not hasattr(env, "_wtw_prev_prev_action"):
        env._wtw_prev_prev_action = torch.zeros_like(prev_action)
    prev_prev_action = env._wtw_prev_prev_action
    penalty = torch.sum(torch.square(action - 2.0 * prev_action + prev_prev_action), dim=1)
    env._wtw_prev_prev_action = prev_action.clone()
    return penalty


def wtw_ji22_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_sensor_cfg: SceneEntityCfg,
    foot_asset_cfg: SceneEntityCfg,
    collision_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma_rew_neg: float = 0.02,
    tracking_sigma: float = 0.25,
    tracking_sigma_yaw: float = 0.25,
    gait_force_sigma: float = 100.0,
    gait_vel_sigma: float = 10.0,
    nominal_base_height: float = 0.265,
    tracking_lin_vel_weight: float = 1.0,
    tracking_ang_vel_weight: float = 0.5,
    tracking_contacts_force_weight: float = 4.0,
    tracking_contacts_vel_weight: float = 4.0,
    jump_weight: float = 10.0,
    orientation_control_weight: float = -5.0,
    raibert_heuristic_weight: float = -10.0,
    feet_clearance_cmd_weight: float = -30.0,
    feet_slip_weight: float = -0.04,
    action_smoothness_1_weight: float = -0.1,
    action_smoothness_2_weight: float = -0.1,
    lin_vel_z_weight: float = -0.02,
    ang_vel_xy_weight: float = -0.001,
    torques_weight: float = -2.0e-5,
    dof_vel_weight: float = -2.0e-5,
    dof_acc_weight: float = -5.0e-9,
    dof_pos_limits_weight: float = -1.0,
    collision_weight: float = -5.0,
) -> torch.Tensor:
    """WTW JI22-style multiplicative reward aggregation.

    The official implementation first multiplies each term by its weight and
    control timestep, splits the weighted terms into positive task reward and
    negative penalties, then returns ``pos * exp(neg / sigma_rew_neg)``. Isaac
    Lab applies the timestep outside this function, so the exponential uses
    ``env.step_dt * neg_per_second`` to preserve the same ordering.
    """

    if sigma_rew_neg <= 0.0:
        raise ValueError(f"sigma_rew_neg must be positive, got {sigma_rew_neg}.")

    command_term = _wtw_command_term(env, command_name)
    command = env.command_manager.get_command(command_name)
    pos_reward = torch.zeros(env.num_envs, device=command.device, dtype=command.dtype)
    neg_reward = torch.zeros_like(pos_reward)

    tracking_lin_vel = wtw_track_lin_vel_xy_exp(
        env, command_name=command_name, sigma=tracking_sigma, asset_cfg=asset_cfg
    )
    tracking_ang_vel = wtw_track_ang_vel_z_exp(
        env, command_name=command_name, sigma=tracking_sigma_yaw, asset_cfg=asset_cfg
    )
    tracking_contacts_force = wtw_tracking_contacts_shaped_force(
        env, command_name=command_name, sensor_cfg=foot_sensor_cfg, sigma=gait_force_sigma
    )
    tracking_contacts_vel = wtw_tracking_contacts_shaped_vel(
        env,
        command_name=command_name,
        sensor_cfg=foot_sensor_cfg,
        asset_cfg=foot_asset_cfg,
        sigma=gait_vel_sigma,
    )
    body_height = wtw_body_height_cmd_l2(
        env, command_name=command_name, nominal_base_height=nominal_base_height, asset_cfg=asset_cfg
    )
    orientation_control = wtw_orientation_control_l2(env, command_name=command_name, asset_cfg=asset_cfg)
    raibert_heuristic = wtw_raibert_heuristic(env, command_name=command_name, asset_cfg=foot_asset_cfg)
    feet_clearance_cmd = wtw_feet_clearance_cmd_linear(
        env, command_name=command_name, sensor_cfg=foot_sensor_cfg, asset_cfg=foot_asset_cfg
    )
    feet_slip = wtw_feet_slip(env, sensor_cfg=foot_sensor_cfg, asset_cfg=foot_asset_cfg)
    action_smoothness_1 = wtw_action_smoothness_1(env)
    action_smoothness_2 = wtw_action_smoothness_2(env)
    lin_vel_z = mdp.lin_vel_z_l2(env, asset_cfg=asset_cfg)
    ang_vel_xy = mdp.ang_vel_xy_l2(env, asset_cfg=asset_cfg)
    torques = mdp.joint_torques_l2(env, asset_cfg=asset_cfg)
    dof_vel = mdp.joint_vel_l2(env, asset_cfg=asset_cfg)
    dof_acc = mdp.joint_acc_l2(env, asset_cfg=asset_cfg)
    dof_pos_limits = mdp.joint_pos_limits(env, asset_cfg=asset_cfg)
    collision = mdp.undesired_contacts(env, sensor_cfg=collision_sensor_cfg, threshold=1.0)

    tracking_lin_vel_term = tracking_lin_vel_weight * tracking_lin_vel
    tracking_ang_vel_term = tracking_ang_vel_weight * tracking_ang_vel
    tracking_contacts_force_term = tracking_contacts_force_weight * tracking_contacts_force
    tracking_contacts_vel_term = tracking_contacts_vel_weight * tracking_contacts_vel
    body_height_term = jump_weight * body_height
    orientation_control_term = orientation_control_weight * orientation_control
    raibert_heuristic_term = raibert_heuristic_weight * raibert_heuristic
    feet_clearance_cmd_term = feet_clearance_cmd_weight * feet_clearance_cmd
    feet_slip_term = feet_slip_weight * feet_slip
    action_smoothness_1_term = action_smoothness_1_weight * action_smoothness_1
    action_smoothness_2_term = action_smoothness_2_weight * action_smoothness_2
    lin_vel_z_term = lin_vel_z_weight * lin_vel_z
    ang_vel_xy_term = ang_vel_xy_weight * ang_vel_xy
    torques_term = torques_weight * torques
    dof_vel_term = dof_vel_weight * dof_vel
    dof_acc_term = dof_acc_weight * dof_acc
    dof_pos_limits_term = dof_pos_limits_weight * dof_pos_limits
    collision_term = collision_weight * collision

    pos_reward += tracking_lin_vel_term
    pos_reward += tracking_ang_vel_term
    neg_reward += tracking_contacts_force_term
    neg_reward += tracking_contacts_vel_term
    neg_reward += body_height_term
    neg_reward += orientation_control_term
    neg_reward += raibert_heuristic_term
    neg_reward += feet_clearance_cmd_term
    neg_reward += feet_slip_term
    neg_reward += action_smoothness_1_term
    neg_reward += action_smoothness_2_term
    neg_reward += lin_vel_z_term
    neg_reward += ang_vel_xy_term
    neg_reward += torques_term
    neg_reward += dof_vel_term
    neg_reward += dof_acc_term
    neg_reward += dof_pos_limits_term
    neg_reward += collision_term

    penalty_gate = torch.exp((env.step_dt * neg_reward) / sigma_rew_neg)
    total_reward = pos_reward * penalty_gate

    # Official command curriculum thresholds are expressed on normalized task
    # success signals. Contact-shaped terms are negative penalties in [-1, 0],
    # so add one to recover the "good contact" score used by WTW curricula.
    env._wtw_curriculum_tracking_lin_vel = tracking_lin_vel.detach()
    env._wtw_curriculum_tracking_ang_vel = tracking_ang_vel.detach()
    env._wtw_curriculum_tracking_contacts_force = torch.clamp(1.0 + tracking_contacts_force, 0.0, 1.0).detach()
    env._wtw_curriculum_tracking_contacts_vel = torch.clamp(1.0 + tracking_contacts_vel, 0.0, 1.0).detach()
    env._wtw_ji22_pos_reward = pos_reward.detach()
    env._wtw_ji22_neg_reward = neg_reward.detach()
    env._wtw_ji22_total_reward = total_reward.detach()
    if hasattr(command_term, "update_reward_metrics"):
        command_term.update_reward_metrics(
            {
                "tracking_lin_vel": tracking_lin_vel_term,
                "tracking_ang_vel": tracking_ang_vel_term,
                "tracking_contacts_force": tracking_contacts_force_term,
                "tracking_contacts_vel": tracking_contacts_vel_term,
                "body_height": body_height_term,
                "orientation": orientation_control_term,
                "raibert": raibert_heuristic_term,
                "feet_clearance": feet_clearance_cmd_term,
                "feet_slip": feet_slip_term,
                "action_smoothness_1": action_smoothness_1_term,
                "action_smoothness_2": action_smoothness_2_term,
                "lin_vel_z": lin_vel_z_term,
                "ang_vel_xy": ang_vel_xy_term,
                "torques": torques_term,
                "dof_vel": dof_vel_term,
                "dof_acc": dof_acc_term,
                "dof_pos_limits": dof_pos_limits_term,
                "collision": collision_term,
                "pos_sum": pos_reward,
                "neg_sum": neg_reward,
                "penalty_gate": penalty_gate,
                "total": total_reward,
            }
        )
    return total_reward


def lin_vel_xy_orthogonal_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal velocity orthogonal to the commanded direction.

    This complements the standard tracking term by explicitly discouraging sideways drift when the robot is asked to
    move in a particular planar direction. The velocity and command are both evaluated in the body frame.
    """

    asset = env.scene[asset_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    body_lin_vel_xy = asset.data.root_lin_vel_b[:, :2]

    command_norm = torch.norm(command_xy, dim=1, keepdim=True)
    moving = command_norm.squeeze(-1) > command_threshold
    command_dir = command_xy / command_norm.clamp(min=1.0e-6)

    commanded_component = torch.sum(body_lin_vel_xy * command_dir, dim=1, keepdim=True) * command_dir
    orthogonal_component = body_lin_vel_xy - commanded_component
    penalty = torch.sum(torch.square(orthogonal_component), dim=1)
    return penalty * moving.float()

# 只在 command generator 抽到 standing env 时，惩罚关节偏离默认站姿。
def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint offsets only for explicitly sampled standing environments.
    ``UniformVelocityCommand`` samples standing environments using
    ``rel_standing_envs`` and stores the mask in ``is_standing_env``. Using
    that mask avoids penalizing ordinary low-speed walking commands. The
    threshold fallback is only for command terms that do not expose the mask.
    """

    command_term = env.command_manager.get_term(command_name)
    standing_envs = getattr(command_term, "is_standing_env", None)
    if standing_envs is None:
        command = env.command_manager.get_command(command_name)
        standing_envs = torch.linalg.norm(command, dim=1) < command_threshold

    return mdp.joint_deviation_l1(env, asset_cfg) * standing_envs.to(dtype=torch.float32)



# def diagonal_gait(
#     env,
#     sensor_cfg: SceneEntityCfg,
#     command_name: str = "base_velocity",
#     contact_threshold: float = 1.0,
#     sync_tolerance: float = 0.10,
# ) -> torch.Tensor:
#     """Reward diagonal trot gait.

#     Foot order must be:
#         0: FL_foot
#         1: FR_foot
#         2: RL_foot
#         3: RR_foot
#     """

#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

#     # Contact state
#     forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
#     contacts = forces.norm(dim=-1).max(dim=1)[0] > contact_threshold

#     contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
#     air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

#     fl, fr, rl, rr = contacts[:, 0], contacts[:, 1], contacts[:, 2], contacts[:, 3]

#     fl_ct, fr_ct, rl_ct, rr_ct = contact_time[:, 0], contact_time[:, 1], contact_time[:, 2], contact_time[:, 3]
#     fl_at, fr_at, rl_at, rr_at = air_time[:, 0], air_time[:, 1], air_time[:, 2], air_time[:, 3]

#     # ------------------------------------------------------------
#     # 1. 对角腿同相：FL-RR 同步，FR-RL 同步
#     # ------------------------------------------------------------
#     diag1_same_contact = fl & rr & (torch.abs(fl_ct - rr_ct) < sync_tolerance)
#     diag1_same_air = (~fl) & (~rr) & (torch.abs(fl_at - rr_at) < sync_tolerance)

#     diag2_same_contact = fr & rl & (torch.abs(fr_ct - rl_ct) < sync_tolerance)
#     diag2_same_air = (~fr) & (~rl) & (torch.abs(fr_at - rl_at) < sync_tolerance)

#     diag_sync = (
#         diag1_same_contact.float()
#         + diag1_same_air.float()
#         + diag2_same_contact.float()
#         + diag2_same_air.float()
#     ) / 4.0

#     # ------------------------------------------------------------
#     # 2. 两组对角腿反相
#     #    理想 trot:
#     #       FL+RR 接触, FR+RL 离地
#     #       或
#     #       FR+RL 接触, FL+RR 离地
#     # ------------------------------------------------------------
#     trot_phase_a = fl & rr & (~fr) & (~rl)
#     trot_phase_b = fr & rl & (~fl) & (~rr)

#     trot_pattern = (trot_phase_a | trot_phase_b).float()

#     # ------------------------------------------------------------
#     # 3. 接触脚数量接近 2，防止四脚全接触或四脚全腾空也拿奖励
#     # ------------------------------------------------------------
#     num_contacts = contacts.float().sum(dim=1)
#     two_feet_contact_score = torch.exp(-torch.square(num_contacts - 2.0))

#     # ------------------------------------------------------------
#     # 总奖励
#     # ------------------------------------------------------------
#     reward = (
#         0.4 * diag_sync
#         + 0.4 * trot_pattern
#         + 0.2 * two_feet_contact_score
#     )

#     # 低速/站立时不强制对角步态
#     command = env.command_manager.get_command(command_name)
#     moving = torch.norm(command[:, :2], dim=1) > 0.15
#     reward *= moving

#     return reward


# # =========================
# # Custom rewards for rough-terrain teacher training
# # =========================

# def joint_torques_l1_custom(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Paper-like torque penalty using L1 norm."""
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


# def base_motion_reward_custom(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Approximate Lee et al. 2020 base-motion reward.

#     Reward = exp(-1.5 * v_orth^2) + exp(-1.5 * ||omega_xy||^2)

#     - v_orth: velocity component orthogonal to commanded xy direction
#     - if command is near zero, use total xy velocity instead
#     """
#     asset: RigidObject = env.scene[asset_cfg.name]

#     v_xy = asset.data.root_lin_vel_b[:, :2]          # [N, 2]
#     omega_xy = asset.data.root_ang_vel_b[:, :2]      # [N, 2]
#     cmd_xy = env.command_manager.get_command(command_name)[:, :2]

#     cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True)
#     moving = cmd_norm.squeeze(-1) > 0.1

#     cmd_dir = cmd_xy / torch.clamp(cmd_norm, min=1e-6)
#     v_proj = torch.sum(v_xy * cmd_dir, dim=1, keepdim=True)
#     v_orth_vec = v_xy - v_proj * cmd_dir
#     v_orth = torch.norm(v_orth_vec, dim=1)

#     total_xy_speed = torch.norm(v_xy, dim=1)
#     v_term = torch.where(moving, v_orth, total_xy_speed)

#     reward = torch.exp(-1.5 * torch.square(v_term)) + torch.exp(
#         -1.5 * torch.sum(torch.square(omega_xy), dim=1)
#     )
#     return reward


def action_smoothness_2nd_custom(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Second-order action smoothness penalty.

    Computes ||a_t - 2 a_{t-1} + a_{t-2}||^2

    Note:
        This is a practical Isaac Lab version of the paper's target smoothness term.
        We store a_{t-2} in the env object.
    """
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    if not hasattr(env, "_prev_prev_action"):
        env._prev_prev_action = torch.zeros_like(prev_action)

    prev_prev_action = env._prev_prev_action

    penalty = torch.sum(torch.square(action - 2.0 * prev_action + prev_prev_action), dim=1)

    # update buffer for next step
    env._prev_prev_action = prev_action.clone()

    return penalty


# def foot_clearance_reward_custom(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     contact_sensor_cfg: SceneEntityCfg,
#     asset_cfg: SceneEntityCfg,
#     height_sensor_cfg: SceneEntityCfg | None = None,
#     contact_threshold: float = 1.0,
#     min_clearance: float = 0.06,
# ) -> torch.Tensor:
#     """Approximate foot-clearance reward for rough terrain.

#     Paper version uses:
#       - swing feet set
#       - 9 scan points around each foot
#       - compare foot height against scanned local terrain heights

#     This Isaac Lab approximation uses:
#       - swing ~= no contact
#       - terrain height ~= mean height from a height scanner around robot
#       - reward swing feet whose clearance exceeds min_clearance

#     Returns a reward in [0, 1].
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
#     asset: Articulation = env.scene[asset_cfg.name]

#     # [N, history, feet, 3]
#     contact_forces_hist = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :]
#     contact_force_mag = torch.norm(contact_forces_hist, dim=-1)                  # [N, H, feet]
#     in_contact = torch.max(contact_force_mag, dim=1)[0] > contact_threshold      # [N, feet]
#     swing = ~in_contact                                                           # [N, feet]

#     # foot heights in world frame
#     foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]               # [N, feet]

#     # terrain height estimate
#     if height_sensor_cfg is not None:
#         ray_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
#         # use mean local terrain height around robot as a coarse estimate
#         terrain_height = torch.mean(ray_sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)  # [N, 1]
#     else:
#         # fallback for flat terrain / no height scanner
#         terrain_height = torch.zeros((env.num_envs, 1), device=env.device)

#     clearance = foot_heights - terrain_height         # [N, feet]
#     is_clear = (clearance > min_clearance).float()    # [N, feet]

#     swing_f = swing.float()
#     swing_count = torch.clamp(torch.sum(swing_f, dim=1), min=1.0)
#     reward = torch.sum(is_clear * swing_f, dim=1) / swing_count

#     moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     reward *= moving.float()

#     return reward


def joint_diagonal_symmetry_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Trot 对角肢体对称性惩罚 (Diagonal Gait Symmetry Penalty)。

    在黄金 Trot 小跑中，对角腿对 [FL+RR] 与 [FR+RL] 必须处于完全一致的运动学相位。
    此项直接惩罚对角大腿 (thigh) 与小腿 (calf) 之间的瞬时角度差平方和：
      (FL_thigh - RR_thigh)^2 + (FR_thigh - RL_thigh)^2 + (FL_calf - RR_calf)^2 + (FR_calf - RL_calf)^2
    彻底扼杀单侧深蹲、对角不同步引发的长短腿跛行与斜向侧漂。
    """
    asset = env.scene[asset_cfg.name]
    j_pos = asset.data.joint_pos  # shape: (num_envs, 12)
    # 关节索引映射:
    # 0: FL_hip,   1: FR_hip,   2: RL_hip,   3: RR_hip
    # 4: FL_thigh, 5: FR_thigh, 6: RL_thigh, 7: RR_thigh
    # 8: FL_calf,  9: FR_calf, 10: RL_calf, 11: RR_calf
    diff_thigh_1 = torch.square(j_pos[:, 4] - j_pos[:, 7])  # FL_thigh - RR_thigh
    diff_thigh_2 = torch.square(j_pos[:, 5] - j_pos[:, 6])  # FR_thigh - RL_thigh
    diff_calf_1 = torch.square(j_pos[:, 8] - j_pos[:, 11])  # FL_calf - RR_calf
    diff_calf_2 = torch.square(j_pos[:, 9] - j_pos[:, 10])  # FR_calf - RL_calf
    diff_hip_1 = torch.square(j_pos[:, 0] + j_pos[:, 1])    # FL_hip + FR_hip 左右镜像展收
    diff_hip_2 = torch.square(j_pos[:, 2] + j_pos[:, 3])    # RL_hip + RR_hip 左右镜像展收
    
    total_diff = diff_thigh_1 + diff_thigh_2 + diff_calf_1 + diff_calf_2 + 0.5 * (diff_hip_1 + diff_hip_2)
    return total_diff


def feet_air_time_variance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
) -> torch.Tensor:
    """四足滞空/支撑时间方差惩罚 (Feet Air-Time Variance Penalty)。

    惩罚四条腿在摆动相滞空时间 (last_air_time) 与支撑相接触时间 (last_contact_time) 的方差。
    强制四条腿的步幅、节奏和蹬地时长保持严格等长对称，杜绝任何单腿常悬或单侧拖地。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    return reward


def hip_pos_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    yaw_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
) -> torch.Tensor:
    """WaxberryS exact hip position lock: penalize hip joint deviation when lateral |vy| and yaw |wz| are below thresholds."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    flag = (
        (torch.abs(command[:, 1]) < float(command_threshold))
        & (torch.abs(command[:, 2]) < float(yaw_threshold))
    ).float()
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return flag * torch.sum(torch.abs(hip_pos), dim=1)


def swing_hip_abduction_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.08,
    yaw_threshold: float = 0.15,
    contact_threshold: float = 1.0,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
) -> torch.Tensor:
    """靶向惩罚摆动相（Swing Phase）大角度向外画弧外甩（Circumduction Gait）。

    【动力学与全向解耦原理】：
    1. 动态门控（Command Gate）：当用户下达侧移 (|vy| >= 0.08) 或 原地自转 (|wz| >= 0.15) 指令时，
       门控值立即清零，100% 释放髋关节外展自由度，保证斜爬台阶与原地自转畅通无阻；
    2. 摆动相隔离（Swing Phase Isolation）：仅对腾空离地（contact force < contact_threshold）的腿生效；
       支撑踩地相完全不受任何约束；
    3. 纯单向外展惩罚（Unilateral Outward Abduction Penalty）：
       只惩罚超出标称位姿的【向外画弧外撇】（左腿 > +0.1 rad, 右腿 < -0.1 rad），
       不惩罚正常范围内的对齐运动，彻底逼迫摆动相把抬腿动作转移到矢状面的正向深屈膝高抬腿。
    """
    command = env.command_manager.get_command(command_name)
    gate = (
        (torch.abs(command[:, 1]) < float(command_threshold))
        & (torch.abs(command[:, 2]) < float(yaw_threshold))
    ).float()

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    in_contact = _contact_states(contact_sensor, contact_sensor_cfg.body_ids, contact_threshold)  # [N, 4]
    is_swing = (~in_contact).float()  # [N, 4]

    asset = env.scene[asset_cfg.name]
    # hip joints order: [FL_hip, FR_hip, RL_hip, RR_hip]
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [N, 4]
    default_hip = asset.data.default_joint_pos[:, asset_cfg.joint_ids]  # [N, 4]

    # FL (idx 0, left), FR (idx 1, right), RL (idx 2, left), RR (idx 3, right)
    side_signs = torch.tensor([1.0, -1.0, 1.0, -1.0], device=env.device).unsqueeze(0)
    outward_deviation = torch.clamp((hip_pos - default_hip) * side_signs, min=0.0)  # [N, 4]

    swing_abduction_penalty = torch.sum(torch.square(outward_deviation) * is_swing, dim=1)  # [N]
    return gate * swing_abduction_penalty


def full_hip_swing_symmetry_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.08,
    yaw_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
) -> torch.Tensor:
    """四肢前后全镜像对称外展约束：同时惩罚前肢对与后肢对髋关节外展幅度的非对称差，彻底消除任何单腿特立独行外甩。"""
    command = env.command_manager.get_command(command_name)
    gate = (
        (torch.abs(command[:, 1]) < float(command_threshold))
        & (torch.abs(command[:, 2]) < float(yaw_threshold))
    ).float()

    asset = env.scene[asset_cfg.name]
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [N, 4]: FL, FR, RL, RR
    default_hip = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # FL (idx 0, left), FR (idx 1, right)
    fl_outward = torch.clamp(hip_pos[:, 0] - default_hip[:, 0], min=0.0)
    fr_outward = torch.clamp(-(hip_pos[:, 1] - default_hip[:, 1]), min=0.0)
    front_diff = torch.square(fl_outward - fr_outward)

    # RL (idx 2, left), RR (idx 3, right)
    rl_outward = torch.clamp(hip_pos[:, 2] - default_hip[:, 2], min=0.0)
    rr_outward = torch.clamp(-(hip_pos[:, 3] - default_hip[:, 3]), min=0.0)
    rear_diff = torch.square(rl_outward - rr_outward)

    return gate * (front_diff + rear_diff)


def rear_hip_swing_symmetry_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.08,
    yaw_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
) -> torch.Tensor:
    """向后兼容别名：直接调用 full_hip_swing_symmetry_l2 实现四肢全对称。"""
    return full_hip_swing_symmetry_l2(env, command_name, command_threshold, yaw_threshold, asset_cfg)


def flat_heading_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    yaw_threshold: float = 0.15,
    target_heading_rad: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚平地运动时的绝对偏航角偏差 (wrap_to_pi(yaw - target_yaw)^2)。
    当要求直行或全向平移 (|wz| < yaw_threshold) 时施加航向死锁；
    当指令要求自转或动态转弯 (|wz| >= yaw_threshold) 时动态门控清零，彻底释放自转自由度。
    """
    command = env.command_manager.get_command(command_name)
    gate = (torch.abs(command[:, 2]) < float(yaw_threshold)).float()
    asset = env.scene[asset_cfg.name]
    heading = asset.data.heading_w
    if command.shape[1] > 3:
        target = command[:, 3]
    else:
        target = torch.full_like(heading, float(target_heading_rad))
    heading_err = wrap_to_pi(heading - target)
    return gate * torch.square(heading_err)


def roll_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """姿态精准解耦惩罚：只惩罚机身横滚角偏差 (Roll)，彻底释放俯仰角 (Pitch)。

    物理机理：
    - projected_gravity_b[:, 1] = gy = -sin(roll) * cos(pitch)
    - 当机身端正无侧倾时，gy = 0，惩罚为 0；
    - 当机身左右倾斜摇摆时，惩罚 gy^2，强力锁定身姿端正不侧晃；
    - 完全不惩罚 gx (Pitch)，机器狗顺应 15°~25° 台阶仰角爬坡完全免除惩罚！
    """
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


def pitch_adaptive_foot_clearance_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    flat_target_clearance: float = 0.04,
    stairs_target_clearance: float = 0.16,
    target_clearance: float = 0.04,
    contact_threshold: float = 1.0,
    top_k: int = 8,
    speed_scale: float = 2.0,
    normalize: bool = True,
) -> torch.Tensor:
    """俯仰自适应单边足端净空惩罚 (Pitch-Adaptive Foot Clearance Penalty).

    物理机理：
    - 平地行走 (|pitch| < 3°): 自动将目标净空门限压低至 flat_target_clearance (0.04m)，
      步态紧凑轻盈，彻底杜绝高抬腿跺地产生的高频地面冲击反力与机身颠簸；
    - 台阶翻越 (pitch > 6°): 当机身仰起攀爬台阶时，自适应扩大门限至 stairs_target_clearance (0.16m)，
      强力引导深屈膝高提膝跨步，杜绝踢阶蹭阶卡死。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # 使用当前接触力识别摆动相
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(contact_forces, dim=-1) > contact_threshold
    swing_feet = ~in_contact

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_heights = foot_pos_w[..., 2]

    # 足端局部坐标转换
    sensor_pos_w = height_sensor.data.pos_w
    sensor_yaw_quat_w = yaw_quat(height_sensor.data.quat_w)

    foot_pos_rel_w = foot_pos_w - sensor_pos_w.unsqueeze(1)
    foot_pos_sensor = quat_apply_inverse(
        sensor_yaw_quat_w.unsqueeze(1)
        .expand(-1, foot_pos_rel_w.shape[1], -1)
        .reshape(-1, 4),
        foot_pos_rel_w.reshape(-1, 3),
    ).reshape_as(foot_pos_rel_w)

    foot_xy_sensor = foot_pos_sensor[..., :2]

    ray_xy_sensor = height_sensor.ray_starts[..., :2]
    if ray_xy_sensor.dim() == 2:
        ray_xy_sensor = ray_xy_sensor.unsqueeze(0).expand(
            env.num_envs, -1, -1
        )

    ray_hit_heights = height_sensor.data.ray_hits_w[..., 2]
    valid_rays = torch.isfinite(ray_hit_heights)

    foot_to_ray_dist = torch.cdist(foot_xy_sensor, ray_xy_sensor)
    foot_to_ray_dist = torch.where(
        valid_rays.unsqueeze(1),
        foot_to_ray_dist,
        torch.full_like(foot_to_ray_dist, float("inf")),
    )

    top_k = min(top_k, foot_to_ray_dist.shape[-1])
    nearest_ids = torch.topk(
        foot_to_ray_dist,
        k=top_k,
        dim=-1,
        largest=False,
    ).indices
    expanded_heights = ray_hit_heights.unsqueeze(1).expand(
        -1, foot_heights.shape[1], -1
    )
    local_heights = torch.gather(
        expanded_heights,
        dim=2,
        index=nearest_ids,
    )
    valid_local = torch.isfinite(local_heights)
    local_heights_safe = torch.where(
        valid_local,
        local_heights,
        torch.full_like(local_heights, -float("inf")),
    )
    nearby_terrain_height = torch.max(
        local_heights_safe,
        dim=2,
    ).values
    has_valid_local = valid_local.any(dim=2)

    # 动态根据机身俯仰角计算目标净空
    # asset.data.projected_gravity_b[:, 0] = -sin(pitch)
    pitch_up = -asset.data.projected_gravity_b[:, 0]  # shape: (num_envs,)
    # 当 pitch_up <= 0.05 (~3°) 时 alpha = 0.0; 当 pitch_up >= 0.20 (~11.5°) 时 alpha = 1.0
    alpha = torch.clamp((pitch_up - 0.05) / 0.15, min=0.0, max=1.0).unsqueeze(1)  # (num_envs, 1)
    target_clearance = flat_target_clearance + alpha * (stairs_target_clearance - flat_target_clearance)  # (num_envs, 1)

    nearby_terrain_height = torch.where(
        has_valid_local,
        nearby_terrain_height,
        foot_heights - target_clearance,
    )

    clearance = foot_heights - nearby_terrain_height
    # 只惩罚离地间隙不足
    clearance_shortfall = torch.clamp(
        target_clearance - clearance,
        min=0.0,
    )
    clearance_error = torch.clamp(clearance_shortfall, max=float(stairs_target_clearance))

    foot_speed_xy = torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2],
        dim=-1,
    )
    speed_weight = torch.tanh(speed_scale * foot_speed_xy)
    swing_count = swing_feet.sum(dim=1).clamp(min=1)
    penalty = torch.sum(
        clearance_error
        * speed_weight
        * swing_feet,
        dim=1,
    ) / swing_count
    command = env.command_manager.get_command(command_name)
    moving = (
        (torch.norm(command[:, :2], dim=1) > 0.1)
        | (torch.abs(command[:, 2]) > 0.1)
    )

    return penalty * moving


def feet_contact_velocity_z_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """足端接地瞬间垂直撞击速度软着陆阻尼惩罚 (Foot Soft Landing Damping Penalty).

    物理机理：
    - 当足端触地 (接触力 > contact_threshold) 时，惩罚足端在世界坐标系下的垂直速度平方 (v_{z}^2)；
    - 迫使策略网络在摆动相即将着地的一瞬主动缓速收腿 (Retraction velocity matching: \dot{z} -> 0)，
      消除高抬腿重砸地面产生的瞬态冲击（GRF Impact Spike）；
    - 从而将平地机身质心的垂直抖动彻底压入 < 10mm 黄金线以内。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = force_history.norm(dim=-1).max(dim=1)[0] > float(contact_threshold)

    asset = env.scene[asset_cfg.name]
    foot_vz = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]

    landing_impact = torch.square(foot_vz) * contacts.float()
    return torch.sum(landing_impact, dim=1)


def feet_contact_force_excess_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_force: float = 85.0,
) -> torch.Tensor:
    """足端地面接触力超限软化缓冲惩罚 (Compliant Ground Contact Force Cushioning).

    物理机理：
    - Go2 机器狗自重 15kg，在对角 Trot 对称支撑下，每只单腿的名义承重约为 75N；
    - 超过 85N 的尖峰力矩代表猛烈剁地冲击，惩罚超额接触力矩的平方；
    - 激励膝关节与踝部在接地后产生柔顺屈伸缓冲（Compliant Cushioning），吸收冲击动能。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    excess = torch.clamp(forces - float(max_contact_force), min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def terrain_adaptive_pitch_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stairs_pitch_threshold: float = 0.08,
) -> torch.Tensor:
    """解耦平地俯仰刚性锁定惩罚 (Decoupled Flat Pitch Rigidity Lock).

    物理机理：
    - asset.data.projected_gravity_b[:, 0] = gx = -sin(pitch)
    - 在平地或小角度行走时 (|gx| < stairs_pitch_threshold, 即 |pitch| < ~4.6°)，gate -> 1.0，
      强力施加 gx^2 惩罚，迫使机身严格保持水平 (pitch 压至 <= 1.5°)，根除跷跷板点头颠簸；
    - 当机器狗遭遇台阶、机身主动仰起攀爬时 (pitch > stairs_pitch_threshold)，
      gate 平滑衰减至 0.0，彻底解除俯仰锁，释放爬阶自由度，严格遵循红线 W4。
    """
    asset = env.scene[asset_cfg.name]
    gx = asset.data.projected_gravity_b[:, 0]
    pitch_up = -gx
    # 当 pitch_up <= 0.05 (~3°) 时 gate = 1.0; 当 pitch_up >= 0.12 (~7°) 时 gate = 0.0
    gate = torch.clamp((0.12 - pitch_up) / 0.07, min=0.0, max=1.0)
    return gate * torch.square(gx)


def dual_bounded_pitch_adaptive_foot_clearance_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    flat_target_clearance: float = 0.022,
    flat_max_clearance: float = 0.028,
    stairs_target_clearance: float = 0.16,
    target_clearance: float = 0.04,
    excess_penalty_scale: float = 3.0,
    contact_threshold: float = 1.0,
    top_k: int = 8,
    speed_scale: float = 2.0,
    normalize: bool = True,
) -> torch.Tensor:
    """双向限幅俯仰自适应足端净空惩罚 (Dual-Bounded Pitch-Adaptive Foot Clearance).

    物理机理：
    1. 平地行走 (alpha -> 0): 
       - 下限约束：离地间隙不足 flat_target_clearance (0.022m) 时产生 shortfall 惩罚（防绊腿拖地）；
       - 上限约束：离地间隙超过 flat_max_clearance (0.028m) 时处以 excess_penalty_scale (3.0x) 超额惩罚！
         严厉斩断平地 35mm 高抬腿大跨步恶习，把平地步态强行按死在 10~25mm 磁悬浮滑行区间，彻底消除跺地撞击！
    2. 台阶翻越 (alpha -> 1):
       - 超额惩罚系数随 (1.0 - alpha) 彻底归零，允许并激励深屈膝高提膝 (达到 0.16m)，100% 翻越连续高台阶。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(contact_forces, dim=-1) > contact_threshold
    swing_feet = ~in_contact

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_heights = foot_pos_w[..., 2]

    sensor_pos_w = height_sensor.data.pos_w
    sensor_yaw_quat_w = yaw_quat(height_sensor.data.quat_w)

    foot_pos_rel_w = foot_pos_w - sensor_pos_w.unsqueeze(1)
    foot_pos_sensor = quat_apply_inverse(
        sensor_yaw_quat_w.unsqueeze(1)
        .expand(-1, foot_pos_rel_w.shape[1], -1)
        .reshape(-1, 4),
        foot_pos_rel_w.reshape(-1, 3),
    ).reshape_as(foot_pos_rel_w)

    foot_xy_sensor = foot_pos_sensor[..., :2]

    ray_xy_sensor = height_sensor.ray_starts[..., :2]
    if ray_xy_sensor.dim() == 2:
        ray_xy_sensor = ray_xy_sensor.unsqueeze(0).expand(
            env.num_envs, -1, -1
        )

    ray_hit_heights = height_sensor.data.ray_hits_w[..., 2]
    valid_rays = torch.isfinite(ray_hit_heights)

    foot_to_ray_dist = torch.cdist(foot_xy_sensor, ray_xy_sensor)
    foot_to_ray_dist = torch.where(
        valid_rays.unsqueeze(1),
        foot_to_ray_dist,
        torch.full_like(foot_to_ray_dist, float("inf")),
    )

    top_k = min(top_k, foot_to_ray_dist.shape[-1])
    nearest_ids = torch.topk(
        foot_to_ray_dist,
        k=top_k,
        dim=-1,
        largest=False,
    ).indices
    expanded_heights = ray_hit_heights.unsqueeze(1).expand(
        -1, foot_heights.shape[1], -1
    )
    local_heights = torch.gather(
        expanded_heights,
        dim=2,
        index=nearest_ids,
    )
    valid_local = torch.isfinite(local_heights)
    local_heights_safe = torch.where(
        valid_local,
        local_heights,
        torch.full_like(local_heights, -float("inf")),
    )
    nearby_terrain_height = torch.max(
        local_heights_safe,
        dim=2,
    ).values
    has_valid_local = valid_local.any(dim=2)

    cur_ground_height = torch.where(
        has_valid_local,
        nearby_terrain_height,
        foot_heights - flat_target_clearance,
    )

    pitch_up = -asset.data.projected_gravity_b[:, 0]
    alpha_pitch = torch.clamp((pitch_up - 0.03) / 0.12, min=0.0, max=1.0).unsqueeze(1)
    # 结合扫描仪局部地形落差：若全图存在台阶高差 (span > 4cm)，作为宏观环境特征
    safe_ray_hits_max = torch.where(valid_rays, ray_hit_heights, torch.full_like(ray_hit_heights, -1e4)).max(dim=-1).values
    safe_ray_hits_min = torch.where(valid_rays, ray_hit_heights, torch.full_like(ray_hit_heights, 1e4)).min(dim=-1).values
    has_any_valid_rays = valid_rays.any(dim=-1)
    span_terrain = torch.where(has_any_valid_rays, safe_ray_hits_max - safe_ray_hits_min, torch.zeros_like(safe_ray_hits_max)).clamp(min=0.0)
    alpha_terrain = torch.clamp((span_terrain - 0.04) / 0.06, min=0.0, max=1.0).unsqueeze(1)

    # --- 核心手术：0.3 秒前瞻自适应提膝激活 (提前感知前方直角立面) ---
    # 机身速度 0.6~1.0m/s 时，0.3s 行程对应足端前方 dx in [0.05, 0.40]m，横向宽度 dy in [-0.15, 0.15]m
    ray_x = ray_xy_sensor[:, :, 0].unsqueeze(1)  # [num_envs, 1, num_rays]
    ray_y = ray_xy_sensor[:, :, 1].unsqueeze(1)  # [num_envs, 1, num_rays]
    foot_x = foot_xy_sensor[:, :, 0].unsqueeze(2)  # [num_envs, 4, 1]
    foot_y = foot_xy_sensor[:, :, 1].unsqueeze(2)  # [num_envs, 4, 1]

    dx_ahead = ray_x - foot_x  # [num_envs, 4, num_rays]
    dy_ahead = ray_y - foot_y  # [num_envs, 4, num_rays]

    ahead_mask = (
        (dx_ahead >= 0.05)
        & (dx_ahead <= 0.40)
        & (torch.abs(dy_ahead) <= 0.15)
        & valid_rays.unsqueeze(1)
    )  # [num_envs, 4, num_rays]

    ahead_heights_safe = torch.where(
        ahead_mask,
        expanded_heights,
        torch.full_like(expanded_heights, -1e4),
    )
    ahead_max_height = torch.max(ahead_heights_safe, dim=-1).values  # [num_envs, 4]
    has_valid_ahead = ahead_mask.any(dim=-1)  # [num_envs, 4]

    # 前方台阶落差：ahead 地形最大高度 - 当前脚底地面高度
    delta_z_ahead = torch.where(
        has_valid_ahead,
        ahead_max_height - cur_ground_height,
        torch.zeros_like(cur_ground_height),
    ).clamp(min=0.0)

    # 提前 0.3 秒前瞻激活门控：当落差超过 4cm (如 10~16cm 台阶) 时，alpha_ahead 迅速拉满至 1.0
    alpha_ahead = torch.clamp((delta_z_ahead - 0.04) / 0.06, min=0.0, max=1.0)  # [num_envs, 4]

    # 综合激活度：前瞻感知 + 机身俯仰 + 宏观地形落差三位一体
    alpha = torch.maximum(torch.maximum(alpha_pitch, alpha_terrain), alpha_ahead)  # [num_envs, 4]
    target_clearance = flat_target_clearance + alpha * (stairs_target_clearance - flat_target_clearance)

    nearby_terrain_height = torch.where(
        has_valid_local,
        nearby_terrain_height,
        foot_heights - target_clearance,
    )

    clearance = foot_heights - nearby_terrain_height
    # 1. 离地间隙不足惩罚（增加下降触地保护门控 Landing Gate，消灭落脚死锁）
    # 当摆动足处于向下降落准备触地阶段时 (foot_vel_z < 0)，离地间隙自然归零，不应遭受 shortfall 惩罚；
    # 只有在抬腿上升/前进腾空阶段 (foot_vel_z >= 0)，才强制要求达到目标离地高度。
    foot_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    landing_gate = torch.clamp((foot_vel_z + 0.15) / 0.20, min=0.0, max=1.0)
    effective_landing_gate = (1.0 - alpha) * 1.0 + alpha * landing_gate

    clearance_shortfall = effective_landing_gate * torch.clamp(
        target_clearance - clearance,
        min=0.0,
    )
    # 2. 平地超额抬腿严惩 (彻底斩断跺地冲击)
    flat_excess = torch.clamp(clearance - float(flat_max_clearance), min=0.0)
    flat_excess_penalty = (1.0 - alpha) * flat_excess * float(excess_penalty_scale)

    clearance_error = torch.clamp(clearance_shortfall + flat_excess_penalty, max=float(stairs_target_clearance))

    foot_speed_xy = torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2],
        dim=-1,
    )
    speed_weight = torch.tanh(speed_scale * foot_speed_xy)
    swing_count = swing_feet.sum(dim=1).clamp(min=1)
    penalty = torch.sum(
        clearance_error
        * speed_weight
        * swing_feet,
        dim=1,
    ) / swing_count
    command = env.command_manager.get_command(command_name)
    moving = (
        (torch.norm(command[:, :2], dim=1) > 0.1)
        | (torch.abs(command[:, 2]) > 0.1)
    )

    return penalty * moving


def bilateral_hip_symmetry_equivariance_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.08,
    yaw_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
) -> torch.Tensor:
    """双向全域等变镜像对称约束 (无任何单边截断)：
    左右前腿与左右后腿髋关节偏差必须完全镜像相反：
    (q_FL - q_nom_FL) + (q_FR - q_nom_FR) == 0
    (q_RL - q_nom_RL) + (q_RR - q_nom_RR) == 0
    完全剔除 min=0.0 单边截断后门，无论是外甩还是内扣，只要左右不对称一律严惩！
    """
    command = env.command_manager.get_command(command_name)
    gate = (
        (torch.abs(command[:, 1]) < float(command_threshold))
        & (torch.abs(command[:, 2]) < float(yaw_threshold))
    ).float()

    asset = env.scene[asset_cfg.name]
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_hip = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    front_mirror_err = (hip_pos[:, 0] - default_hip[:, 0]) + (hip_pos[:, 1] - default_hip[:, 1])
    rear_mirror_err = (hip_pos[:, 2] - default_hip[:, 2]) + (hip_pos[:, 3] - default_hip[:, 3])

    return gate * (torch.square(front_mirror_err) + torch.square(rear_mirror_err))


def front_hip_neutral_pos_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.08,
    yaw_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
) -> torch.Tensor:
    """前腿髋关节双向中立零位锁：惩罚前肢偏离名义基准（无论是内扣还是外展），彻底将左前内扣拉直。"""
    command = env.command_manager.get_command(command_name)
    gate = (
        (torch.abs(command[:, 1]) < float(command_threshold))
        & (torch.abs(command[:, 2]) < float(yaw_threshold))
    ).float()

    asset = env.scene[asset_cfg.name]
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_hip = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    fl_dev = hip_pos[:, 0] - default_hip[:, 0]
    fr_dev = hip_pos[:, 1] - default_hip[:, 1]

    return gate * (torch.square(fl_dev) + torch.square(fr_dev))


def power_distribution_variance_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """四肢功率方差惩罚：强制四腿机械功率对称均衡分配，杜绝单腿内扣偷懒或单侧负荷过载。"""
    asset = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel
    qfrc = asset.data.applied_torque
    power = torch.abs(qvel * qfrc)
    return torch.var(power, dim=-1)


def straight_ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    yaw_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """平地直行偏航角速度刚性锁：在直行指令下 (|wz| < 0.10) 严惩任何偏航晃动，彻底锁死直线。"""
    command = env.command_manager.get_command(command_name)
    gate = (torch.abs(command[:, 2]) < float(yaw_threshold)).float()
    asset = env.scene[asset_cfg.name]
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    return gate * torch.square(ang_vel_z)


def soft_contact_velocity_damping_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 2.0,
) -> torch.Tensor:
    """连续接触速度阻尼：只惩罚高速粗暴撞阶，允许低速贴阶滑行，消灭碰阶回缩假死与卡顿。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces, dim=-1)
    in_contact = force_norm > float(contact_force_threshold)

    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :]
    vel_sq = torch.sum(torch.square(body_vel), dim=-1)

    return torch.sum(in_contact.float() * vel_sq, dim=-1)


def rear_hip_neutral_pos_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.08,
    yaw_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
) -> torch.Tensor:
    """后腿髋关节双向中立零位锁：惩罚后肢髋角偏离名义基准，消灭后驱单侧偏向。"""
    command = env.command_manager.get_command(command_name)
    gate = (
        (torch.abs(command[:, 1]) < float(command_threshold))
        & (torch.abs(command[:, 2]) < float(yaw_threshold))
    ).float()

    asset = env.scene[asset_cfg.name]
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_hip = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    rl_dev = hip_pos[:, 2] - default_hip[:, 2]
    rr_dev = hip_pos[:, 3] - default_hip[:, 3]

    return gate * (torch.square(rl_dev) + torch.square(rr_dev))


