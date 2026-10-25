# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, ContactSensor, Imu, RayCaster, RayCasterCamera, TiledCamera
from dreamwaq_lab.sensors.depth_preprocessing import (
    DEPTH_TEMPORAL_ABLATIONS,
    DistinctDepthFrameBuffer,
    apply_depth_near_plane,
    apply_depth_temporal_ablation,
    preprocess_depth_image,
    validate_depth_max_distance,
)
from .spatial_scan import encode_vertical_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)


def _get_body_shape_slices(asset: Articulation) -> list[slice]:
    """Return cached shape-index slices for each body of an articulation."""

    cache_name = "_lq_body_shape_slices"
    if hasattr(asset, cache_name):
        return getattr(asset, cache_name)

    body_shape_slices: list[slice] = []
    start_idx = 0
    for link_path in asset.root_physx_view.link_paths[0]:
        link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
        num_shapes = int(link_physx_view.max_shapes)
        body_shape_slices.append(slice(start_idx, start_idx + num_shapes))
        start_idx += num_shapes

    if start_idx != int(asset.root_physx_view.max_shapes):
        raise RuntimeError(
            "Failed to resolve articulation body-to-shape mapping for friction observations: "
            f"parsed {start_idx} shapes, expected {asset.root_physx_view.max_shapes}."
        )

    setattr(asset, cache_name, body_shape_slices)
    return body_shape_slices


def _get_local_foot_state_in_sensor_frame(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> tuple[RayCaster, Articulation, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project foot positions into the yaw-aligned height-sensor frame."""

    height_sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    sensor_pos_w = height_sensor.data.pos_w
    sensor_yaw_quat_w = math_utils.yaw_quat(height_sensor.data.quat_w)

    foot_pos_rel_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - sensor_pos_w.unsqueeze(1)
    foot_pos_rel_sensor = math_utils.quat_apply_inverse(
        sensor_yaw_quat_w.unsqueeze(1).expand(-1, foot_pos_rel_w.shape[1], -1).reshape(-1, 4),
        foot_pos_rel_w.reshape(-1, 3),
    ).reshape_as(foot_pos_rel_w)

    foot_xy_sensor = foot_pos_rel_sensor[..., :2]
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    ray_xy_sensor = height_sensor.ray_starts[..., :2]
    if ray_xy_sensor.dim() == 2:
        ray_xy_sensor = ray_xy_sensor.unsqueeze(0).expand(env.num_envs, -1, -1)

    return height_sensor, asset, foot_xy_sensor, foot_heights, ray_xy_sensor


def _build_foot_scan_offsets(
    *,
    radii: Sequence[float],
    num_angles: int,
    include_center: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create 2-D query offsets for a foot-centric height scan."""

    if num_angles <= 0:
        raise ValueError(f"num_angles must be positive, got {num_angles}.")

    offsets: list[torch.Tensor] = []
    if include_center:
        offsets.append(torch.zeros(1, 2, device=device, dtype=dtype))

    angles = torch.linspace(0.0, 2.0 * torch.pi, steps=num_angles + 1, device=device, dtype=dtype)[:-1]
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    for radius in radii:
        if radius <= 0.0:
            if not include_center:
                offsets.append(torch.zeros(1, 2, device=device, dtype=dtype))
            continue
        offsets.append(directions * radius)

    if not offsets:
        raise ValueError("At least one foot-local height-scan sample must be configured.")

    return torch.cat(offsets, dim=0)


def _latest_depth_plane(depth: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Normalize supported camera layouts to ``[N,H,W]`` using the latest frame."""

    if depth.ndim == 5 and depth.shape[0] == num_envs:
        depth = depth[:, -1]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    elif depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
    if depth.ndim != 3 or depth.shape[0] != num_envs:
        raise RuntimeError(f"Expected depth image [N,H,W], got {tuple(depth.shape)} for N={num_envs}.")
    return depth


def depth_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    max_distance: float = 3.0,    # 最远距离
    near_plane: float = 0.0,
    min_distance: float = 0.0,
    normalization_distance: float | None = None,
    normalization_mode: str = "near_offset",
    flatten: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the latest RayCasterCamera depth image as an observation tensor.

    ``near_plane`` emulates the physical near clipping plane at the policy
    input boundary because Isaac Lab's RayCasterCamera has no native near-plane
    configuration.
    """

    camera: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    output = getattr(camera.data, "output", None)
    if not isinstance(output, dict):
        raise RuntimeError(f"Sensor '{sensor_cfg.name}' does not expose a data.output dictionary.")
    if data_type not in output:
        raise RuntimeError(
            f"Depth data type '{data_type}' is not available from sensor '{sensor_cfg.name}'. "
            f"Available keys: {list(output.keys())}."
        )

    if normalization_distance not in (None, max_distance):
        raise ValueError(
            "PIE depth preprocessing no longer supports an independent normalization_distance; "
            "normalization is defined by the physical clipping range."
        )
    depth = _latest_depth_plane(output[data_type], env.num_envs)
    depth = apply_depth_near_plane(depth, near_plane=near_plane, max_distance=max_distance)
    if normalize:
        depth = preprocess_depth_image(
            depth,
            min_distance=min_distance,
            max_distance=max_distance,
            normalization_mode=normalization_mode,
        )
    else:
        depth = torch.nan_to_num(depth.float(), nan=max_distance, posinf=max_distance, neginf=max_distance)
        depth = torch.clamp(depth, min=min_distance, max=max_distance)
    if flatten:
        depth = depth.reshape(env.num_envs, -1)
    return depth


class depth_frame_history(ManagerTermBase):
    """Maintain the latest distinct camera frames using the camera frame counter.

    The observation manager may call this term at 50 Hz, but the buffer advances
    only when ``RayCasterCamera.frame`` changes. This implements the same
    zero-order hold expected from a 10 Hz hardware camera feeding a 50 Hz policy.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        camera: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
        history_length = int(cfg.params.get("history_length", 2))
        if history_length != 2:
            raise ValueError("The PIE reproduction baseline requires exactly two distinct depth frames.")
        height, width = camera.image_shape
        self.history_length = history_length
        self.height = int(height)
        self.width = int(width)
        self.buffer = DistinctDepthFrameBuffer(
            num_streams=env.num_envs,
            height=self.height,
            width=self.width,
            history_length=history_length,
            device=env.device,
        )
        self.temporal_ablation = str(cfg.params.get("temporal_ablation", "distinct_pair")).strip().lower()
        if self.temporal_ablation not in DEPTH_TEMPORAL_ABLATIONS:
            raise ValueError(
                f"Unsupported PIE depth temporal ablation {self.temporal_ablation!r}; "
                f"expected {DEPTH_TEMPORAL_ABLATIONS}."
            )
        self.diagnostics_interval = max(int(cfg.params.get("diagnostics_interval", 250)), 0)
        self.diagnostics_max_samples = max(int(cfg.params.get("diagnostics_max_samples", 262_144)), 1)
        self._last_policy_step = -1
        self._last_diagnostics_step = -1
        self._five_step_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        self._five_step_length = 0
        self._new_frames_per_five_steps = torch.zeros((), device=env.device)
        self._cached_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def frame_buffer(self) -> torch.Tensor:
        return self.buffer.frames

    @property
    def frame_timestamps(self) -> torch.Tensor:
        return self.buffer.timestamps

    @property
    def last_frame_ids(self) -> torch.Tensor:
        return self.buffer.last_frame_ids

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.buffer.reset(env_ids)
        self._five_step_counts[env_ids] = 0.0

    @staticmethod
    def _as_frame_ids(camera: RayCasterCamera, env: ManagerBasedRLEnv) -> torch.Tensor:
        frame_ids = camera.frame
        if not isinstance(frame_ids, torch.Tensor):
            frame_ids = torch.full((env.num_envs,), int(frame_ids), device=env.device, dtype=torch.long)
        return frame_ids.to(device=env.device, dtype=torch.long).reshape(-1)

    @staticmethod
    def _as_timestamps(camera: RayCasterCamera, env: ManagerBasedRLEnv) -> torch.Tensor:
        timestamps = getattr(camera, "_timestamp_last_update", None)
        if timestamps is None:
            timestamp = float(getattr(env, "common_step_counter", 0)) * float(env.step_dt)
            return torch.full((env.num_envs,), timestamp, device=env.device, dtype=torch.float32)
        if not isinstance(timestamps, torch.Tensor):
            timestamps = torch.full(
                (env.num_envs,), float(timestamps), device=env.device, dtype=torch.float32
            )
        return timestamps.to(device=env.device, dtype=torch.float32).reshape(-1)

    def _update_five_step_counter(self, policy_step: int, new_frame_mask: torch.Tensor) -> None:
        if policy_step == self._last_policy_step:
            return
        self._last_policy_step = policy_step
        self._five_step_counts += new_frame_mask.to(dtype=torch.float32)
        self._five_step_length += 1
        if self._five_step_length >= 5:
            self._new_frames_per_five_steps = self._five_step_counts.mean().detach()
            self._five_step_counts.zero_()
            self._five_step_length = 0

    def _update_diagnostics(
        self,
        env: ManagerBasedRLEnv,
        raw_depth: torch.Tensor,
        frame_ids: torch.Tensor,
        sensor_timestamps: torch.Tensor,
        distinct_history: torch.Tensor,
    ) -> None:
        policy_step = int(getattr(env, "common_step_counter", 0))
        should_refresh = self._last_diagnostics_step < 0 or (
            self.diagnostics_interval > 0
            and policy_step - self._last_diagnostics_step >= self.diagnostics_interval
        )
        if should_refresh:
            self._last_diagnostics_step = policy_step
            flat = distinct_history[:, -1].reshape(-1)
            stride = max((flat.numel() + self.diagnostics_max_samples - 1) // self.diagnostics_max_samples, 1)
            sample = flat[::stride]
            quantiles = torch.quantile(
                sample,
                torch.tensor((0.01, 0.50, 0.99), device=sample.device, dtype=sample.dtype),
            )
            timestamps = self.buffer.timestamps
            valid_pair = (timestamps[:, 0] >= 0.0) & (timestamps[:, 1] >= 0.0)
            timestamp_delta = torch.where(
                valid_pair,
                timestamps[:, 1] - timestamps[:, 0],
                torch.zeros_like(timestamps[:, 1]),
            )
            sim_time = float(policy_step) * float(env.step_dt)
            newest_age = torch.clamp(
                torch.full_like(sensor_timestamps, sim_time) - sensor_timestamps,
                min=0.0,
            )
            raw_nan = torch.isnan(raw_depth)
            raw_inf = torch.isinf(raw_depth)
            raw_invalid = (~torch.isfinite(raw_depth)) | (raw_depth <= 0.0)
            temporal_difference = torch.abs(distinct_history[:, 1] - distinct_history[:, 0])
            self._cached_diagnostics = {
                "Depth/camera_frame": frame_ids.float().mean().detach(),
                "Depth/timestamp_delta": timestamp_delta.mean().detach(),
                "Depth/newest_frame_age": newest_age.mean().detach(),
                "Depth/mean_abs_new_minus_old": temporal_difference.mean().detach(),
                "Depth/p01": quantiles[0].detach(),
                "Depth/p50": quantiles[1].detach(),
                "Depth/p99": quantiles[2].detach(),
                "Depth/zero_ratio": (flat == 0.0).float().mean().detach(),
                "Depth/one_ratio": (flat == 1.0).float().mean().detach(),
                "Depth/raw_invalid_count": raw_invalid.sum().float().detach(),
                "Depth/raw_nan_count": raw_nan.sum().float().detach(),
                "Depth/raw_inf_count": raw_inf.sum().float().detach(),
                "Depth/new_frames_per_5_policy_steps": self._new_frames_per_five_steps.detach(),
                "Depth/diagnostic_sample_count": torch.tensor(
                    float(sample.numel()), device=sample.device, dtype=torch.float32
                ),
            }

        if self._cached_diagnostics:
            env.extras.setdefault("log", {}).update(self._cached_diagnostics)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        data_type: str = "distance_to_image_plane",
        max_distance: float = 3.0,
        near_plane: float = 0.0,
        min_distance: float = 0.1,
        normalization_mode: str = "near_offset",
        history_length: int = 2,
        temporal_ablation: str = "distinct_pair",
        diagnostics_interval: int = 250,
        diagnostics_max_samples: int = 262_144,
        flatten: bool = True,
    ) -> torch.Tensor:
        if history_length != self.history_length:
            raise ValueError(
                f"Configured depth history changed from {self.history_length} to {history_length}."
            )
        temporal_ablation = str(temporal_ablation).strip().lower()
        if temporal_ablation != self.temporal_ablation:
            raise ValueError(
                f"Configured depth temporal ablation changed from {self.temporal_ablation!r} "
                f"to {temporal_ablation!r}."
            )
        if int(diagnostics_interval) != self.diagnostics_interval:
            raise ValueError(
                f"Configured depth diagnostics interval changed from {self.diagnostics_interval} "
                f"to {diagnostics_interval}."
            )
        if int(diagnostics_max_samples) != self.diagnostics_max_samples:
            raise ValueError(
                f"Configured depth diagnostic sample limit changed from {self.diagnostics_max_samples} "
                f"to {diagnostics_max_samples}."
            )
        max_distance = validate_depth_max_distance(max_distance)
        camera: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
        output = camera.data.output
        if data_type not in output:
            raise RuntimeError(f"Depth type {data_type!r} is unavailable; keys: {list(output.keys())}.")
        raw_depth = _latest_depth_plane(output[data_type], env.num_envs)
        raw_depth = apply_depth_near_plane(raw_depth, near_plane=near_plane, max_distance=max_distance)
        current_depth = preprocess_depth_image(
            raw_depth,
            min_distance=min_distance,
            max_distance=max_distance,
            normalization_mode=normalization_mode,
        )
        if tuple(current_depth.shape[-2:]) != (self.height, self.width):
            raise RuntimeError(
                f"Depth camera shape changed from {(self.height, self.width)} to {tuple(current_depth.shape[-2:])}."
            )
        frame_ids = self._as_frame_ids(camera, env)
        sensor_timestamps = self._as_timestamps(camera, env)
        new_frame_mask = frame_ids != self.buffer.last_frame_ids
        self._update_five_step_counter(int(getattr(env, "common_step_counter", 0)), new_frame_mask)
        distinct_history = self.buffer.update(current_depth, frame_ids, sensor_timestamps)
        self._update_diagnostics(env, raw_depth, frame_ids, sensor_timestamps, distinct_history)
        output_history = apply_depth_temporal_ablation(distinct_history, temporal_ablation)
        return output_history.reshape(env.num_envs, -1) if flatten else output_history


def joint_acc(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint accelerations for the selected articulation joints."""

    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return binary contact states for the selected bodies.

    This is used as privileged information for the teacher. A body is considered in contact when the
    maximum contact-force magnitude over the contact sensor history exceeds ``threshold``.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_force_magnitude = contact_forces.norm(dim=-1).max(dim=1)[0]
    return (contact_force_magnitude > threshold).float()


def contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = 1.0,
    clip_max: float | None = None,
) -> torch.Tensor:
    """Return per-body contact-force magnitudes for the selected bodies.

    The output uses the maximum force magnitude over the contact sensor history so the teacher can access
    a stable contact summary instead of a single-step impulse.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_force_magnitude = contact_forces.norm(dim=-1).max(dim=1)[0]

    # Raw contact forces can be much larger than the rest of the observation vector and easily destabilize
    # PPO when observation normalization is disabled. Expose an optional bounded version for policy inputs.
    if scale != 1.0:
        contact_force_magnitude = contact_force_magnitude / scale
    if clip_max is not None:
        contact_force_magnitude = torch.clamp(contact_force_magnitude, min=0.0, max=clip_max)
    return contact_force_magnitude


def foot_contact_force_norm(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return reference-style foot-contact magnitudes with history.

    For a contact sensor with history length ``T``, the output concatenates
    the maximum magnitude over the history and every history sample, producing
    ``(T + 1) * num_feet`` values. With the Go2 sensor's three-frame history,
    this produces 16 privileged observations.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_force_norm = torch.linalg.vector_norm(contact_force_history, dim=-1)
    max_contact_force_norm = torch.max(contact_force_norm, dim=1).values
    return torch.cat((max_contact_force_norm.unsqueeze(1), contact_force_norm), dim=1).flatten(start_dim=1)


def contact_normal(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return per-body contact normals approximated from the strongest recent contact force.

    Isaac Lab's default contact sensor exposes net contact forces but not explicit contact normals. To keep the
    implementation lightweight, we use the direction of the strongest recent contact force as a stable proxy.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_force_magnitude = contact_forces.norm(dim=-1)

    max_history_ids = contact_force_magnitude.argmax(dim=1, keepdim=True)
    strongest_forces = torch.gather(
        contact_forces,
        dim=1,
        index=max_history_ids.unsqueeze(-1).expand(-1, 1, -1, 3),
    ).squeeze(1)
    max_force_magnitude = contact_force_magnitude.max(dim=1).values

    normals = strongest_forces / max_force_magnitude.unsqueeze(-1).clamp(min=1.0e-6)
    normals = torch.where(max_force_magnitude.unsqueeze(-1) > threshold, normals, torch.zeros_like(normals))
    return normals.reshape(env.num_envs, -1)


def foot_friction_coefficients(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return one static-friction coefficient per selected body.

    The current environment randomizes body materials only at startup, so we cache the resulting static friction
    per body after the first query to avoid repeated CPU-side material fetches.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    cache_name = "_lq_cached_static_friction_per_body"
    if not hasattr(asset, cache_name):
        materials = asset.root_physx_view.get_material_properties().to(device=asset.device, dtype=torch.float32)
        body_shape_slices = _get_body_shape_slices(asset)
        static_friction = torch.zeros((env.num_envs, asset.num_bodies), device=asset.device, dtype=torch.float32)
        for body_id, shape_slice in enumerate(body_shape_slices):
            body_materials = materials[:, shape_slice, 0]
            if body_materials.numel() == 0:
                continue
            static_friction[:, body_id] = body_materials.mean(dim=1)
        setattr(asset, cache_name, static_friction)

    static_friction = getattr(asset, cache_name)
    return static_friction[:, asset_cfg.body_ids]


def body_restitution_coefficients(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return one restitution coefficient per selected body.

    Walk These Ways uses friction/restitution as the supervised CSE target in
    its public configuration. Isaac Lab stores both values in the rigid-shape
    material table, so we expose restitution with the same body-to-shape cache
    used by :func:`foot_friction_coefficients`.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    cache_name = "_lq_cached_restitution_per_body"
    if not hasattr(asset, cache_name):
        materials = asset.root_physx_view.get_material_properties().to(device=asset.device, dtype=torch.float32)
        body_shape_slices = _get_body_shape_slices(asset)
        restitution = torch.zeros((env.num_envs, asset.num_bodies), device=asset.device, dtype=torch.float32)
        for body_id, shape_slice in enumerate(body_shape_slices):
            body_materials = materials[:, shape_slice, 2]
            if body_materials.numel() == 0:
                continue
            restitution[:, body_id] = body_materials.mean(dim=1)
        setattr(asset, cache_name, restitution)

    restitution = getattr(asset, cache_name)
    return restitution[:, asset_cfg.body_ids]


def previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the action from one policy step before ``mdp.last_action``."""

    return env.action_manager.prev_action


def wtw_clock_inputs(env: ManagerBasedRLEnv, command_name: str = "walk_these_ways") -> torch.Tensor:
    """Return the four sinusoidal WTW gait clock inputs."""

    return env.command_manager.get_term(command_name).clock_inputs


def wtw_desired_contact_states(env: ManagerBasedRLEnv, command_name: str = "walk_these_ways") -> torch.Tensor:
    """Return smooth desired contact probabilities for FL, FR, RL, RR."""

    return env.command_manager.get_term(command_name).desired_contact_states


def wtw_foot_indices(env: ManagerBasedRLEnv, command_name: str = "walk_these_ways") -> torch.Tensor:
    """Return normalized per-foot WTW phase indices for FL, FR, RL, RR."""

    return env.command_manager.get_term(command_name).foot_indices


def base_external_wrench(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the composed external force/torque applied to the selected robot body.

    This exposes the wrench created by Isaac Lab's permanent wrench composer. Velocity-based pushes are not part of
    this signal and can be added later through a dedicated privileged disturbance term if needed.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    body_force = asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids].reshape(env.num_envs, -1)
    body_torque = (
        asset.permanent_wrench_composer.composed_torque_as_torch[:, asset_cfg.body_ids].reshape(env.num_envs, -1)
    )
    return torch.cat((body_force, body_torque), dim=-1)


def base_external_force(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the composed external force applied to the selected robot body."""

    asset: Articulation = env.scene[asset_cfg.name]
    return asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids].reshape(env.num_envs, -1)


def foot_local_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    radii: Sequence[float] = (0.04, 0.08, 0.12, 0.16, 0.20),
    num_angles: int = 4,
    include_center: bool = True,
    top_k: int = 4,
    clip_value: float | None = 1.0,
) -> torch.Tensor:
    """Return a foot-centric terrain scan built from the existing base-mounted height raycaster.

    The output is a flattened tensor of terrain heights around each foot, expressed relative to the current foot
    height. Positive values indicate terrain that sits above the foot plane, while negative values indicate holes.
    """

    height_sensor, _, foot_xy_sensor, foot_heights, ray_xy_sensor = _get_local_foot_state_in_sensor_frame(
        env=env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    )

    offsets = _build_foot_scan_offsets(
        radii=radii,
        num_angles=num_angles,
        include_center=include_center,
        device=foot_xy_sensor.device,
        dtype=foot_xy_sensor.dtype,
    )

    query_xy_sensor = foot_xy_sensor.unsqueeze(2) + offsets.view(1, 1, -1, 2)
    num_envs, num_feet, num_queries_per_foot, _ = query_xy_sensor.shape

    foot_query_dist = torch.cdist(query_xy_sensor.reshape(num_envs, -1, 2), ray_xy_sensor)
    top_k = min(top_k, foot_query_dist.shape[-1])
    nearest_ray_ids = torch.topk(foot_query_dist, k=top_k, dim=-1, largest=False).indices

    ray_hit_heights = height_sensor.data.ray_hits_w[..., 2]
    expanded_heights = ray_hit_heights.unsqueeze(1).expand(-1, num_feet * num_queries_per_foot, -1)
    sampled_heights = torch.gather(expanded_heights, dim=2, index=nearest_ray_ids)

    finite_mask = torch.isfinite(sampled_heights)
    sampled_heights = torch.where(finite_mask, sampled_heights, torch.zeros_like(sampled_heights))
    valid_counts = finite_mask.sum(dim=2).clamp(min=1)
    local_terrain_height = sampled_heights.sum(dim=2) / valid_counts
    local_terrain_height = local_terrain_height.view(num_envs, num_feet, num_queries_per_foot)

    relative_height_scan = local_terrain_height - foot_heights.unsqueeze(-1)
    if clip_value is not None:
        relative_height_scan = torch.clamp(relative_height_scan, min=-clip_value, max=clip_value)
    return relative_height_scan.reshape(env.num_envs, -1)


def foot_clearance_target(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    top_k: int = 4,
    clip_value: float | None = 1.0,
) -> torch.Tensor:
    """Return terrain-relative foot clearances for FL/FR/RL/RR supervision."""

    height_sensor, _, foot_xy_sensor, foot_heights, ray_xy_sensor = _get_local_foot_state_in_sensor_frame(
        env=env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    )

    foot_ray_dist = torch.cdist(foot_xy_sensor, ray_xy_sensor)
    top_k = min(max(int(top_k), 1), foot_ray_dist.shape[-1])
    nearest_ray_ids = torch.topk(foot_ray_dist, k=top_k, dim=-1, largest=False).indices

    ray_hit_heights = height_sensor.data.ray_hits_w[..., 2]
    expanded_heights = ray_hit_heights.unsqueeze(1).expand(-1, foot_xy_sensor.shape[1], -1)
    sampled_heights = torch.gather(expanded_heights, dim=2, index=nearest_ray_ids)

    finite_mask = torch.isfinite(sampled_heights)
    sampled_heights = torch.where(finite_mask, sampled_heights, torch.zeros_like(sampled_heights))
    valid_counts_raw = finite_mask.sum(dim=2)
    terrain_height = sampled_heights.sum(dim=2) / valid_counts_raw.clamp(min=1)
    terrain_height = torch.where(valid_counts_raw > 0, terrain_height, foot_heights)

    clearance = foot_heights - terrain_height
    if clip_value is not None:
        clearance = torch.clamp(clearance, min=-clip_value, max=clip_value)
    return clearance


def _semantic_vertical_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    *,
    mode: str,
    offset: float,
    min_distance: float,
    max_distance: float,
) -> torch.Tensor:
    """Encode semantic mesh hits as signed heights relative to ``Robot/base``."""

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    data = sensor.data
    expected_mesh = "/World/floor" if mode == "floor" else "/World/overhead"
    if list(sensor.cfg.mesh_prim_paths) != [expected_mesh]:
        raise ValueError(
            f"LOCO3D {mode} scan must target only {expected_mesh!r}, got {sensor.cfg.mesh_prim_paths!r}."
        )
    base_z_w = data.pos_w[..., 2].unsqueeze(-1)
    ray_hit_valid = torch.isfinite(data.ray_hits_w).all(dim=-1)
    return encode_vertical_distance(
        base_z_w,
        data.ray_hits_w[..., 2],
        ray_hit_valid,
        mode=mode,
        offset=offset,
        min_distance=min_distance,
        max_distance=max_distance,
    )


def floor_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.4,
    min_distance: float = -1.0,
    max_distance: float = 1.0,
) -> torch.Tensor:
    """Return signed support-surface height relative to ``Robot/base``.

    The value is ``clip(base.z - floor.z - 0.4, -1, 1)`` in meters by
    default.  Zero represents support 0.4 m below base.  Positive means support
    is farther down than the zero plane, negative means it is closer/higher,
    and no hit maps to the positive limit.  The physical ray origin is
    independent of this network-facing offset.
    """

    return _semantic_vertical_scan(
        env,
        sensor_cfg,
        mode="floor",
        offset=offset,
        min_distance=min_distance,
        max_distance=max_distance,
    )


def ceiling_clearance_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.1,
    min_distance: float = -1.0,
    max_distance: float = 1.0,
) -> torch.Tensor:
    """Return signed overhead-underside height relative to ``Robot/base``.

    The value is ``clip(overhead.z - base.z - 0.1, -1, 1)`` in meters by
    default.  Zero represents an underside 0.1 m above base.  Positive means
    more room than the zero plane, negative means less room, and no hit maps to
    the positive limit.  The function keeps its historical name even though
    this remains base-relative rather than Head_upper-relative.
    """

    return _semantic_vertical_scan(
        env,
        sensor_cfg,
        mode="ceiling",
        offset=offset,
        min_distance=min_distance,
        max_distance=max_distance,
    )


def privileged_spatial_scan(
    env: ManagerBasedRLEnv,
    floor_sensor_cfg: SceneEntityCfg,
    ceiling_sensor_cfg: SceneEntityCfg,
    floor_offset: float = 0.4,
    floor_min_distance: float = -1.0,
    floor_max_distance: float = 1.0,
    ceiling_offset: float = 0.1,
    ceiling_min_distance: float = -1.0,
    ceiling_max_distance: float = 1.0,
) -> torch.Tensor:
    """Concatenate signed ``[floor_scan, ceiling_scan]`` for LOCO3D only."""

    floor = floor_scan(
        env,
        floor_sensor_cfg,
        offset=floor_offset,
        min_distance=floor_min_distance,
        max_distance=floor_max_distance,
    )
    ceiling = ceiling_clearance_scan(
        env,
        ceiling_sensor_cfg,
        offset=ceiling_offset,
        min_distance=ceiling_min_distance,
        max_distance=ceiling_max_distance,
    )
    if floor.shape != ceiling.shape:
        raise RuntimeError(
            f"LOCO3D floor/ceiling grid shapes differ: {tuple(floor.shape)} and {tuple(ceiling.shape)}."
        )
    return torch.cat((floor, ceiling), dim=-1)
