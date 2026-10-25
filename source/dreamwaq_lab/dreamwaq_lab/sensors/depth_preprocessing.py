"""Shared metric-depth preprocessing for PIE simulation and deployment."""

from __future__ import annotations

import torch


DEPTH_TEMPORAL_ABLATIONS = ("distinct_pair", "latest_duplicate", "single_latest")
SUPPORTED_DEPTH_MAX_DISTANCES = (2.0, 2.5, 3.0)
DEPTH_NORMALIZATION_MODES = ("near_offset", "robot_parkour")


def validate_depth_max_distance(max_distance: float) -> float:
    """Validate the finite clipping distances used by the PIE ablations."""

    max_distance = float(max_distance)
    if not any(abs(max_distance - candidate) < 1.0e-6 for candidate in SUPPORTED_DEPTH_MAX_DISTANCES):
        raise ValueError(
            f"Unsupported PIE depth_max_distance={max_distance}. "
            f"Expected one of {SUPPORTED_DEPTH_MAX_DISTANCES}."
        )
    return max_distance


def apply_depth_temporal_ablation(history: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a two-channel history for the requested temporal ablation.

    ``single_latest`` deliberately preserves the network-facing two-channel
    shape by duplicating the latest frame. It is kept as a separate label from
    ``latest_duplicate`` so experiment metadata can distinguish the intended
    single-frame baseline from a pair-corruption control.
    """

    mode = str(mode).strip().lower()
    if mode not in DEPTH_TEMPORAL_ABLATIONS:
        raise ValueError(f"Unsupported depth temporal ablation {mode!r}; expected {DEPTH_TEMPORAL_ABLATIONS}.")
    if history.ndim != 4 or history.shape[1] != 2:
        raise ValueError(f"Expected a two-frame depth history [N,2,H,W], got {tuple(history.shape)}.")
    if mode == "distinct_pair":
        return history
    latest = history[:, -1:].clone()
    return latest.expand(-1, 2, -1, -1).clone()


def apply_depth_near_plane(
    depth: torch.Tensor,
    near_plane: float,
    max_distance: float,
) -> torch.Tensor:
    """Mask finite returns in front of a camera's physical near plane.

    Isaac Lab's :class:`RayCasterCamera` supports a far clipping distance but
    no near-plane setting.  A zero value is used for masked returns so the
    subsequent Robot Parkour ``[0, max_distance]`` normalization preserves the
    unavailable-pixel convention used by that pipeline.  Non-finite values are
    left untouched for the normal invalid-value handling to process.
    """

    near_plane = float(near_plane)
    max_distance = float(max_distance)
    if not 0.0 <= near_plane < max_distance:
        raise ValueError(
            f"Expected 0 <= near_plane < max_distance, got {near_plane} and {max_distance}."
        )
    if near_plane == 0.0:
        return depth
    return torch.where(
        torch.isfinite(depth) & (depth < near_plane),
        torch.zeros_like(depth),
        depth,
    )


class DistinctDepthFrameBuffer:
    """Keep the latest distinct frames while holding output between updates."""

    def __init__(
        self,
        num_streams: int,
        height: int,
        width: int,
        history_length: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_streams <= 0 or height <= 0 or width <= 0 or history_length <= 0:
            raise ValueError("Depth buffer dimensions and history_length must be positive.")
        self.num_streams = int(num_streams)
        self.height = int(height)
        self.width = int(width)
        self.history_length = int(history_length)
        self.device = torch.device(device)
        self.frames = torch.zeros(
            self.num_streams,
            self.history_length,
            self.height,
            self.width,
            device=self.device,
        )
        self.timestamps = torch.full(
            (self.num_streams, self.history_length),
            -1.0,
            device=self.device,
        )
        self.last_frame_ids = torch.full(
            (self.num_streams,),
            -1,
            device=self.device,
            dtype=torch.long,
        )

    def reset(self, stream_ids: torch.Tensor | list[int] | slice | None = None) -> None:
        if stream_ids is None:
            stream_ids = slice(None)
        self.frames[stream_ids] = 0.0
        self.timestamps[stream_ids] = -1.0
        self.last_frame_ids[stream_ids] = -1

    def update(
        self,
        frames: torch.Tensor,
        frame_ids: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Insert only newer frame IDs and return the zero-order-held history."""

        expected_frame_shape = (self.num_streams, self.height, self.width)
        if tuple(frames.shape) != expected_frame_shape:
            raise ValueError(f"Expected frames {expected_frame_shape}, got {tuple(frames.shape)}.")
        frame_ids = frame_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        timestamps = timestamps.to(device=self.device, dtype=torch.float32).reshape(-1)
        if frame_ids.numel() != self.num_streams or timestamps.numel() != self.num_streams:
            raise ValueError("frame_ids and timestamps must contain one value per stream.")
        frames = frames.to(device=self.device, dtype=torch.float32)
        new_frame = frame_ids != self.last_frame_ids
        first_frame = new_frame & (self.last_frame_ids < 0)
        subsequent_frame = new_frame & ~first_frame
        if torch.any(subsequent_frame):
            if torch.any(frame_ids[subsequent_frame] <= self.last_frame_ids[subsequent_frame]):
                raise RuntimeError("Depth frame IDs must increase monotonically between resets.")
            if torch.any(timestamps[subsequent_frame] + 1.0e-6 < self.timestamps[subsequent_frame, -1]):
                raise RuntimeError("Depth frame timestamps moved backwards.")
            self.frames[subsequent_frame, :-1] = self.frames[subsequent_frame, 1:].clone()
            self.timestamps[subsequent_frame, :-1] = self.timestamps[subsequent_frame, 1:].clone()
            self.frames[subsequent_frame, -1] = frames[subsequent_frame]
            self.timestamps[subsequent_frame, -1] = timestamps[subsequent_frame]
        if torch.any(first_frame):
            self.frames[first_frame] = frames[first_frame].unsqueeze(1).expand(
                -1, self.history_length, -1, -1
            )
            self.timestamps[first_frame] = timestamps[first_frame].unsqueeze(1).expand(
                -1, self.history_length
            )
        self.last_frame_ids[new_frame] = frame_ids[new_frame]
        return self.frames.clone()


def preprocess_depth_image(
    depth: torch.Tensor,
    min_distance: float,
    max_distance: float,
    normalization_mode: str = "near_offset",
) -> torch.Tensor:
    """Clip metric depth and normalize it to ``[0, 1]`` using a fixed physical range.

    ``near_offset`` preserves the existing Loco3D-compatible mapping:
    ``(clip(d, min_distance, max_distance) - min_distance) / (max_distance - min_distance)``,
    with invalid returns mapped to the far plane.

    ``robot_parkour`` follows the Robot Parkour visual pipeline:
    ``clip(d, 0, max_distance) / max_distance``. Positive infinity and NaN map
    to the far plane, while negative infinity and finite non-positive values map
    to zero. The explicit NaN conversion keeps Isaac Lab's no-hit values from
    contaminating network inputs.
    """

    min_distance = float(min_distance)
    max_distance = float(max_distance)
    if not 0.0 <= min_distance < max_distance:
        raise ValueError(
            f"Expected 0 <= min_distance < max_distance, got {min_distance} and {max_distance}."
        )
    normalization_mode = str(normalization_mode).strip().lower()
    if normalization_mode not in DEPTH_NORMALIZATION_MODES:
        raise ValueError(
            f"Unsupported depth normalization mode {normalization_mode!r}; "
            f"expected one of {DEPTH_NORMALIZATION_MODES}."
        )
    depth = depth.to(dtype=torch.float32)

    if normalization_mode == "robot_parkour":
        depth = torch.nan_to_num(
            depth,
            nan=max_distance,
            posinf=max_distance,
            neginf=0.0,
        )
        return torch.clamp(depth, min=0.0, max=max_distance) / max_distance

    invalid = ~torch.isfinite(depth) | (depth <= 0.0)
    depth = torch.nan_to_num(
        depth,
        nan=max_distance,
        posinf=max_distance,
        neginf=max_distance,
    )
    depth = torch.where(invalid, torch.full_like(depth, max_distance), depth)
    depth = torch.clamp(depth, min=min_distance, max=max_distance)
    return (depth - min_distance) / (max_distance - min_distance)


__all__ = [
    "DEPTH_TEMPORAL_ABLATIONS",
    "DEPTH_NORMALIZATION_MODES",
    "SUPPORTED_DEPTH_MAX_DISTANCES",
    "DistinctDepthFrameBuffer",
    "apply_depth_near_plane",
    "apply_depth_temporal_ablation",
    "preprocess_depth_image",
    "validate_depth_max_distance",
]
