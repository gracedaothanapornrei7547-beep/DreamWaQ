"""Pure tensor helpers for semantic vertical spatial scans.

The helpers in this module deliberately do not import Isaac Sim.  They define
the finite distance encoding shared by the LOCO3D semantic-mesh observations
and CPU-only tests.
"""

from __future__ import annotations

import math
from typing import Literal

import torch


VerticalSurfaceMode = Literal["floor", "ceiling"]


def encode_vertical_distance(
    reference_z_w: torch.Tensor,
    hit_z_w: torch.Tensor,
    valid: torch.Tensor,
    *,
    mode: VerticalSurfaceMode,
    offset: float,
    min_distance: float,
    max_distance: float,
) -> torch.Tensor:
    """Encode a finite signed surface height relative to a reference plane.

    Floor uses ``reference_z - hit_z - offset`` and ceiling uses
    ``hit_z - reference_z - offset``.  A missing/invalid hit maps directly to
    ``max_distance`` after zero-centering, which explicitly means that no
    constraining surface was found in the corresponding direction.
    """

    offset = float(offset)
    min_distance = float(min_distance)
    max_distance = float(max_distance)
    if not math.isfinite(offset):
        raise ValueError(f"offset must be finite, got {offset}.")
    if min_distance >= max_distance:
        raise ValueError(
            f"min_distance must be less than max_distance, got [{min_distance}, {max_distance}]."
        )
    reference_z_w = torch.as_tensor(reference_z_w, device=hit_z_w.device, dtype=hit_z_w.dtype)
    valid = valid.to(device=hit_z_w.device, dtype=torch.bool)
    if mode == "floor":
        distance = reference_z_w - hit_z_w
    elif mode == "ceiling":
        distance = hit_z_w - reference_z_w
    else:
        raise ValueError(f"Unsupported vertical surface mode {mode!r}.")

    distance = distance - offset
    finite_valid = valid & torch.isfinite(distance)
    distance = torch.where(finite_valid, distance, torch.full_like(distance, max_distance))
    distance = torch.nan_to_num(
        distance,
        nan=max_distance,
        posinf=max_distance,
        neginf=min_distance,
    )
    return distance.clamp(min=min_distance, max=max_distance)


__all__ = [
    "VerticalSurfaceMode",
    "encode_vertical_distance",
]
