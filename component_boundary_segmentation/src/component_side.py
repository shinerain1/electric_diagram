"""Shared stage-2 component-side classification policy."""

from __future__ import annotations

import math

import numpy as np

from evaluate_exploded_svg_conductors import Primitive


def component_side_probability(role_probability: np.ndarray) -> np.ndarray:
    """Collapse either a binary or three-role output into P(component-side)."""
    if role_probability.ndim != 2:
        raise ValueError("role_probability must have shape [N, C]")
    if role_probability.shape[1] == 2:
        return np.asarray(role_probability[:, 1], dtype=float).copy()
    if role_probability.shape[1] == 3:
        return np.asarray(
            role_probability[:, 0] + role_probability[:, 1], dtype=float
        ).copy()
    raise ValueError("stage-2 model must have two or three output classes")


def long_axis_aligned_conductor_mask(
    primitives: list[Primitive],
    scale: float,
    minimum_length_h: float = 1.5,
    direction_tolerance_degrees: float = 1.0,
) -> np.ndarray:
    """Return high-confidence long carrier strokes fixed as non-component."""
    output = np.zeros(len(primitives), dtype=bool)
    minimum_length = float(minimum_length_h) * float(scale)
    tangent = math.tan(math.radians(float(direction_tolerance_degrees)))
    for index, primitive in enumerate(primitives):
        if (
            primitive.kind != "line"
            or primitive.closed
            or primitive.start is None
            or primitive.length < minimum_length
        ):
            continue
        dx = abs(primitive.end[0] - primitive.start[0])
        dy = abs(primitive.end[1] - primitive.start[1])
        if min(dx, dy) <= tangent * max(dx, dy, 1e-9):
            output[index] = True
    return output


def apply_stage2_policy(
    role_probability: np.ndarray,
    primitives: list[Primitive],
    scale: float,
    minimum_length_h: float = 1.5,
    direction_tolerance_degrees: float = 1.0,
) -> np.ndarray:
    score = component_side_probability(role_probability)
    score[long_axis_aligned_conductor_mask(
        primitives, scale, minimum_length_h, direction_tolerance_degrees
    )] = 0.0
    return score
