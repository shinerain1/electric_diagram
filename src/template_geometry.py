"""Geometry normalization and sampling used by component templates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def sample_line(
    start: tuple[float, float],
    end: tuple[float, float],
    count: int = 48,
) -> np.ndarray:
    values = np.linspace(0.0, 1.0, count)
    return np.column_stack(
        (
            start[0] + values * (end[0] - start[0]),
            start[1] + values * (end[1] - start[1]),
        )
    )


def sample_circle(
    center: tuple[float, float],
    radius: float,
    count: int = 96,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return np.column_stack(
        (
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        )
    )


def sample_polyline(
    points: list[tuple[float, float]],
    closed: bool,
) -> np.ndarray:
    materialized = list(points)
    if closed and materialized and materialized[0] != materialized[-1]:
        materialized.append(materialized[0])
    samples = [
        sample_line(start, end, 32)
        for start, end in zip(materialized, materialized[1:])
    ]
    return np.vstack(samples) if samples else np.empty((0, 2))


def normalize_points(points: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    scale = max(float((maximum - minimum).max()), 1e-12)
    return (points - center) / scale


def dihedral_variants(
    points: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    for mirrored in (False, True):
        base = points.copy()
        if mirrored:
            base[:, 0] *= -1.0
        for turns in range(4):
            angle = turns * math.pi / 2.0
            rotation = np.array(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ]
            )
            variants.append(
                (
                    f"{'mirror_x+' if mirrored else ''}"
                    f"rotate_{turns * 90}",
                    base @ rotation.T,
                )
            )
    return variants


def count_similarity(
    candidate_counts: dict[str, int],
    template_counts: dict[str, int],
) -> float:
    names = set(candidate_counts) | set(template_counts)
    intersection = sum(
        min(
            candidate_counts.get(name, 0),
            template_counts.get(name, 0),
        )
        for name in names
    )
    union = sum(
        max(
            candidate_counts.get(name, 0),
            template_counts.get(name, 0),
        )
        for name in names
    )
    return intersection / union if union else 0.0


def sample_template(
    record: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    samples: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for primitive in record.get("normalized_primitives", []):
        kind = primitive["type"]
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "line":
            samples.append(
                sample_line(
                    tuple(primitive["start"]),
                    tuple(primitive["end"]),
                )
            )
        elif kind == "circle":
            samples.append(
                sample_circle(
                    tuple(primitive["center"]),
                    float(primitive["radius"]),
                )
            )
        elif kind in {"polygon", "polyline"}:
            points = [tuple(point) for point in primitive["points"]]
            samples.append(
                sample_polyline(points, closed=kind == "polygon")
            )
    if not samples:
        raise RuntimeError(
            f"template has no sampleable geometry: {record['symbol_id']}"
        )
    return normalize_points(np.vstack(samples)), counts
