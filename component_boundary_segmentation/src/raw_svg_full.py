#!/usr/bin/env python3
"""Build full-drawing SVG training data without semantic input filtering.

The blind parser creates the model input.  SVG semantic groups and paired XML
are read only afterwards to attach training/evaluation labels to geometrically
identical anonymous primitives.  Geometry outside the electrical truth set is
labelled background instead of being removed before inference.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_exploded_svg_conductors import (
    FEATURE_NAMES,
    Primitive,
    extract_drawing,
    feature_rows,
)
from segment_svg_blind import svg_primitives_blind


RAW_LABEL_NAMES = {
    0: "electrical_component_or_wire",
    1: "background_frame_or_annotation",
}

BACKGROUND_FEATURE_NAMES = [
    "center_x_normalized",
    "center_y_normalized",
    "bbox_width_drawing_ratio",
    "bbox_height_drawing_ratio",
    "length_drawing_span_ratio",
    "distance_left_ratio",
    "distance_right_ratio",
    "distance_top_ratio",
    "distance_bottom_ratio",
    "nearest_outer_edge_ratio",
    "long_axis_aligned_near_outer_edge",
    "outer_frame_rectangle_rule",
]

RAW_FOREGROUND_FEATURE_NAMES = FEATURE_NAMES + BACKGROUND_FEATURE_NAMES


def primitive_signature(
    primitive: Primitive,
    quantum: float,
) -> tuple[Any, ...]:
    """Quantized visual signature used only to transfer training labels."""

    def q(value: float) -> int:
        return round(float(value) / quantum)

    if primitive.kind == "circle":
        return (
            "circle",
            q(primitive.center[0]),
            q(primitive.center[1]),
            *(q(value) for value in primitive.bbox),
        )
    endpoints = sorted((primitive.start, primitive.end))
    return (
        "line",
        *(q(value) for point in endpoints for value in point),
    )


def _drawing_bounds(
    primitives: list[Primitive],
) -> tuple[float, float, float, float]:
    if not primitives:
        return 0.0, 0.0, 1.0, 1.0
    return (
        min(item.bbox[0] for item in primitives),
        min(item.bbox[1] for item in primitives),
        max(item.bbox[2] for item in primitives),
        max(item.bbox[3] for item in primitives),
    )


def outer_frame_rule(primitives: list[Primitive]) -> np.ndarray:
    """Find only high-confidence outer rectangular frame strokes.

    A single long busbar is never enough.  Hard frame removal is enabled only
    when long axis-aligned strokes provide both horizontal and vertical outer
    border evidence.  The learned foreground model receives the same rule as a
    feature and can additionally reject title-block and annotation geometry.
    """

    count = len(primitives)
    result = np.zeros(count, dtype=bool)
    if count < 4:
        return result
    left, bottom, right, top = _drawing_bounds(primitives)
    width = max(right - left, 1e-9)
    height = max(top - bottom, 1e-9)
    horizontal: list[int] = []
    vertical: list[int] = []
    for index, primitive in enumerate(primitives):
        if primitive.kind != "line" or primitive.start is None:
            continue
        dx = abs(primitive.end[0] - primitive.start[0])
        dy = abs(primitive.end[1] - primitive.start[1])
        if dx >= 0.25 * width and dy <= 0.01 * max(dx, width):
            y = primitive.center[1]
            if min(abs(y - bottom), abs(y - top)) <= 0.015 * height:
                horizontal.append(index)
        if dy >= 0.25 * height and dx <= 0.01 * max(dy, height):
            x = primitive.center[0]
            if min(abs(x - left), abs(x - right)) <= 0.015 * width:
                vertical.append(index)
    if len(horizontal) >= 2 and len(vertical) >= 2:
        result[horizontal] = True
        result[vertical] = True
    return result


def background_feature_rows(primitives: list[Primitive]) -> np.ndarray:
    if not primitives:
        return np.empty((0, len(BACKGROUND_FEATURE_NAMES)), dtype=float)
    left, bottom, right, top = _drawing_bounds(primitives)
    width = max(right - left, 1e-9)
    height = max(top - bottom, 1e-9)
    span = max(width, height)
    frame = outer_frame_rule(primitives)
    rows = []
    for index, primitive in enumerate(primitives):
        bbox_width = primitive.bbox[2] - primitive.bbox[0]
        bbox_height = primitive.bbox[3] - primitive.bbox[1]
        center_x = (primitive.center[0] - left) / width
        center_y = (primitive.center[1] - bottom) / height
        edge_distances = (
            max(primitive.bbox[0] - left, 0.0) / width,
            max(right - primitive.bbox[2], 0.0) / width,
            max(primitive.bbox[1] - bottom, 0.0) / height,
            max(top - primitive.bbox[3], 0.0) / height,
        )
        axis_aligned = (
            primitive.kind == "line"
            and min(bbox_width, bbox_height)
            <= 0.02 * max(bbox_width, bbox_height, 1e-9)
        )
        long_outer = (
            axis_aligned
            and primitive.length >= 0.20 * span
            and min(edge_distances) <= 0.02
        )
        rows.append(
            [
                center_x,
                center_y,
                bbox_width / width,
                bbox_height / height,
                primitive.length / span,
                *edge_distances,
                min(edge_distances),
                float(long_outer),
                float(frame[index]),
            ]
        )
    return np.asarray(rows, dtype=float)


def raw_foreground_feature_rows(
    primitives: list[Primitive],
    scale: float,
) -> np.ndarray:
    return np.hstack(
        [
            feature_rows(primitives, scale),
            background_feature_rows(primitives),
        ]
    )


def labeled_full_drawing(
    svg_path: Path,
    xml_path: Path,
) -> tuple[list[Primitive], np.ndarray, dict[str, Any]]:
    """Return blind full-drawing primitives plus post-hoc binary labels.

    Label 0 is an electrical component/wire primitive.  Label 1 is visible
    geometry excluded from the electrical semantic truth, including drawing
    frames and annotation graphics.  Labels are never included in features.
    """

    raw, raw_audit = svg_primitives_blind(svg_path)
    truth, truth_audit = extract_drawing(
        svg_path,
        xml_path,
        include_truth_terminals=True,
    )
    quantum = max(
        min(float(raw_audit["scale"]), float(truth_audit["scale"])) * 1e-6,
        1e-8,
    )
    by_signature: dict[tuple[Any, ...], deque[Primitive]] = defaultdict(deque)
    for primitive in truth:
        by_signature[primitive_signature(primitive, quantum)].append(primitive)

    labels = np.ones(len(raw), dtype=np.int8)
    matched_truth = 0
    for index, primitive in enumerate(raw):
        candidates = by_signature.get(primitive_signature(primitive, quantum))
        if not candidates:
            primitive.label = 2
            continue
        source = candidates.popleft()
        labels[index] = 0
        primitive.label = source.label
        primitive.truth_component_id = source.truth_component_id
        primitive.truth_component_class = source.truth_component_class
        matched_truth += 1
    missing_truth = sum(len(items) for items in by_signature.values())
    audit = {
        **raw_audit,
        "semantic_truth_primitive_count": len(truth),
        "matched_electrical_primitive_count": matched_truth,
        "missing_semantic_truth_primitive_count": missing_truth,
        "background_primitive_count": int(np.sum(labels == 1)),
        "truth_match_recall": matched_truth / max(len(truth), 1),
        "semantic_scale_for_audit_only": float(truth_audit["scale"]),
        "component_terminals_for_labeling_only": truth_audit.get(
            "component_terminals", {}
        ),
        "label_contract": (
            "blind full geometry is created first; SVG/XML semantics are "
            "used only to attach post-hoc training/evaluation labels"
        ),
    }
    return raw, labels, audit


def binary_metrics(
    truth_electrical: np.ndarray,
    predicted_electrical: np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(truth_electrical, dtype=bool)
    prediction = np.asarray(predicted_electrical, dtype=bool)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "electrical_precision": precision,
        "electrical_recall": recall,
        "electrical_f1": f1,
        "background_recall": tn / max(tn + fp, 1),
    }


def stable_sample_indices(
    labels: np.ndarray,
    primitive_ids: list[str],
    drawing: str,
    seed: int,
    electrical_limit: int = 1200,
    background_limit: int = 1200,
) -> np.ndarray:
    """Deterministic balanced per-drawing sample for the foreground model."""

    import hashlib

    selected = []
    for label, limit in ((0, electrical_limit), (1, background_limit)):
        candidates = np.flatnonzero(labels == label).tolist()
        candidates.sort(
            key=lambda index: hashlib.sha256(
                f"{seed}:{drawing}:{primitive_ids[index]}:{label}".encode(
                    "utf-8"
                )
            ).digest()
        )
        selected.extend(candidates[:limit])
    return np.asarray(selected, dtype=np.int64)


__all__ = [
    "BACKGROUND_FEATURE_NAMES",
    "RAW_FOREGROUND_FEATURE_NAMES",
    "RAW_LABEL_NAMES",
    "background_feature_rows",
    "binary_metrics",
    "labeled_full_drawing",
    "outer_frame_rule",
    "raw_foreground_feature_rows",
    "stable_sample_indices",
]
