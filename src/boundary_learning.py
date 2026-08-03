"""Category-agnostic component-boundary features and model inference.

The boundary model deliberately receives no component family, symbol name,
XML object identifier, or template score.  It learns whether a subset of
nearby vector primitives forms a coherent component boundary.  Classification
remains a separate downstream task.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np


MODEL_SCHEMA = "category-agnostic-boundary-ranker-v1"


FEATURE_NAMES = [
    "candidate_count",
    "group_count",
    "candidate_group_count_ratio",
    "line_fraction",
    "circle_fraction",
    "polygon_fraction",
    "polyline_fraction",
    "path_fraction",
    "closed_fraction",
    "axis_line_fraction",
    "diagonal_line_fraction",
    "candidate_aspect_log",
    "candidate_group_width_ratio",
    "candidate_group_height_ratio",
    "candidate_group_area_ratio",
    "candidate_center_offset",
    "length_mean_scale_ratio",
    "length_std_scale_ratio",
    "length_max_scale_ratio",
    "length_max_median_ratio",
    "internal_distance_mean",
    "internal_distance_max",
    "internal_touch_fraction",
    "internal_near_fraction",
    "internal_graph_density",
    "internal_component_ratio",
    "excluded_count",
    "excluded_distance_min",
    "excluded_distance_mean",
    "excluded_near_fraction",
    "cut_edge_count_ratio",
    "candidate_margin_left",
    "candidate_margin_right",
    "candidate_margin_bottom",
    "candidate_margin_top",
]


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _handle(item: Any) -> str:
    return str(_value(item, "handle", ""))


def _kind(item: Any) -> str:
    return str(_value(item, "kind", "other")).lower()


def _bbox(item: Any) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in _value(item, "bbox"))


def _length(item: Any) -> float:
    return max(float(_value(item, "length", 0.0)), 0.0)


def _closed(item: Any) -> bool:
    return bool(_value(item, "closed", False))


def bbox_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def merge_bbox(
    items: Iterable[Any],
) -> tuple[float, float, float, float]:
    boxes = [_bbox(item) for item in items]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _connected_component_count(
    items: list[Any],
    tolerance: float,
) -> int:
    if not items:
        return 0
    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if bbox_distance(_bbox(items[left]), _bbox(items[right])) <= tolerance:
                union(left, right)
    return len({find(index) for index in range(len(items))})


def candidate_feature_mapping(
    group: list[Any],
    owned_handles: Iterable[str],
) -> dict[str, float]:
    """Describe a candidate subset relative to its complete local group."""
    owned = {str(handle) for handle in owned_handles}
    candidate = [item for item in group if _handle(item) in owned]
    excluded = [item for item in group if _handle(item) not in owned]
    if not candidate:
        raise ValueError("boundary candidate contains no owned primitives")

    candidate_box = merge_bbox(candidate)
    group_box = merge_bbox(group)
    candidate_width = max(candidate_box[2] - candidate_box[0], 1e-9)
    candidate_height = max(candidate_box[3] - candidate_box[1], 1e-9)
    candidate_scale = max(candidate_width, candidate_height, 1e-9)
    group_width = max(group_box[2] - group_box[0], 1e-9)
    group_height = max(group_box[3] - group_box[1], 1e-9)
    group_scale = max(group_width, group_height, 1e-9)

    count = float(len(candidate))
    group_count = float(len(group))
    kinds = [_kind(item) for item in candidate]
    lines = [
        item for item in candidate if _kind(item) in {"line", "polyline"}
    ]
    axis_lines = 0
    for item in lines:
        box = _bbox(item)
        width = box[2] - box[0]
        height = box[3] - box[1]
        if min(width, height) <= max(width, height, 1e-9) * 0.02:
            axis_lines += 1

    lengths = np.asarray([_length(item) for item in candidate], dtype=float)
    positive_lengths = lengths[lengths > 1e-12]
    if len(positive_lengths):
        length_mean = float(positive_lengths.mean())
        length_std = float(positive_lengths.std())
        length_max = float(positive_lengths.max())
        length_median = float(np.median(positive_lengths))
    else:
        length_mean = length_std = length_max = 0.0
        length_median = 1e-9

    pair_distances = []
    touches = 0
    near = 0
    for left in range(len(candidate)):
        for right in range(left + 1, len(candidate)):
            current = bbox_distance(
                _bbox(candidate[left]),
                _bbox(candidate[right]),
            ) / candidate_scale
            pair_distances.append(current)
            if current <= 1e-6:
                touches += 1
            if current <= 0.08:
                near += 1
    pair_count = max(len(pair_distances), 1)
    graph_density_denominator = max(len(candidate) * (len(candidate) - 1) / 2, 1)
    component_count = _connected_component_count(
        candidate,
        tolerance=candidate_scale * 0.08,
    )

    excluded_distances = [
        bbox_distance(candidate_box, _bbox(item)) / group_scale
        for item in excluded
    ]
    cut_edges = 0
    for selected in candidate:
        for other in excluded:
            if bbox_distance(_bbox(selected), _bbox(other)) <= group_scale * 0.08:
                cut_edges += 1

    candidate_center = _center(candidate_box)
    group_center = _center(group_box)
    candidate_area = candidate_width * candidate_height
    group_area = group_width * group_height

    mapping = {
        "candidate_count": count,
        "group_count": group_count,
        "candidate_group_count_ratio": count / max(group_count, 1.0),
        "line_fraction": sum(kind == "line" for kind in kinds) / count,
        "circle_fraction": sum(kind == "circle" for kind in kinds) / count,
        "polygon_fraction": sum(kind == "polygon" for kind in kinds) / count,
        "polyline_fraction": sum(kind == "polyline" for kind in kinds) / count,
        "path_fraction": sum(kind == "path" for kind in kinds) / count,
        "closed_fraction": sum(_closed(item) for item in candidate) / count,
        "axis_line_fraction": axis_lines / max(float(len(lines)), 1.0),
        "diagonal_line_fraction": (
            max(len(lines) - axis_lines, 0) / max(float(len(lines)), 1.0)
        ),
        "candidate_aspect_log": abs(math.log(candidate_width / candidate_height)),
        "candidate_group_width_ratio": candidate_width / group_width,
        "candidate_group_height_ratio": candidate_height / group_height,
        "candidate_group_area_ratio": candidate_area / max(group_area, 1e-9),
        "candidate_center_offset": math.dist(
            candidate_center,
            group_center,
        )
        / group_scale,
        "length_mean_scale_ratio": length_mean / candidate_scale,
        "length_std_scale_ratio": length_std / candidate_scale,
        "length_max_scale_ratio": length_max / candidate_scale,
        "length_max_median_ratio": length_max / max(length_median, 1e-9),
        "internal_distance_mean": (
            float(np.mean(pair_distances)) if pair_distances else 0.0
        ),
        "internal_distance_max": max(pair_distances, default=0.0),
        "internal_touch_fraction": touches / pair_count,
        "internal_near_fraction": near / pair_count,
        "internal_graph_density": near / graph_density_denominator,
        "internal_component_ratio": component_count / count,
        "excluded_count": float(len(excluded)),
        "excluded_distance_min": min(excluded_distances, default=1.0),
        "excluded_distance_mean": (
            float(np.mean(excluded_distances))
            if excluded_distances
            else 1.0
        ),
        "excluded_near_fraction": (
            sum(value <= 0.08 for value in excluded_distances)
            / max(float(len(excluded_distances)), 1.0)
        ),
        "cut_edge_count_ratio": cut_edges / max(count, 1.0),
        "candidate_margin_left": (candidate_box[0] - group_box[0]) / group_width,
        "candidate_margin_right": (group_box[2] - candidate_box[2]) / group_width,
        "candidate_margin_bottom": (
            candidate_box[1] - group_box[1]
        )
        / group_height,
        "candidate_margin_top": (group_box[3] - candidate_box[3]) / group_height,
    }
    return mapping


def candidate_feature_vector(
    group: list[Any],
    owned_handles: Iterable[str],
) -> np.ndarray:
    mapping = candidate_feature_mapping(group, owned_handles)
    return np.asarray([mapping[name] for name in FEATURE_NAMES], dtype=float)


def load_boundary_model(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle.get("schema_version") != MODEL_SCHEMA:
        raise ValueError(
            f"unsupported boundary model schema: {bundle.get('schema_version')}"
        )
    if list(bundle.get("feature_names") or []) != FEATURE_NAMES:
        raise ValueError("boundary model feature schema does not match runtime")
    bundle["path"] = str(path)
    return bundle


def predict_boundary_quality(
    bundle: dict[str, Any],
    group: list[Any],
    owned_handles: Iterable[str],
) -> float:
    vector = candidate_feature_vector(group, owned_handles)
    prediction = float(bundle["model"].predict(vector.reshape(1, -1))[0])
    return min(max(prediction, 0.0), 1.0)


def acceptance_threshold(bundle: dict[str, Any]) -> float:
    metadata = bundle.get("metadata") or {}
    return float(metadata.get("acceptance_threshold", 0.80))
