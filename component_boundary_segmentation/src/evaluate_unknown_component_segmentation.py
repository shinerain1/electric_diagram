#!/usr/bin/env python3
"""Category-agnostic component boundary segmentation on exploded SVG/XML.

The experiment deliberately withholds several equipment classes from conductor
model training and segmentation-parameter selection.  Component class and
instance IDs are retained only for evaluation after a boundary has been cut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from evaluate_exploded_svg_conductors import (
    FEATURE_NAMES,
    Primitive,
    extract_drawing,
    feature_rows,
)


HELD_OUT_CLASSES = {
    "PowerTransformerClass",
    "PTClass",
    "ArresterClass",
    "DisconnectorClass",
    "LoadSwitchClass",
}


@dataclass(frozen=True)
class SegmentationConfig:
    name: str
    core_wire_probability: float
    join_gap_h: float
    reclaim_wire_probability: float
    reclaim_gap_h: float
    reclaim_max_length_h: float
    hard_reclaim_max_length_h: float
    region_reclaim_gap_h: float
    max_cluster_span_h: float
    interface_lead_max_length_h: float
    split_interface_bridges: bool


CONFIGS = [
    SegmentationConfig(
        "strict_binary",
        0.35, 0.025, 0.70, 0.05, 0.80, 0.25, 0.0, 0.0, 0.0, False,
    ),
    SegmentationConfig(
        "wide_gap_binary",
        0.60, 0.12, 0.88, 0.18, 1.80, 0.45, 0.0, 0.0, 0.0, False,
    ),
    SegmentationConfig(
        "bridge_split_0_25h",
        0.60, 0.12, 0.88, 0.18, 1.80, 0.45, 0.0, 0.0, 0.25, True,
    ),
    SegmentationConfig(
        "bridge_split_0_40h",
        0.60, 0.12, 0.88, 0.18, 1.80, 0.45, 0.0, 0.0, 0.40, True,
    ),
    SegmentationConfig(
        "bridge_split_0_70h",
        0.60, 0.12, 0.88, 0.18, 2.00, 0.55, 0.0, 0.0, 0.70, True,
    ),
    SegmentationConfig(
        "bridge_split_1_00h",
        0.65, 0.10, 0.92, 0.18, 2.50, 0.70, 0.0, 0.0, 1.00, True,
    ),
]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def bbox_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-18:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = (
        (point[0] - start[0]) * vx + (point[1] - start[1]) * vy
    ) / length_sq
    fraction = min(1.0, max(0.0, fraction))
    closest = (start[0] + fraction * vx, start[1] + fraction * vy)
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


def orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    tolerance = 1e-9
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 * o2 < -tolerance and o3 * o4 < -tolerance:
        return True
    return (
        point_segment_distance(a, c, d) <= tolerance
        or point_segment_distance(b, c, d) <= tolerance
        or point_segment_distance(c, a, b) <= tolerance
        or point_segment_distance(d, a, b) <= tolerance
    )


def primitive_distance(left: Primitive, right: Primitive) -> float:
    if bbox_distance(left.bbox, right.bbox) > 0.0:
        bbox_gap = bbox_distance(left.bbox, right.bbox)
    else:
        bbox_gap = 0.0
    left_segments = (
        list(zip(left.geometry_points, left.geometry_points[1:]))
        if len(left.geometry_points) >= 2
        else []
    )
    right_segments = (
        list(zip(right.geometry_points, right.geometry_points[1:]))
        if len(right.geometry_points) >= 2
        else []
    )
    if left_segments and right_segments:
        best = float("inf")
        for left_start, left_end in left_segments:
            for right_start, right_end in right_segments:
                if segments_intersect(
                    left_start, left_end, right_start, right_end
                ):
                    return 0.0
                best = min(
                    best,
                    point_segment_distance(
                        left_start, right_start, right_end
                    ),
                    point_segment_distance(
                        left_end, right_start, right_end
                    ),
                    point_segment_distance(
                        right_start, left_start, left_end
                    ),
                    point_segment_distance(
                        right_end, left_start, left_end
                    ),
                )
        return best
    if left.start is not None and right.start is not None:
        if segments_intersect(left.start, left.end, right.start, right.end):
            return 0.0
        return min(
            point_segment_distance(left.start, right.start, right.end),
            point_segment_distance(left.end, right.start, right.end),
            point_segment_distance(right.start, left.start, left.end),
            point_segment_distance(right.end, left.start, left.end),
        )
    if left.kind == "circle" and right.kind == "circle":
        left_radius = max(
            left.bbox[2] - left.bbox[0],
            left.bbox[3] - left.bbox[1],
        ) / 2.0
        right_radius = max(
            right.bbox[2] - right.bbox[0],
            right.bbox[3] - right.bbox[1],
        ) / 2.0
        return max(
            0.0,
            math.hypot(
                left.center[0] - right.center[0],
                left.center[1] - right.center[1],
            )
            - left_radius
            - right_radius,
        )
    circle = left if left.kind == "circle" else right
    line = right if left.kind == "circle" else left
    if line.start is not None:
        radius = max(
            circle.bbox[2] - circle.bbox[0],
            circle.bbox[3] - circle.bbox[1],
        ) / 2.0
        return max(
            0.0,
            point_segment_distance(circle.center, line.start, line.end)
            - radius,
        )
    return bbox_gap


def candidate_pairs(
    primitives: list[Primitive],
    indices: Iterable[int],
    gap: float,
) -> Iterable[tuple[int, int]]:
    indices = list(indices)
    if len(indices) < 2:
        return
    cell_size = max(gap * 4.0, 1e-6)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    compared: set[tuple[int, int]] = set()
    for index in indices:
        bbox = primitives[index].bbox
        x0 = math.floor((bbox[0] - gap) / cell_size)
        y0 = math.floor((bbox[1] - gap) / cell_size)
        x1 = math.floor((bbox[2] + gap) / cell_size)
        y1 = math.floor((bbox[3] + gap) / cell_size)
        cells = [
            (x, y)
            for x in range(x0, x1 + 1)
            for y in range(y0, y1 + 1)
        ]
        for cell in cells:
            for other in grid[cell]:
                pair = (other, index) if other < index else (index, other)
                if pair in compared:
                    continue
                compared.add(pair)
                if bbox_distance(
                    primitives[pair[0]].bbox,
                    primitives[pair[1]].bbox,
                ) <= gap:
                    yield pair
        for cell in cells:
            grid[cell].append(index)


def obvious_component_seed(primitive: Primitive, feature: np.ndarray) -> bool:
    return bool(
        primitive.closed
        or primitive.kind == "circle"
        or feature[9] >= 0.5  # diagonal
    )


def interface_lead_candidate(
    primitive: Primitive,
    feature: np.ndarray,
    config: SegmentationConfig,
) -> bool:
    """Identify open, short, axis-aligned strokes as provisional port leads."""
    return bool(
        config.interface_lead_max_length_h > 0.0
        and primitive.kind == "line"
        and not primitive.closed
        and feature[6] >= 0.5
        and feature[0] <= config.interface_lead_max_length_h
    )


def identify_interface_leads(
    primitives: list[Primitive],
    features: np.ndarray,
    scale: float,
    config: SegmentationConfig,
) -> set[int]:
    """Keep only short strokes attached to exactly one obvious body core."""
    raw_candidates = {
        index
        for index, primitive in enumerate(primitives)
        if interface_lead_candidate(
            primitive,
            features[index],
            config,
        )
    }
    if not raw_candidates:
        return set()
    obvious = {
        index
        for index, primitive in enumerate(primitives)
        if obvious_component_seed(primitive, features[index])
    }
    if not obvious:
        return set()
    join_gap = config.join_gap_h * scale
    seed_union = UnionFind(len(primitives))
    for left, right in candidate_pairs(primitives, obvious, join_gap):
        if primitive_distance(primitives[left], primitives[right]) <= join_gap:
            seed_union.union(left, right)
    detection_gap = config.reclaim_gap_h * scale
    nearby_seed_roots: dict[int, set[int]] = defaultdict(set)
    search_indices = raw_candidates | obvious
    for left, right in candidate_pairs(
        primitives,
        search_indices,
        detection_gap,
    ):
        if primitive_distance(primitives[left], primitives[right]) > detection_gap:
            continue
        if left in raw_candidates and right in obvious:
            nearby_seed_roots[left].add(seed_union.find(right))
        elif right in raw_candidates and left in obvious:
            nearby_seed_roots[right].add(seed_union.find(left))
    return {
        index
        for index in raw_candidates
        if len(nearby_seed_roots[index]) == 1
    }


def split_interface_bridge_clusters(
    primitives: list[Primitive],
    features: np.ndarray,
    assignment: np.ndarray,
    interface_leads: set[int],
    scale: float,
    config: SegmentationConfig,
) -> None:
    """Split only clusters containing two bodies joined through short leads."""
    if not config.split_interface_bridges or not interface_leads:
        return
    clusters: dict[int, list[int]] = defaultdict(list)
    for index, cluster_value in enumerate(assignment):
        if cluster_value >= 0:
            clusters[int(cluster_value)].append(index)
    next_cluster = int(np.max(assignment)) + 1 if len(assignment) else 0
    join_gap = config.join_gap_h * scale
    attach_gap = config.reclaim_gap_h * scale
    for _, members in clusters.items():
        leads = set(members) & interface_leads
        if not leads:
            continue
        body = [index for index in members if index not in leads]
        if len(body) < 2:
            continue
        body_union = UnionFind(len(primitives))
        for left, right in candidate_pairs(primitives, body, join_gap):
            if primitive_distance(primitives[left], primitives[right]) <= join_gap:
                body_union.union(left, right)
        body_parts: dict[int, list[int]] = defaultdict(list)
        for index in body:
            body_parts[body_union.find(index)].append(index)
        substantial = [
            indices
            for indices in body_parts.values()
            if any(
                obvious_component_seed(
                    primitives[index],
                    features[index],
                )
                for index in indices
            )
        ]
        if len(substantial) < 2:
            continue
        new_clusters = []
        for indices in substantial:
            cluster_id = next_cluster
            next_cluster += 1
            new_clusters.append((cluster_id, indices))
            for index in indices:
                assignment[index] = cluster_id
        assigned_body = {
            index for _, indices in new_clusters for index in indices
        }
        leftovers = [
            index for index in body if index not in assigned_body
        ]
        for index in leftovers:
            distances = [
                (
                    min(
                        primitive_distance(
                            primitives[index],
                            primitives[member],
                        )
                        for member in indices
                    ),
                    cluster_id,
                )
                for cluster_id, indices in new_clusters
            ]
            distances.sort()
            assignment[index] = (
                distances[0][1]
                if distances and distances[0][0] <= attach_gap
                else -1
            )
        for index in leads:
            distances = [
                (
                    min(
                        primitive_distance(
                            primitives[index],
                            primitives[member],
                        )
                        for member in indices
                    ),
                    cluster_id,
                )
                for cluster_id, indices in new_clusters
            ]
            distances.sort()
            if not distances or distances[0][0] > attach_gap:
                assignment[index] = -1
                continue
            if (
                len(distances) >= 2
                and distances[1][0] - distances[0][0] <= 0.02 * scale
            ):
                assignment[index] = -1
                continue
            assignment[index] = distances[0][1]


def segment_components(
    primitives: list[Primitive],
    features: np.ndarray,
    wire_probability: np.ndarray,
    scale: float,
    config: SegmentationConfig,
    predicted_interface_leads: set[int] | None = None,
) -> np.ndarray:
    """Return a predicted cluster ID per primitive, or -1 for conductor."""
    count = len(primitives)
    assignment = np.full(count, -1, dtype=np.int32)
    interface_leads = identify_interface_leads(
        primitives,
        features,
        scale,
        config,
    )
    if predicted_interface_leads:
        interface_leads.update(predicted_interface_leads)
    core = [
        index
        for index, primitive in enumerate(primitives)
        if (
            obvious_component_seed(primitive, features[index])
            or (
                wire_probability[index] < config.core_wire_probability
            )
        )
    ]
    union_find = UnionFind(count)
    join_gap = config.join_gap_h * scale
    for left, right in candidate_pairs(primitives, core, join_gap):
        if primitive_distance(primitives[left], primitives[right]) <= join_gap:
            union_find.union(left, right)

    root_to_cluster: dict[int, int] = {}
    next_cluster = 0
    for index in core:
        root = union_find.find(index)
        if root not in root_to_cluster:
            root_to_cluster[root] = next_cluster
            next_cluster += 1
        assignment[index] = root_to_cluster[root]

    split_interface_bridge_clusters(
        primitives,
        features,
        assignment,
        interface_leads,
        scale,
        config,
    )

    reclaim_gap = config.reclaim_gap_h * scale
    ambiguous = [
        index
        for index in range(count)
        if assignment[index] < 0
        and features[index, 0] <= config.reclaim_max_length_h
    ]
    neighbor_pairs = list(
        candidate_pairs(
            primitives,
            range(count),
            reclaim_gap,
        )
    )
    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in neighbor_pairs:
        if primitive_distance(primitives[left], primitives[right]) <= reclaim_gap:
            adjacency[left].append(right)
            adjacency[right].append(left)

    # Reclaim short ambiguous stems only when they touch one existing component.
    # A primitive touching two components is kept as a conductor, preventing it
    # from becoming an accidental bridge.
    for _ in range(3):
        updates: list[tuple[int, int]] = []
        for index in ambiguous:
            if assignment[index] >= 0:
                continue
            clusters = {
                int(assignment[neighbor])
                for neighbor in adjacency.get(index, [])
                if assignment[neighbor] >= 0
            }
            if len(clusters) != 1:
                continue
            permitted = (
                wire_probability[index]
                < config.reclaim_wire_probability
                or features[index, 0]
                <= config.hard_reclaim_max_length_h
            )
            if permitted:
                updates.append((index, next(iter(clusters))))
        if not updates:
            break
        for index, cluster in updates:
            assignment[index] = cluster

    # Second-pass size confirmation. Some symbols contain short strokes that
    # are intentionally separated from the main glyph. They cannot be found by
    # connected components alone, so recover them inside a generic H-sized
    # envelope. Class names and templates remain unavailable here.
    if config.region_reclaim_gap_h > 0.0:
        region_gap = config.region_reclaim_gap_h * scale
        maximum_span = config.max_cluster_span_h * scale
        for _ in range(2):
            cluster_bboxes: dict[int, list[float]] = {}
            for index, cluster_value in enumerate(assignment):
                if cluster_value < 0:
                    continue
                cluster = int(cluster_value)
                bbox = primitives[index].bbox
                if cluster not in cluster_bboxes:
                    cluster_bboxes[cluster] = list(bbox)
                else:
                    current = cluster_bboxes[cluster]
                    current[0] = min(current[0], bbox[0])
                    current[1] = min(current[1], bbox[1])
                    current[2] = max(current[2], bbox[2])
                    current[3] = max(current[3], bbox[3])
            updates = []
            for index in ambiguous:
                if assignment[index] >= 0:
                    continue
                permitted = (
                    wire_probability[index]
                    < config.reclaim_wire_probability
                    or features[index, 0]
                    <= config.hard_reclaim_max_length_h
                )
                if not permitted:
                    continue
                primitive_bbox = primitives[index].bbox
                compatible = []
                for cluster, cluster_bbox_list in cluster_bboxes.items():
                    cluster_bbox = tuple(cluster_bbox_list)
                    if bbox_distance(primitive_bbox, cluster_bbox) > region_gap:
                        continue
                    combined = (
                        min(primitive_bbox[0], cluster_bbox[0]),
                        min(primitive_bbox[1], cluster_bbox[1]),
                        max(primitive_bbox[2], cluster_bbox[2]),
                        max(primitive_bbox[3], cluster_bbox[3]),
                    )
                    if max(
                        combined[2] - combined[0],
                        combined[3] - combined[1],
                    ) <= maximum_span:
                        compatible.append(cluster)
                if len(compatible) == 1:
                    updates.append((index, compatible[0]))
            if not updates:
                break
            for index, cluster in updates:
                assignment[index] = cluster
    return assignment


def truth_maps(
    primitives: list[Primitive],
    include_classes: set[str] | None = None,
) -> tuple[dict[str, set[int]], dict[str, str], set[int]]:
    instances: dict[str, set[int]] = defaultdict(set)
    classes = {}
    allowed_indices = set()
    for index, primitive in enumerate(primitives):
        if primitive.label == 1:
            allowed_indices.add(index)
            continue
        if not primitive.truth_component_id:
            continue
        if (
            include_classes is not None
            and primitive.truth_component_class not in include_classes
        ):
            continue
        instances[primitive.truth_component_id].add(index)
        classes[primitive.truth_component_id] = primitive.truth_component_class
        allowed_indices.add(index)
    return dict(instances), classes, allowed_indices


def round_metric(value: float) -> float:
    return round(float(value), 6)


def evaluate_segmentation(
    primitives: list[Primitive],
    assignment: np.ndarray,
    include_classes: set[str] | None = None,
    ignore_other_component_classes: bool = False,
) -> dict[str, float | int]:
    truth, _, allowed = truth_maps(primitives, include_classes)
    predictions: dict[int, set[int]] = defaultdict(set)
    for index, cluster in enumerate(assignment):
        if cluster < 0:
            continue
        if ignore_other_component_classes and index not in allowed:
            continue
        predictions[int(cluster)].add(index)
    predictions = {
        key: value for key, value in predictions.items() if value
    }
    if include_classes is not None:
        relevant_truth_indices = {
            index for indices in truth.values() for index in indices
        }
        predictions = {
            key: value
            for key, value in predictions.items()
            if value & relevant_truth_indices
        }

    intersections: Counter[tuple[str, int]] = Counter()
    index_truth = {
        index: truth_id
        for truth_id, indices in truth.items()
        for index in indices
    }
    for pred_id, indices in predictions.items():
        for index in indices:
            if index in index_truth:
                intersections[(index_truth[index], pred_id)] += 1

    truth_best = {truth_id: 0.0 for truth_id in truth}
    pred_best = {pred_id: 0.0 for pred_id in predictions}
    truth_overlap_count: Counter[str] = Counter()
    pred_overlap_count: Counter[int] = Counter()
    for (truth_id, pred_id), overlap in intersections.items():
        score = 2.0 * overlap / (
            len(truth[truth_id]) + len(predictions[pred_id])
        )
        truth_best[truth_id] = max(truth_best[truth_id], score)
        pred_best[pred_id] = max(pred_best[pred_id], score)
        truth_overlap_count[truth_id] += 1
        pred_overlap_count[pred_id] += 1

    truth_values = list(truth_best.values())
    pred_values = list(pred_best.values())
    truth_component_indices = {
        index for indices in truth.values() for index in indices
    }
    predicted_indices = {
        index for indices in predictions.values() for index in indices
    }
    primitive_tp = len(truth_component_indices & predicted_indices)
    primitive_fp = len(predicted_indices - truth_component_indices)
    primitive_fn = len(truth_component_indices - predicted_indices)
    primitive_precision = primitive_tp / max(primitive_tp + primitive_fp, 1)
    primitive_recall = primitive_tp / max(primitive_tp + primitive_fn, 1)
    primitive_f1 = (
        2 * primitive_precision * primitive_recall
        / max(primitive_precision + primitive_recall, 1e-12)
    )
    return {
        "truth_instance_count": len(truth),
        "predicted_cluster_count": len(predictions),
        "mean_truth_best_boundary_f1": round_metric(
            np.mean(truth_values) if truth_values else 0.0
        ),
        "mean_prediction_best_boundary_f1": round_metric(
            np.mean(pred_values) if pred_values else 0.0
        ),
        "truth_recall_boundary_f1_0_50": round_metric(
            np.mean(np.asarray(truth_values) >= 0.50)
            if truth_values
            else 0.0
        ),
        "truth_recall_boundary_f1_0_80": round_metric(
            np.mean(np.asarray(truth_values) >= 0.80)
            if truth_values
            else 0.0
        ),
        "truth_recall_boundary_f1_0_95": round_metric(
            np.mean(np.asarray(truth_values) >= 0.95)
            if truth_values
            else 0.0
        ),
        "truth_split_rate": round_metric(
            np.mean(
                [truth_overlap_count[item] > 1 for item in truth]
            )
            if truth
            else 0.0
        ),
        "prediction_merge_rate": round_metric(
            np.mean(
                [pred_overlap_count[item] > 1 for item in predictions]
            )
            if predictions
            else 0.0
        ),
        "component_primitive_precision": round_metric(primitive_precision),
        "component_primitive_recall": round_metric(primitive_recall),
        "component_primitive_f1": round_metric(primitive_f1),
    }


def combine_metrics(rows: list[dict[str, float | int]]) -> dict[str, float]:
    if not rows:
        return {}
    weights = np.asarray(
        [max(int(row["truth_instance_count"]), 1) for row in rows],
        dtype=float,
    )
    keys = [
        "mean_truth_best_boundary_f1",
        "mean_prediction_best_boundary_f1",
        "truth_recall_boundary_f1_0_50",
        "truth_recall_boundary_f1_0_80",
        "truth_recall_boundary_f1_0_95",
        "truth_split_rate",
        "prediction_merge_rate",
        "component_primitive_precision",
        "component_primitive_recall",
        "component_primitive_f1",
    ]
    output = {
        "drawing_count": len(rows),
        "truth_instance_count": int(
            sum(int(row["truth_instance_count"]) for row in rows)
        ),
        "predicted_cluster_count": int(
            sum(int(row["predicted_cluster_count"]) for row in rows)
        ),
    }
    for key in keys:
        output[key] = round_metric(
            np.average([float(row[key]) for row in rows], weights=weights)
        )
    return output


def selection_score(metrics: dict[str, float]) -> float:
    return (
        0.45 * metrics["mean_truth_best_boundary_f1"]
        + 0.25 * metrics["mean_prediction_best_boundary_f1"]
        + 0.20 * metrics["truth_recall_boundary_f1_0_80"]
        + 0.10 * metrics["component_primitive_f1"]
        - 0.05 * metrics["prediction_merge_rate"]
    )


def paths_by_stem(directory: Path, suffix: str) -> dict[str, Path]:
    return {path.stem: path for path in directory.glob(f"*{suffix}")}


def sample_indices(
    primitives: list[Primitive],
    label: int,
    maximum: int,
    drawing: str,
    seed: int,
) -> list[int]:
    candidates = [
        index
        for index, primitive in enumerate(primitives)
        if primitive.label == label
        and not (
            label == 0
            and primitive.truth_component_class in HELD_OUT_CLASSES
        )
    ]
    return sorted(
        candidates,
        key=lambda index: stable_fraction(
            f"{drawing}:{primitives[index].primitive_id}:{label}",
            seed,
        ),
    )[:maximum]


def conductor_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = probability >= threshold
    return {
        "primitive_count": int(len(truth)),
        "wire_precision": round_metric(
            precision_score(truth, prediction, zero_division=0)
        ),
        "wire_recall": round_metric(
            recall_score(truth, prediction, zero_division=0)
        ),
        "wire_f1": round_metric(
            f1_score(truth, prediction, zero_division=0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--svg-dir",
        type=Path,
        default=Path("data/svg"),
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=Path("data/xml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/exploded_svg_conductors/split_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/unknown_component_segmentation"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--tuning-drawings", type=int, default=60)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split_names = {
        split: [str(row["drawing"]) for row in manifest[split]]
        for split in ("train", "validation", "test")
    }
    svg_by_stem = paths_by_stem(args.svg_dir, ".svg")
    xml_by_stem = paths_by_stem(args.xml_dir, ".xml")

    # Train the primitive separator without any component examples belonging
    # to the held-out equipment classes.
    train_x = []
    train_y = []
    excluded_training_component_primitives = 0
    for number, drawing in enumerate(split_names["train"], 1):
        primitives, audit = extract_drawing(
            svg_by_stem[drawing],
            xml_by_stem[drawing],
        )
        matrix = feature_rows(primitives, float(audit["scale"]))
        known_component = sample_indices(
            primitives, 0, 400, drawing, args.seed
        )
        wire = sample_indices(primitives, 1, 250, drawing, args.seed)
        chosen = known_component + wire
        if chosen:
            train_x.append(matrix[chosen])
            train_y.append(
                np.asarray([primitives[index].label for index in chosen])
            )
        excluded_training_component_primitives += sum(
            primitive.label == 0
            and primitive.truth_component_class in HELD_OUT_CLASSES
            for primitive in primitives
        )
        if number % 200 == 0:
            print(f"training extraction {number}/{len(split_names['train'])}")
    x_train = np.vstack(train_x)
    y_train = np.concatenate(train_y)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=0.5,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(x_train, y_train)

    validation_cache = []
    validation_truth = []
    validation_probability = []
    tuning_names = set(
        sorted(
            split_names["validation"],
            key=lambda drawing: stable_fraction(
                f"tune:{drawing}", args.seed
            ),
        )[: args.tuning_drawings]
    )
    for number, drawing in enumerate(split_names["validation"], 1):
        primitives, audit = extract_drawing(
            svg_by_stem[drawing],
            xml_by_stem[drawing],
        )
        matrix = feature_rows(primitives, float(audit["scale"]))
        probability = model.predict_proba(matrix)[:, 1]
        validation_truth.append(
            np.asarray([primitive.label for primitive in primitives])
        )
        validation_probability.append(probability)
        if drawing in tuning_names:
            validation_cache.append(
                (
                    drawing,
                    primitives,
                    matrix,
                    probability,
                    float(audit["scale"]),
                )
            )
        if number % 50 == 0:
            print(
                f"validation extraction {number}/"
                f"{len(split_names['validation'])}"
            )
    validation_truth_array = np.concatenate(validation_truth)
    validation_probability_array = np.concatenate(validation_probability)
    thresholds = np.linspace(0.30, 0.97, 68)
    wire_threshold = float(
        max(
            thresholds,
            key=lambda threshold: f1_score(
                validation_truth_array,
                validation_probability_array >= threshold,
                zero_division=0,
            ),
        )
    )

    known_classes = {
        primitive.truth_component_class
        for _, primitives, _, _, _ in validation_cache
        for primitive in primitives
        if primitive.label == 0
        and primitive.truth_component_class not in HELD_OUT_CLASSES
    }
    tuning_results = []
    for config in CONFIGS:
        rows = []
        for _, primitives, matrix, probability, scale in validation_cache:
            assignment = segment_components(
                primitives,
                matrix,
                probability,
                scale,
                config,
            )
            rows.append(
                evaluate_segmentation(
                    primitives,
                    assignment,
                    include_classes=known_classes,
                    ignore_other_component_classes=True,
                )
            )
        metrics = combine_metrics(rows)
        score = selection_score(metrics)
        tuning_results.append(
            {
                "config": asdict(config),
                "known_class_validation_metrics": metrics,
                "selection_score": round_metric(score),
            }
        )
        print(config.name, round_metric(score), metrics)
    selected = max(
        tuning_results,
        key=lambda item: item["selection_score"],
    )
    selected_config = SegmentationConfig(**selected["config"])

    def evaluate_split(split: str) -> dict[str, object]:
        overall_rows = []
        held_out_rows = []
        per_class_rows: dict[str, list[dict[str, float | int]]] = defaultdict(list)
        conductor_truth = []
        conductor_probability = []
        for number, drawing in enumerate(split_names[split], 1):
            primitives, audit = extract_drawing(
                svg_by_stem[drawing],
                xml_by_stem[drawing],
            )
            matrix = feature_rows(primitives, float(audit["scale"]))
            probability = model.predict_proba(matrix)[:, 1]
            assignment = segment_components(
                primitives,
                matrix,
                probability,
                float(audit["scale"]),
                selected_config,
            )
            overall_rows.append(
                evaluate_segmentation(primitives, assignment)
            )
            held_out_rows.append(
                evaluate_segmentation(
                    primitives,
                    assignment,
                    include_classes=HELD_OUT_CLASSES,
                    ignore_other_component_classes=False,
                )
            )
            present_classes = {
                primitive.truth_component_class
                for primitive in primitives
                if primitive.truth_component_class in HELD_OUT_CLASSES
            }
            for class_name in present_classes:
                per_class_rows[class_name].append(
                    evaluate_segmentation(
                        primitives,
                        assignment,
                        include_classes={class_name},
                        ignore_other_component_classes=False,
                    )
                )
            conductor_truth.append(
                np.asarray([primitive.label for primitive in primitives])
            )
            conductor_probability.append(probability)
            if number % 50 == 0:
                print(
                    f"{split} segmentation {number}/"
                    f"{len(split_names[split])}"
                )
        return {
            "overall": combine_metrics(overall_rows),
            "held_out_unknown_classes": combine_metrics(held_out_rows),
            "held_out_by_class": {
                class_name: combine_metrics(rows)
                for class_name, rows in sorted(per_class_rows.items())
            },
            "conductor_classification": conductor_metrics(
                np.concatenate(conductor_truth),
                np.concatenate(conductor_probability),
                wire_threshold,
            ),
        }

    validation_result = evaluate_split("validation")
    test_result = evaluate_split("test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "description": (
                "Class-agnostic three-layer boundary cutting: corrected SVG "
                "path expansion, conductor probability prefilter, component "
                "body clustering, interface-lead bridge splitting, then "
                "conservative reclaim of ambiguous short stems."
            ),
            "feature_names": FEATURE_NAMES,
            "svg_path_commands_supported": (
                "M/L/H/V/C/S/Q/T/A/Z, absolute and relative, multiple subpaths"
            ),
            "separates_component_body_interface_lead_and_main_conductor": True,
            "component_type_or_template_used_for_segmentation": False,
            "truth_instance_id_used_only_for_evaluation": True,
            "held_out_classes_absent_from_conductor_training_and_parameter_selection": sorted(
                HELD_OUT_CLASSES
            ),
        },
        "data": {
            "train_drawings": len(split_names["train"]),
            "validation_drawings": len(split_names["validation"]),
            "test_drawings": len(split_names["test"]),
            "tuning_validation_drawings": len(tuning_names),
            "train_sample_primitives": int(len(y_train)),
            "train_sample_component_primitives": int(np.sum(y_train == 0)),
            "train_sample_wire_primitives": int(np.sum(y_train == 1)),
            "excluded_held_out_training_component_primitives": int(
                excluded_training_component_primitives
            ),
        },
        "wire_probability_threshold_selected_on_validation": round_metric(
            wire_threshold
        ),
        "validation_conductor_classification": conductor_metrics(
            validation_truth_array,
            validation_probability_array,
            wire_threshold,
        ),
        "segmentation_parameter_search": tuning_results,
        "selected_segmentation_config": asdict(selected_config),
        "validation": validation_result,
        "test": test_result,
        "interpretation": [
            (
                "Boundary F1 compares the set of exploded primitives in a "
                "predicted cluster with one SVG equipment instance."
            ),
            (
                "Held-out classes were removed from separator training and "
                "from parameter selection; their test metrics therefore "
                "measure transfer to unseen equipment categories."
            ),
            (
                "No SVG class name, symbol ID, XML device type, or standard "
                "component template is available to the segmentation code."
            ),
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model_path = args.output_dir / "unknown_category_conductor_model.joblib"
    joblib.dump(
        {
            "schema_version": "1.0",
            "feature_names": FEATURE_NAMES,
            "model": model,
            "wire_probability_threshold": wire_threshold,
            "held_out_component_classes": sorted(HELD_OUT_CLASSES),
            "segmentation_config": asdict(selected_config),
        },
        model_path,
    )
    print(f"report: {report_path}")
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
