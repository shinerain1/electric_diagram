#!/usr/bin/env python3
"""Learned edge decisions for cutting touching component boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluate_exploded_svg_conductors import Primitive
from network_boundary import UnionFind, component_node_mask


EDGE_JOIN_FEATURE_NAMES = [
    "contact_distance",
    "endpoint_endpoint_contact",
    "endpoint_to_interior_contact",
    "interior_crossing",
    "direction_parallel",
    "direction_perpendicular",
    "log_length_ratio_abs",
    "body_probability_min",
    "body_probability_max",
    "interface_probability_min",
    "interface_probability_max",
    "conductor_probability_min",
    "conductor_probability_max",
    "electrical_probability_min",
    "electrical_probability_max",
    "base_wire_probability_min",
    "base_wire_probability_max",
    "length_h_min",
    "length_h_max",
    "closed_min",
    "closed_max",
    "axis_aligned_min",
    "axis_aligned_max",
    "diagonal_min",
    "diagonal_max",
    "endpoint_degree_sum_min",
    "endpoint_degree_sum_max",
    "neighbor_count_1h_min",
    "neighbor_count_1h_max",
    "short_to_long_ratio",
    "short_lead_long_carrier_pattern",
]


class FeatureSubsetProbabilityModel:
    """Apply a fitted sklearn classifier to a declared feature subset.

    Keeping the transform in the serialized model prevents training/inference
    skew while retaining the ordinary ``predict_proba`` interface.
    """

    def __init__(self, model: Any, feature_indices: list[int]) -> None:
        self.model = model
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.classes_ = model.classes_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features[:, self.feature_indices])


def undirected_edge_rows(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if edge_index.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(
            (0, edge_attr.shape[1] if edge_attr.ndim == 2 else 7),
            dtype=np.float32,
        )
    chosen = np.flatnonzero(edge_index[0] < edge_index[1])
    return edge_index[:, chosen].T.astype(np.int64), edge_attr[chosen]


def edge_join_feature_rows(
    pairs: np.ndarray,
    attributes: np.ndarray,
    role_probability: np.ndarray,
    base_features: np.ndarray,
    base_wire_probability: np.ndarray,
    electrical_probability: np.ndarray,
) -> np.ndarray:
    if not len(pairs):
        return np.empty((0, len(EDGE_JOIN_FEATURE_NAMES)), dtype=np.float32)
    rows = []
    for (left, right), relation in zip(pairs, attributes):
        left = int(left)
        right = int(right)
        roles_left = role_probability[left]
        roles_right = role_probability[right]

        def minmax(a: float, b: float) -> list[float]:
            return [min(float(a), float(b)), max(float(a), float(b))]

        length_left = float(base_features[left, 0])
        length_right = float(base_features[right, 0])
        short_length = min(length_left, length_right)
        long_length = max(length_left, length_right)
        ratio = short_length / max(long_length, 1e-8)
        if length_left <= length_right:
            short_index, long_index = left, right
        else:
            short_index, long_index = right, left
        pattern = float(
            short_length <= 0.75
            and long_length >= 1.50
            and base_features[long_index, 6] >= 0.5
            and role_probability[short_index, 1] >= 0.20
            and role_probability[long_index, 2] >= 0.20
        )
        rows.append(
            [
                *[float(value) for value in relation],
                *minmax(roles_left[0], roles_right[0]),
                *minmax(roles_left[1], roles_right[1]),
                *minmax(roles_left[2], roles_right[2]),
                *minmax(
                    electrical_probability[left],
                    electrical_probability[right],
                ),
                *minmax(
                    base_wire_probability[left],
                    base_wire_probability[right],
                ),
                *minmax(length_left, length_right),
                *minmax(base_features[left, 10], base_features[right, 10]),
                *minmax(base_features[left, 6], base_features[right, 6]),
                *minmax(base_features[left, 9], base_features[right, 9]),
                *minmax(base_features[left, 15], base_features[right, 15]),
                *minmax(base_features[left, 16], base_features[right, 16]),
                ratio,
                pattern,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def structural_carrier_mask(
    primitives: list[Primitive],
    pairs: np.ndarray,
    base_features: np.ndarray,
) -> np.ndarray:
    """High-confidence long carrier with multiple substantially shorter taps."""

    neighbors: list[list[int]] = [[] for _ in primitives]
    for left, right in pairs:
        neighbors[int(left)].append(int(right))
        neighbors[int(right)].append(int(left))
    result = np.zeros(len(primitives), dtype=bool)
    for index, primitive in enumerate(primitives):
        length_h = float(base_features[index, 0])
        if (
            primitive.kind != "line"
            or base_features[index, 6] < 0.5
            or length_h < 1.50
            or len(neighbors[index]) < 2
        ):
            continue
        short_taps = sum(
            float(base_features[neighbor, 0]) <= 0.60 * length_h
            for neighbor in neighbors[index]
        )
        result[index] = short_taps >= 2
    return result


def edge_join_labels(
    foreground_primitives: list[Primitive],
    pairs: np.ndarray,
) -> np.ndarray:
    output = []
    for left, right in pairs:
        left_id = foreground_primitives[int(left)].truth_component_id
        right_id = foreground_primitives[int(right)].truth_component_id
        output.append(
            int(left_id is not None and left_id == right_id)
        )
    return np.asarray(output, dtype=np.int8)


def learned_edge_connected_components(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    role_probability: np.ndarray,
    foreground_primitives: list[Primitive],
    base_features: np.ndarray,
    base_wire_probability: np.ndarray,
    electrical_probability: np.ndarray,
    edge_payload: dict[str, Any],
    conductor_threshold: float,
    interface_threshold: float,
    force_structural_carriers: bool | None = None,
    precomputed_same_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    pairs, attributes = undirected_edge_rows(edge_index, edge_attr)
    features = edge_join_feature_rows(
        pairs,
        attributes,
        role_probability,
        base_features,
        base_wire_probability,
        electrical_probability,
    )
    model = edge_payload["model"]
    same_class = int(edge_payload.get("same_component_class", 1))
    column = int(np.flatnonzero(model.classes_ == same_class)[0])
    same_probability = (
        np.asarray(precomputed_same_probability, dtype=float)
        if precomputed_same_probability is not None
        else model.predict_proba(features)[:, column]
        if len(features)
        else np.empty(0, dtype=float)
    )
    threshold = float(edge_payload["same_component_probability_threshold"])
    component = component_node_mask(
        role_probability, conductor_threshold, interface_threshold
    )
    use_carrier_rule = (
        bool(edge_payload.get("force_structural_carriers", False))
        if force_structural_carriers is None
        else force_structural_carriers
    )
    carriers = structural_carrier_mask(
        foreground_primitives, pairs, base_features
    )
    if use_carrier_rule:
        component &= ~carriers

    union_find = UnionFind(len(foreground_primitives))
    accepted = 0
    candidate = 0
    for edge_number, (left, right) in enumerate(pairs):
        left = int(left)
        right = int(right)
        if not (component[left] and component[right]):
            continue
        candidate += 1
        if same_probability[edge_number] >= threshold:
            union_find.union(left, right)
            accepted += 1
    assignment = np.full(len(foreground_primitives), -1, dtype=np.int32)
    root_to_cluster: dict[int, int] = {}
    for index in np.flatnonzero(component):
        root = union_find.find(int(index))
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        assignment[index] = root_to_cluster[root]
    return assignment, {
        "edge_pair_count": len(pairs),
        "component_candidate_edge_count": candidate,
        "accepted_same_component_edge_count": accepted,
        "structural_carrier_count": int(np.sum(carriers)),
        "force_structural_carriers": use_carrier_rule,
    }


__all__ = [
    "EDGE_JOIN_FEATURE_NAMES",
    "FeatureSubsetProbabilityModel",
    "edge_join_feature_rows",
    "edge_join_labels",
    "learned_edge_connected_components",
    "structural_carrier_mask",
    "undirected_edge_rows",
]
