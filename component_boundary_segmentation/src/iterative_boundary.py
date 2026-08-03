#!/usr/bin/env python3
"""Stage-3 component boundary growth from soft stage-2 probabilities.

The algorithm is category agnostic.  It never reads component names, template
IDs or truth instance IDs.  High-confidence component-side primitives form
seeds; uncertain neighboring primitives may be reabsorbed when contact-edge,
size and carrier constraints agree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from component_side import long_axis_aligned_conductor_mask
from evaluate_exploded_svg_conductors import Primitive
from evaluate_unknown_component_segmentation import candidate_pairs, primitive_distance
from logical_context import LOGIC_FEATURE_NAMES, logical_feature_rows
from network_boundary import UnionFind
from same_component_edge import (
    edge_join_feature_rows,
    structural_carrier_mask,
    undirected_edge_rows,
)


@dataclass(frozen=True)
class IterativeBoundaryConfig:
    """Conservative, scale-relative boundary-growth configuration."""

    seed_probability_floor: float = 0.0
    candidate_probability_ratio: float = 0.50
    candidate_probability_floor: float = 0.002
    rescue_edge_probability: float = 0.90
    maximum_candidate_length_h: float = 1.50
    profile_name: str = "generic_safe"
    region_reclaim_gap_h: float = 0.12
    region_unique_margin_h: float = 0.15
    maximum_boundary_span_h: float = 6.0
    cluster_merge_gap_h: float = 0.0
    maximum_merged_boundary_span_h: float = 6.0
    maximum_rounds: int = 8
    long_axis_length_h: float = 1.50
    long_axis_tolerance_degrees: float = 1.0
    allow_multi_cluster_bridge: bool = False
    same_component_join_threshold: float | None = None

    @classmethod
    def dxf_fixed_h_profile(cls) -> "IterativeBoundaryConfig":
        """DXF profile for the project convention H=100.

        DXF composite glyphs in the current corpus span many H units, unlike
        the SVG scale estimator.  Keeping this explicit prevents domain-scale
        parameters from being silently reused on SVG drawings.
        """
        return cls(
            profile_name="dxf_fixed_h_100",
            seed_probability_floor=0.50,
            candidate_probability_ratio=0.25,
            rescue_edge_probability=0.65,
            region_reclaim_gap_h=2.0,
            maximum_boundary_span_h=10.0,
            cluster_merge_gap_h=3.5,
            maximum_merged_boundary_span_h=30.0,
        )

    @classmethod
    def svg_template_disjoint_profile(cls) -> "IterativeBoundaryConfig":
        """Selection-set calibrated SVG profile for the strict edge model."""
        return cls(
            profile_name="svg_template_disjoint",
            candidate_probability_ratio=0.25,
            rescue_edge_probability=0.80,
            region_reclaim_gap_h=0.0,
            cluster_merge_gap_h=0.0,
            same_component_join_threshold=0.20,
        )


def _same_component_probability(
    pairs: np.ndarray,
    attributes: np.ndarray,
    role_probability: np.ndarray,
    base_features: np.ndarray,
    base_wire_probability: np.ndarray,
    electrical_probability: np.ndarray,
    edge_payload: dict[str, Any] | None,
) -> tuple[np.ndarray, float]:
    """Return learned join probabilities or a conservative geometric proxy."""
    if not len(pairs):
        return np.empty(0, dtype=float), 1.0
    if edge_payload is not None:
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
        return (
            model.predict_proba(features)[:, column],
            float(edge_payload["same_component_probability_threshold"]),
        )

    # This fallback is intentionally conservative and is used by unit tests or
    # deployments where the optional edge model is absent.
    endpoint = attributes[:, 1]
    endpoint_to_interior = attributes[:, 2]
    crossing = attributes[:, 3]
    distance = attributes[:, 0]
    probability = (
        0.35
        + 0.35 * endpoint
        + 0.20 * endpoint_to_interior
        - 0.25 * crossing
        - 0.10 * np.minimum(distance, 1.0)
    )
    return np.clip(probability, 0.0, 1.0), 0.60


def _cluster_bbox(
    primitives: list[Primitive], indices: list[int]
) -> tuple[float, float, float, float]:
    return (
        min(primitives[index].bbox[0] for index in indices),
        min(primitives[index].bbox[1] for index in indices),
        max(primitives[index].bbox[2] for index in indices),
        max(primitives[index].bbox[3] for index in indices),
    )


def _projected_span_h(
    primitives: list[Primitive], indices: list[int], candidate: int, scale: float
) -> float:
    box = _cluster_bbox(primitives, [*indices, candidate])
    return max(box[2] - box[0], box[3] - box[1]) / max(scale, 1e-9)


def iterative_component_boundaries(
    primitives: list[Primitive],
    scale: float,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    role_probability: np.ndarray,
    component_probability: np.ndarray,
    component_threshold: float,
    base_features: np.ndarray,
    base_wire_probability: np.ndarray,
    electrical_probability: np.ndarray | None = None,
    edge_payload: dict[str, Any] | None = None,
    config: IterativeBoundaryConfig | None = None,
    source_group_ids: list[str] | None = None,
    precomputed_same_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create boundaries and recover uncertain primitives after stage 2.

    Returns compact component IDs and an audit containing each primitive's
    membership source (seed, reabsorbed, or excluded).
    """
    config = config or IterativeBoundaryConfig()
    count = len(primitives)
    if role_probability.shape != (count, 3):
        raise ValueError("role_probability must have shape [N, 3]")
    component_probability = np.asarray(component_probability, dtype=float)
    if component_probability.shape != (count,):
        raise ValueError("component_probability must have shape [N]")
    if electrical_probability is None:
        electrical_probability = np.ones(count, dtype=float)

    pairs, attributes = undirected_edge_rows(edge_index, edge_attr)
    if precomputed_same_probability is None:
        same_probability, join_threshold = _same_component_probability(
            pairs,
            attributes,
            role_probability,
            base_features,
            base_wire_probability,
            np.asarray(electrical_probability, dtype=float),
            edge_payload,
        )
    else:
        same_probability = np.asarray(precomputed_same_probability, dtype=float)
        if same_probability.shape != (len(pairs),):
            raise ValueError("precomputed_same_probability must have one value per undirected edge")
        join_threshold = float(
            edge_payload.get("same_component_probability_threshold", 0.5)
            if edge_payload is not None
            else 0.5
        )
    if config.same_component_join_threshold is not None:
        join_threshold = float(config.same_component_join_threshold)
    neighbors: list[list[tuple[int, float, int]]] = [[] for _ in range(count)]
    for edge_number, (left_value, right_value) in enumerate(pairs):
        left, right = int(left_value), int(right_value)
        probability = float(same_probability[edge_number])
        neighbors[left].append((right, probability, edge_number))
        neighbors[right].append((left, probability, edge_number))
    proximity_neighbors: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    region_gap = config.region_reclaim_gap_h * scale
    if region_gap > 0.0:
        for left, right in candidate_pairs(primitives, range(count), region_gap):
            distance = primitive_distance(primitives[left], primitives[right])
            if distance <= region_gap:
                proximity_neighbors[left].append((right, distance))
                proximity_neighbors[right].append((left, distance))

    logic = logical_feature_rows(primitives, scale, edge_index, edge_attr)
    symbol_column = LOGIC_FEATURE_NAMES.index(
        "local_closed_or_circle_count_0_5h"
    )
    long_axis = long_axis_aligned_conductor_mask(
        primitives,
        scale,
        config.long_axis_length_h,
        config.long_axis_tolerance_degrees,
    )
    structural = structural_carrier_mask(primitives, pairs, base_features)
    hard_carrier = structural | (
        long_axis & (logic[:, symbol_column] <= 0.0)
    )

    seed_threshold = max(component_threshold, config.seed_probability_floor)
    active = (component_probability >= seed_threshold) & ~hard_carrier
    membership_source = np.full(count, "excluded", dtype=object)
    membership_source[active] = "seed"
    union_find = UnionFind(count)
    for edge_number, (left_value, right_value) in enumerate(pairs):
        left, right = int(left_value), int(right_value)
        if (
            active[left]
            and active[right]
            and same_probability[edge_number] >= join_threshold
        ):
            union_find.union(left, right)
    source_group_union_count = 0
    if source_group_ids is not None:
        if len(source_group_ids) != count:
            raise ValueError("source_group_ids must have length N")
        grouped: dict[str, list[int]] = {}
        for index, group_id in enumerate(source_group_ids):
            if active[index] and group_id:
                grouped.setdefault(str(group_id), []).append(index)
        for indices in grouped.values():
            # A closed DXF polyline is one source entity even though the parser
            # represents its sides separately.  Keep its selected sides in one
            # boundary; open wiring polylines are not force-joined here.
            if len(indices) < 2 or not any(primitives[index].closed for index in indices):
                continue
            anchor = indices[0]
            for index in indices[1:]:
                if union_find.find(anchor) != union_find.find(index):
                    union_find.union(anchor, index)
                    source_group_union_count += 1

    minimum_candidate_probability = max(
        config.candidate_probability_floor,
        config.candidate_probability_ratio * component_threshold,
    )
    rounds: list[dict[str, int]] = []
    rejected_multi_cluster = 0
    rejected_span = 0
    contact_reabsorbed = 0
    region_reabsorbed = 0
    for round_number in range(1, config.maximum_rounds + 1):
        cluster_members: dict[int, list[int]] = {}
        for index in np.flatnonzero(active):
            cluster_members.setdefault(union_find.find(int(index)), []).append(int(index))
        accepted: list[tuple[int, int, list[int], str]] = []
        for candidate in np.flatnonzero(~active & ~hard_carrier):
            candidate = int(candidate)
            primitive = primitives[candidate]
            symbol_like = bool(
                primitive.closed
                or primitive.kind in {"circle", "arc", "ellipse", "spline"}
                or float(base_features[candidate, 9]) >= 0.5
            )
            short_enough = (
                float(base_features[candidate, 0])
                <= config.maximum_candidate_length_h
            )
            if not (short_enough or symbol_like):
                continue
            if (
                component_probability[candidate] < minimum_candidate_probability
                and not symbol_like
            ):
                continue
            contacts: dict[int, list[tuple[int, float]]] = {}
            for other, edge_score, _ in neighbors[candidate]:
                if not active[other] or edge_score < config.rescue_edge_probability:
                    continue
                root = union_find.find(other)
                contacts.setdefault(root, []).append((other, edge_score))
            method = "contact"
            if contacts:
                ordered = sorted(
                    contacts.items(),
                    key=lambda item: max(score for _, score in item[1]),
                    reverse=True,
                )
                if len(ordered) > 1 and not config.allow_multi_cluster_bridge:
                    rejected_multi_cluster += 1
                    continue
                root, contact_rows = ordered[0]
            else:
                # Some valid symbols contain intentional gaps.  Search only a
                # bounded H-sized neighborhood and require the nearest existing
                # boundary to be unambiguous.  This does not join seed clusters;
                # it only recovers a primitive omitted by stage 2.
                region_contacts: dict[int, list[tuple[int, float]]] = {}
                for other, distance in proximity_neighbors[candidate]:
                    if not active[other]:
                        continue
                    root = union_find.find(other)
                    region_contacts.setdefault(root, []).append((other, distance))
                if not region_contacts:
                    continue
                nearest = sorted(
                    (
                        min(distance for _, distance in values),
                        root,
                        values,
                    )
                    for root, values in region_contacts.items()
                )
                if (
                    len(nearest) > 1
                    and nearest[1][0] - nearest[0][0]
                    < config.region_unique_margin_h * scale
                ):
                    rejected_multi_cluster += 1
                    continue
                if (
                    component_probability[candidate] < component_threshold
                    and not symbol_like
                ):
                    continue
                _, root, region_rows = nearest[0]
                contact_rows = [
                    (other, 1.0 - min(distance / max(region_gap, 1e-9), 1.0))
                    for other, distance in region_rows
                    if distance <= nearest[0][0] + config.region_unique_margin_h * scale
                ]
                method = "region"
            members = cluster_members[root]
            if (
                _projected_span_h(primitives, members, candidate, scale)
                > config.maximum_boundary_span_h
            ):
                rejected_span += 1
                continue
            accepted.append(
                (candidate, root, [other for other, _ in contact_rows], method)
            )
        for candidate, root, contacts, method in accepted:
            active[candidate] = True
            membership_source[candidate] = "reabsorbed"
            if method == "contact":
                contact_reabsorbed += 1
            else:
                region_reabsorbed += 1
            for other in contacts:
                if union_find.find(other) == root:
                    union_find.union(candidate, other)
        rounds.append({
            "round": round_number,
            "reabsorbed_primitive_count": len(accepted),
        })
        if not accepted:
            break

    # Merge disconnected glyph parts conservatively.  Two clusters must be
    # mutual nearest neighbors, both must contain symbol-like geometry, and the
    # merged envelope must remain local.  Plain line clusters are deliberately
    # excluded so nearby conductors cannot merge components.
    cluster_merge_rounds: list[dict[str, int]] = []
    cluster_merge_gap = config.cluster_merge_gap_h * scale
    merge_round_limit = config.maximum_rounds if cluster_merge_gap > 0.0 else 0
    for round_number in range(1, merge_round_limit + 1):
        cluster_members: dict[int, list[int]] = {}
        for index in np.flatnonzero(active):
            cluster_members.setdefault(union_find.find(int(index)), []).append(int(index))
        roots = sorted(cluster_members)
        if len(roots) < 2:
            break
        primitive_root = {
            index: root for root, indices in cluster_members.items() for index in indices
        }
        root_symbol_like = {
            root: any(
                primitives[index].closed
                or primitives[index].kind in {"circle", "arc", "ellipse", "spline"}
                or float(base_features[index, 9]) >= 0.5
                for index in indices
            )
            for root, indices in cluster_members.items()
        }
        distances: dict[tuple[int, int], float] = {}
        for left, right in candidate_pairs(
            primitives, np.flatnonzero(active), cluster_merge_gap
        ):
            left_root, right_root = primitive_root[left], primitive_root[right]
            if left_root == right_root:
                continue
            pair = (min(left_root, right_root), max(left_root, right_root))
            distance = primitive_distance(primitives[left], primitives[right])
            if distance <= cluster_merge_gap:
                distances[pair] = min(distances.get(pair, float("inf")), distance)
        nearest: dict[int, tuple[float, int]] = {}
        for (left_root, right_root), distance in distances.items():
            for owner, other in ((left_root, right_root), (right_root, left_root)):
                if owner not in nearest or (distance, other) < nearest[owner]:
                    nearest[owner] = (distance, other)
        accepted_pairs = []
        consumed: set[int] = set()
        for root in roots:
            if root in consumed or root not in nearest:
                continue
            _, other = nearest[root]
            if other in consumed or nearest.get(other, (None, None))[1] != root:
                continue
            if not (root_symbol_like[root] and root_symbol_like[other]):
                continue
            combined = cluster_members[root] + cluster_members[other]
            box = _cluster_bbox(primitives, combined)
            span_h = max(box[2] - box[0], box[3] - box[1]) / max(scale, 1e-9)
            if span_h > config.maximum_merged_boundary_span_h:
                continue
            accepted_pairs.append((root, other))
            consumed.update((root, other))
        for left_root, right_root in accepted_pairs:
            union_find.union(left_root, right_root)
        cluster_merge_rounds.append({
            "round": round_number,
            "merged_cluster_pair_count": len(accepted_pairs),
        })
        if not accepted_pairs:
            break

    assignment = np.full(count, -1, dtype=np.int32)
    root_to_cluster: dict[int, int] = {}
    for index in np.flatnonzero(active):
        root = union_find.find(int(index))
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        assignment[index] = root_to_cluster[root]
    audit = {
        "method": "soft_stage2_seed_iterative_reabsorption",
        "config": asdict(config),
        "component_threshold": float(component_threshold),
        "seed_probability_threshold": float(seed_threshold),
        "candidate_probability_threshold": float(minimum_candidate_probability),
        "same_component_join_threshold": float(join_threshold),
        "initial_seed_count": int(np.sum(membership_source == "seed")),
        "reabsorbed_primitive_count": int(np.sum(membership_source == "reabsorbed")),
        "contact_reabsorbed_primitive_count": contact_reabsorbed,
        "region_reabsorbed_primitive_count": region_reabsorbed,
        "hard_carrier_count": int(np.sum(hard_carrier)),
        "closed_source_entity_union_count": source_group_union_count,
        "component_count": len(root_to_cluster),
        "rejected_multi_cluster_candidate_count": rejected_multi_cluster,
        "rejected_span_candidate_count": rejected_span,
        "rounds": rounds,
        "cluster_merge_rounds": cluster_merge_rounds,
        "merged_cluster_pair_count": sum(
            row["merged_cluster_pair_count"] for row in cluster_merge_rounds
        ),
        "membership_source": membership_source.tolist(),
        "hard_carrier_mask": hard_carrier.tolist(),
    }
    return assignment, audit


__all__ = ["IterativeBoundaryConfig", "iterative_component_boundaries"]
