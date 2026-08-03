"""Drawing-level conductor-chain and local-symbol logic features."""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

from evaluate_exploded_svg_conductors import Primitive
from evaluate_unknown_component_segmentation import candidate_pairs


LOGIC_FEATURE_NAMES = [
    "chain_total_length_h",
    "chain_span_h",
    "log_chain_segment_count",
    "chain_continuity_ratio",
    "chain_external_contact_count",
    "chain_branch_contact_count",
    "chain_endpoint_to_interior_count",
    "chain_drawing_span_ratio",
    "chain_long_carrier_candidate",
    "local_nonaxis_or_curve_count_0_5h",
    "local_closed_or_circle_count_0_5h",
    "local_primitive_count_1h",
]

# Kept separate from LOGIC_FEATURE_NAMES so that previously saved 30-feature
# models remain loadable.  New models can opt into these relation features.
RELATION_LOGIC_FEATURE_NAMES = [
    "inverse_hops_to_nearest_carrier",
    "direct_carrier_contact",
    "parallel_to_nearest_carrier",
    "perpendicular_to_nearest_carrier",
    "oblique_to_nearest_carrier",
    "carrier_endpoint_to_interior_contact",
    "carrier_interior_crossing",
    "noncarrier_neighbor_count",
    "symbol_like_neighbor_count",
    "short_line_candidate",
    "short_perpendicular_branch_candidate",
    "short_oblique_component_candidate",
]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def axis_kind(primitive: Primitive, tolerance_degrees: float = 5.0) -> int:
    """Return 1 horizontal, 2 vertical, or 0 non-axis/non-line."""
    if primitive.kind != "line" or primitive.start is None:
        return 0
    dx = abs(primitive.end[0] - primitive.start[0])
    dy = abs(primitive.end[1] - primitive.start[1])
    tangent = math.tan(math.radians(tolerance_degrees))
    if dy <= tangent * max(dx, 1e-9):
        return 1
    if dx <= tangent * max(dy, 1e-9):
        return 2
    return 0


def _interval_and_offset(primitive: Primitive, axis: int):
    if axis == 1:
        return (
            min(primitive.start[0], primitive.end[0]),
            max(primitive.start[0], primitive.end[0]),
            (primitive.start[1] + primitive.end[1]) / 2.0,
        )
    return (
        min(primitive.start[1], primitive.end[1]),
        max(primitive.start[1], primitive.end[1]),
        (primitive.start[0] + primitive.end[0]) / 2.0,
    )


def collinear_chains(
    primitives: list[Primitive],
    scale: float,
    maximum_gap_h: float = 0.15,
    offset_tolerance_h: float = 0.03,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Group nearly collinear axis-aligned segments without semantic input."""
    axes = np.asarray([axis_kind(item) for item in primitives], dtype=np.int8)
    union = UnionFind(len(primitives))
    maximum_gap = maximum_gap_h * scale
    offset_tolerance = offset_tolerance_h * scale
    line_indices = [index for index, axis in enumerate(axes) if axis]
    for left, right in candidate_pairs(primitives, line_indices, maximum_gap):
        axis = int(axes[left])
        if axis == 0 or int(axes[right]) != axis:
            continue
        left_start, left_end, left_offset = _interval_and_offset(primitives[left], axis)
        right_start, right_end, right_offset = _interval_and_offset(primitives[right], axis)
        interval_gap = max(left_start - right_end, right_start - left_end, 0.0)
        if (
            abs(left_offset - right_offset) <= offset_tolerance
            and interval_gap <= maximum_gap
        ):
            union.union(left, right)
    members: dict[int, list[int]] = defaultdict(list)
    chain_id = np.full(len(primitives), -1, dtype=np.int32)
    for index in line_indices:
        members[union.find(index)].append(index)
    normalized = {}
    for number, (_, indices) in enumerate(sorted(members.items()), 0):
        normalized[number] = indices
        chain_id[indices] = number
    return chain_id, normalized


def logical_feature_rows(
    primitives: list[Primitive],
    scale: float,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> np.ndarray:
    """Return per-primitive chain/global logic without labels or IDs."""
    count = len(primitives)
    if not count:
        return np.empty((0, len(LOGIC_FEATURE_NAMES)), dtype=np.float32)
    chain_id, chains = collinear_chains(primitives, scale)
    drawing_left = min(item.bbox[0] for item in primitives)
    drawing_bottom = min(item.bbox[1] for item in primitives)
    drawing_right = max(item.bbox[2] for item in primitives)
    drawing_top = max(item.bbox[3] for item in primitives)
    drawing_span = max(drawing_right - drawing_left, drawing_top - drawing_bottom, scale)

    external: dict[int, set[int]] = defaultdict(set)
    branches: dict[int, set[int]] = defaultdict(set)
    endpoint_to_interior: dict[int, set[int]] = defaultdict(set)
    # contact_graph is directed; process each unordered pair once.
    observed_pairs = set()
    for edge_number, (left, right) in enumerate(edge_index.T):
        left, right = int(left), int(right)
        pair = (min(left, right), max(left, right))
        if pair in observed_pairs:
            continue
        observed_pairs.add(pair)
        left_chain, right_chain = int(chain_id[left]), int(chain_id[right])
        for owner, other, owner_chain, other_chain in (
            (left, right, left_chain, right_chain),
            (right, left, right_chain, left_chain),
        ):
            if owner_chain < 0 or owner_chain == other_chain:
                continue
            external[owner_chain].add(other)
            row = edge_attr[edge_number]
            if float(row[5]) >= 0.70 or float(row[2]) >= 0.5 or float(row[3]) >= 0.5:
                branches[owner_chain].add(other)
            if float(row[2]) >= 0.5:
                endpoint_to_interior[owner_chain].add(other)

    chain_values: dict[int, tuple[float, ...]] = {}
    for identifier, indices in chains.items():
        axis = axis_kind(primitives[indices[0]])
        intervals = [_interval_and_offset(primitives[index], axis) for index in indices]
        start = min(item[0] for item in intervals)
        end = max(item[1] for item in intervals)
        span = max(end - start, 1e-9)
        total_length = sum(primitives[index].length for index in indices)
        continuity = min(total_length / span, 2.0) / 2.0
        external_count = len(external[identifier])
        branch_count = len(branches[identifier])
        endpoint_interior_count = len(endpoint_to_interior[identifier])
        span_h = span / max(scale, 1e-9)
        drawing_ratio = span / drawing_span
        carrier = float(
            span_h >= 1.5
            and (
                branch_count >= 1
                or len(indices) >= 2
                or drawing_ratio >= 0.15
            )
        )
        chain_values[identifier] = (
            total_length / max(scale, 1e-9),
            span_h,
            math.log1p(len(indices)),
            continuity,
            min(external_count, 12) / 12.0,
            min(branch_count, 8) / 8.0,
            min(endpoint_interior_count, 8) / 8.0,
            min(drawing_ratio, 1.0),
            carrier,
        )

    local_nonaxis = np.zeros(count, dtype=float)
    local_closed_circle = np.zeros(count, dtype=float)
    local_count = np.zeros(count, dtype=float)
    for left, right in candidate_pairs(primitives, range(count), 1.0 * scale):
        dx = primitives[left].center[0] - primitives[right].center[0]
        dy = primitives[left].center[1] - primitives[right].center[1]
        center_distance = math.hypot(dx, dy)
        if center_distance <= 1.0 * scale:
            local_count[left] += 1
            local_count[right] += 1
        if center_distance <= 0.5 * scale:
            for owner, other in ((left, right), (right, left)):
                item = primitives[other]
                if axis_kind(item) == 0:
                    local_nonaxis[owner] += 1
                if item.closed or item.kind == "circle":
                    local_closed_circle[owner] += 1

    rows = np.zeros((count, len(LOGIC_FEATURE_NAMES)), dtype=np.float32)
    for index in range(count):
        chain = int(chain_id[index])
        if chain >= 0:
            rows[index, :9] = chain_values[chain]
        rows[index, 9:] = (
            min(local_nonaxis[index], 12) / 12.0,
            min(local_closed_circle[index], 8) / 8.0,
            min(local_count[index], 24) / 24.0,
        )
    return rows


def _line_relation(left: Primitive, right: Primitive) -> tuple[float, float, float]:
    """Return parallel, perpendicular and oblique strengths in [0, 1]."""
    if (
        left.kind != "line" or right.kind != "line"
        or left.start is None or right.start is None
    ):
        return 0.0, 0.0, 0.0
    left_dx = left.end[0] - left.start[0]
    left_dy = left.end[1] - left.start[1]
    right_dx = right.end[0] - right.start[0]
    right_dy = right.end[1] - right.start[1]
    left_length = math.hypot(left_dx, left_dy)
    right_length = math.hypot(right_dx, right_dy)
    if left_length <= 1e-12 or right_length <= 1e-12:
        return 0.0, 0.0, 0.0
    parallel = abs(
        left_dx * right_dx + left_dy * right_dy
    ) / (left_length * right_length)
    parallel = min(max(parallel, 0.0), 1.0)
    perpendicular = math.sqrt(max(0.0, 1.0 - parallel * parallel))
    # Both parallel and perpendicular are weak near 45 degrees.  Normalize so
    # that 0/90 degrees -> 0 and 45 degrees -> 1.
    oblique = max(0.0, (1.0 - max(parallel, perpendicular)) / (1.0 - 2 ** -0.5))
    return parallel, perpendicular, min(oblique, 1.0)


def relation_logical_feature_rows(
    primitives: list[Primitive],
    scale: float,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    base_logic: np.ndarray | None = None,
) -> np.ndarray:
    """Describe each primitive relative to nearby long conductor carriers.

    This encodes the drawing convention as evidence rather than a hard rule:
    a short perpendicular T branch is wire-like, while an oblique short stroke
    leading from a carrier toward symbol geometry is component-like.
    """
    count = len(primitives)
    rows = np.zeros((count, len(RELATION_LOGIC_FEATURE_NAMES)), dtype=np.float32)
    if not count:
        return rows

    if base_logic is None:
        base_logic = logical_feature_rows(primitives, scale, edge_index, edge_attr)
    carrier_column = LOGIC_FEATURE_NAMES.index("chain_long_carrier_candidate")
    carriers = {
        index for index in range(count)
        if float(base_logic[index, carrier_column]) >= 0.5
    }

    adjacency: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    observed_pairs: set[tuple[int, int]] = set()
    for edge_number, (left_value, right_value) in enumerate(edge_index.T):
        left, right = int(left_value), int(right_value)
        pair = (min(left, right), max(left, right))
        if pair in observed_pairs:
            continue
        observed_pairs.add(pair)
        relation = edge_attr[edge_number]
        adjacency[left].append((right, relation))
        adjacency[right].append((left, relation))

    nearest_carrier = np.full(count, -1, dtype=np.int32)
    hops = np.full(count, 99, dtype=np.int16)
    queue: deque[int] = deque()
    for carrier in carriers:
        nearest_carrier[carrier] = carrier
        hops[carrier] = 0
        queue.append(carrier)
    while queue:
        current = queue.popleft()
        if hops[current] >= 3:
            continue
        for neighbor, _ in adjacency.get(current, []):
            if hops[neighbor] > hops[current] + 1:
                hops[neighbor] = hops[current] + 1
                nearest_carrier[neighbor] = nearest_carrier[current]
                queue.append(neighbor)

    for index, primitive in enumerate(primitives):
        direct_relations = [
            (other, relation)
            for other, relation in adjacency.get(index, [])
            if other in carriers and other != index
        ]
        carrier = int(nearest_carrier[index])
        if direct_relations:
            # Prefer the strongest T/perpendicular carrier contact.
            carrier, carrier_edge = max(
                direct_relations,
                key=lambda item: float(item[1][2]) + float(item[1][5]),
            )
        else:
            carrier_edge = None
        if carrier >= 0 and carrier != index:
            parallel, perpendicular, oblique = _line_relation(
                primitive, primitives[carrier]
            )
        else:
            parallel = perpendicular = oblique = 0.0

        noncarrier_neighbors = [
            other for other, _ in adjacency.get(index, []) if other not in carriers
        ]
        symbol_like = sum(
            1 for other in noncarrier_neighbors
            if (
                primitives[other].closed
                or primitives[other].kind == "circle"
                or axis_kind(primitives[other]) == 0
            )
        )
        short = float(primitive.kind == "line" and primitive.length <= 1.5 * scale)
        direct = float(bool(direct_relations))
        endpoint_to_interior = float(carrier_edge[2]) if carrier_edge is not None else 0.0
        interior_crossing = float(carrier_edge[3]) if carrier_edge is not None else 0.0
        perpendicular_branch = float(
            short and direct and perpendicular >= 0.95
            and (endpoint_to_interior >= 0.5 or interior_crossing >= 0.5)
        )
        oblique_component = float(
            short and carrier >= 0 and hops[index] <= 2
            and oblique >= 0.35 and (symbol_like > 0 or len(noncarrier_neighbors) > 0)
        )
        rows[index] = (
            1.0 / (1.0 + float(hops[index])) if hops[index] <= 3 else 0.0,
            direct,
            parallel,
            perpendicular,
            oblique,
            endpoint_to_interior,
            interior_crossing,
            min(len(noncarrier_neighbors), 8) / 8.0,
            min(symbol_like, 6) / 6.0,
            short,
            perpendicular_branch,
            oblique_component,
        )
    return rows
