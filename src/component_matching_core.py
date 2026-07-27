"""Reusable candidate grouping and component-template scoring.

This module contains only the geometry needed by the production recognizer.
It intentionally has no drawing-specific constants, truth readers, reports,
or command-line experiment entry points.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from multiscale_template_matching import fast_chamfer, prepare_templates
from template_geometry import (
    count_similarity,
    normalize_points,
    sample_circle,
    sample_line,
    sample_polyline,
)


Point = tuple[float, float]
Box = tuple[float, float, float, float]
STUB_RATIOS = (0.75, 1.0, 1.125, 1.25, 1.5)


@dataclass
class Shape:
    handle: str
    entity: Any
    entity_type: str
    layer: str
    kind: str
    bbox: Box
    points: np.ndarray
    length: float
    closed: bool


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


def distance(left: Point, right: Point) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def bbox_distance(left: Box, right: Box) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def merge_bbox(boxes: Iterable[Box]) -> Box:
    materialized = list(boxes)
    return (
        min(box[0] for box in materialized),
        min(box[1] for box in materialized),
        max(box[2] for box in materialized),
        max(box[3] for box in materialized),
    )


def bbox_center(box: Box) -> Point:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def shape_from_entity(entity: Any) -> Shape | None:
    entity_type = entity.dxftype()
    handle = entity.dxf.handle or ""
    layer = entity.dxf.layer or ""
    if entity_type == "LINE":
        start = (float(entity.dxf.start.x), float(entity.dxf.start.y))
        end = (float(entity.dxf.end.x), float(entity.dxf.end.y))
        length = distance(start, end)
        if length <= 1e-9:
            return None
        return Shape(
            handle,
            entity,
            entity_type,
            layer,
            "line",
            (
                min(start[0], end[0]),
                min(start[1], end[1]),
                max(start[0], end[0]),
                max(start[1], end[1]),
            ),
            sample_line(start, end),
            length,
            False,
        )
    if entity_type == "CIRCLE":
        center = (
            float(entity.dxf.center.x),
            float(entity.dxf.center.y),
        )
        radius = float(entity.dxf.radius)
        return Shape(
            handle,
            entity,
            entity_type,
            layer,
            "circle",
            (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ),
            sample_circle(center, radius),
            2.0 * math.pi * radius,
            True,
        )
    if entity_type == "LWPOLYLINE":
        raw = [
            (float(x), float(y))
            for x, y in entity.get_points("xy")
        ]
        if len(raw) < 2:
            return None
        closed = bool(entity.closed)
        points = sample_polyline(raw, closed)
        if len(points) == 0:
            return None
        pairs = list(zip(raw, raw[1:]))
        if closed:
            pairs.append((raw[-1], raw[0]))
        xs = [point[0] for point in raw]
        ys = [point[1] for point in raw]
        return Shape(
            handle,
            entity,
            entity_type,
            layer,
            "polygon" if closed else "polyline",
            (min(xs), min(ys), max(xs), max(ys)),
            points,
            sum(distance(left, right) for left, right in pairs),
            closed,
        )
    return None


def cluster_shapes(
    shapes: list[Shape],
    gap: float,
) -> list[list[Shape]]:
    union = UnionFind(len(shapes))
    for left in range(len(shapes)):
        for right in range(left + 1, len(shapes)):
            if bbox_distance(shapes[left].bbox, shapes[right].bbox) <= gap:
                union.union(left, right)
    groups: dict[int, list[Shape]] = defaultdict(list)
    for index, shape in enumerate(shapes):
        groups[union.find(index)].append(shape)
    return sorted(
        groups.values(),
        key=lambda group: (
            min(shape.bbox[0] for shape in group),
            min(shape.bbox[1] for shape in group),
        ),
    )


def clipped_line_from_circle(
    line: Shape,
    circles: list[Shape],
    length: float,
) -> np.ndarray | None:
    start = (
        float(line.entity.dxf.start.x),
        float(line.entity.dxf.start.y),
    )
    end = (
        float(line.entity.dxf.end.x),
        float(line.entity.dxf.end.y),
    )
    ranked: list[tuple[float, Point, Point]] = []
    for near, far in ((start, end), (end, start)):
        gap = min(
            abs(
                distance(
                    near,
                    (
                        float(circle.entity.dxf.center.x),
                        float(circle.entity.dxf.center.y),
                    ),
                )
                - float(circle.entity.dxf.radius)
            )
            for circle in circles
        )
        ranked.append((gap, near, far))
    gap, near, far = min(ranked, key=lambda item: item[0])
    radius = statistics.mean(
        float(circle.entity.dxf.radius) for circle in circles
    )
    if gap > radius * 0.20:
        return None
    dx = far[0] - near[0]
    dy = far[1] - near[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return None
    clipped_end = (
        near[0] + dx / norm * min(length, norm),
        near[1] + dy / norm * min(length, norm),
    )
    return sample_line(near, clipped_end)


def clipped_line_from_shape(
    line: Shape,
    target: Shape,
    length: float,
) -> np.ndarray | None:
    start = (
        float(line.entity.dxf.start.x),
        float(line.entity.dxf.start.y),
    )
    end = (
        float(line.entity.dxf.end.x),
        float(line.entity.dxf.end.y),
    )
    ranked = []
    for near, far in ((start, end), (end, start)):
        gap = float(
            np.linalg.norm(
                target.points - np.asarray(near),
                axis=1,
            ).min()
        )
        ranked.append((gap, near, far))
    gap, near, far = min(ranked, key=lambda item: item[0])
    scale = max(
        target.bbox[2] - target.bbox[0],
        target.bbox[3] - target.bbox[1],
    )
    if gap > scale * 0.20:
        return None
    dx = far[0] - near[0]
    dy = far[1] - near[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return None
    clipped_end = (
        near[0] + dx / norm * min(length, norm),
        near[1] + dy / norm * min(length, norm),
    )
    return sample_line(near, clipped_end)


def add_candidate(
    output: list[dict[str, Any]],
    seen: set[str],
    group_id: str,
    mode: str,
    shapes: list[Shape],
    extra_arrays: list[np.ndarray] | None = None,
    extra_handles: list[str] | None = None,
) -> None:
    extra_arrays = extra_arrays or []
    extra_handles = extra_handles or []
    if not shapes and not extra_arrays:
        return
    counts = Counter(shape.kind for shape in shapes)
    if extra_arrays:
        counts["line"] += len(extra_arrays)
    handles = sorted(
        {shape.handle for shape in shapes} | set(extra_handles)
    )
    signature = json.dumps(
        {
            "handles": handles,
            "mode": mode,
            "counts": counts,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if signature in seen:
        return
    arrays = [shape.points for shape in shapes] + extra_arrays
    seen.add(signature)
    output.append(
        {
            "candidate_id": f"{group_id}_C{len(output) + 1:03d}",
            "group_id": group_id,
            "mode": mode,
            "source_handles": handles,
            "owned_handles": sorted(
                {shape.handle for shape in shapes}
            ),
            "primitive_counts": dict(counts),
            "points": normalize_points(np.vstack(arrays)),
            "bbox": list(
                merge_bbox(shape.bbox for shape in shapes)
            )
            if shapes
            else [],
        }
    )


def connected_subset(shapes: list[Shape], gap: float) -> bool:
    if len(shapes) <= 1:
        return True
    visited = {0}
    changed = True
    while changed:
        changed = False
        for left in list(visited):
            for right in range(len(shapes)):
                if right in visited:
                    continue
                if (
                    bbox_distance(
                        shapes[left].bbox,
                        shapes[right].bbox,
                    )
                    <= gap
                ):
                    visited.add(right)
                    changed = True
    return len(visited) == len(shapes)


def generate_group_candidates(
    group_id: str,
    group: list[Shape],
    all_lines: list[Shape],
    typical_text_height: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    add_candidate(output, seen, group_id, "complete_group", group)

    if len(group) <= 6:
        for mask in range(1, 1 << len(group)):
            subset = [
                shape
                for index, shape in enumerate(group)
                if mask & (1 << index)
            ]
            if len(subset) == len(group):
                continue
            if connected_subset(
                subset,
                typical_text_height * 0.25,
            ):
                add_candidate(
                    output,
                    seen,
                    group_id,
                    f"connected_subset_{len(subset)}",
                    subset,
                )
    else:
        for shape in group:
            add_candidate(
                output,
                seen,
                group_id,
                "single_primitive",
                [shape],
            )

    circles = [shape for shape in group if shape.kind == "circle"]
    for left_index, left in enumerate(circles):
        for right in circles[left_index + 1 :]:
            left_radius = float(left.entity.dxf.radius)
            right_radius = float(right.entity.dxf.radius)
            radius_error = abs(left_radius - right_radius) / max(
                left_radius,
                right_radius,
            )
            center_gap = distance(
                (
                    float(left.entity.dxf.center.x),
                    float(left.entity.dxf.center.y),
                ),
                (
                    float(right.entity.dxf.center.x),
                    float(right.entity.dxf.center.y),
                ),
            )
            mean_radius = (left_radius + right_radius) / 2.0
            if (
                radius_error > 0.15
                or abs(center_gap / mean_radius - 1.0) > 0.35
            ):
                continue
            pair = [left, right]
            add_candidate(
                output,
                seen,
                group_id,
                "equal_radius_circle_pair",
                pair,
            )
            for line in all_lines:
                if line.handle in {left.handle, right.handle}:
                    continue
                for ratio in STUB_RATIOS:
                    stub = clipped_line_from_circle(
                        line,
                        pair,
                        mean_radius * ratio,
                    )
                    if stub is not None:
                        add_candidate(
                            output,
                            seen,
                            group_id,
                            f"circle_pair_with_stub_{ratio:.3f}",
                            pair,
                            [stub],
                            [line.handle],
                        )

    polygons = [
        shape for shape in group if shape.kind == "polygon"
    ]
    for polygon in polygons:
        scale = max(
            polygon.bbox[2] - polygon.bbox[0],
            polygon.bbox[3] - polygon.bbox[1],
        )
        for line in all_lines:
            if line.handle == polygon.handle:
                continue
            for ratio in (0.75, 1.0, 1.25, 1.5):
                stub = clipped_line_from_shape(
                    line,
                    polygon,
                    scale * ratio,
                )
                if stub is not None:
                    add_candidate(
                        output,
                        seen,
                        group_id,
                        f"closed_shape_with_stub_{ratio:.3f}",
                        [polygon],
                        [stub],
                        [line.handle],
                    )
    return output


def classify_text_prior(text: str) -> str:
    compact = text.replace(" ", "")
    if "配变" in compact or "变压器" in compact:
        return "PowerTransformer"
    if compact.upper() == "PT" or "电压互感" in compact:
        return "PT"
    return ""


def nearest_text_prior(
    box: Box,
    texts: list[dict[str, Any]],
    maximum: float,
) -> dict[str, Any] | None:
    center = bbox_center(box)
    ranked = []
    for text in texts:
        family = classify_text_prior(text["text"])
        if not family:
            continue
        physical = distance(center, (text["x"], text["y"]))
        if physical <= maximum:
            ranked.append((physical, text, family))
    if not ranked:
        return None
    physical, text, family = min(
        ranked,
        key=lambda item: item[0],
    )
    return {
        "text": text["text"],
        "handle": text["handle"],
        "family": family,
        "distance": round(physical, 3),
        "bonus": 5.0,
    }


def score_candidate(
    candidate: dict[str, Any],
    templates: list[dict[str, Any]],
    prior: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = []
    for prepared in templates:
        record = prepared["record"]
        if not prepared["supported"]:
            continue
        best_distance = float("inf")
        best_transform = ""
        for transform_name, transformed in prepared["transforms"]:
            current = fast_chamfer(
                candidate["points"],
                transformed,
            )
            if current < best_distance:
                best_distance = current
                best_transform = transform_name
        geometry = math.exp(-8.0 * best_distance)
        primitive = count_similarity(
            candidate["primitive_counts"],
            record.get("primitive_counts", {}),
        )
        base = 100.0 * (0.85 * geometry + 0.15 * primitive)
        bonus = (
            5.0
            if prior is not None
            and record["family"] == prior["family"]
            else 0.0
        )
        rows.append(
            {
                "template_id": record["symbol_id"],
                "template_name": record["name"],
                "family": record["family"],
                "combined_score": round(
                    min(100.0, base + bonus),
                    2,
                ),
                "base_score": round(base, 2),
                "geometry_score": round(geometry * 100.0, 2),
                "primitive_count_score": round(
                    primitive * 100.0,
                    2,
                ),
                "text_bonus": bonus,
                "best_transform": best_transform,
                "chamfer_distance": round(best_distance, 6),
            }
        )
    rows.sort(
        key=lambda item: (
            -item["combined_score"],
            item["template_id"],
        )
    )
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def select_group_result(
    group: list[Shape],
    candidates: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    group_handles = {shape.handle for shape in group}
    full_group_scores = [
        evaluation["ranked"][0]["combined_score"]
        for evaluation in evaluations
        if set(evaluation["candidate"]["owned_handles"])
        == group_handles
        and "circle_pair" not in evaluation["candidate"]["mode"]
        and "closed_shape_with_stub"
        not in evaluation["candidate"]["mode"]
    ]
    full_group_score = max(full_group_scores, default=0.0)
    ranked = []
    for evaluation in evaluations:
        candidate = evaluation["candidate"]
        top = evaluation["ranked"][0]
        coverage = len(
            set(candidate["owned_handles"]) & group_handles
        ) / max(len(group_handles), 1)
        boundary_motif = (
            "circle_pair" in candidate["mode"]
            or "closed_shape_with_stub" in candidate["mode"]
        )
        if (
            len(group_handles) > 1
            and coverage < 1.0
            and not boundary_motif
            and (
                coverage < 0.5
                or top["combined_score"] < full_group_score + 8.0
            )
        ):
            continue
        single_penalty = (
            5.0
            if len(candidate["owned_handles"]) == 1
            and len(group_handles) > 1
            else 0.0
        )
        selection_score = (
            top["combined_score"]
            + 4.0 * math.sqrt(coverage)
            - single_penalty
        )
        ranked.append(
            (
                selection_score,
                top["combined_score"],
                coverage,
                evaluation,
            )
        )
    if not ranked:
        raise RuntimeError(
            "no component candidate survived coverage filtering"
        )
    _, _, coverage, best = max(
        ranked,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            -len(item[3]["candidate"]["owned_handles"]),
        ),
    )
    family_top: dict[str, dict[str, Any]] = {}
    for item in best["ranked"]:
        family_top.setdefault(item["family"], item)
    alternatives = sorted(
        family_top.values(),
        key=lambda item: -item["combined_score"],
    )[:5]
    top = best["ranked"][0]
    second_family_score = (
        alternatives[1]["combined_score"]
        if len(alternatives) > 1
        else 0.0
    )
    margin = top["combined_score"] - second_family_score
    if top["combined_score"] >= 85.0 and margin >= 5.0:
        confidence = "high"
    elif top["combined_score"] >= 72.0:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "candidate": best["candidate"],
        "top": top,
        "family_alternatives": alternatives,
        "coverage": round(coverage, 3),
        "family_margin": round(margin, 2),
        "confidence": confidence,
        "candidate_count": len(candidates),
    }


__all__ = [
    "Shape",
    "bbox_center",
    "bbox_distance",
    "cluster_shapes",
    "generate_group_candidates",
    "merge_bbox",
    "nearest_text_prior",
    "prepare_templates",
    "score_candidate",
    "select_group_result",
    "shape_from_entity",
]
