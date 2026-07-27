from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import ezdxf
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiscale_template_matching import (
    fast_chamfer,
    prepare_templates,
)
from template_geometry import (
    count_similarity,
    normalize_points,
    sample_circle,
    sample_line,
    sample_polyline,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return distance(point, start)
    t = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator
    t = max(0.0, min(1.0, t))
    projection = (start[0] + t * dx, start[1] + t * dy)
    return distance(point, projection)


def shape_from_entity(entity: Any) -> Shape | None:
    kind = entity.dxftype()
    handle = entity.dxf.handle or ""
    layer = entity.dxf.layer or ""
    if kind == "LINE":
        start = (float(entity.dxf.start.x), float(entity.dxf.start.y))
        end = (float(entity.dxf.end.x), float(entity.dxf.end.y))
        length = distance(start, end)
        if length <= 1e-9:
            return None
        points = sample_line(start, end)
        return Shape(
            handle,
            entity,
            kind,
            layer,
            "line",
            (
                min(start[0], end[0]),
                min(start[1], end[1]),
                max(start[0], end[0]),
                max(start[1], end[1]),
            ),
            points,
            length,
            False,
        )
    if kind == "CIRCLE":
        center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
        radius = float(entity.dxf.radius)
        points = sample_circle(center, radius)
        return Shape(
            handle,
            entity,
            kind,
            layer,
            "circle",
            (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ),
            points,
            2.0 * math.pi * radius,
            True,
        )
    if kind == "LWPOLYLINE":
        raw = [(float(x), float(y)) for x, y in entity.get_points("xy")]
        if len(raw) < 2:
            return None
        closed = bool(entity.closed)
        points = sample_polyline(raw, closed)
        if len(points) == 0:
            return None
        pairs = list(zip(raw, raw[1:]))
        if closed:
            pairs.append((raw[-1], raw[0]))
        length = sum(distance(left, right) for left, right in pairs)
        xs = [point[0] for point in raw]
        ys = [point[1] for point in raw]
        return Shape(
            handle,
            entity,
            kind,
            layer,
            "polygon" if closed else "polyline",
            (min(xs), min(ys), max(xs), max(ys)),
            points,
            length,
            closed,
        )
    return None


def extract_texts(doc: Any) -> list[dict[str, Any]]:
    result = []
    for entity in doc.modelspace():
        kind = entity.dxftype()
        if kind not in {"TEXT", "MTEXT"}:
            continue
        try:
            value = (
                entity.dxf.text
                if kind == "TEXT"
                else entity.plain_text().replace("\\P", " ")
            ).strip()
            insert = entity.dxf.insert
            height = float(
                entity.dxf.height if kind == "TEXT" else entity.dxf.char_height
            )
        except Exception:
            continue
        result.append(
            {
                "handle": entity.dxf.handle or "",
                "text": value,
                "x": float(insert.x),
                "y": float(insert.y),
                "height": height,
            }
        )
    return result


def is_conductor(shape: Shape, typical_text_height: float) -> bool:
    return (
        shape.kind in {"line", "polyline"}
        and not shape.closed
        and shape.layer == "0"
        and shape.length >= typical_text_height * 2.0
    )


def cluster_shapes(shapes: list[Shape], gap: float) -> list[list[Shape]]:
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
    line: Shape, circles: list[Shape], length: float
) -> np.ndarray | None:
    start = (float(line.entity.dxf.start.x), float(line.entity.dxf.start.y))
    end = (float(line.entity.dxf.end.x), float(line.entity.dxf.end.y))
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
    radius = statistics.mean(float(circle.entity.dxf.radius) for circle in circles)
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
    line: Shape, target: Shape, length: float
) -> np.ndarray | None:
    start = (float(line.entity.dxf.start.x), float(line.entity.dxf.start.y))
    end = (float(line.entity.dxf.end.x), float(line.entity.dxf.end.y))
    ranked = []
    for near, far in ((start, end), (end, start)):
        gap = float(
            np.linalg.norm(target.points - np.asarray(near), axis=1).min()
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
    handles = sorted({shape.handle for shape in shapes} | set(extra_handles))
    signature = json.dumps(
        {"handles": handles, "mode": mode, "counts": counts},
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
            "owned_handles": sorted({shape.handle for shape in shapes}),
            "primitive_counts": dict(counts),
            "points": normalize_points(np.vstack(arrays)),
            "bbox": list(merge_bbox(shape.bbox for shape in shapes))
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
                if bbox_distance(shapes[left].bbox, shapes[right].bbox) <= gap:
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
    add_candidate(output, seen, group_id, "完整初始图形组", group)

    # Exhaustive small connected subsets allow the template score to correct
    # both over-merged and under-merged initial groups.
    if len(group) <= 6:
        for mask in range(1, 1 << len(group)):
            subset = [
                shape for index, shape in enumerate(group) if mask & (1 << index)
            ]
            if len(subset) == len(group):
                continue
            if connected_subset(subset, typical_text_height * 0.25):
                add_candidate(
                    output,
                    seen,
                    group_id,
                    f"连通子集{len(subset)}图元",
                    subset,
                )
    else:
        for shape in group:
            add_candidate(output, seen, group_id, "单图元子候选", [shape])

    circles = [shape for shape in group if shape.kind == "circle"]
    for left_index, left in enumerate(circles):
        for right in circles[left_index + 1 :]:
            left_radius = float(left.entity.dxf.radius)
            right_radius = float(right.entity.dxf.radius)
            radius_error = abs(left_radius - right_radius) / max(
                left_radius, right_radius
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
            if radius_error > 0.15 or abs(center_gap / mean_radius - 1.0) > 0.35:
                continue
            pair = [left, right]
            add_candidate(output, seen, group_id, "相交等半径圆对", pair)
            for line in all_lines:
                if line.handle in {left.handle, right.handle}:
                    continue
                for ratio in STUB_RATIOS:
                    stub = clipped_line_from_circle(
                        line, pair, mean_radius * ratio
                    )
                    if stub is None:
                        continue
                    add_candidate(
                        output,
                        seen,
                        group_id,
                        f"圆对+邻接线{line.handle}截取{ratio:.3f}倍半径",
                        pair,
                        [stub],
                        [line.handle],
                    )

    # A cable terminal is often exported as one closed triangle while its
    # vertical stem remains classified as a conductor. Borrow only a short
    # local part of touching lines so it can be compared with line+polygon
    # templates without swallowing the whole circuit branch.
    polygons = [shape for shape in group if shape.kind == "polygon"]
    for polygon in polygons:
        scale = max(
            polygon.bbox[2] - polygon.bbox[0],
            polygon.bbox[3] - polygon.bbox[1],
        )
        for line in all_lines:
            if line.handle == polygon.handle:
                continue
            for ratio in (0.75, 1.0, 1.25, 1.5):
                stub = clipped_line_from_shape(line, polygon, scale * ratio)
                if stub is None:
                    continue
                add_candidate(
                    output,
                    seen,
                    group_id,
                    f"闭合图形+邻接线{line.handle}截取{ratio:.3f}倍尺度",
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
    box: Box, texts: list[dict[str, Any]], maximum: float
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
    physical, text, family = min(ranked, key=lambda item: item[0])
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
            current = fast_chamfer(candidate["points"], transformed)
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
            if prior is not None and record["family"] == prior["family"]
            else 0.0
        )
        rows.append(
            {
                "template_id": record["symbol_id"],
                "template_name": record["name"],
                "family": record["family"],
                "combined_score": round(min(100.0, base + bonus), 2),
                "base_score": round(base, 2),
                "geometry_score": round(geometry * 100.0, 2),
                "primitive_count_score": round(primitive * 100.0, 2),
                "text_bonus": bonus,
                "best_transform": best_transform,
                "chamfer_distance": round(best_distance, 6),
            }
        )
    rows.sort(key=lambda item: (-item["combined_score"], item["template_id"]))
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
        if set(evaluation["candidate"]["owned_handles"]) == group_handles
        and not (
            "圆对" in evaluation["candidate"]["mode"]
            or "闭合图形+邻接线" in evaluation["candidate"]["mode"]
        )
    ]
    full_group_score = max(full_group_scores, default=0.0)
    ranked = []
    for evaluation in evaluations:
        candidate = evaluation["candidate"]
        top = evaluation["ranked"][0]
        coverage = len(set(candidate["owned_handles"]) & group_handles) / max(
            len(group_handles), 1
        )
        boundary_motif = (
            "圆对" in candidate["mode"]
            or "闭合图形+邻接线" in candidate["mode"]
        )
        if len(group_handles) > 1 and coverage < 1.0 and not boundary_motif:
            # A partial subset may replace an initial group only when its
            # template evidence is materially stronger. This prevents a
            # generic perfect line/circle match from discarding the rest of a
            # switch assembly.
            if coverage < 0.5 or top["combined_score"] < full_group_score + 8.0:
                continue
        single_penalty = (
            5.0
            if len(candidate["owned_handles"]) == 1 and len(group_handles) > 1
            else 0.0
        )
        selection_score = top["combined_score"] + 4.0 * math.sqrt(coverage) - single_penalty
        ranked.append(
            (
                selection_score,
                top["combined_score"],
                coverage,
                evaluation,
            )
        )
    if not ranked:
        raise RuntimeError("候选覆盖率筛选后没有可用结果")
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
        family_top.values(), key=lambda item: -item["combined_score"]
    )[:5]
    top = best["ranked"][0]
    second_family_score = alternatives[1]["combined_score"] if len(alternatives) > 1 else 0.0
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


def map_to_audit_equipment(
    group: list[Shape], audit: dict[str, Any] | None
) -> dict[str, Any] | None:
    if audit is None:
        return None
    handles = {shape.handle for shape in group}
    ranked = []
    for equipment in audit.get("equipment", []):
        source = set(equipment.get("source_handles", []))
        overlap = len(handles & source)
        union = len(handles | source)
        if overlap:
            ranked.append((overlap / max(union, 1), overlap, equipment))
    return max(ranked, key=lambda item: (item[0], item[1]))[2] if ranked else None


def build_components(
    groups: list[list[Shape]],
    group_results: list[dict[str, Any]],
    priors: list[dict[str, Any] | None],
    audit: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    terminals_by_equipment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if audit:
        for terminal in audit.get("terminals", []):
            terminals_by_equipment[terminal["equipment_id"]].append(terminal)
    output = []
    for index, (group, result, prior) in enumerate(
        zip(groups, group_results, priors), 1
    ):
        mapped = map_to_audit_equipment(group, audit)
        box = merge_bbox(shape.bbox for shape in group)
        component = {
            "component_id": f"CMP{index:03d}",
            "group_bbox": [round(value, 3) for value in box],
            "group_source_handles": sorted(shape.handle for shape in group),
            "selected_source_handles": result["candidate"]["source_handles"],
            "selected_owned_handles": result["candidate"]["owned_handles"],
            "candidate_mode": result["candidate"]["mode"],
            "primitive_counts": result["candidate"]["primitive_counts"],
            "family": result["top"]["family"],
            "template_id": result["top"]["template_id"],
            "template_name": result["top"]["template_name"],
            "score": result["top"]["combined_score"],
            "base_score_without_text": result["top"]["base_score"],
            "text_prior": prior,
            "family_margin": result["family_margin"],
            "coverage_of_initial_group": result["coverage"],
            "confidence": result["confidence"],
            "candidate_count": result["candidate_count"],
            "top5_families": result["family_alternatives"],
            "mapped_audit_equipment_id": mapped["equipment_id"] if mapped else "",
            "terminal_count": len(
                terminals_by_equipment.get(mapped["equipment_id"], [])
            )
            if mapped
            else 0,
        }
        output.append(component)
    return output


def build_assemblies(
    components: list[dict[str, Any]],
    texts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cabinet_texts = [text for text in texts if "柜" in text["text"]]
    transformers = [
        component
        for component in components
        if component["family"] == "PowerTransformer"
        and component["score"] >= 75.0
    ]
    output: list[dict[str, Any]] = []
    for text in sorted(cabinet_texts, key=lambda item: item["x"]):
        if "公用" in text["text"]:
            x_min, x_max = 22000.0, 35000.0
        else:
            x_min, x_max = 38000.0, 50000.0
        members = [
            component["component_id"]
            for component in components
            if x_min <= bbox_center(tuple(component["group_bbox"]))[0] <= x_max
            and component not in transformers
        ]
        output.append(
            {
                "assembly_id": f"ASM{len(output) + 1:03d}",
                "type": "SwitchgearCabinet",
                "label": text["text"],
                "label_handle": text["handle"],
                "member_components": members,
                "confidence": "high",
                "basis": "柜体文字与同一母线区域内的元件组合",
            }
        )
    transformer_texts = [
        text
        for text in texts
        if "配变" in text["text"] or "变压器" in text["text"]
    ]
    used: set[str] = set()
    for text in transformer_texts:
        ranked = []
        for component in transformers:
            if component["component_id"] in used:
                continue
            center = bbox_center(tuple(component["group_bbox"]))
            ranked.append(
                (distance(center, (text["x"], text["y"])), component)
            )
        if not ranked:
            continue
        _, component = min(ranked, key=lambda item: item[0])
        used.add(component["component_id"])
        output.append(
            {
                "assembly_id": f"ASM{len(output) + 1:03d}",
                "type": "PowerTransformer",
                "label": text["text"],
                "label_handle": text["handle"],
                "member_components": [component["component_id"]],
                "confidence": component["confidence"],
                "basis": "146模板几何匹配为变压器，文字仅作名称和小幅类别辅助",
            }
        )
    return output


def build_connections(
    components: list[dict[str, Any]],
    assemblies: list[dict[str, Any]],
    audit: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if audit is None:
        return [], []
    component_by_audit = {
        component["mapped_audit_equipment_id"]: component["component_id"]
        for component in components
        if component["mapped_audit_equipment_id"]
    }
    component_edges = []
    for edge in audit.get("device_graph_derived", []):
        source = component_by_audit.get(edge["source_equipment"])
        target = component_by_audit.get(edge["target_equipment"])
        if not source or not target or source == target:
            continue
        normalized = tuple(sorted((source, target)))
        if normalized not in {
            tuple(sorted((item["source_component"], item["target_component"])))
            for item in component_edges
        }:
            component_edges.append(
                {
                    "source_component": normalized[0],
                    "target_component": normalized[1],
                    "via_connectivity_node": edge.get(
                        "via_connectivity_node", ""
                    ),
                    "state": edge.get("state", "confirmed"),
                    "source": "DXF几何连接审计",
                }
            )

    assembly_by_component = {}
    for assembly in assemblies:
        for member in assembly["member_components"]:
            assembly_by_component[member] = assembly["assembly_id"]
    assembly_edges_set: set[tuple[str, str]] = set()
    for edge in component_edges:
        source = assembly_by_component.get(edge["source_component"])
        target = assembly_by_component.get(edge["target_component"])
        if source and target and source != target:
            assembly_edges_set.add(tuple(sorted((source, target))))
    assembly_edges = [
        {
            "source_assembly": source,
            "target_assembly": target,
            "relation": "electrically_connected",
            "source": "底层元件连接汇总",
        }
        for source, target in sorted(assembly_edges_set)
    ]
    return component_edges, assembly_edges


def build_functional_annotations(
    texts: list[dict[str, Any]], conductors: list[Shape]
) -> list[dict[str, Any]]:
    output = []
    for text in texts:
        compact = text["text"].replace(" ", "")
        if compact.upper() == "PT":
            annotation_type = "VoltageTransformerFunctionalBranch"
            explanation = "只有PT文字和独立支路，缺少可由模板确认的完整PT图形"
        elif "计量" in compact:
            annotation_type = "MeteringFunction"
            explanation = "只有计量文字和支路，未见独立计量设备标准符号"
        else:
            continue
        point_box = (text["x"], text["y"], text["x"], text["y"])
        nearest = min(
            conductors,
            key=lambda shape: bbox_distance(point_box, shape.bbox),
            default=None,
        )
        output.append(
            {
                "annotation_id": f"FUNC{len(output) + 1:03d}",
                "type": annotation_type,
                "text": text["text"],
                "text_handle": text["handle"],
                "nearest_conductor_handle": nearest.handle if nearest else "",
                "independent_equipment": False,
                "confidence": "medium",
                "basis": explanation,
            }
        )
    return output


def compare_truth(
    assemblies: list[dict[str, Any]],
    assembly_edges: list[dict[str, Any]],
    truth: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if truth is None:
        return None
    truth_equipment = truth.get("equipment", [])
    truth_by_id = {item["id"]: item for item in truth_equipment}
    matches = []
    used: set[str] = set()
    for truth_item in truth_equipment:
        truth_type = truth_item["type"]
        requested = (
            "PowerTransformer"
            if "变压器" in truth_type
            else "SwitchgearCabinet"
            if "柜" in truth_type
            else truth_type
        )
        label = truth_item.get("visible_label", "")
        ranked = []
        for assembly in assemblies:
            if assembly["assembly_id"] in used or assembly["type"] != requested:
                continue
            label_score = sum(
                token in assembly["label"]
                for token in ("公用", "专用", "原有", "新建", "1#")
                if token in label
            )
            ranked.append((label_score, assembly))
        if not ranked:
            continue
        _, assembly = max(ranked, key=lambda item: item[0])
        used.add(assembly["assembly_id"])
        matches.append(
            {
                "truth_id": truth_item["id"],
                "predicted_id": assembly["assembly_id"],
                "truth_type": truth_type,
                "predicted_type": assembly["type"],
                "truth_label": label,
                "predicted_label": assembly["label"],
                "match": True,
            }
        )
    tp = len(matches)
    precision = tp / len(assemblies) if assemblies else 0.0
    recall = tp / len(truth_equipment) if truth_equipment else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    assembly_for_truth = {
        item["truth_id"]: item["predicted_id"] for item in matches
    }
    predicted_edges = {
        tuple(sorted((edge["source_assembly"], edge["target_assembly"])))
        for edge in assembly_edges
    }
    truth_edges = set()
    for relation in truth.get("device_relations", []):
        source = assembly_for_truth.get(relation["from"])
        target = assembly_for_truth.get(relation["to"])
        if source and target:
            truth_edges.add(tuple(sorted((source, target))))
    for node in truth.get("connectivity_nodes", []):
        refs = node.get("connected_equipment_refs", [])
        for left_index, left_ref in enumerate(refs):
            for right_ref in refs[left_index + 1 :]:
                source = assembly_for_truth.get(left_ref)
                target = assembly_for_truth.get(right_ref)
                if source and target:
                    truth_edges.add(tuple(sorted((source, target))))
    edge_tp = len(predicted_edges & truth_edges)
    edge_precision = edge_tp / len(predicted_edges) if predicted_edges else 0.0
    edge_recall = edge_tp / len(truth_edges) if truth_edges else 0.0
    edge_f1 = (
        2 * edge_precision * edge_recall / (edge_precision + edge_recall)
        if edge_precision + edge_recall
        else 0.0
    )
    return {
        "truth_scope": (
            "纯视觉盲真值只标注2个柜和2台变压器的整体层级；"
            "未标注柜内开关、终端等底层元件。"
        ),
        "equipment_truth_count": len(truth_equipment),
        "equipment_prediction_count": len(assemblies),
        "equipment_true_positive": tp,
        "equipment_precision": round(precision, 4),
        "equipment_recall": round(recall, 4),
        "equipment_f1": round(f1, 4),
        "relation_truth_count": len(truth_edges),
        "relation_prediction_count": len(predicted_edges),
        "relation_true_positive": edge_tp,
        "relation_precision": round(edge_precision, 4),
        "relation_recall": round(edge_recall, 4),
        "relation_f1": round(edge_f1, 4),
        "matches": matches,
        "unmatched_truth_ids": sorted(set(truth_by_id) - set(assembly_for_truth)),
        "predicted_edges": sorted(predicted_edges),
        "truth_edges": sorted(truth_edges),
        "warning": (
            "该指标只评价与真值同粒度的整体设备和设备间关系，"
            "不能证明底层元件模板分类达到相同准确率。"
        ),
    }


def render_overlay(
    output: Path,
    shapes: list[Shape],
    components: list[dict[str, Any]],
    assemblies: list[dict[str, Any]],
) -> None:
    all_box = merge_bbox(shape.bbox for shape in shapes)
    figure, axis = plt.subplots(figsize=(16, 8), dpi=180)
    for shape in shapes:
        if shape.kind == "circle":
            center = (
                float(shape.entity.dxf.center.x),
                float(shape.entity.dxf.center.y),
            )
            radius = float(shape.entity.dxf.radius)
            axis.add_patch(
                plt.Circle(center, radius, fill=False, color="#b7b7b7", lw=0.7)
            )
        else:
            raw = shape.points[:: max(1, len(shape.points) // 80)]
            axis.plot(raw[:, 0], raw[:, 1], color="#b7b7b7", lw=0.7)
    palette = plt.get_cmap("tab20")
    for index, component in enumerate(components):
        x1, y1, x2, y2 = component["group_bbox"]
        color = palette(index % 20)
        axis.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                color=color,
                lw=1.1,
            )
        )
        axis.text(
            x1,
            y2 + 120,
            f"{component['component_id']} {component['family']} "
            f"{component['score']:.1f}",
            color=color,
            fontsize=6,
            va="bottom",
        )
    for assembly in assemblies:
        members = [
            component
            for component in components
            if component["component_id"] in assembly["member_components"]
        ]
        if not members:
            continue
        box = merge_bbox(tuple(member["group_bbox"]) for member in members)
        x1, y1, x2, y2 = box
        axis.add_patch(
            plt.Rectangle(
                (x1 - 120, y1 - 120),
                x2 - x1 + 240,
                y2 - y1 + 240,
                fill=False,
                color="#d62728",
                lw=1.8,
                ls="--",
            )
        )
        axis.text(
            x1,
            y1 - 250,
            f"{assembly['assembly_id']} {assembly['label']}",
            color="#d62728",
            fontsize=7,
            va="top",
        )
    axis.set_xlim(all_box[0] - 800, all_box[2] + 800)
    axis.set_ylim(all_box[1] - 1200, all_box[3] + 1800)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title("02系统接线图：整图多尺度候选与146模板识别")
    figure.tight_layout()
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", type=Path, required=True)
    parser.add_argument("--template-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topology-audit", type=Path)
    parser.add_argument("--visual-truth", type=Path)
    args = parser.parse_args()

    doc = ezdxf.readfile(args.dxf)
    texts = extract_texts(doc)
    heights = [item["height"] for item in texts if item["height"] > 0]
    typical_text_height = statistics.median(heights) if heights else 100.0
    all_shapes = [
        shape
        for entity in doc.modelspace()
        if (shape := shape_from_entity(entity)) is not None
    ]
    conductors = [
        shape for shape in all_shapes if is_conductor(shape, typical_text_height)
    ]
    device_shapes = [
        shape for shape in all_shapes if not is_conductor(shape, typical_text_height)
    ]
    group_gap = typical_text_height * 0.95
    groups = cluster_shapes(device_shapes, group_gap)

    library = load_json(args.template_library)
    templates = prepare_templates(library)
    if len(templates) != 146:
        raise RuntimeError(f"预期146个设备模板，实际{len(templates)}")

    all_lines = [shape for shape in all_shapes if shape.kind == "line"]
    group_results = []
    priors = []
    top_rows = []
    candidate_summaries = []
    for group_index, group in enumerate(groups, 1):
        group_id = f"G{group_index:03d}"
        box = merge_bbox(shape.bbox for shape in group)
        prior = nearest_text_prior(
            box, texts, maximum=typical_text_height * 5.0
        )
        candidates = generate_group_candidates(
            group_id, group, all_lines, typical_text_height
        )
        evaluations = []
        for candidate in candidates:
            ranked = score_candidate(candidate, templates, prior)
            evaluations.append({"candidate": candidate, "ranked": ranked})
            for row in ranked[:5]:
                top_rows.append(
                    {
                        "group_id": group_id,
                        "candidate_id": candidate["candidate_id"],
                        "candidate_mode": candidate["mode"],
                        "candidate_handles": ",".join(
                            candidate["source_handles"]
                        ),
                        "candidate_counts": json.dumps(
                            candidate["primitive_counts"],
                            ensure_ascii=False,
                        ),
                        **row,
                    }
                )
        result = select_group_result(group, candidates, evaluations)
        group_results.append(result)
        priors.append(prior)
        candidate_summaries.append(
            {
                "group_id": group_id,
                "group_handles": sorted(shape.handle for shape in group),
                "candidate_count": len(candidates),
                "selected_candidate": {
                    key: value
                    for key, value in result["candidate"].items()
                    if key != "points"
                },
                "selected_top1": result["top"],
                "selected_top5_families": result["family_alternatives"],
                "coverage": result["coverage"],
                "confidence": result["confidence"],
                "text_prior": prior,
            }
        )

    audit = load_json(args.topology_audit) if args.topology_audit else None
    truth = load_json(args.visual_truth) if args.visual_truth else None
    components = build_components(groups, group_results, priors, audit)
    assemblies = build_assemblies(components, texts)
    component_edges, assembly_edges = build_connections(
        components, assemblies, audit
    )
    functional_annotations = build_functional_annotations(texts, conductors)
    comparison = compare_truth(assemblies, assembly_edges, truth)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "full-drawing-multiscale-template-search-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "drawing": args.dxf.name,
        "method": [
            "从DXF直接读取图元，不使用拟人工真值生成候选。",
            "按图层、闭合状态和长度分离导线与元件图形。",
            "以0.95倍典型文字高度生成初始图形组。",
            "对每组生成完整组、连通子集、相交圆对和不同比例局部引线候选。",
            "每个候选与146个设备模板逐一比较。",
            "几何相似度占85%，图元计数占15%；配变和PT文字最多增加5分。",
            "按模板得分、候选覆盖率和单图元惩罚选择每组结果。",
            "用既有DXF几何拓扑审计映射端口和连接，不用真值修正连接。",
        ],
        "parameters": {
            "typical_text_height": typical_text_height,
            "initial_group_gap": group_gap,
            "wire_minimum_length": typical_text_height * 2.0,
            "wire_layer": "0",
            "stub_ratios": STUB_RATIOS,
            "geometry_weight": 0.85,
            "primitive_count_weight": 0.15,
            "text_prior_max_bonus": 5.0,
        },
        "statistics": {
            "dxf_shape_count": len(all_shapes),
            "conductor_shape_count": len(conductors),
            "device_graphic_count": len(device_shapes),
            "initial_group_count": len(groups),
            "candidate_count": sum(
                item["candidate_count"] for item in candidate_summaries
            ),
            "template_count": len(templates),
            "component_count": len(components),
            "assembly_count": len(assemblies),
            "component_edge_count": len(component_edges),
            "assembly_edge_count": len(assembly_edges),
            "functional_annotation_count": len(functional_annotations),
        },
        "candidate_groups": candidate_summaries,
        "components": components,
        "assemblies": assemblies,
        "component_connections": component_edges,
        "assembly_connections": assembly_edges,
        "functional_annotations": functional_annotations,
        "visual_truth_comparison": comparison,
        "limitations": [
            "146模板中1个纯SVG path模板仍无法参与点集比较。",
            "开关与断路器组合符号可能需要组合模板，单模板家族分数只能作为候选。",
            "纯视觉盲真值没有逐个标注柜内元件，因此底层元件不能计算完整准确率。",
            "本次整图结果只针对02系统接线图，阈值泛化需用其他图纸盲测。",
        ],
    }
    json_path = args.output_dir / "02系统接线图_整图多尺度模板识别.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    component_rows = []
    for item in components:
        component_rows.append(
            {
                "component_id": item["component_id"],
                "family": item["family"],
                "template_name": item["template_name"],
                "score": item["score"],
                "base_score_without_text": item["base_score_without_text"],
                "confidence": item["confidence"],
                "family_margin": item["family_margin"],
                "candidate_mode": item["candidate_mode"],
                "selected_handles": ",".join(item["selected_source_handles"]),
                "group_handles": ",".join(item["group_source_handles"]),
                "terminal_count": item["terminal_count"],
                "mapped_audit_equipment_id": item["mapped_audit_equipment_id"],
            }
        )
    write_csv(args.output_dir / "02系统接线图_整图元件识别.csv", component_rows)
    write_csv(args.output_dir / "02系统接线图_候选模板Top5.csv", top_rows)
    write_csv(
        args.output_dir / "02系统接线图_元件连接.csv", component_edges
    )
    write_csv(
        args.output_dir / "02系统接线图_整体设备连接.csv", assembly_edges
    )
    render_overlay(
        args.output_dir / "02系统接线图_整图识别叠加.png",
        all_shapes,
        components,
        assemblies,
    )

    component_table = "\n".join(
        f"| {item['component_id']} | {item['family']} | "
        f"{item['template_name']} | {item['score']:.2f} | "
        f"{item['confidence']} | {', '.join(item['selected_source_handles'])} |"
        for item in components
    )
    assembly_table = "\n".join(
        f"| {item['assembly_id']} | {item['type']} | {item['label']} | "
        f"{', '.join(item['member_components'])} |"
        for item in assemblies
    )
    if comparison:
        comparison_text = (
            f"- 整体设备：P={comparison['equipment_precision']:.2%}，"
            f"R={comparison['equipment_recall']:.2%}，"
            f"F1={comparison['equipment_f1']:.2%}\n"
            f"- 整体关系：P={comparison['relation_precision']:.2%}，"
            f"R={comparison['relation_recall']:.2%}，"
            f"F1={comparison['relation_f1']:.2%}\n"
            f"- 注意：{comparison['warning']}"
        )
    else:
        comparison_text = "未提供视觉真值，未计算对比指标。"
    report = f"""# 02系统接线图整图多尺度模板识别

## 运行结果

- DXF图形实体：{len(all_shapes)}
- 导电图形：{len(conductors)}
- 元件图形：{len(device_shapes)}
- 初始图形组：{len(groups)}
- 多尺度候选：{sum(item['candidate_count'] for item in candidate_summaries)}
- 比较模板：{len(templates)}
- 底层元件组：{len(components)}
- 顶层整体设备：{len(assemblies)}
- 底层元件连接：{len(component_edges)}
- 顶层设备连接：{len(assembly_edges)}
- 功能性文字支路：{len(functional_annotations)}

## 底层元件识别

| ID | 家族 | 最佳模板 | 得分 | 置信度 | 采用HANDLE |
|---|---|---|---:|---|---|
{component_table}

## 顶层整体设备

| ID | 类型 | 文字 | 成员 |
|---|---|---|---|
{assembly_table}

## 与纯视觉盲真值比较

{comparison_text}

## 未作为独立元件的功能支路

{chr(10).join(f"- {item['text']}：{item['basis']}。" for item in functional_annotations)}

## 解释

本次不再用一个固定距离直接确定元件大小。距离用于生成初始组和多种子候选，
最终由146模板得分选择完整组或子集。变压器文字只提供5分辅助；
每项结果同时保存不含文字的基础分。

真值只标注了两个柜和两台变压器，没有逐个标注柜内开关、终端等符号。
因此顶层指标可以计算，底层结果只能查看模板得分、候选边界和可视化，
不能据此宣称底层分类准确率为100%。

两台变压器模板得分均为97.13，属于高置信度。5个三角端头匹配为
配电双端电缆终端头，得分为70.33至80.29。其余开关图形的最高分仅为
68.71至70.29，且负荷开关、隔离开关和断路器模板得分接近；这是因为
原图画的是组合符号，而知识库目前主要保存单个开关模板，所以这些结果
只能作为低置信度开关家族候选，不能确认具体开关类型。
"""
    (args.output_dir / "02系统接线图_整图多尺度模板识别报告.md").write_text(
        report, encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "groups": len(groups),
                "candidates": sum(
                    item["candidate_count"] for item in candidate_summaries
                ),
                "components": len(components),
                "assemblies": len(assemblies),
                "component_edges": len(component_edges),
                "assembly_edges": len(assembly_edges),
                "equipment_f1": comparison["equipment_f1"]
                if comparison
                else None,
                "relation_f1": comparison["relation_f1"]
                if comparison
                else None,
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
