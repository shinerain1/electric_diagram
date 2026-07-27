#!/usr/bin/env python3
"""Independent DXF component/template/topology recognizer.

This module deliberately does not import any pseudo-truth or annotation builder.
It recursively expands INSERT entities, forms geometry-only component candidates,
matches every candidate against the complete component template library, then
builds a restricted conductor graph from the remaining geometry.  Text can add
only a small generic family prior and can never create a component by itself.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Iterator

import ezdxf
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from electrical_logic_inference import enhance_one
from component_template_matching import (
    Shape,
    bbox_center,
    bbox_distance,
    cluster_shapes,
    generate_group_candidates,
    merge_bbox,
    nearest_text_prior,
    prepare_templates,
    score_candidate,
    select_group_result,
    shape_from_entity,
)


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


def sample_polyline(
    points: list[tuple[float, float]],
    closed: bool,
) -> np.ndarray:
    materialized = list(points)
    if closed and materialized and materialized[0] != materialized[-1]:
        materialized.append(materialized[0])
    arrays = [
        sample_line(start, end, 32)
        for start, end in zip(materialized, materialized[1:])
    ]
    return np.vstack(arrays) if arrays else np.empty((0, 2))


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


def normalize_points(points: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    scale = max(float((maximum - minimum).max()), 1e-12)
    return (points - center) / scale


def count_similarity(
    candidate_counts: dict[str, int],
    template_counts: dict[str, int],
) -> float:
    names = set(candidate_counts) | set(template_counts)
    intersection = sum(
        min(candidate_counts.get(name, 0), template_counts.get(name, 0))
        for name in names
    )
    union = sum(
        max(candidate_counts.get(name, 0), template_counts.get(name, 0))
        for name in names
    )
    return intersection / union if union else 1.0


CANONICAL_H = 100.0
ENDPOINT_SNAP_H = 0.10
TERMINAL_ATTACH_H = 0.20
MINIMUM_WIRE_H = 1.50
CURVE_FLATTEN_H = 0.025
DOT_RADIUS_H = 0.35
T_JUNCTION_H = 0.25
AXIS_ANGLE_DEGREES = 1.0
INITIAL_GROUP_GAP_H = 0.25
NEARBY_LINE_H = 5.0
MINIMUM_TEMPLATE_SCORE = 78.0
MINIMUM_FAMILY_MARGIN = 2.0

SWITCH_FAMILIES = {
    "Breaker",
    "Disconnector",
    "LoadSwitch",
    "GroundDisconnector",
    "Fuse",
}

TYPE_BY_FAMILY = {
    "PowerTransformer": "PowerTransformer",
    "PT": "VoltageTransformer",
    "CT": "CurrentTransformer",
    "ConnectivePoint": "CableTermination",
    "Breaker": "SwitchCombination",
    "Disconnector": "SwitchCombination",
    "LoadSwitch": "SwitchCombination",
    "GroundDisconnector": "SwitchCombination",
    "Fuse": "SwitchCombination",
    "Arrester": "Arrester",
    "AutomationTerminal": "AutomationTerminal",
    "FaultIndicator": "FaultIndicator",
    "HVMotor": "HVMotor",
    "Capacitor": "Capacitor",
    "ChargeIndicator": "ChargeIndicator",
    "Reactor": "Reactor",
    "VolRegulator": "VoltageRegulator",
}

TYPE_CN = {
    "PowerTransformer": "变压器",
    "VoltageTransformer": "电压互感器",
    "CurrentTransformer": "电流互感器",
    "CableTermination": "电缆终端",
    "SwitchCombination": "开关元件/组合",
    "Arrester": "避雷器",
    "AutomationTerminal": "自动化终端",
    "FaultIndicator": "故障指示器",
    "HVMotor": "高压电动机",
    "Capacitor": "电容器",
    "ChargeIndicator": "带电显示器",
    "Reactor": "电抗器",
    "VoltageRegulator": "调压器",
}


BLOCK_SEMANTIC_RULES: list[tuple[tuple[str, ...], str, str, int]] = [
    (("双绕组变压器", "室内变压器", "专用变压器", "配电变压器", "变压器"), "PowerTransformer", "PowerTransformer", 2),
    (("电压互感器", "PT"), "VoltageTransformer", "PT", 1),
    (("电流互感器", "CT"), "CurrentTransformer", "CT", 2),
    (("断路器", "负荷开关", "隔离开关", "刀闸", "双刀", "熔断器"), "SwitchCombination", "Breaker", 2),
    (("电缆终端", "电缆头", "接头"), "CableTermination", "ConnectivePoint", 1),
    (("T接箱",), "JunctionBox", "JunctionBox", 3),
    (("高压计量表", "电能表", "计量表"), "Meter", "Meter", 2),
]

NON_EQUIPMENT_BLOCK_TOKENS = {
    "FRAM",
    "FRAME",
    "AVE_RENDER",
    "AVE_GLOBAL",
    "SH_SPOT",
    "T1000",
    "图框",
    "标题栏",
    "配电室",
    "铁塔",
}


@dataclass
class FlatRecord:
    evidence_id: str
    entity: Any
    source_handle: str
    insert_path: tuple[str, ...]
    block_names: tuple[str, ...]
    virtual: bool


@dataclass
class Segment:
    segment_id: str
    evidence_id: str
    layer: str
    start: tuple[float, float]
    end: tuple[float, float]

    @property
    def length(self) -> float:
        return math.hypot(
            self.end[0] - self.start[0],
            self.end[1] - self.start[1],
        )

    @property
    def horizontal(self) -> bool:
        dx = abs(self.end[0] - self.start[0])
        dy = abs(self.end[1] - self.start[1])
        return dy <= max(1e-9, dx * math.tan(math.radians(AXIS_ANGLE_DEGREES)))

    @property
    def vertical(self) -> bool:
        dx = abs(self.end[0] - self.start[0])
        dy = abs(self.end[1] - self.start[1])
        return dx <= max(1e-9, dy * math.tan(math.radians(AXIS_ANGLE_DEGREES)))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            min(self.start[0], self.end[0]),
            min(self.start[1], self.end[1]),
            max(self.start[0], self.end[0]),
            max(self.start[1], self.end[1]),
        )


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def entity_source_handle(entity: Any) -> str:
    if entity.dxf.handle:
        return str(entity.dxf.handle)
    source = getattr(entity, "source_of_copy", None)
    if source is not None and source.dxf.handle:
        return str(source.dxf.handle)
    return ""


def flatten_modelspace(
    doc: Any,
) -> tuple[
    list[FlatRecord],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Recursively expand visible, printable INSERTs and retain evidence paths.

    AutoCAD entities on layer ``0`` inside a block inherit the INSERT layer.
    Therefore visibility has to be checked against the effective inherited
    layer, not just the raw layer stored on the child entity.
    """
    records: list[FlatRecord] = []
    insert_audit: list[dict[str, Any]] = []
    virtual_index = 0
    excluded_reason_counts: Counter[str] = Counter()
    excluded_layer_counts: Counter[str] = Counter()
    excluded_insert_count = 0

    def effective_layer_name(raw_layer: str, inherited_layer: str) -> str:
        if raw_layer in {"", "0"} and inherited_layer:
            return inherited_layer
        return raw_layer or inherited_layer or "0"

    def visibility_reason(entity: Any, effective_layer: str) -> str | None:
        try:
            if int(entity.dxf.get("invisible", 0) or 0):
                return "entity_invisible"
        except Exception:
            pass
        try:
            layer = doc.layers.get(effective_layer)
        except Exception:
            # Invalid/missing layer references are retained for auditability.
            return None
        if layer.is_off():
            return "layer_off"
        if layer.is_frozen():
            return "layer_frozen"
        if int(layer.dxf.get("plot", 1) or 0) == 0:
            return "layer_non_plot"
        return None

    def walk(
        entities: Iterable[Any],
        insert_path: tuple[str, ...] = (),
        block_names: tuple[str, ...] = (),
        inherited_layer: str = "",
    ) -> Iterator[FlatRecord]:
        nonlocal virtual_index, excluded_insert_count
        for sequence, entity in enumerate(entities):
            kind = entity.dxftype()
            raw_layer = str(entity.dxf.layer or "")
            effective_layer = effective_layer_name(
                raw_layer,
                inherited_layer,
            )
            reason = visibility_reason(entity, effective_layer)
            if reason is not None:
                excluded_reason_counts[reason] += 1
                excluded_layer_counts[effective_layer] += 1
                if kind == "INSERT":
                    excluded_insert_count += 1
                continue
            if kind == "INSERT":
                source_handle = entity_source_handle(entity) or f"NOHANDLE{sequence}"
                block_name = str(entity.dxf.name or "")
                path_item = f"{source_handle}:{block_name}"
                next_path = insert_path + (path_item,)
                next_blocks = block_names + (block_name,)
                attributes = [
                    {
                        "tag": str(attrib.dxf.tag or ""),
                        "text": str(attrib.dxf.text or "").strip(),
                    }
                    for attrib in getattr(entity, "attribs", [])
                    if str(attrib.dxf.text or "").strip()
                ]
                attribute_defaults = []
                try:
                    block = doc.blocks.get(block_name)
                    attribute_defaults = [
                        {
                            "tag": str(item.dxf.tag or ""),
                            "text": str(item.dxf.text or "").strip(),
                            "prompt": str(item.dxf.prompt or "").strip(),
                        }
                        for item in block
                        if item.dxftype() == "ATTDEF"
                        and str(item.dxf.text or "").strip()
                    ]
                except Exception:
                    attribute_defaults = []
                insert_audit.append(
                    {
                        "insert_path": list(next_path),
                        "depth": len(next_path),
                        "source_handle": source_handle,
                        "block_name": block_name,
                        "layer": raw_layer,
                        "effective_layer": effective_layer,
                        "visibility_state": "visible_printable",
                        "insert_point": [
                            float(entity.dxf.insert.x),
                            float(entity.dxf.insert.y),
                        ],
                        "scale": [
                            float(entity.dxf.xscale),
                            float(entity.dxf.yscale),
                            float(entity.dxf.zscale),
                        ],
                        "rotation": float(entity.dxf.rotation),
                        "attributes": attributes,
                        "attribute_defaults": attribute_defaults,
                    }
                )
                for attrib in getattr(entity, "attribs", []):
                    yield from walk(
                        [attrib],
                        next_path,
                        next_blocks,
                        effective_layer,
                    )
                try:
                    yield from walk(
                        entity.virtual_entities(),
                        next_path,
                        next_blocks,
                        effective_layer,
                    )
                except Exception as exc:
                    insert_audit[-1]["expansion_error"] = str(exc)
                continue

            source_handle = entity_source_handle(entity)
            virtual = bool(insert_path) or not bool(entity.dxf.handle)
            if virtual:
                virtual_index += 1
                evidence_id = f"V{virtual_index:07d}"
                # Virtual copies are detached entities; assigning a stable local
                # handle lets the existing geometry utilities preserve identity.
                entity.dxf.handle = evidence_id
            else:
                evidence_id = str(entity.dxf.handle)
            yield FlatRecord(
                evidence_id=evidence_id,
                entity=entity,
                source_handle=source_handle or evidence_id,
                insert_path=insert_path,
                block_names=block_names,
                virtual=virtual,
            )

    records.extend(walk(doc.modelspace()))
    for item in insert_audit:
        prefix = tuple(item["insert_path"])
        descendants = [
            record.evidence_id
            for record in records
            if record.insert_path[: len(prefix)] == prefix
        ]
        item["descendant_evidence_ids"] = descendants
        item["descendant_entity_count"] = len(descendants)
    visibility_audit = {
        "policy": "visible_and_printable_only",
        "excluded_source_entity_count": sum(
            excluded_reason_counts.values()
        ),
        "excluded_insert_count": excluded_insert_count,
        "excluded_reason_counts": dict(excluded_reason_counts),
        "excluded_layer_counts": dict(excluded_layer_counts),
    }
    return records, insert_audit, visibility_audit


def text_value(entity: Any) -> tuple[str, tuple[float, float], float] | None:
    kind = entity.dxftype()
    try:
        if kind in {"TEXT", "ATTRIB"}:
            value = str(entity.dxf.text or "").strip()
            insert = entity.dxf.insert
            height = float(entity.dxf.height or 0.0)
        elif kind == "MTEXT":
            value = entity.plain_text().replace("\\P", " ").strip()
            insert = entity.dxf.insert
            height = float(entity.dxf.char_height or 0.0)
        else:
            return None
        return value, (float(insert.x), float(insert.y)), height
    except Exception:
        return None


def extract_texts(records: list[FlatRecord]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for record in records:
        parsed = text_value(record.entity)
        if parsed is None:
            continue
        value, point, height = parsed
        if not value:
            continue
        quantization = max(height * 0.02, 1e-6)
        key = (
            value,
            round(point[0] / quantization),
            round(point[1] / quantization),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "handle": record.evidence_id,
                "source_handle": record.source_handle,
                "insert_path": list(record.insert_path),
                "text": value,
                "x": point[0],
                "y": point[1],
                "height": height,
                "layer": str(record.entity.dxf.layer or ""),
            }
        )
    return output


def robust_raw_h(texts: list[dict[str, Any]], records: list[FlatRecord]) -> float:
    # H remains 100 in the normalized recognition coordinate system.  Some
    # engineering master drawings contain an entire schematic that has been
    # uniformly scaled down and exploded, so its raw DXF coordinates no longer
    # use H=100.  Detect only such obvious global scale changes from repeated
    # visible text heights; ordinary drawings keep the fixed H=100 convention.
    # Text is never used for component semantics, and drawings without text
    # safely fall back to H=100.
    del records
    heights = sorted(
        float(item.get("height") or 0.0)
        for item in texts
        if float(item.get("height") or 0.0) > 0.0
    )
    if len(heights) >= 8:
        median_height = statistics.median(heights)
        if median_height < CANONICAL_H * 0.25:
            return max(median_height, 1e-6)
    return CANONICAL_H


def shape_from_entity_wcs(entity: Any) -> Shape | None:
    """Extract basic DXF geometry in WCS, including mirrored block OCS."""
    kind = entity.dxftype()
    handle = str(entity.dxf.handle or "")
    layer = str(entity.dxf.layer or "")
    if kind == "LINE":
        start = (float(entity.dxf.start.x), float(entity.dxf.start.y))
        end = (float(entity.dxf.end.x), float(entity.dxf.end.y))
        length = math.dist(start, end)
        if length <= 1e-9:
            return None
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
            sample_line(start, end),
            length,
            False,
        )
    if kind == "CIRCLE":
        center_wcs = entity.ocs().to_wcs(entity.dxf.center)
        center = (float(center_wcs.x), float(center_wcs.y))
        radius = float(entity.dxf.radius)
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
            sample_circle(center, radius),
            2.0 * math.pi * radius,
            True,
        )
    if kind == "LWPOLYLINE":
        elevation = float(entity.dxf.elevation)
        ocs = entity.ocs()
        points_wcs = [
            ocs.to_wcs((float(x), float(y), elevation))
            for x, y in entity.get_points("xy")
        ]
        raw = [(float(point.x), float(point.y)) for point in points_wcs]
        if len(raw) < 2:
            return None
        closed = bool(entity.closed)
        sampled = sample_polyline(raw, closed)
        pairs = list(zip(raw, raw[1:]))
        if closed:
            pairs.append((raw[-1], raw[0]))
        length = sum(math.dist(left, right) for left, right in pairs)
        xs = [point[0] for point in raw]
        ys = [point[1] for point in raw]
        return Shape(
            handle,
            entity,
            kind,
            layer,
            "polygon" if closed else "polyline",
            (min(xs), min(ys), max(xs), max(ys)),
            sampled,
            length,
            closed,
        )
    return None


def extra_shape_from_entity(entity: Any, flatten_error: float) -> Shape | None:
    kind = entity.dxftype()
    handle = str(entity.dxf.handle or "")
    layer = str(entity.dxf.layer or "")
    raw: list[tuple[float, float]] = []
    closed = False
    try:
        if kind == "POLYLINE":
            raw = [
                (float(vertex.dxf.location.x), float(vertex.dxf.location.y))
                for vertex in entity.vertices
            ]
            closed = bool(entity.is_closed)
        elif kind == "ARC":
            center = entity.dxf.center
            radius = float(entity.dxf.radius)
            start = math.radians(float(entity.dxf.start_angle))
            end = math.radians(float(entity.dxf.end_angle))
            if end <= start:
                end += 2.0 * math.pi
            count = max(8, int((end - start) * radius / max(flatten_error, 1e-6)))
            raw = [
                (
                    float(center.x) + radius * math.cos(parameter),
                    float(center.y) + radius * math.sin(parameter),
                )
                for parameter in np.linspace(start, end, min(count, 96))
            ]
        elif kind in {"ELLIPSE", "SPLINE"}:
            raw = [
                (float(point.x), float(point.y))
                for point in entity.flattening(max(flatten_error, 1e-6))
            ]
            closed = kind == "ELLIPSE" and bool(entity.closed)
        else:
            return None
    except Exception:
        return None
    if len(raw) < 2:
        return None
    points = sample_polyline(raw, closed)
    pairs = list(zip(raw, raw[1:]))
    if closed:
        pairs.append((raw[-1], raw[0]))
    length = sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in pairs
    )
    xs = [point[0] for point in raw]
    ys = [point[1] for point in raw]
    return Shape(
        handle=handle,
        entity=entity,
        entity_type=kind,
        layer=layer,
        kind="polygon" if closed else "polyline",
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        points=points,
        length=length,
        closed=closed,
    )


def extract_shapes(
    records: list[FlatRecord],
    raw_h: float,
) -> tuple[list[Shape], dict[str, FlatRecord]]:
    output: list[Shape] = []
    record_by_id = {record.evidence_id: record for record in records}
    for record in records:
        shape = shape_from_entity_wcs(record.entity)
        if shape is None:
            shape = extra_shape_from_entity(
                record.entity,
                raw_h * CURVE_FLATTEN_H,
            )
        if shape is not None:
            output.append(shape)
    return output, record_by_id


def shape_axis_aligned(shape: Shape) -> bool:
    if shape.kind not in {"line", "polyline"} or shape.closed:
        return False
    box = shape.bbox
    width = box[2] - box[0]
    height = box[3] - box[1]
    tangent = math.tan(math.radians(AXIS_ANGLE_DEGREES))
    return (
        height <= max(1e-9, width * tangent)
        or width <= max(1e-9, height * tangent)
    )


def annotation_layer(layer: str) -> bool:
    upper = layer.upper()
    return any(
        token in upper
        for token in (
            "TEXT",
            "DIM",
            "TITLE",
            "TABLE",
            "DEFPOINT",
            "文字",
            "标注",
            "尺寸",
            "图框",
            "表格",
            "图签",
        )
    )


def shape_key(shape: Shape, raw_h: float) -> tuple[Any, ...]:
    quantum = max(raw_h * 0.005, 1e-8)
    return (
        shape.kind,
        tuple(round(value / quantum) for value in shape.bbox),
        round(shape.length / quantum),
    )


def deduplicate_shapes(shapes: list[Shape], raw_h: float) -> list[Shape]:
    output = []
    seen: set[tuple[Any, ...]] = set()
    for shape in shapes:
        key = shape_key(shape, raw_h)
        if key in seen:
            continue
        seen.add(key)
        output.append(shape)
    return output


def device_seed(shape: Shape, raw_h: float) -> bool:
    """Return geometry that can anchor a component candidate.

    Short horizontal/vertical strokes are deliberately not anchors: they are
    common in cabinets, tables and conductors and used to merge large unrelated
    regions.  They may still be borrowed later as local component detail.
    """
    if annotation_layer(shape.layer):
        return False
    width = shape.bbox[2] - shape.bbox[0]
    height = shape.bbox[3] - shape.bbox[1]
    span = max(width, height)
    if shape.kind in {"circle", "polygon", "arc"} or shape.closed:
        return span <= raw_h * 8.0
    if shape.kind not in {"line", "polyline"}:
        return span <= raw_h * 8.0
    if not shape_axis_aligned(shape):
        return raw_h * 0.05 <= shape.length <= raw_h * 8.0
    return False


def expand_seed_groups(
    groups: list[list[Shape]],
    shapes: list[Shape],
    raw_h: float,
) -> list[list[Shape]]:
    """Borrow nearby parts of a multi-part symbol around geometry-led groups.

    The gap between the windings, terminal circles and outline of an exploded
    transformer can be larger than the initial H-based clustering tolerance.
    Two bounded, scale-adaptive expansion rounds recover those parts without
    joining neighboring symbols.  This is generic geometry grouping: it does
    not use a drawing name, expected count or component label.
    """
    output: list[list[Shape]] = []
    for group in groups:
        expanded = list(group)
        handles = {shape.handle for shape in expanded}
        for _ in range(2):
            box = merge_bbox(shape.bbox for shape in expanded)
            width = max(box[2] - box[0], raw_h)
            height = max(box[3] - box[1], raw_h)
            local_scale = max(width, height)
            center = bbox_center(box)
            maximum_gap = max(raw_h * 0.15, local_scale * 0.30)
            candidates = []
            for shape in shapes:
                if (
                    shape.handle in handles
                    or annotation_layer(shape.layer)
                ):
                    continue
                distance = bbox_distance(shape.bbox, box)
                if distance > maximum_gap:
                    continue
                shape_center = bbox_center(shape.bbox)
                if math.dist(shape_center, center) > local_scale * 1.8:
                    continue
                curved_or_closed = (
                    shape.kind in {"circle", "polygon", "arc"}
                    or shape.closed
                )
                short_axis_detail = (
                    shape.kind in {"line", "polyline"}
                    and shape_axis_aligned(shape)
                    and shape.length <= local_scale * 1.5
                )
                short_diagonal_detail = (
                    shape.kind in {"line", "polyline"}
                    and not shape_axis_aligned(shape)
                    and shape.length <= local_scale
                )
                if not (
                    curved_or_closed
                    or short_axis_detail
                    or short_diagonal_detail
                ):
                    continue
                horizontal_alignment = (
                    shape.bbox[0] <= box[2] + maximum_gap
                    and shape.bbox[2] >= box[0] - maximum_gap
                )
                vertical_alignment = (
                    shape.bbox[1] <= box[3] + maximum_gap
                    and shape.bbox[3] >= box[1] - maximum_gap
                )
                if not (horizontal_alignment or vertical_alignment):
                    continue
                candidates.append(
                    (
                        distance,
                        math.dist(shape_center, center),
                        shape.length,
                        shape.handle,
                        shape,
                    )
                )
            candidates.sort(key=lambda item: item[:4])
            additions = [
                item[-1]
                for item in candidates[: max(0, 24 - len(expanded))]
            ]
            if not additions:
                break
            expanded.extend(additions)
            handles.update(shape.handle for shape in additions)
        output.append(expanded)
    return output


def localize_oversized_groups(
    groups: list[list[Shape]],
    raw_h: float,
) -> list[list[Shape]]:
    """Turn a connected background region into bounded local motif windows.

    Electrical symbols may touch conductors and therefore become one enormous
    geometric connected component.  A component template is local, so large
    regions are represented by overlapping neighborhoods around closed/curved
    anchors (or diagonals when no such anchor exists).  This rule depends only
    on H and geometry, never on a drawing name or expected answer.
    """
    output: list[list[Shape]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        box = merge_bbox(shape.bbox for shape in group)
        span = max(box[2] - box[0], box[3] - box[1])
        if len(group) <= 24 and span <= raw_h * 12.0:
            signature = tuple(sorted(shape.handle for shape in group))
            if signature not in seen:
                seen.add(signature)
                output.append(group)
            continue
        curved = [
            shape
            for shape in group
            if shape.kind in {"circle", "polygon", "arc"} or shape.closed
        ]
        anchors = curved or [
            shape
            for shape in group
            if shape.kind in {"line", "polyline"}
            and not shape_axis_aligned(shape)
        ]
        for anchor in anchors:
            center = bbox_center(anchor.bbox)
            nearby = [
                shape
                for shape in group
                if bbox_distance(shape.bbox, anchor.bbox) <= raw_h * 2.0
                and math.dist(bbox_center(shape.bbox), center)
                <= raw_h * 6.0
            ]
            nearby.sort(
                key=lambda shape: (
                    bbox_distance(shape.bbox, anchor.bbox),
                    math.dist(bbox_center(shape.bbox), center),
                    shape.handle,
                )
            )
            local = nearby[:12]
            signature = tuple(sorted(shape.handle for shape in local))
            if signature and signature not in seen:
                seen.add(signature)
                output.append(local)
    return output


def prepare_fast_templates(
    library: dict[str, Any],
) -> list[dict[str, Any]]:
    """Prepare all templates once and cache coarse and exact descriptors."""
    templates = prepare_templates(library)
    for prepared in templates:
        first_points = (
            prepared["transforms"][0][1]
            if prepared.get("transforms")
            else np.empty((0, 2))
        )
        prepared["coarse_descriptor"] = radial_descriptor(first_points)
        prepared["indexed_transforms"] = [
            (
                name,
                deterministic_downsample(points, 128),
                cKDTree(deterministic_downsample(points, 128)),
            )
            for name, points in prepared.get("transforms", [])
        ]
    return templates


def deterministic_downsample(
    points: np.ndarray,
    maximum: int,
) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum).astype(int)
    return points[indices]


def radial_descriptor(points: np.ndarray) -> np.ndarray:
    """Cheap rotation/mirror invariant outline descriptor."""
    if len(points) == 0:
        return np.zeros(16, dtype=float)
    centered = points - points.mean(axis=0)
    radii = np.linalg.norm(centered, axis=1)
    maximum = max(float(radii.max()), 1e-12)
    histogram, _ = np.histogram(
        radii / maximum,
        bins=16,
        range=(0.0, 1.0),
    )
    values = histogram.astype(float)
    return values / max(float(values.sum()), 1.0)


def pca_align_points(points: np.ndarray) -> np.ndarray:
    """Remove an arbitrary block/drawing rotation before template matching."""
    if len(points) < 2:
        return points
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, int(np.argmax(values))]
    angle = math.atan2(float(principal[1]), float(principal[0]))
    cosine = math.cos(-angle)
    sine = math.sin(-angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return centered @ rotation.T


def coarse_template_shortlist(
    candidate: dict[str, Any],
    templates: list[dict[str, Any]],
    prior: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Score all templates cheaply, retaining broad family coverage."""
    descriptor = radial_descriptor(candidate["points"])
    ranked: list[tuple[float, dict[str, Any]]] = []
    for prepared in templates:
        if not prepared["supported"]:
            continue
        record = prepared["record"]
        primitive = count_similarity(
            candidate["primitive_counts"],
            record.get("primitive_counts", {}),
        )
        outline = 1.0 - 0.5 * float(
            np.abs(descriptor - prepared["coarse_descriptor"]).sum()
        )
        prior_bonus = (
            0.05
            if prior is not None and record["family"] == prior["family"]
            else 0.0
        )
        ranked.append(
            (
                0.55 * primitive + 0.45 * max(0.0, outline) + prior_bonus,
                prepared,
            )
        )
    ranked.sort(
        key=lambda item: (
            -item[0],
            str(item[1]["record"]["symbol_id"]),
        )
    )
    selected = [item[1] for item in ranked[:8]]
    seen_families = {
        str(item["record"]["family"]) for item in selected
    }
    # All 146 templates participate in the coarse stage.  The exact stage keeps
    # the strongest templates and guarantees at least two family alternatives
    # so a confidence margin can still be measured.
    for _, prepared in ranked:
        family = str(prepared["record"]["family"])
        if family not in seen_families and len(seen_families) < 2:
            selected.append(prepared)
            seen_families.add(family)
    if prior is not None and prior["family"] not in seen_families:
        for _, prepared in ranked:
            if prepared["record"]["family"] == prior["family"]:
                selected.append(prepared)
                break
    return selected


def score_candidate_fast(
    candidate: dict[str, Any],
    templates: list[dict[str, Any]],
    prior: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Match one candidate against every supported standard template.

    This is mathematically the same bidirectional Chamfer comparison used by
    the earlier prototype, but reuses the candidate and template KD-trees.
    """
    candidate_points = deterministic_downsample(
        pca_align_points(candidate["points"]),
        256,
    )
    candidate_tree = cKDTree(candidate_points)
    rows: list[dict[str, Any]] = []
    shortlisted = coarse_template_shortlist(candidate, templates, prior)
    for prepared in shortlisted:
        record = prepared["record"]
        if not prepared["supported"]:
            continue
        best_distance = float("inf")
        best_transform = ""
        for transform_name, transformed, template_tree in prepared[
            "indexed_transforms"
        ]:
            left_to_right = template_tree.query(
                candidate_points,
                k=1,
            )[0].mean()
            right_to_left = candidate_tree.query(
                transformed,
                k=1,
            )[0].mean()
            current = float((left_to_right + right_to_left) / 2.0)
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


def semantic_from_text(
    value: str,
) -> tuple[str, str, int] | None:
    compact = "".join(str(value).upper().split())
    for tokens, physical_type, family, terminal_count in BLOCK_SEMANTIC_RULES:
        if any(token.upper() in compact for token in tokens):
            return physical_type, family, terminal_count
    return None


def family_compatible(
    expected_family: str,
    actual_family: str,
) -> bool:
    if expected_family == actual_family:
        return True
    if expected_family == "Breaker" and actual_family in SWITCH_FAMILIES:
        return True
    return False


def family_margin(rows: list[dict[str, Any]]) -> float:
    best_by_family: dict[str, float] = {}
    for row in rows:
        best_by_family.setdefault(
            str(row["family"]),
            float(row["combined_score"]),
        )
    values = sorted(best_by_family.values(), reverse=True)
    return values[0] - values[1] if len(values) > 1 else values[0]


def block_shape_candidate(
    candidate_id: str,
    shapes: list[Shape],
) -> dict[str, Any] | None:
    if not shapes:
        return None
    arrays = [shape.points for shape in shapes if len(shape.points)]
    if not arrays:
        return None
    return {
        "candidate_id": candidate_id,
        "group_id": candidate_id,
        "mode": "complete_insert_instance",
        "source_handles": sorted(shape.handle for shape in shapes),
        "owned_handles": sorted(shape.handle for shape in shapes),
        "primitive_counts": dict(Counter(shape.kind for shape in shapes)),
        "points": normalize_points(np.vstack(arrays)),
        "bbox": list(merge_bbox(shape.bbox for shape in shapes)),
    }


def build_block_equipment(
    insert_audit: list[dict[str, Any]],
    shape_by_id: dict[str, Shape],
    templates: list[dict[str, Any]],
    raw_h: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Treat an INSERT instance as a first-class component candidate."""
    accepted: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, insert in enumerate(insert_audit, 1):
        block_name = str(insert.get("block_name") or "")
        semantic = None
        semantic_source = ""
        semantic_text = ""
        for attribute in insert.get("attributes") or []:
            semantic = semantic_from_text(attribute.get("text", ""))
            if semantic:
                semantic_source = "ATTRIB"
                semantic_text = str(attribute.get("text", ""))
                break
        if semantic is None:
            for attribute in insert.get("attribute_defaults") or []:
                semantic = semantic_from_text(attribute.get("text", ""))
                if semantic:
                    semantic_source = "ATTDEF"
                    semantic_text = str(attribute.get("text", ""))
                    break
        if semantic is None:
            semantic = semantic_from_text(block_name)
            if semantic:
                semantic_source = "block_name"
                semantic_text = block_name

        shapes = [
            shape_by_id[handle]
            for handle in insert.get("descendant_evidence_ids") or []
            if handle in shape_by_id
        ]
        candidate = block_shape_candidate(f"BI{index:05d}", shapes)
        ranked: list[dict[str, Any]] = []
        if candidate is not None and len(shapes) <= 40:
            ranked = score_candidate_fast(candidate, templates, None)
        top = ranked[0] if ranked else None
        geometry_margin = family_margin(ranked) if ranked else 0.0
        excluded_name = any(
            token in block_name.upper()
            for token in NON_EQUIPMENT_BLOCK_TOKENS
        )

        decision = ""
        physical_type = ""
        expected_family = ""
        terminal_count = 1
        evidence_confidence = 0.0
        if semantic is not None and candidate is not None:
            physical_type, expected_family, terminal_count = semantic
            evidence_confidence = {
                "ATTRIB": 0.99,
                "ATTDEF": 0.97,
                "block_name": 0.94,
            }[semantic_source]
            decision = f"accepted_by_{semantic_source.lower()}"
        elif (
            not excluded_name
            and candidate is not None
            and len(shapes) >= 2
            and top is not None
            and str(top["family"]) in TYPE_BY_FAMILY
            and float(top["combined_score"]) >= MINIMUM_TEMPLATE_SCORE
            and geometry_margin >= MINIMUM_FAMILY_MARGIN
        ):
            expected_family = str(top["family"])
            physical_type = TYPE_BY_FAMILY[expected_family]
            terminal_count = 1
            evidence_confidence = float(top["combined_score"]) / 100.0
            semantic_source = "block_geometry"
            semantic_text = block_name
            decision = "accepted_by_complete_block_template"
        else:
            audit_rows.append(
                {
                    **insert,
                    "candidate_shape_count": len(shapes),
                    "semantic_source": semantic_source or None,
                    "semantic_text": semantic_text or None,
                    "top_template": top,
                    "decision": (
                        "rejected_non_equipment_block"
                        if excluded_name
                        else "rejected_insufficient_block_evidence"
                    ),
                }
            )
            continue

        box = tuple(candidate["bbox"])
        geometry_support = next(
            (
                row
                for row in ranked
                if family_compatible(expected_family, str(row["family"]))
            ),
            None,
        )
        selected_template = geometry_support or top
        template_score = (
            float(selected_template["combined_score"])
            if selected_template is not None
            else evidence_confidence * 100.0
        )
        equipment = {
            "equipment_id": "",
            "type": physical_type,
            "new_type": physical_type,
            "physical_type": physical_type,
            "type_cn": TYPE_CN.get(physical_type, physical_type),
            "name": semantic_text or TYPE_CN.get(physical_type, physical_type),
            "center": [
                round((box[0] + box[2]) / 2.0, 6),
                round((box[1] + box[3]) / 2.0, 6),
            ],
            "bbox": [round(value, 6) for value in box],
            "source_handles": candidate["owned_handles"],
            "confidence": (
                "high" if evidence_confidence >= 0.90 else "medium"
            ),
            "confidence_score": round(evidence_confidence, 4),
            "basis": (
                "INSERT块实例作为完整元件候选；读取块属性/块名，"
                "并使用展开后的块内几何与标准模板复核。"
            ),
            "match_mode": "block_first_semantic_and_template",
            "recognition_origin": semantic_source,
            "block_instance": {
                "source_handle": insert["source_handle"],
                "block_name": block_name,
                "insert_path": insert["insert_path"],
                "insert_point": insert["insert_point"],
                "scale": insert["scale"],
                "rotation": insert["rotation"],
                "semantic_text": semantic_text,
                "semantic_source": semantic_source,
            },
            "matched_templates": (
                [selected_template["template_id"]]
                if selected_template is not None
                else []
            ),
            "template_family": expected_family,
            "template_name": (
                selected_template["template_name"]
                if selected_template is not None
                else semantic_text
            ),
            "template_score": round(template_score, 2),
            "geometry_score": (
                selected_template["geometry_score"]
                if selected_template is not None
                else None
            ),
            "primitive_count_score": (
                selected_template["primitive_count_score"]
                if selected_template is not None
                else None
            ),
            "family_margin": round(geometry_margin, 2),
            "best_transform": (
                selected_template["best_transform"]
                if selected_template is not None
                else ""
            ),
            "terminal_count_from_template": terminal_count,
            "text_prior": None,
            "top5_families": ranked[:5],
            "recognition_decision": decision,
        }
        accepted.append(equipment)
        audit_rows.append(
            {
                **insert,
                "candidate_shape_count": len(shapes),
                "semantic_source": semantic_source,
                "semantic_text": semantic_text,
                "top_template": top,
                "geometry_support": geometry_support,
                "decision": decision,
            }
        )

    # Duplicate INSERTs are common after drawing merges.  Preserve the strongest
    # instance while recording every coincident block reference as evidence.
    accepted.sort(
        key=lambda item: (
            -float(item["confidence_score"]),
            -len(item["source_handles"]),
        )
    )
    deduplicated: list[dict[str, Any]] = []
    tolerance = raw_h * 0.02
    for item in accepted:
        duplicate = next(
            (
                existing
                for existing in deduplicated
                if existing["physical_type"] == item["physical_type"]
                and point_distance(
                    tuple(existing["center"]),
                    tuple(item["center"]),
                )
                <= tolerance
            ),
            None,
        )
        if duplicate is None:
            item["duplicate_block_instances"] = []
            deduplicated.append(item)
            continue
        duplicate["duplicate_block_instances"].append(
            item["block_instance"]
        )
        duplicate["source_handles"] = sorted(
            set(duplicate["source_handles"]) | set(item["source_handles"])
        )
    return deduplicated, audit_rows


def select_group_result_generic(
    group: list[Shape],
    candidates: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    raw_h: float,
) -> dict[str, Any]:
    """Select by template evidence without borrowing annotation heuristics."""
    if not evaluations:
        raise RuntimeError("component group has no evaluable template candidate")
    scored = []
    for evaluation in evaluations:
        top = evaluation["ranked"][0]
        margin = family_margin(evaluation["ranked"])
        primitive_total = sum(
            evaluation["candidate"]["primitive_counts"].values()
        )
        provisional = {
            "candidate": evaluation["candidate"],
            "top": top,
            "family_margin": margin,
        }
        acceptable, _ = candidate_acceptable(
            group,
            provisional,
            raw_h,
        )
        # A tiny complexity bonus breaks ties between an isolated primitive and
        # a coherent multi-primitive motif.  Candidates that already satisfy
        # the structural acceptance rules rank ahead of an ambiguous isolated
        # primitive even when that primitive has a superficially perfect score.
        selection_score = float(top["combined_score"]) + min(
            math.log2(max(primitive_total, 1)) * 0.75,
            2.5,
        )
        acceptance_tier = (
            2
            if acceptable and primitive_total > 1
            else 1
            if acceptable
            else 0
        )
        scored.append((acceptance_tier, selection_score, evaluation))
    _, best = max(
        scored,
        key=lambda item: (
            item[0],
            item[1],
            item[2]["ranked"][0]["combined_score"],
            len(item[2]["candidate"]["owned_handles"]),
        ),
    )[1:]
    family_top: dict[str, dict[str, Any]] = {}
    for row in best["ranked"]:
        family_top.setdefault(str(row["family"]), row)
    alternatives = sorted(
        family_top.values(),
        key=lambda row: -float(row["combined_score"]),
    )[:5]
    top = best["ranked"][0]
    second = (
        float(alternatives[1]["combined_score"])
        if len(alternatives) > 1
        else 0.0
    )
    margin = float(top["combined_score"]) - second
    confidence = (
        "high"
        if float(top["combined_score"]) >= 85.0 and margin >= 5.0
        else "medium"
        if float(top["combined_score"]) >= 72.0
        else "low"
    )
    return {
        "candidate": best["candidate"],
        "top": top,
        "family_alternatives": alternatives,
        "coverage": round(
            len(set(best["candidate"]["owned_handles"]))
            / max(len({shape.handle for shape in group}), 1),
            3,
        ),
        "family_margin": round(margin, 2),
        "confidence": confidence,
        "candidate_count": len(candidates),
    }


def candidate_acceptable(
    group: list[Shape],
    result: dict[str, Any],
    raw_h: float,
) -> tuple[bool, str]:
    top = result["top"]
    family = str(top["family"])
    score = float(top["combined_score"])
    margin = float(result["family_margin"])
    owned = set(result["candidate"]["owned_handles"])
    owned_shapes = [shape for shape in group if shape.handle in owned]
    closed_count = sum(shape.closed for shape in owned_shapes)
    circle_count = sum(shape.kind == "circle" for shape in owned_shapes)
    box = merge_bbox(shape.bbox for shape in owned_shapes)
    span = max(box[2] - box[0], box[3] - box[1])
    if family not in TYPE_BY_FAMILY:
        return False, "template_family_not_in_equipment_scope"
    if span < raw_h * 0.20 or span > raw_h * 12.0:
        return False, "component_scale_outside_h_range"
    if family == "PowerTransformer":
        if circle_count < 2:
            return False, "transformer_requires_two_circle_primitives"
        if score < 55.0 or margin < 3.0:
            return False, "transformer_template_evidence_too_weak"
        return True, "two_circle_transformer_template_accepted"
    if score < MINIMUM_TEMPLATE_SCORE:
        return False, "template_score_below_threshold"
    if margin < MINIMUM_FAMILY_MARGIN and score < 90.0:
        return False, "template_family_margin_too_small"
    if len(owned) == 1:
        if (
            family == "ConnectivePoint"
            and closed_count
            and not circle_count
            and raw_h * 0.50 <= span <= raw_h * 3.0
        ):
            return True, "single_closed_connective_symbol"
        if family in SWITCH_FAMILIES and closed_count and score >= 88.0:
            return True, "single_closed_switch_symbol"
        return False, "single_generic_primitive_rejected"
    return True, "full_template_candidate_accepted"


def template_record_index(
    library: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for item in library.get("templates", []):
        key = (str(item.get("symbol_id")), str(item.get("family")))
        output.setdefault(key, item)
    return output


def build_equipment(
    groups: list[list[Shape]],
    group_results: list[dict[str, Any]],
    priors: list[dict[str, Any] | None],
    library: dict[str, Any],
    raw_h: float,
    reserved_handles: set[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]:
    equipment: list[dict[str, Any]] = []
    consumed: set[str] = set(reserved_handles or set())
    rejected: list[dict[str, Any]] = []
    records = template_record_index(library)
    ranked_items = sorted(
        zip(groups, group_results, priors),
        key=lambda item: (
            -float(item[1]["top"]["combined_score"]),
            -len(item[1]["candidate"]["owned_handles"]),
        ),
    )
    for group, result, prior in ranked_items:
        accepted, reason = candidate_acceptable(group, result, raw_h)
        selected_handles = set(result["candidate"]["owned_handles"])
        if selected_handles & consumed:
            accepted = False
            reason = "overlaps_higher_scoring_component"
        if not accepted:
            rejected.append(
                {
                    "top_family": result["top"]["family"],
                    "top_template": result["top"]["template_id"],
                    "score": result["top"]["combined_score"],
                    "family_margin": result["family_margin"],
                    "selected_handles": sorted(selected_handles),
                    "reason": reason,
                }
            )
            continue

        family = str(result["top"]["family"])
        physical_type = TYPE_BY_FAMILY[family]
        selected_shapes = [
            shape for shape in group if shape.handle in selected_handles
        ]
        if not selected_shapes:
            continue
        box = merge_bbox(shape.bbox for shape in selected_shapes)
        template = records.get(
            (str(result["top"]["template_id"]), family),
            {},
        )
        terminal_count = int(template.get("terminal_count") or 1)
        equipment_id = f"EQ{len(equipment) + 1:05d}"
        confidence = (
            "high"
            if result["top"]["combined_score"] >= 88.0
            and result["family_margin"] >= 5.0
            else "medium"
        )
        equipment.append(
            {
                "equipment_id": equipment_id,
                "type": physical_type,
                "new_type": physical_type,
                "physical_type": physical_type,
                "type_cn": TYPE_CN.get(physical_type, physical_type),
                "name": TYPE_CN.get(physical_type, physical_type),
                "center": [
                    round((box[0] + box[2]) / 2.0, 6),
                    round((box[1] + box[3]) / 2.0, 6),
                ],
                "bbox": [round(value, 6) for value in box],
                "source_handles": sorted(selected_handles),
                "confidence": confidence,
                "confidence_score": round(
                    float(result["top"]["combined_score"]) / 100.0,
                    4,
                ),
                "basis": (
                    "DXF局部几何与全部标准模板逐一相似度比较；"
                    "文字仅提供最多5分的通用类别先验。"
                ),
                "match_mode": "all_template_geometry_similarity",
                "matched_templates": [result["top"]["template_id"]],
                "template_family": family,
                "template_name": result["top"]["template_name"],
                "template_score": result["top"]["combined_score"],
                "geometry_score": result["top"]["geometry_score"],
                "primitive_count_score": result["top"][
                    "primitive_count_score"
                ],
                "family_margin": result["family_margin"],
                "best_transform": result["top"]["best_transform"],
                "terminal_count_from_template": terminal_count,
                "text_prior": prior,
                "top5_families": result["family_alternatives"],
                "recognition_decision": reason,
            }
        )
        consumed.update(selected_handles)
    equipment.sort(key=lambda item: (item["center"][0], item["center"][1]))
    for index, item in enumerate(equipment, 1):
        item["equipment_id"] = f"EQ{index:05d}"
    return equipment, consumed, rejected


def complete_transformer_components(
    equipment: list[dict[str, Any]],
    shapes: list[Shape],
    raw_h: float,
) -> set[str]:
    """Assign the complete local transformer motif to the transformer.

    Template matching intentionally uses the most distinctive two winding
    circles.  After acceptance, this pass claims the aligned phase circles,
    triangle and short stems that belong to the same symbol.  Claimed geometry
    cannot later become a spurious standalone component or conductor.
    """
    newly_consumed: set[str] = set()
    shape_by_handle = {shape.handle: shape for shape in shapes}
    for item in equipment:
        if item.get("physical_type") != "PowerTransformer":
            continue
        owned = [
            shape_by_handle[handle]
            for handle in item.get("source_handles") or []
            if handle in shape_by_handle
        ]
        main_circles = sorted(
            (shape for shape in owned if shape.kind == "circle"),
            key=lambda shape: -max(
                shape.bbox[2] - shape.bbox[0],
                shape.bbox[3] - shape.bbox[1],
            ),
        )[:2]
        if len(main_circles) < 2:
            continue
        main_box = merge_bbox(shape.bbox for shape in main_circles)
        center_x = (main_box[0] + main_box[2]) / 2.0
        diameter = statistics.median(
            max(
                shape.bbox[2] - shape.bbox[0],
                shape.bbox[3] - shape.bbox[1],
            )
            for shape in main_circles
        )
        diameter = max(diameter, raw_h)
        zone = (
            center_x - diameter * 1.05,
            main_box[1] - diameter * 2.30,
            center_x + diameter * 1.05,
            main_box[3] + diameter * 0.15,
        )
        claimed = list(main_circles)
        for shape in shapes:
            if annotation_layer(shape.layer):
                continue
            shape_center = bbox_center(shape.bbox)
            if not (
                zone[0] <= shape_center[0] <= zone[2]
                and zone[1] <= shape_center[1] <= zone[3]
            ):
                continue
            span = max(
                shape.bbox[2] - shape.bbox[0],
                shape.bbox[3] - shape.bbox[1],
            )
            closed_detail = (
                shape.kind in {"circle", "polygon", "arc"}
                or shape.closed
            ) and span <= diameter * 1.20
            short_stem = (
                shape.kind in {"line", "polyline"}
                and span <= diameter * 3.0
                and abs(shape_center[0] - center_x)
                <= diameter * 0.25
            )
            if closed_detail or short_stem:
                claimed.append(shape)
        claimed_by_handle = {
            shape.handle: shape for shape in claimed
        }
        completed = list(claimed_by_handle.values())
        completed_box = merge_bbox(shape.bbox for shape in completed)
        handles = sorted(claimed_by_handle)
        original = set(item.get("source_handles") or [])
        additions = set(handles) - original
        newly_consumed.update(additions)
        item["source_handles"] = handles
        item["component_owned_handles"] = handles
        item["absorbed_component_handles"] = sorted(additions)
        item["bbox"] = [
            round(value, 6) for value in completed_box
        ]
        item["center"] = [
            round((completed_box[0] + completed_box[2]) / 2.0, 6),
            round((completed_box[1] + completed_box[3]) / 2.0, 6),
        ]
        item["terminal_count_from_template"] = 2
        item["component_completion"] = {
            "method": "scale_adaptive_aligned_motif_ownership",
            "main_winding_circle_count": 2,
            "absorbed_primitive_count": len(additions),
            "local_diameter": round(diameter, 6),
        }
    return newly_consumed


def attach_visible_equipment_names(
    equipment: list[dict[str, Any]],
    texts: list[dict[str, Any]],
    raw_h: float,
) -> None:
    """Attach nearby visible designations after geometry has fixed the type."""
    transformer_labels = [
        text
        for text in texts
        if (
            "变" in str(text.get("text") or "")
            and any(
                token in str(text.get("text") or "")
                for token in ("kVA", "KVA", "#专", "专变")
            )
            and "柜" not in str(text.get("text") or "")
        )
    ]
    used: set[str] = set()
    for item in equipment:
        if item.get("physical_type") != "PowerTransformer":
            continue
        center = tuple(map(float, item["center"]))
        candidates = []
        for text in transformer_labels:
            if text["handle"] in used:
                continue
            point = (float(text["x"]), float(text["y"]))
            gap = math.dist(center, point)
            if gap <= raw_h * 40.0:
                candidates.append((gap, text))
        if not candidates:
            continue
        _, selected = min(candidates, key=lambda item: item[0])
        used.add(selected["handle"])
        visible = str(selected["text"]).replace("\n", " ").strip()
        item["name"] = visible
        item["visible_label"] = visible
        item["text_label_evidence"] = {
            "handle": selected["handle"],
            "text": selected["text"],
            "point": [selected["x"], selected["y"]],
            "distance_h": round(
                math.dist(
                    center,
                    (float(selected["x"]), float(selected["y"])),
                )
                / raw_h,
                3,
            ),
        }


def shape_raw_vertices(shape: Shape) -> list[tuple[float, float]]:
    entity = shape.entity
    if entity.dxftype() == "LINE":
        return [
            (float(entity.dxf.start.x), float(entity.dxf.start.y)),
            (float(entity.dxf.end.x), float(entity.dxf.end.y)),
        ]
    if entity.dxftype() == "LWPOLYLINE":
        elevation = float(entity.dxf.elevation)
        ocs = entity.ocs()
        return [
            (
                float(point.x),
                float(point.y),
            )
            for point in (
                ocs.to_wcs((float(x), float(y), elevation))
                for x, y in entity.get_points("xy")
            )
        ]
    if entity.dxftype() == "POLYLINE":
        return [
            (float(vertex.dxf.location.x), float(vertex.dxf.location.y))
            for vertex in entity.vertices
        ]
    return []


def build_segments(
    shapes: list[Shape],
    consumed_handles: set[str],
    raw_h: float,
) -> tuple[list[Segment], dict[str, int]]:
    layer_counts: Counter[str] = Counter()
    segments: list[Segment] = []
    seen: set[tuple[int, int, int, int]] = set()
    quantum = max(raw_h * 0.01, 1e-8)
    for shape in shapes:
        if shape.handle in consumed_handles:
            continue
        if annotation_layer(shape.layer):
            continue
        if shape.kind not in {"line", "polyline"} or shape.closed:
            continue
        vertices = shape_raw_vertices(shape)
        for segment_index, (start, end) in enumerate(
            zip(vertices, vertices[1:])
        ):
            segment = Segment(
                segment_id=f"{shape.handle}:{segment_index}",
                evidence_id=shape.handle,
                layer=shape.layer,
                start=start,
                end=end,
            )
            if segment.length < raw_h * MINIMUM_WIRE_H:
                continue
            ordered = sorted((start, end))
            key = tuple(
                round(value / quantum)
                for point in ordered
                for value in point
            )
            if key in seen:
                continue
            seen.add(key)
            segments.append(segment)
            layer_counts[shape.layer] += 1
    return segments, dict(layer_counts)


def point_distance(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def point_segment_projection(
    point: tuple[float, float],
    segment: Segment,
) -> tuple[float, tuple[float, float], float]:
    dx = segment.end[0] - segment.start[0]
    dy = segment.end[1] - segment.start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return point_distance(point, segment.start), segment.start, 0.0
    parameter = (
        (point[0] - segment.start[0]) * dx
        + (point[1] - segment.start[1]) * dy
    ) / denominator
    parameter = max(0.0, min(1.0, parameter))
    projected = (
        segment.start[0] + parameter * dx,
        segment.start[1] + parameter * dy,
    )
    return point_distance(point, projected), projected, parameter


def bbox_overlaps(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    return not (
        left[2] + tolerance < right[0]
        or right[2] + tolerance < left[0]
        or left[3] + tolerance < right[1]
        or right[3] + tolerance < left[1]
    )


def build_conductor_graph(
    segments: list[Segment],
    marker_shapes: list[Shape],
    raw_h: float,
) -> tuple[UnionFind, list[dict[str, Any]]]:
    union = UnionFind(len(segments))
    snap = raw_h * ENDPOINT_SNAP_H
    t_tolerance = raw_h * T_JUNCTION_H
    dot_radius = raw_h * DOT_RADIUS_H
    markers = []
    for shape in marker_shapes:
        if shape.kind != "circle":
            continue
        try:
            radius = max(
                shape.bbox[2] - shape.bbox[0],
                shape.bbox[3] - shape.bbox[1],
            ) / 2.0
            if radius <= dot_radius:
                markers.append(bbox_center(shape.bbox))
        except Exception:
            continue
    events: dict[tuple[str, int, int], dict[str, Any]] = {}
    quantum = max(raw_h * 0.005, 1e-8)

    def add_event(
        kind: str,
        point: tuple[float, float],
        left: Segment,
        right: Segment,
        state: str,
    ) -> None:
        key = (
            kind,
            round(point[0] / quantum),
            round(point[1] / quantum),
        )
        if key not in events:
            events[key] = {
                "event_type": kind,
                "kind": kind,
                "point": [round(point[0], 6), round(point[1], 6)],
                "source_handles": sorted(
                    {left.evidence_id, right.evidence_id}
                ),
                "state": state,
            }

    for left_index, right_index in combinations(range(len(segments)), 2):
        left = segments[left_index]
        right = segments[right_index]
        if not bbox_overlaps(left.bbox, right.bbox, t_tolerance):
            continue

        endpoint_pairs = [
            (point_distance(a, b), a, b)
            for a in (left.start, left.end)
            for b in (right.start, right.end)
        ]
        endpoint_gap, left_point, right_point = min(endpoint_pairs)
        if endpoint_gap <= snap:
            point = (
                (left_point[0] + right_point[0]) / 2.0,
                (left_point[1] + right_point[1]) / 2.0,
            )
            union.union(left_index, right_index)
            add_event(
                "endpoint_connection",
                point,
                left,
                right,
                "connected",
            )
            continue

        left_vector = (
            left.end[0] - left.start[0],
            left.end[1] - left.start[1],
        )
        right_vector = (
            right.end[0] - right.start[0],
            right.end[1] - right.start[1],
        )
        offset = (
            right.start[0] - left.start[0],
            right.start[1] - left.start[1],
        )

        def cross(
            first: tuple[float, float],
            second: tuple[float, float],
        ) -> float:
            return first[0] * second[1] - first[1] * second[0]

        denominator = cross(left_vector, right_vector)
        parallel_limit = (
            left.length
            * right.length
            * math.sin(math.radians(AXIS_ANGLE_DEGREES))
        )
        if abs(denominator) <= parallel_limit:
            # Parallel strokes connect only when they are also collinear and
            # their projected intervals overlap.
            line_distance = abs(cross(offset, left_vector)) / max(
                left.length,
                1e-12,
            )
            if line_distance > snap:
                continue
            unit = (
                left_vector[0] / left.length,
                left_vector[1] / left.length,
            )
            projections = [
                (
                    (point[0] - left.start[0]) * unit[0]
                    + (point[1] - left.start[1]) * unit[1]
                )
                for point in (right.start, right.end)
            ]
            right_min, right_max = sorted(projections)
            if max(0.0, right_min) <= min(left.length, right_max) + snap:
                union.union(left_index, right_index)
            continue

        left_parameter = cross(offset, right_vector) / denominator
        right_parameter = cross(offset, left_vector) / denominator
        left_slack = t_tolerance / max(left.length, 1e-12)
        right_slack = t_tolerance / max(right.length, 1e-12)
        if not (
            -left_slack <= left_parameter <= 1.0 + left_slack
            and -right_slack <= right_parameter <= 1.0 + right_slack
        ):
            continue
        point = (
            left.start[0] + left_parameter * left_vector[0],
            left.start[1] + left_parameter * left_vector[1],
        )
        left_endpoint = min(
            point_distance(point, left.start),
            point_distance(point, left.end),
        )
        right_endpoint = min(
            point_distance(point, right.start),
            point_distance(point, right.end),
        )
        if left_endpoint <= t_tolerance or right_endpoint <= t_tolerance:
            union.union(left_index, right_index)
            add_event("t_junction", point, left, right, "connected")
            continue
        has_marker = any(
            point_distance(point, marker) <= snap for marker in markers
        )
        if has_marker:
            union.union(left_index, right_index)
            add_event("x_connected", point, left, right, "connected")
        else:
            add_event("x_not_connected", point, left, right, "not_connected")
    return union, list(events.values())


def boundary_port_candidates(
    equipment: dict[str, Any],
) -> list[tuple[float, float]]:
    left, bottom, right, top = map(float, equipment["bbox"])
    center_x = (left + right) / 2.0
    center_y = (bottom + top) / 2.0
    return [
        (center_x, top),
        (center_x, bottom),
        (left, center_y),
        (right, center_y),
    ]


def attach_terminals(
    equipment: list[dict[str, Any]],
    segments: list[Segment],
    union: UnionFind,
    raw_h: float,
) -> tuple[list[dict[str, Any]], dict[int, list[str]], list[str]]:
    terminals: list[dict[str, Any]] = []
    root_terminals: dict[int, list[str]] = defaultdict(list)
    issues: list[str] = []
    strict_distance = raw_h * TERMINAL_ATTACH_H
    review_distance = raw_h * 2.0
    for item in equipment:
        count = max(1, min(int(item["terminal_count_from_template"]), 4))
        candidates = boundary_port_candidates(item)
        ranked_ports = []
        for port in candidates:
            ranked_segments = []
            for segment_index, segment in enumerate(segments):
                gap, projection, _ = point_segment_projection(port, segment)
                ranked_segments.append(
                    (gap, segment_index, projection, segment)
                )
            if ranked_segments:
                ranked_ports.append(
                    (*min(ranked_segments, key=lambda row: row[0]), port)
                )
        ranked_ports.sort(key=lambda row: row[0])
        used_segments: set[int] = set()
        selected = []
        for ranked in ranked_ports:
            gap, segment_index, projection, segment, port = ranked
            if segment_index in used_segments and len(segments) > 1:
                continue
            selected.append(ranked)
            used_segments.add(segment_index)
            if len(selected) >= count:
                break
        if not selected:
            issues.append(f"{item['equipment_id']}: no conductor available")
            continue
        for gap, segment_index, projection, segment, port in selected:
            if gap > review_distance:
                issues.append(
                    f"{item['equipment_id']}: terminal too far from conductor "
                    f"({gap / raw_h:.3f}H)"
                )
                continue
            terminal_id = f"T{len(terminals) + 1:05d}"
            root = union.find(segment_index)
            node_id = f"CN_ROOT_{root}"
            terminal = {
                "terminal_id": terminal_id,
                "equipment_id": item["equipment_id"],
                "role": "template_port",
                "point": [round(port[0], 6), round(port[1], 6)],
                "port_position": [
                    round(port[0], 6),
                    round(port[1], 6),
                ],
                "wire_attach_position": [
                    round(projection[0], 6),
                    round(projection[1], 6),
                ],
                "attachment_distance": round(gap, 6),
                "attachment_distance_h": round(gap / raw_h, 6),
                "conductor_handle": segment.evidence_id,
                "conductor_segment_id": segment.segment_id,
                "connectivity_node": node_id,
                "confidence": (
                    "high" if gap <= strict_distance else "review"
                ),
                "source": "template_boundary_to_nearest_filtered_conductor",
            }
            terminals.append(terminal)
            root_terminals[root].append(terminal_id)
    return terminals, root_terminals, issues


def build_nodes_and_edges(
    equipment: list[dict[str, Any]],
    terminals: list[dict[str, Any]],
    root_terminals: dict[int, list[str]],
    segments: list[Segment],
    union: UnionFind,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    terminal_by_id = {item["terminal_id"]: item for item in terminals}
    equipment_by_terminal = {
        item["terminal_id"]: item["equipment_id"] for item in terminals
    }
    segments_by_root: dict[int, list[str]] = defaultdict(list)
    for index, segment in enumerate(segments):
        segments_by_root[union.find(index)].append(segment.evidence_id)
    nodes = []
    active_roots: set[int] = set()
    node_id_by_root = {}
    for root, terminal_ids in sorted(root_terminals.items()):
        if not terminal_ids:
            continue
        active_roots.add(root)
        node_id = f"CN{len(nodes) + 1:05d}"
        node_id_by_root[root] = node_id
        equipment_ids = sorted(
            {equipment_by_terminal[terminal_id] for terminal_id in terminal_ids}
        )
        nodes.append(
            {
                "connectivity_node_id": node_id,
                "terminal_ids": sorted(terminal_ids),
                "equipment_ids": equipment_ids,
                "conductor_handles": sorted(
                    set(segments_by_root.get(root, []))
                ),
                "confidence": "automatic_geometry",
            }
        )
        for terminal_id in terminal_ids:
            terminal_by_id[terminal_id]["connectivity_node"] = node_id

    edge_set: set[tuple[str, str, str]] = set()
    for node in nodes:
        for left, right in combinations(node["equipment_ids"], 2):
            edge_set.add(
                (left, right, node["connectivity_node_id"])
            )
    edges = [
        {
            "source_equipment": left,
            "target_equipment": right,
            "via_connectivity_node": node_id,
            "state": "automatic_geometry",
        }
        for left, right, node_id in sorted(edge_set)
    ]
    return nodes, edges, active_roots


def filter_active_events(
    events: list[dict[str, Any]],
    segments: list[Segment],
    union: UnionFind,
    active_roots: set[int],
) -> list[dict[str, Any]]:
    roots_by_handle: dict[str, set[int]] = defaultdict(set)
    for index, segment in enumerate(segments):
        roots_by_handle[segment.evidence_id].add(union.find(index))
    output = []
    for event in events:
        roots = {
            root
            for handle in event.get("source_handles") or []
            for root in roots_by_handle.get(handle, set())
        }
        if roots & active_roots:
            output.append(copy.deepcopy(event))
    output.sort(
        key=lambda item: (
            item["point"][0],
            item["point"][1],
            item["event_type"],
        )
    )
    for index, item in enumerate(output, 1):
        item["crossing_id"] = f"X{index:05d}"
    return output


def normalize_connection_events(
    events: list[dict[str, Any]],
    raw_h: float,
) -> list[dict[str, Any]]:
    """Resolve multiple pairwise events reported at the same coordinate."""
    quantum = max(raw_h * 0.02, 1e-8)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        point = event["point"]
        grouped[
            (
                round(float(point[0]) / quantum),
                round(float(point[1]) / quantum),
            )
        ].append(event)
    priority = {
        "t_junction": 5,
        "endpoint_connection": 4,
        "x_connected": 3,
        "x_not_connected": 1,
    }
    output = []
    for same_point in grouped.values():
        chosen = max(
            same_point,
            key=lambda item: (
                priority.get(str(item.get("event_type")), 0),
                item.get("state") == "connected",
            ),
        )
        merged = copy.deepcopy(chosen)
        merged["source_handles"] = sorted(
            {
                handle
                for item in same_point
                for handle in item.get("source_handles") or []
            }
        )
        raw_types = sorted(
            {str(item.get("event_type")) for item in same_point}
        )
        raw_states = sorted(
            {str(item.get("state")) for item in same_point}
        )
        merged["raw_event_types"] = raw_types
        merged["raw_states"] = raw_states
        merged["conflict_resolved"] = len(raw_states) > 1
        if any(item.get("state") == "connected" for item in same_point):
            merged["state"] = "connected"
        output.append(merged)
    output.sort(
        key=lambda item: (
            item["point"][0],
            item["point"][1],
            item["event_type"],
        )
    )
    for index, item in enumerate(output, 1):
        item["crossing_id"] = f"X{index:05d}"
    return output


def cabinet_semantics(value: str) -> tuple[str, str] | None:
    compact = "".join(str(value).split())
    if "柜" not in compact:
        return None
    if any(
        token in compact
        for token in ("引来", "引至", "来自", "接至", "去往")
    ):
        return None
    if "环网柜" in compact:
        return "RingMainUnit", "专用环网柜"
    if "DTU柜" in compact:
        return "DTUCabinet", "公用DTU柜"
    if "智能" in compact:
        return "SmartPublicCabinet", "公用智能柜"
    if "公用柜" in compact:
        return "PublicCabinet", "公用柜"
    return "Cabinet", "柜体"


def interval_distance(
    value: float,
    left: float,
    right: float,
) -> float:
    if left <= value <= right:
        return 0.0
    return min(abs(value - left), abs(value - right))


def segment_midpoint(segment: Segment) -> tuple[float, float]:
    return (
        (segment.start[0] + segment.end[0]) / 2.0,
        (segment.start[1] + segment.end[1]) / 2.0,
    )


def detect_cabinet_containers(
    texts: list[dict[str, Any]],
    shapes: list[Shape],
    segments: list[Segment],
    union: UnionFind,
    equipment: list[dict[str, Any]],
    raw_h: float,
) -> list[dict[str, Any]]:
    """Detect cabinet containers from labels, frames and local bus geometry."""
    labels = []
    for text in texts:
        semantic = cabinet_semantics(str(text.get("text") or ""))
        if semantic is None:
            continue
        family, type_cn = semantic
        labels.append(
            {
                **text,
                "container_family": family,
                "container_type_cn": type_cn,
            }
        )

    frames = []
    for shape in shapes:
        if not (shape.closed or shape.kind == "polygon"):
            continue
        width = shape.bbox[2] - shape.bbox[0]
        height = shape.bbox[3] - shape.bbox[1]
        if (
            raw_h * 12.0 <= width <= raw_h * 80.0
            and raw_h * 12.0 <= height <= raw_h * 80.0
            and 0.35 <= width / max(height, 1e-9) <= 3.0
        ):
            frames.append(shape)

    horizontal_buses = [
        (index, segment)
        for index, segment in enumerate(segments)
        if segment.horizontal and segment.length >= raw_h * 15.0
    ]
    containers = []
    used_frames: set[str] = set()
    for label in labels:
        x = float(label["x"])
        y = float(label["y"])
        height = max(float(label.get("height") or 0.0), raw_h)
        frame_candidates = []
        for frame in frames:
            left, bottom, right, top = frame.bbox
            if (
                left - raw_h * 10.0 <= x <= right + raw_h * 10.0
                and bottom - raw_h * 3.0
                <= y
                <= top + raw_h * 12.0
            ):
                gap = interval_distance(x, left, right) + abs(y - top)
                frame_candidates.append((gap, frame))
        frame = (
            min(frame_candidates, key=lambda item: item[0])[1]
            if frame_candidates
            else None
        )
        bus_indices: list[int] = []
        basis = []
        if frame is not None and frame.handle not in used_frames:
            used_frames.add(frame.handle)
            box = tuple(frame.bbox)
            basis.append("closed_cabinet_frame")
            for index, segment in horizontal_buses:
                midpoint = segment_midpoint(segment)
                if (
                    box[0] - raw_h <= midpoint[0] <= box[2] + raw_h
                    and box[1] - raw_h <= midpoint[1] <= box[3] + raw_h
                ):
                    bus_indices.append(index)
        else:
            primary_candidates = []
            for index, segment in horizontal_buses:
                bus_y = (segment.start[1] + segment.end[1]) / 2.0
                vertical_gap = y - bus_y
                horizontal_gap = interval_distance(
                    x,
                    segment.bbox[0],
                    segment.bbox[2],
                )
                if (
                    raw_h * 2.0 <= vertical_gap <= raw_h * 60.0
                    and horizontal_gap <= raw_h * 30.0
                ):
                    primary_candidates.append(
                        (
                            vertical_gap + horizontal_gap * 1.5,
                            index,
                            segment,
                        )
                    )
            if not primary_candidates:
                continue
            _, primary_index, primary = min(
                primary_candidates,
                key=lambda item: item[0],
            )
            bus_indices = [primary_index]
            target_y = (
                primary.start[1] + primary.end[1]
            ) / 2.0
            intervals = [
                (primary.bbox[0], primary.bbox[2])
            ]
            changed = True
            while changed:
                changed = False
                current_left = min(item[0] for item in intervals)
                current_right = max(item[1] for item in intervals)
                for index, segment in horizontal_buses:
                    if index in bus_indices:
                        continue
                    bus_y = (
                        segment.start[1] + segment.end[1]
                    ) / 2.0
                    gap = max(
                        current_left - segment.bbox[2],
                        segment.bbox[0] - current_right,
                        0.0,
                    )
                    if (
                        abs(bus_y - target_y) <= raw_h * 0.20
                        and gap <= raw_h * 40.0
                    ):
                        bus_indices.append(index)
                        intervals.append(
                            (segment.bbox[0], segment.bbox[2])
                        )
                        changed = True
            bus_segments = [segments[index] for index in bus_indices]
            left = min(segment.bbox[0] for segment in bus_segments)
            right = max(segment.bbox[2] for segment in bus_segments)
            bottom = target_y - raw_h * 18.0
            top = max(y + height, target_y + raw_h * 4.0)
            box = (
                left - raw_h * 2.0,
                bottom,
                right + raw_h * 2.0,
                top,
            )
            basis.append("label_associated_horizontal_bus")

        roots = sorted(
            {union.find(index) for index in bus_indices}
        )
        member_ids = []
        for item in equipment:
            cx, cy = map(float, item["center"])
            if (
                box[0] <= cx <= box[2]
                and box[1] <= cy <= box[3]
                and item.get("physical_type") != "PowerTransformer"
            ):
                member_ids.append(item["equipment_id"])
        containers.append(
            {
                "container_id": "",
                "type": "Cabinet",
                "container_family": label["container_family"],
                "type_cn": label["container_type_cn"],
                "name": str(label["text"]).replace("\n", " "),
                "bbox": [round(value, 6) for value in box],
                "center": [
                    round((box[0] + box[2]) / 2.0, 6),
                    round((box[1] + box[3]) / 2.0, 6),
                ],
                "label_evidence": {
                    "handle": label["handle"],
                    "text": label["text"],
                    "point": [x, y],
                },
                "frame_handle": frame.handle if frame is not None else None,
                "bus_segment_ids": [
                    segments[index].segment_id for index in bus_indices
                ],
                "bus_roots": roots,
                "member_equipment_ids": sorted(member_ids),
                "recognition_basis": basis,
                "confidence": "high" if frame is not None else "medium",
            }
        )
    containers.sort(
        key=lambda item: (item["center"][1], item["center"][0])
    )
    for index, container in enumerate(containers, 1):
        container["container_id"] = f"CAB{index:05d}"
    return containers


def vertical_path_coverage(
    x: float,
    bottom: float,
    top: float,
    segments: list[Segment],
    raw_h: float,
) -> tuple[float, list[str]]:
    intervals = []
    evidence = []
    for segment in segments:
        if not segment.vertical:
            continue
        sx = (segment.start[0] + segment.end[0]) / 2.0
        if abs(sx - x) > raw_h * 0.75:
            continue
        low = max(bottom, segment.bbox[1])
        high = min(top, segment.bbox[3])
        if high <= low:
            continue
        intervals.append((low, high))
        evidence.append(segment.segment_id)
    if not intervals or top <= bottom:
        return 0.0, []
    intervals.sort()
    merged = [list(intervals[0])]
    for low, high in intervals[1:]:
        if low <= merged[-1][1] + raw_h * 0.25:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    covered = sum(high - low for low, high in merged)
    return min(1.0, covered / (top - bottom)), sorted(set(evidence))


def build_engineering_topology(
    containers: list[dict[str, Any]],
    equipment: list[dict[str, Any]],
    segments: list[Segment],
    union: UnionFind,
    raw_h: float,
) -> dict[str, Any]:
    transformers = [
        item
        for item in equipment
        if item.get("physical_type") == "PowerTransformer"
    ]
    engineering_equipment = []
    engineering_id_by_source: dict[str, str] = {}
    for container in containers:
        engineering_id = f"ENG{len(engineering_equipment) + 1:05d}"
        engineering_id_by_source[container["container_id"]] = engineering_id
        engineering_equipment.append(
            {
                "engineering_equipment_id": engineering_id,
                "type": container["container_family"],
                "type_cn": container["type_cn"],
                "name": container["name"],
                "bbox": container["bbox"],
                "center": container["center"],
                "source_container_id": container["container_id"],
                "detailed_member_equipment_ids": container[
                    "member_equipment_ids"
                ],
                "confidence": container["confidence"],
            }
        )
    for transformer in transformers:
        engineering_id = f"ENG{len(engineering_equipment) + 1:05d}"
        engineering_id_by_source[
            transformer["equipment_id"]
        ] = engineering_id
        engineering_equipment.append(
            {
                "engineering_equipment_id": engineering_id,
                "type": "PowerTransformer",
                "type_cn": "变压器",
                "name": transformer.get("name") or "变压器",
                "bbox": transformer["bbox"],
                "center": transformer["center"],
                "source_detailed_equipment_id": transformer["equipment_id"],
                "confidence": transformer["confidence"],
            }
        )

    segment_index_by_id = {
        segment.segment_id: index
        for index, segment in enumerate(segments)
    }
    relations = []
    transformer_bus_assignment: dict[str, tuple[str, int]] = {}
    for transformer in transformers:
        x = float(transformer["center"][0])
        transformer_top = float(transformer["bbox"][3])
        candidates = []
        for container in containers:
            for segment_id in container["bus_segment_ids"]:
                segment_index = segment_index_by_id.get(segment_id)
                if segment_index is None:
                    continue
                segment = segments[segment_index]
                bus_y = (
                    segment.start[1] + segment.end[1]
                ) / 2.0
                vertical_gap = bus_y - transformer_top
                horizontal_gap = interval_distance(
                    x,
                    segment.bbox[0],
                    segment.bbox[2],
                )
                if (
                    0.0 <= vertical_gap <= raw_h * 50.0
                    and horizontal_gap <= raw_h * 2.0
                ):
                    candidates.append(
                        (
                            vertical_gap + horizontal_gap * 2.0,
                            container,
                            segment_index,
                            segment,
                        )
                    )
        if not candidates:
            continue
        _, container, segment_index, bus = min(
            candidates,
            key=lambda item: item[0],
        )
        bus_y = (bus.start[1] + bus.end[1]) / 2.0
        coverage, path_evidence = vertical_path_coverage(
            x,
            transformer_top,
            bus_y,
            segments,
            raw_h,
        )
        cabinet_engineering_id = engineering_id_by_source[
            container["container_id"]
        ]
        transformer_engineering_id = engineering_id_by_source[
            transformer["equipment_id"]
        ]
        root = union.find(segment_index)
        relation_id = f"ER{len(relations) + 1:05d}"
        relations.append(
            {
                "relation_id": relation_id,
                "from_equipment": cabinet_engineering_id,
                "to_equipment": transformer_engineering_id,
                "relation": "feeds",
                "via_bus_root": root,
                "state": "automatic_geometry_and_hierarchy",
                "vertical_path_coverage": round(coverage, 3),
                "evidence_segment_ids": sorted(
                    set(path_evidence + [bus.segment_id])
                ),
                "confidence": "high" if coverage >= 0.55 else "medium",
            }
        )
        transformer_bus_assignment[transformer["equipment_id"]] = (
            container["container_id"],
            root,
        )

    # At engineering level, adjacent cabinets connected by the same external
    # route form one network group even though their internal switches split
    # the strict conductor graph.  Determine those groups only from conductor
    # roots touching two container regions in the same drawing band.
    container_union = UnionFind(len(containers))
    roots_touching_containers: dict[int, set[int]] = defaultdict(set)
    for segment_index, segment in enumerate(segments):
        root = union.find(segment_index)
        for container_index, container in enumerate(containers):
            if bbox_overlaps(
                segment.bbox,
                tuple(container["bbox"]),
                raw_h * 2.0,
            ):
                roots_touching_containers[root].add(container_index)
    route_evidence_by_pair: dict[tuple[int, int], set[int]] = defaultdict(set)
    for root, touched in roots_touching_containers.items():
        for left, right in combinations(sorted(touched), 2):
            left_y = float(containers[left]["center"][1])
            right_y = float(containers[right]["center"][1])
            if abs(left_y - right_y) > raw_h * 50.0:
                continue
            container_union.union(left, right)
            route_evidence_by_pair[(left, right)].add(root)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(containers)):
        clusters[container_union.find(index)].append(index)
    relations_by_cabinet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        relations_by_cabinet[relation["from_equipment"]].append(relation)

    nodes = []
    for member_indices in sorted(
        clusters.values(),
        key=lambda members: min(
            containers[index]["center"][1] for index in members
        ),
    ):
        if len(member_indices) > 1:
            attached = {
                engineering_id_by_source[
                    containers[index]["container_id"]
                ]
                for index in member_indices
            }
            evidence_roots = set()
            for left, right in combinations(sorted(member_indices), 2):
                evidence_roots.update(
                    route_evidence_by_pair.get((left, right), set())
                )
            for cabinet_id in list(attached):
                attached.update(
                    relation["to_equipment"]
                    for relation in relations_by_cabinet.get(
                        cabinet_id,
                        [],
                    )
                )
            nodes.append(
                {
                    "engineering_node_id": "",
                    "conductor_roots": sorted(evidence_roots),
                    "connected_equipment_ids": sorted(attached),
                    "state": "automatic_inter_cabinet_route",
                }
            )
            continue

        container = containers[member_indices[0]]
        cabinet_id = engineering_id_by_source[
            container["container_id"]
        ]
        relations_for_cabinet = relations_by_cabinet.get(
            cabinet_id,
            [],
        )
        roots = sorted(
            set(container["bus_roots"])
            | {
                int(relation["via_bus_root"])
                for relation in relations_for_cabinet
            }
        )
        if not roots:
            nodes.append(
                {
                    "engineering_node_id": "",
                    "conductor_roots": [],
                    "connected_equipment_ids": [cabinet_id],
                    "state": "automatic_isolated_cabinet",
                }
            )
            continue
        for root in roots:
            attached = {cabinet_id}
            attached.update(
                relation["to_equipment"]
                for relation in relations_for_cabinet
                if int(relation["via_bus_root"]) == root
            )
            nodes.append(
                {
                    "engineering_node_id": "",
                    "conductor_roots": [root],
                    "connected_equipment_ids": sorted(attached),
                    "state": "automatic_engineering_bus_section",
                }
            )
    for index, node in enumerate(nodes, 1):
        node["engineering_node_id"] = f"EN{index:05d}"
    hierarchy = [
        {
            "container_id": container["container_id"],
            "engineering_equipment_id": engineering_id_by_source[
                container["container_id"]
            ],
            "detailed_member_equipment_ids": container[
                "member_equipment_ids"
            ],
        }
        for container in containers
    ]
    return {
        "schema_version": "two-level-engineering-topology-v1",
        "equipment": engineering_equipment,
        "containers": containers,
        "equipment_hierarchy": hierarchy,
        "connectivity_nodes": nodes,
        "device_relations": relations,
        "statistics": {
            "cabinet_count": len(containers),
            "transformer_count": len(transformers),
            "equipment_count": len(engineering_equipment),
            "connectivity_node_count": len(nodes),
            "device_relation_count": len(relations),
        },
    }


def provenance_for_handles(
    handles: Iterable[str],
    record_by_id: dict[str, FlatRecord],
) -> list[dict[str, Any]]:
    output = []
    for handle in handles:
        record = record_by_id.get(handle)
        if record is None:
            continue
        output.append(
            {
                "evidence_id": handle,
                "source_handle": record.source_handle,
                "insert_path": list(record.insert_path),
                "block_names": list(record.block_names),
                "entity_type": record.entity.dxftype(),
                "layer": str(record.entity.dxf.layer or ""),
                "virtual": record.virtual,
            }
        )
    return output


def render_overlay(
    path: Path,
    shapes: list[Shape],
    equipment: list[dict[str, Any]],
    segments: list[Segment],
    active_roots: set[int],
    union: UnionFind,
) -> None:
    figure, axis = plt.subplots(figsize=(15, 10))
    active_indices = {
        index
        for index in range(len(segments))
        if union.find(index) in active_roots
    }
    for index in active_indices:
        segment = segments[index]
        axis.plot(
            [segment.start[0], segment.end[0]],
            [segment.start[1], segment.end[1]],
            color="#8b95a5",
            linewidth=0.8,
            zorder=1,
        )
    colors = {
        "PowerTransformer": "#d62728",
        "VoltageTransformer": "#9467bd",
        "CableTermination": "#ff7f0e",
        "SwitchCombination": "#1f77b4",
    }
    for item in equipment:
        left, bottom, right, top = item["bbox"]
        color = colors.get(item["physical_type"], "#2ca02c")
        axis.add_patch(
            plt.Rectangle(
                (left, bottom),
                right - left,
                top - bottom,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
                zorder=3,
            )
        )
        axis.text(
            left,
            top,
            f"{item['equipment_id']} {item['physical_type']} "
            f"{item['template_score']:.1f}",
            fontsize=7,
            color=color,
            zorder=4,
        )
    if equipment:
        boxes = [tuple(item["bbox"]) for item in equipment]
        left, bottom, right, top = merge_bbox(boxes)
        margin = max(right - left, top - bottom, 1.0) * 0.08
        axis.set_xlim(left - margin, right + margin)
        axis.set_ylim(bottom - margin, top + margin)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title("独立自动识别：全部模板匹配与活动导线网络")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def render_engineering_overlay(
    path: Path,
    segments: list[Segment],
    engineering: dict[str, Any],
) -> None:
    """Render cabinet containers, transformers and engineering relations."""
    figure, axis = plt.subplots(figsize=(18, 10))
    for segment in segments:
        axis.plot(
            [segment.start[0], segment.end[0]],
            [segment.start[1], segment.end[1]],
            color="#c7ccd4",
            linewidth=0.45,
            zorder=1,
        )
    equipment_by_id = {
        item["engineering_equipment_id"]: item
        for item in engineering.get("equipment", [])
    }
    for item in engineering.get("equipment", []):
        left, bottom, right, top = map(float, item["bbox"])
        transformer = item["type"] == "PowerTransformer"
        color = "#d62728" if transformer else "#1f77b4"
        axis.add_patch(
            plt.Rectangle(
                (left, bottom),
                right - left,
                top - bottom,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
                zorder=3,
            )
        )
        axis.text(
            left,
            top,
            f'{item["engineering_equipment_id"]} {item["name"]}',
            fontsize=6.5,
            color=color,
            zorder=5,
        )
    for relation in engineering.get("device_relations", []):
        source = equipment_by_id.get(relation["from_equipment"])
        target = equipment_by_id.get(relation["to_equipment"])
        if source is None or target is None:
            continue
        sx, sy = map(float, source["center"])
        tx, ty = map(float, target["center"])
        axis.plot(
            [sx, tx],
            [sy, ty],
            color="#2ca02c",
            linewidth=1.2,
            linestyle="--",
            zorder=2,
        )
    boxes = [
        tuple(map(float, item["bbox"]))
        for item in engineering.get("equipment", [])
    ]
    if boxes:
        left, bottom, right, top = merge_bbox(boxes)
        margin = max(right - left, top - bottom, 1.0) * 0.08
        axis.set_xlim(left - margin, right + margin)
        axis.set_ylim(bottom - margin, top + margin)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title("两级拓扑：柜体工程层（蓝）与变压器（红）")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def recognize_one(
    dxf_path: Path,
    component_library_path: Path,
    component_library: dict[str, Any],
    logic_library_path: Path,
    logic_library: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    doc = ezdxf.readfile(dxf_path)
    flat_records, insert_audit, visibility_audit = flatten_modelspace(doc)
    texts = extract_texts(flat_records)
    raw_h = robust_raw_h(texts, flat_records)
    expanded_shapes, record_by_id = extract_shapes(flat_records, raw_h)
    shapes = deduplicate_shapes(expanded_shapes, raw_h)

    templates = prepare_fast_templates(component_library)
    shape_by_id = {shape.handle: shape for shape in expanded_shapes}
    block_equipment, block_candidate_audit = build_block_equipment(
        insert_audit,
        shape_by_id,
        templates,
        raw_h,
    )
    block_reserved_handles = {
        handle
        for item in block_equipment
        for handle in item["source_handles"]
    }
    free_shapes = [
        shape
        for shape in shapes
        if shape.handle not in block_reserved_handles
    ]

    seeds = [
        shape for shape in free_shapes if device_seed(shape, raw_h)
    ]
    groups = cluster_shapes(seeds, raw_h * INITIAL_GROUP_GAP_H)
    groups = localize_oversized_groups(groups, raw_h)
    groups = expand_seed_groups(groups, free_shapes, raw_h)
    maximum_template_primitives = max(
        (
            sum(
                int(value)
                for value in prepared["record"]
                .get("primitive_counts", {})
                .values()
            )
            for prepared in templates
            if prepared["supported"]
        ),
        default=20,
    )
    all_lines = [
        shape for shape in free_shapes if shape.kind == "line"
    ]
    group_results = []
    priors = []
    candidate_audit = []
    for group_index, group in enumerate(groups, 1):
        group_id = f"G{group_index:05d}"
        box = merge_bbox(shape.bbox for shape in group)
        prior = nearest_text_prior(
            box,
            texts,
            maximum=raw_h * 5.0,
        )
        nearby_lines = [
            shape
            for shape in all_lines
            if bbox_distance(shape.bbox, box) <= raw_h * NEARBY_LINE_H
        ]
        candidates = generate_group_candidates(
            group_id,
            group,
            nearby_lines,
            raw_h,
        )
        generated_candidate_count = len(candidates)
        group_shape_by_handle = {
            shape.handle: shape for shape in group
        }
        candidates = [
            candidate
            for candidate in candidates
            if sum(candidate["primitive_counts"].values())
            <= maximum_template_primitives * 2
            and (
                not candidate["bbox"]
                or max(
                    candidate["bbox"][2] - candidate["bbox"][0],
                    candidate["bbox"][3] - candidate["bbox"][1],
                )
                <= raw_h * 12.0
                and max(
                    candidate["bbox"][2] - candidate["bbox"][0],
                    candidate["bbox"][3] - candidate["bbox"][1],
                )
                >= raw_h * 0.20
            )
            and not (
                len(candidate["owned_handles"]) == 1
                and candidate["owned_handles"][0] in group_shape_by_handle
                and group_shape_by_handle[
                    candidate["owned_handles"][0]
                ].kind
                in {"line", "polyline"}
                and not group_shape_by_handle[
                    candidate["owned_handles"][0]
                ].closed
            )
        ]
        if not candidates:
            # Single seeds remain auditable even when the enclosing seed group
            # is an oversized background structure.
            candidates = generate_group_candidates(
                group_id,
                group[:1],
                nearby_lines,
                raw_h,
            )
        evaluations = []
        for candidate in candidates:
            ranked = score_candidate_fast(candidate, templates, prior)
            evaluations.append(
                {"candidate": candidate, "ranked": ranked}
            )
        result = select_group_result_generic(
            group,
            candidates,
            evaluations,
            raw_h,
        )
        group_results.append(result)
        priors.append(prior)
        candidate_audit.append(
            {
                "group_id": group_id,
                "group_handles": sorted(
                    shape.handle for shape in group
                ),
                "generated_candidate_count": generated_candidate_count,
                "candidate_count": len(candidates),
                "selected_handles": result["candidate"]["owned_handles"],
                "selected_top": result["top"],
                "top5_families": result["family_alternatives"],
                "family_margin": result["family_margin"],
                "text_prior": prior,
            }
        )

    geometry_equipment, consumed, rejected = build_equipment(
        groups,
        group_results,
        priors,
        component_library,
        raw_h,
        block_reserved_handles,
    )
    equipment = block_equipment + geometry_equipment
    consumed.update(
        complete_transformer_components(
            equipment,
            shapes,
            raw_h,
        )
    )
    consumed.update(
        handle
        for item in equipment
        for handle in item.get("source_handles") or []
    )
    equipment.sort(
        key=lambda item: (item["center"][0], item["center"][1])
    )
    attach_visible_equipment_names(equipment, texts, raw_h)
    for index, item in enumerate(equipment, 1):
        item["equipment_id"] = f"EQ{index:05d}"
        item["source_evidence"] = provenance_for_handles(
            item["source_handles"],
            record_by_id,
        )

    segments, conductor_layer_counts = build_segments(
        shapes,
        consumed,
        raw_h,
    )
    union, all_events = build_conductor_graph(
        segments,
        shapes,
        raw_h,
    )
    terminals, root_terminals, terminal_issues = attach_terminals(
        equipment,
        segments,
        union,
        raw_h,
    )
    nodes, edges, active_roots = build_nodes_and_edges(
        equipment,
        terminals,
        root_terminals,
        segments,
        union,
    )
    crossings = filter_active_events(
        all_events,
        segments,
        union,
        active_roots,
    )
    crossings = normalize_connection_events(crossings, raw_h)
    containers = detect_cabinet_containers(
        texts,
        shapes,
        segments,
        union,
        equipment,
        raw_h,
    )
    engineering_topology = build_engineering_topology(
        containers,
        equipment,
        segments,
        union,
        raw_h,
    )
    base = {
        "schema_version": "independent-block-template-topology-v4",
        "generated_at": datetime.now().astimezone().isoformat(),
        "drawing": {
            "file": dxf_path.name,
            "sha256": sha256(dxf_path),
            "dxf_version": doc.dxfversion,
            "modelspace_entity_count": len(doc.modelspace()),
        },
        "recognition_strategy": {
            "name": "block_first_recursive_geometry_fallback",
            "truth_used_during_recognition": False,
            "pseudo_truth_module_imported": False,
            "block_policy": {
                "insert_is_first_class_component_candidate": True,
                "visible_printable_layers_only": True,
                "semantic_priority": [
                    "ATTRIB",
                    "ATTDEF",
                    "block_name",
                    "complete_block_template",
                ],
                "recursive_expansion_use": [
                    "geometry verification",
                    "port location",
                    "conductor attachment",
                ],
                "free_geometry_fallback": True,
            },
            "steps": [
                "排除关闭、冻结和不打印图层，再递归展开可见INSERT并应用块变换。",
                "从闭合、圆形、斜线和短图元生成几何候选。",
                "每个候选与全部设备模板逐一比较。",
                "先确认元件并占用图元，再从剩余图元生成导线。",
                "仅保留连接到元件接口的活动导线网络及交叉事件。",
                "最后使用电气逻辑知识库增加连接上下文语义。",
                "将柜体作为容器建立柜内详细拓扑和柜体工程拓扑两级结果。",
            ],
        },
        "interface_definition": {
            "standard": "component_side",
            "canonical_coordinate_field": "port_position",
            "conductor_coordinate_field": "wire_attach_position",
            "connectivity_field": "connectivity_node",
        },
        "truth_used_during_recognition": False,
        "equipment": equipment,
        "terminals": terminals,
        "connectivity_nodes": nodes,
        "crossings": crossings,
        "functional_annotations": [],
        "derived_device_edges": edges,
        "engineering_topology": engineering_topology,
        "issues": [
            {
                "issue_id": f"ISS{index:05d}",
                "category": "terminal_attachment",
                "description": description,
            }
            for index, description in enumerate(terminal_issues, 1)
        ],
        "automatic_statistics": {
            "canonical_h": CANONICAL_H,
            "raw_units_per_h": raw_h / CANONICAL_H,
            "raw_text_height_reference": raw_h,
            "modelspace_entity_count": len(doc.modelspace()),
            "visibility_policy": visibility_audit["policy"],
            "excluded_hidden_or_non_plot_source_entity_count": (
                visibility_audit["excluded_source_entity_count"]
            ),
            "excluded_hidden_or_non_plot_insert_count": (
                visibility_audit["excluded_insert_count"]
            ),
            "excluded_visibility_reason_counts": (
                visibility_audit["excluded_reason_counts"]
            ),
            "excluded_visibility_layer_counts": (
                visibility_audit["excluded_layer_counts"]
            ),
            "expanded_entity_count": len(flat_records),
            "expanded_insert_count": len(insert_audit),
            "virtual_entity_count": sum(
                record.virtual for record in flat_records
            ),
            "shape_count_after_dedup": len(shapes),
            "device_seed_count": len(seeds),
            "block_insert_candidate_count": len(insert_audit),
            "accepted_block_equipment_count": len(block_equipment),
            "accepted_free_geometry_equipment_count": len(
                geometry_equipment
            ),
            "deduplicated_block_instance_count": sum(
                len(item.get("duplicate_block_instances") or [])
                for item in block_equipment
            ),
            "candidate_group_count": len(groups),
            "template_count_compared_per_candidate": len(templates),
            "candidate_count": sum(
                item["candidate_count"] for item in candidate_audit
            ),
            "accepted_equipment_count": len(equipment),
            "rejected_candidate_count": len(rejected),
            "conductor_candidate_count": len(segments),
            "active_conductor_component_count": len(active_roots),
            "terminal_count": len(terminals),
            "connectivity_node_count": len(nodes),
            "derived_device_edge_count": len(edges),
            "active_crossing_count": len(crossings),
            "engineering_cabinet_count": engineering_topology[
                "statistics"
            ]["cabinet_count"],
            "engineering_equipment_count": engineering_topology[
                "statistics"
            ]["equipment_count"],
            "engineering_connectivity_node_count": engineering_topology[
                "statistics"
            ]["connectivity_node_count"],
            "engineering_device_relation_count": engineering_topology[
                "statistics"
            ]["device_relation_count"],
            "conductor_layer_counts": conductor_layer_counts,
        },
        "parameters": {
            "canonical_h": CANONICAL_H,
            "endpoint_snap": "0.10H",
            "terminal_attach": "0.20H; up to 2.0H retained for review",
            "minimum_wire_length": "1.50H",
            "curve_flatten_error": "0.025H",
            "dot_radius": "0.35H",
            "t_junction_tolerance": "0.25H",
            "axis_angle_degrees": AXIS_ANGLE_DEGREES,
            "initial_component_group_gap": "0.25H",
            "minimum_template_score": MINIMUM_TEMPLATE_SCORE,
            "minimum_family_margin": MINIMUM_FAMILY_MARGIN,
            "text_prior_maximum_bonus": 5.0,
        },
    }
    enhanced = enhance_one(
        base,
        component_library_path,
        component_library,
        logic_library_path,
        logic_library,
    )
    for item in enhanced["equipment"]:
        item["functional_role"] = (
            (item.get("logic_inference") or {}).get("functional_role")
        )
        item["physical_recognition_status"] = "recognized"
    enhanced["detailed_topology"] = {
        "schema_version": "detailed-component-topology-v1",
        "equipment": copy.deepcopy(enhanced["equipment"]),
        "terminals": copy.deepcopy(enhanced["terminals"]),
        "connectivity_nodes": copy.deepcopy(
            enhanced["connectivity_nodes"]
        ),
        "derived_device_edges": copy.deepcopy(
            enhanced["derived_device_edges"]
        ),
        "crossings": copy.deepcopy(enhanced["crossings"]),
    }

    audit = {
        "schema_version": "independent-block-template-audit-v4",
        "drawing": enhanced["drawing"],
        "visibility_filter": visibility_audit,
        "insert_expansion": insert_audit,
        "block_candidates": block_candidate_audit,
        "candidate_groups": candidate_audit,
        "rejected_candidates": rejected,
        "statistics": enhanced["automatic_statistics"],
    }
    return enhanced, audit


def save_report(path: Path, result: dict[str, Any]) -> None:
    stats = result["automatic_statistics"]
    type_counts = Counter(
        item["physical_type"] for item in result["equipment"]
    )
    lines = [
        f"# {Path(result['drawing']['file']).stem} 独立自动识别",
        "",
        "- 拟人工标注参与识别：否",
        "- 标注生成模块导入：否",
        "- 图层范围：仅识别可见且可打印的图层",
        "- 块引用：递归展开并保留来源路径",
        "- 元件匹配：每个候选与全部标准模板逐一比较",
        "- 导线范围：确认元件后，仅从剩余非标注图层的长轴向图元生成",
        "- 交叉范围：仅保存与元件接口相连的活动导线网络事件",
        "",
        "## 统计",
        "",
        f"- 模型空间实体：{stats['modelspace_entity_count']}",
        (
            "- 排除的隐藏/冻结/不打印源图元："
            f"{stats['excluded_hidden_or_non_plot_source_entity_count']}"
        ),
        (
            "- 其中排除的INSERT："
            f"{stats['excluded_hidden_or_non_plot_insert_count']}"
        ),
        f"- 块展开后实体：{stats['expanded_entity_count']}",
        f"- 展开的INSERT：{stats['expanded_insert_count']}",
        f"- 虚拟图元：{stats['virtual_entity_count']}",
        f"- 元件候选组：{stats['candidate_group_count']}",
        f"- 模板数：{stats['template_count_compared_per_candidate']}",
        f"- 接受元件：{stats['accepted_equipment_count']}",
        f"- 导线候选：{stats['conductor_candidate_count']}",
        f"- 活动连接节点：{stats['connectivity_node_count']}",
        f"- 活动交叉事件：{stats['active_crossing_count']}",
        f"- 工程层柜体：{stats['engineering_cabinet_count']}",
        f"- 工程层设备：{stats['engineering_equipment_count']}",
        (
            "- 工程层连接节点："
            f"{stats['engineering_connectivity_node_count']}"
        ),
        (
            "- 工程层设备关系："
            f"{stats['engineering_device_relation_count']}"
        ),
        "",
        "## 元件类型",
        "",
    ]
    for name, count in sorted(type_counts.items()):
        lines.append(f"- {name}: {count}")
    lines += [
        "",
        "低于模板阈值、类型间差距过小、只含单条普通直线/圆或与高分元件重叠的"
        "候选均不进入正式拓扑，而保留在审计JSON中。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf-dir", required=True, type=Path)
    parser.add_argument("--component-library", required=True, type=Path)
    parser.add_argument("--logic-library", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--drawings", nargs="+", required=True)
    args = parser.parse_args()

    component_library = read_json(args.component_library)
    logic_library = read_json(args.logic_library)
    summaries = []
    for drawing in args.drawings:
        dxf_path = args.dxf_dir / f"{drawing}.dxf"
        result, audit = recognize_one(
            dxf_path,
            args.component_library,
            component_library,
            args.logic_library,
            logic_library,
        )
        drawing_dir = args.output_dir / drawing
        automatic_dir = drawing_dir / "automatic"
        audit_dir = drawing_dir / "audit"
        automatic_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            automatic_dir / f"{drawing}_自动识别.json",
            result,
        )
        write_json(
            audit_dir / f"{drawing}_独立模板识别审计.json",
            audit,
        )
        save_report(
            automatic_dir / f"{drawing}_自动识别报告.md",
            result,
        )
        write_csv(
            automatic_dir / f"{drawing}_自动识别元件.csv",
            [
                {
                    "equipment_id": item["equipment_id"],
                    "physical_type": item["physical_type"],
                    "template_family": item["template_family"],
                    "template_name": item["template_name"],
                    "template_score": item["template_score"],
                    "family_margin": item["family_margin"],
                    "source_handles": ",".join(item["source_handles"]),
                }
                for item in result["equipment"]
            ],
        )
        engineering = result["engineering_topology"]
        write_csv(
            automatic_dir / f"{drawing}_工程层设备.csv",
            [
                {
                    "engineering_equipment_id": item[
                        "engineering_equipment_id"
                    ],
                    "type": item["type"],
                    "type_cn": item["type_cn"],
                    "name": item["name"],
                    "source_container_id": item.get(
                        "source_container_id",
                        "",
                    ),
                    "source_detailed_equipment_id": item.get(
                        "source_detailed_equipment_id",
                        "",
                    ),
                }
                for item in engineering["equipment"]
            ],
        )
        write_csv(
            automatic_dir / f"{drawing}_工程层关系.csv",
            engineering["device_relations"],
        )
        # Rebuild the conductor graph objects for a compact visual review.
        doc = ezdxf.readfile(dxf_path)
        flat_records, _, _ = flatten_modelspace(doc)
        texts = extract_texts(flat_records)
        raw_h = robust_raw_h(texts, flat_records)
        shapes, _ = extract_shapes(flat_records, raw_h)
        shapes = deduplicate_shapes(shapes, raw_h)
        consumed = {
            handle
            for item in result["equipment"]
            for handle in item["source_handles"]
        }
        segments, _ = build_segments(shapes, consumed, raw_h)
        union, _ = build_conductor_graph(segments, shapes, raw_h)
        terminal_roots = set()
        segment_index = {
            segment.segment_id: index
            for index, segment in enumerate(segments)
        }
        for terminal in result["terminals"]:
            index = segment_index.get(
                terminal.get("conductor_segment_id", "")
            )
            if index is not None:
                terminal_roots.add(union.find(index))
        render_overlay(
            automatic_dir / f"{drawing}_自动识别复核.png",
            shapes,
            result["equipment"],
            segments,
            terminal_roots,
            union,
        )
        render_engineering_overlay(
            automatic_dir / f"{drawing}_工程拓扑复核.png",
            segments,
            engineering,
        )
        summaries.append(
            {
                "drawing": drawing,
                **result["automatic_statistics"],
                "equipment_type_counts": dict(
                    Counter(
                        item["physical_type"]
                        for item in result["equipment"]
                    )
                ),
            }
        )
        print(
            drawing,
            json.dumps(summaries[-1], ensure_ascii=False),
            flush=True,
        )
    write_json(
        args.output_dir / "独立自动识别汇总.json",
        summaries,
    )


if __name__ == "__main__":
    main()
