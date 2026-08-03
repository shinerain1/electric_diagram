#!/usr/bin/env python3
"""Evaluate conductor-vs-component primitive separation on paired SVG/XML.

The semantic SVG groups are used only to create labels.  Equipment ``<use>``
instances and line-like geometry are exploded into anonymous primitives before
features are computed.  Files, rather than primitives, are split into
train/validation/test sets to prevent drawing leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import joblib
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
PATH_TOKEN_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|"
    r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
)
TRANSFORM_RE = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
RDF_ID = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}ID"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

WIRE_CLASSES = {"ACLineSegmentClass", "ConnectiveLineClass"}
IGNORED_CLASSES = {
    "HeadClass",
    "TextClass",
    "OtherClass",
    "SubstationClass",
}

FEATURE_NAMES = [
    "length_h",
    "log_length_h",
    "length_file_median_ratio",
    "bbox_width_h",
    "bbox_height_h",
    "aspect_log",
    "axis_aligned",
    "horizontal",
    "vertical",
    "diagonal",
    "closed",
    "line_kind",
    "circle_kind",
    "endpoint_degree_min",
    "endpoint_degree_max",
    "endpoint_degree_sum",
    "neighbor_count_1h",
    "neighbor_count_3h",
]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def number(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def point_pairs(value: str | None) -> list[tuple[float, float]]:
    values = [float(item) for item in NUMBER_RE.findall(value or "")]
    return [
        (values[index], values[index + 1])
        for index in range(0, len(values) - 1, 2)
    ]


def distance(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _curve_point(
    points: tuple[tuple[float, float], ...],
    t: float,
) -> tuple[float, float]:
    if len(points) == 3:
        start, control, end = points
        inverse = 1.0 - t
        return (
            inverse * inverse * start[0]
            + 2.0 * inverse * t * control[0]
            + t * t * end[0],
            inverse * inverse * start[1]
            + 2.0 * inverse * t * control[1]
            + t * t * end[1],
        )
    start, control_1, control_2, end = points
    inverse = 1.0 - t
    return (
        inverse**3 * start[0]
        + 3.0 * inverse * inverse * t * control_1[0]
        + 3.0 * inverse * t * t * control_2[0]
        + t**3 * end[0],
        inverse**3 * start[1]
        + 3.0 * inverse * inverse * t * control_1[1]
        + 3.0 * inverse * t * t * control_2[1]
        + t**3 * end[1],
    )


def _arc_points(
    start: tuple[float, float],
    rx: float,
    ry: float,
    rotation_degrees: float,
    large_arc: bool,
    sweep: bool,
    end: tuple[float, float],
) -> list[tuple[float, float]]:
    """Approximate one SVG elliptical arc using the SVG endpoint formula."""
    rx = abs(rx)
    ry = abs(ry)
    if rx <= 1e-12 or ry <= 1e-12 or distance(start, end) <= 1e-12:
        return [end]
    phi = math.radians(rotation_degrees % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    dx = (start[0] - end[0]) / 2.0
    dy = (start[1] - end[1]) / 2.0
    x_prime = cos_phi * dx + sin_phi * dy
    y_prime = -sin_phi * dx + cos_phi * dy
    radii_scale = (
        x_prime * x_prime / (rx * rx)
        + y_prime * y_prime / (ry * ry)
    )
    if radii_scale > 1.0:
        scale = math.sqrt(radii_scale)
        rx *= scale
        ry *= scale
    numerator = max(
        0.0,
        rx * rx * ry * ry
        - rx * rx * y_prime * y_prime
        - ry * ry * x_prime * x_prime,
    )
    denominator = max(
        rx * rx * y_prime * y_prime
        + ry * ry * x_prime * x_prime,
        1e-18,
    )
    coefficient = math.sqrt(numerator / denominator)
    if large_arc == sweep:
        coefficient = -coefficient
    center_x_prime = coefficient * rx * y_prime / ry
    center_y_prime = -coefficient * ry * x_prime / rx
    center_x = (
        cos_phi * center_x_prime
        - sin_phi * center_y_prime
        + (start[0] + end[0]) / 2.0
    )
    center_y = (
        sin_phi * center_x_prime
        + cos_phi * center_y_prime
        + (start[1] + end[1]) / 2.0
    )

    def vector_angle(
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        dot = left[0] * right[0] + left[1] * right[1]
        cross = left[0] * right[1] - left[1] * right[0]
        return math.atan2(cross, dot)

    unit_start = (
        (x_prime - center_x_prime) / rx,
        (y_prime - center_y_prime) / ry,
    )
    unit_end = (
        (-x_prime - center_x_prime) / rx,
        (-y_prime - center_y_prime) / ry,
    )
    start_angle = vector_angle((1.0, 0.0), unit_start)
    delta_angle = vector_angle(unit_start, unit_end)
    if not sweep and delta_angle > 0.0:
        delta_angle -= 2.0 * math.pi
    elif sweep and delta_angle < 0.0:
        delta_angle += 2.0 * math.pi
    segment_count = max(
        2,
        int(math.ceil(abs(delta_angle) / math.radians(15.0))),
    )
    output = []
    for index in range(1, segment_count + 1):
        angle = start_angle + delta_angle * index / segment_count
        local_x = rx * math.cos(angle)
        local_y = ry * math.sin(angle)
        output.append(
            (
                center_x + cos_phi * local_x - sin_phi * local_y,
                center_y + sin_phi * local_x + cos_phi * local_y,
            )
        )
    output[-1] = end
    return output


def svg_path_geometry(value: str | None) -> list[dict[str, Any]]:
    """Parse SVG path commands into separate polyline subpaths."""
    tokens = PATH_TOKEN_RE.findall(value or "")
    if not tokens:
        return []
    parameter_counts = {
        "M": 2,
        "L": 2,
        "H": 1,
        "V": 1,
        "C": 6,
        "S": 4,
        "Q": 4,
        "T": 2,
        "A": 7,
    }
    output: list[dict[str, Any]] = []
    current = (0.0, 0.0)
    subpath_start = current
    points: list[tuple[float, float]] = []
    command: str | None = None
    previous_command = ""
    previous_cubic_control: tuple[float, float] | None = None
    previous_quadratic_control: tuple[float, float] | None = None
    index = 0

    def flush(closed: bool = False) -> None:
        nonlocal points
        if len(points) >= 2:
            output.append(
                {
                    "kind": "line",
                    "closed": closed,
                    "points": points,
                }
            )
        points = []

    def absolute_point(
        x: float,
        y: float,
        relative: bool,
    ) -> tuple[float, float]:
        if relative:
            return current[0] + x, current[1] + y
        return x, y

    while index < len(tokens):
        token = tokens[index]
        if len(token) == 1 and token.isalpha():
            command = token
            index += 1
            if command.upper() == "Z":
                flush(closed=True)
                current = subpath_start
                previous_command = "Z"
                previous_cubic_control = None
                previous_quadratic_control = None
                command = None
            continue
        if command is None:
            index += 1
            continue
        upper = command.upper()
        count = parameter_counts.get(upper)
        if count is None or index + count > len(tokens):
            command = None
            continue
        if any(
            len(item) == 1 and item.isalpha()
            for item in tokens[index : index + count]
        ):
            command = None
            continue
        values = [float(item) for item in tokens[index : index + count]]
        index += count
        relative = command.islower()
        if upper == "M":
            if points:
                flush()
            current = absolute_point(values[0], values[1], relative)
            subpath_start = current
            points = [current]
            command = "l" if relative else "L"
            previous_cubic_control = None
            previous_quadratic_control = None
        elif upper == "L":
            current = absolute_point(values[0], values[1], relative)
            if not points:
                points = [current]
                subpath_start = current
            else:
                points.append(current)
        elif upper == "H":
            current = (
                current[0] + values[0] if relative else values[0],
                current[1],
            )
            points.append(current)
        elif upper == "V":
            current = (
                current[0],
                current[1] + values[0] if relative else values[0],
            )
            points.append(current)
        elif upper == "C":
            control_1 = absolute_point(values[0], values[1], relative)
            control_2 = absolute_point(values[2], values[3], relative)
            end = absolute_point(values[4], values[5], relative)
            points.extend(
                _curve_point((current, control_1, control_2, end), step / 12.0)
                for step in range(1, 13)
            )
            current = end
            previous_cubic_control = control_2
            previous_quadratic_control = None
        elif upper == "S":
            if previous_command.upper() in {"C", "S"} and previous_cubic_control:
                control_1 = (
                    2.0 * current[0] - previous_cubic_control[0],
                    2.0 * current[1] - previous_cubic_control[1],
                )
            else:
                control_1 = current
            control_2 = absolute_point(values[0], values[1], relative)
            end = absolute_point(values[2], values[3], relative)
            points.extend(
                _curve_point((current, control_1, control_2, end), step / 12.0)
                for step in range(1, 13)
            )
            current = end
            previous_cubic_control = control_2
            previous_quadratic_control = None
        elif upper == "Q":
            control = absolute_point(values[0], values[1], relative)
            end = absolute_point(values[2], values[3], relative)
            points.extend(
                _curve_point((current, control, end), step / 8.0)
                for step in range(1, 9)
            )
            current = end
            previous_quadratic_control = control
            previous_cubic_control = None
        elif upper == "T":
            if (
                previous_command.upper() in {"Q", "T"}
                and previous_quadratic_control
            ):
                control = (
                    2.0 * current[0] - previous_quadratic_control[0],
                    2.0 * current[1] - previous_quadratic_control[1],
                )
            else:
                control = current
            end = absolute_point(values[0], values[1], relative)
            points.extend(
                _curve_point((current, control, end), step / 8.0)
                for step in range(1, 9)
            )
            current = end
            previous_quadratic_control = control
            previous_cubic_control = None
        elif upper == "A":
            end = absolute_point(values[5], values[6], relative)
            points.extend(
                _arc_points(
                    current,
                    values[0],
                    values[1],
                    values[2],
                    bool(values[3]),
                    bool(values[4]),
                    end,
                )
            )
            current = end
            previous_cubic_control = None
            previous_quadratic_control = None
        previous_command = upper
        if upper not in {"C", "S"}:
            previous_cubic_control = None
        if upper not in {"Q", "T"}:
            previous_quadratic_control = None
    flush()
    return output


@dataclass
class Primitive:
    drawing: str
    label: int
    kind: str
    closed: bool
    start: tuple[float, float] | None
    end: tuple[float, float] | None
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    length: float
    # Evaluation-only provenance. These fields are deliberately excluded from
    # FEATURE_NAMES and are never exposed to the conductor classifier.
    truth_component_id: str | None = None
    truth_component_class: str = ""
    primitive_id: str = ""
    # Optional computational geometry for a primitive that must stay one
    # semantic object (for example one DXF ARC).  These points are used only
    # for distance/intersection checks and rendering; they do not create
    # additional graph nodes or model features.
    geometry_points: tuple[tuple[float, float], ...] = ()


def segment_primitive(
    drawing: str,
    label: int,
    start: tuple[float, float],
    end: tuple[float, float],
    kind: str = "line",
    closed: bool = False,
) -> Primitive | None:
    length = distance(start, end)
    if length <= 1e-9:
        return None
    return Primitive(
        drawing=drawing,
        label=label,
        kind=kind,
        closed=closed,
        start=start,
        end=end,
        center=((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0),
        bbox=(
            min(start[0], end[0]),
            min(start[1], end[1]),
            max(start[0], end[0]),
            max(start[1], end[1]),
        ),
        length=length,
    )


def element_geometry(
    element: ET.Element,
) -> list[dict[str, Any]]:
    tag = local_name(element.tag)
    attributes = element.attrib
    if tag == "line":
        return [
            {
                "kind": "line",
                "closed": False,
                "points": [
                    (number(attributes.get("x1")), number(attributes.get("y1"))),
                    (number(attributes.get("x2")), number(attributes.get("y2"))),
                ],
            }
        ]
    if tag in {"polyline", "polygon"}:
        points = point_pairs(attributes.get("points"))
        if len(points) < 2:
            return []
        return [
            {
                "kind": "line",
                "closed": tag == "polygon",
                "points": points,
            }
        ]
    if tag == "rect":
        x = number(attributes.get("x"))
        y = number(attributes.get("y"))
        width = number(attributes.get("width"))
        height = number(attributes.get("height"))
        return [
            {
                "kind": "line",
                "closed": True,
                "points": [
                    (x, y),
                    (x + width, y),
                    (x + width, y + height),
                    (x, y + height),
                ],
            }
        ]
    if tag == "path":
        return svg_path_geometry(attributes.get("d"))
    if tag in {"circle", "ellipse"}:
        cx = number(attributes.get("cx"))
        cy = number(attributes.get("cy"))
        rx = number(attributes.get("r"))
        ry = rx
        if tag == "ellipse":
            rx = number(attributes.get("rx"))
            ry = number(attributes.get("ry"))
        if rx <= 0.0 or ry <= 0.0:
            return []
        return [
            {
                "kind": "circle",
                "closed": True,
                "center": (cx, cy),
                "rx": rx,
                "ry": ry,
            }
        ]
    return []


def identity(point: tuple[float, float]) -> tuple[float, float]:
    return point


def transform_function(
    use: ET.Element,
    view_box: tuple[float, float, float, float],
) -> Callable[[tuple[float, float]], tuple[float, float]]:
    origin_x, origin_y, view_width, view_height = view_box
    use_x = number(use.attrib.get("x"))
    use_y = number(use.attrib.get("y"))
    use_width = number(use.attrib.get("width"), view_width)
    use_height = number(use.attrib.get("height"), view_height)
    scale_x = use_width / max(view_width, 1e-9)
    scale_y = use_height / max(view_height, 1e-9)
    operations = [
        (
            name.lower(),
            [float(item) for item in NUMBER_RE.findall(arguments)],
        )
        for name, arguments in TRANSFORM_RE.findall(
            str(use.attrib.get("transform") or "")
        )
    ]

    def apply(point: tuple[float, float]) -> tuple[float, float]:
        x = use_x + (point[0] - origin_x) * scale_x
        y = use_y + (point[1] - origin_y) * scale_y
        for name, values in operations:
            if name == "rotate" and values:
                angle = math.radians(values[0])
                cx = values[1] if len(values) >= 3 else 0.0
                cy = values[2] if len(values) >= 3 else 0.0
                dx = x - cx
                dy = y - cy
                x = cx + dx * math.cos(angle) - dy * math.sin(angle)
                y = cy + dx * math.sin(angle) + dy * math.cos(angle)
            elif name == "scale" and values:
                sx = values[0]
                sy = values[1] if len(values) >= 2 else sx
                x *= sx
                y *= sy
            elif name == "translate" and values:
                x += values[0]
                y += values[1] if len(values) >= 2 else 0.0
        return x, y

    return apply


def geometry_to_primitives(
    drawing: str,
    label: int,
    geometries: Iterable[dict[str, Any]],
    transform: Callable[
        [tuple[float, float]],
        tuple[float, float],
    ],
    truth_component_id: str | None = None,
    truth_component_class: str = "",
) -> list[Primitive]:
    output = []
    for geometry in geometries:
        if geometry["kind"] == "circle":
            center = transform(tuple(geometry["center"]))
            x_point = transform(
                (
                    geometry["center"][0] + geometry["rx"],
                    geometry["center"][1],
                )
            )
            y_point = transform(
                (
                    geometry["center"][0],
                    geometry["center"][1] + geometry["ry"],
                )
            )
            rx = distance(center, x_point)
            ry = distance(center, y_point)
            if rx <= 1e-9 or ry <= 1e-9:
                continue
            output.append(
                Primitive(
                    drawing=drawing,
                    label=label,
                    kind="circle",
                    closed=True,
                    start=None,
                    end=None,
                    center=center,
                    bbox=(
                        center[0] - rx,
                        center[1] - ry,
                        center[0] + rx,
                        center[1] + ry,
                    ),
                    length=math.pi
                    * (
                        3.0 * (rx + ry)
                        - math.sqrt(
                            max(
                                (3.0 * rx + ry)
                                * (rx + 3.0 * ry),
                                0.0,
                            )
                        )
                    ),
                    truth_component_id=truth_component_id,
                    truth_component_class=truth_component_class,
                )
            )
            continue
        points = [transform(tuple(point)) for point in geometry["points"]]
        pairs = list(zip(points, points[1:]))
        if geometry["closed"] and len(points) >= 3:
            pairs.append((points[-1], points[0]))
        for start, end in pairs:
            primitive = segment_primitive(
                drawing,
                label,
                start,
                end,
                kind="line",
                closed=bool(geometry["closed"]),
            )
            if primitive is not None:
                primitive.truth_component_id = truth_component_id
                primitive.truth_component_class = truth_component_class
                output.append(primitive)
    return output


def xml_object_types(path: Path) -> dict[str, str]:
    output = {}
    for _, element in ET.iterparse(path, events=("end",)):
        object_id = element.attrib.get(RDF_ID)
        if object_id:
            output[str(object_id)] = local_name(element.tag)
        element.clear()
    return output


def extract_drawing(
    svg_path: Path,
    xml_path: Path,
    include_truth_terminals: bool = False,
) -> tuple[list[Primitive], dict[str, Any]]:
    root = ET.parse(svg_path).getroot()
    drawing = svg_path.stem
    xml_types = xml_object_types(xml_path)
    symbols: dict[
        str,
        tuple[
            tuple[float, float, float, float],
            list[dict[str, Any]],
            list[tuple[float, float]],
        ],
    ] = {}
    for element in root.iter():
        if local_name(element.tag) != "symbol":
            continue
        symbol_id = str(element.attrib.get("id") or "")
        view_values = [
            float(item)
            for item in NUMBER_RE.findall(
                str(element.attrib.get("viewBox") or "0 0 1 1")
            )
        ]
        view_box = (
            tuple(view_values[:4])
            if len(view_values) >= 4
            else (0.0, 0.0, 1.0, 1.0)
        )
        geometries = []
        terminal_points = []
        for child in element.iter():
            if child is element:
                continue
            if local_name(child.tag) == "use":
                child_href = str(
                    child.attrib.get(XLINK_HREF)
                    or child.attrib.get("href")
                    or ""
                ).lstrip("#")
                if child_href.startswith("terminal:"):
                    terminal_points.append(
                        (
                            number(child.attrib.get("x")),
                            number(child.attrib.get("y")),
                        )
                    )
                    continue
            geometries.extend(element_geometry(child))
        symbols[symbol_id] = (view_box, geometries, terminal_points)

    primitives = []
    component_scales = []
    svg_object_count = 0
    xml_matched_object_count = 0
    class_counts: Counter[str] = Counter()
    component_terminals: dict[
        str,
        list[tuple[float, float]],
    ] = defaultdict(list)
    for class_group in list(root):
        if local_name(class_group.tag) != "g":
            continue
        class_name = str(class_group.attrib.get("id") or "")
        if not class_name.endswith("Class"):
            continue
        if class_name in IGNORED_CLASSES:
            continue
        if class_name in WIRE_CLASSES:
            label = 1
        else:
            label = 0
        class_counts[class_name] += 1
        for object_index, object_group in enumerate(list(class_group)):
            if local_name(object_group.tag) != "g":
                continue
            svg_object_count += 1
            object_ids = {
                str(item.attrib.get("ObjectID"))
                for item in object_group.iter()
                if local_name(item.tag) == "PSR_Ref"
                and item.attrib.get("ObjectID")
            }
            if any(object_id in xml_types for object_id in object_ids):
                xml_matched_object_count += 1
            truth_component_id = None
            if label == 0:
                source_id = (
                    sorted(object_ids)[0]
                    if object_ids
                    else str(object_group.attrib.get("id") or object_index)
                )
                truth_component_id = f"{class_name}:{source_id}"
            uses = [
                item
                for item in object_group.iter()
                if local_name(item.tag) == "use"
            ]
            for use in uses:
                href = str(
                    use.attrib.get(XLINK_HREF)
                    or use.attrib.get("href")
                    or ""
                ).lstrip("#")
                if href not in symbols or href.startswith("terminal:"):
                    continue
                view_box, geometries, terminal_points = symbols[href]
                transform = transform_function(use, view_box)
                current = geometry_to_primitives(
                    drawing,
                    label,
                    geometries,
                    transform,
                    truth_component_id=truth_component_id,
                    truth_component_class=class_name if label == 0 else "",
                )
                primitives.extend(current)
                if label == 0:
                    if truth_component_id and include_truth_terminals:
                        component_terminals[truth_component_id].extend(
                            transform(point) for point in terminal_points
                        )
                    component_scales.append(
                        max(
                            number(use.attrib.get("width"), view_box[2]),
                            number(use.attrib.get("height"), view_box[3]),
                        )
                    )
            for child in object_group.iter():
                if local_name(child.tag) in {
                    "g",
                    "use",
                    "metadata",
                    "PSR_Ref",
                    "text",
                }:
                    continue
                primitives.extend(
                    geometry_to_primitives(
                        drawing,
                        label,
                        element_geometry(child),
                        identity,
                        truth_component_id=truth_component_id,
                        truth_component_class=class_name if label == 0 else "",
                    )
                )

    scale = (
        statistics.median(component_scales)
        if component_scales
        else statistics.median(
            [primitive.length for primitive in primitives]
        )
        if primitives
        else 1.0
    )
    scale = max(float(scale), 1e-6)
    for primitive_index, primitive in enumerate(primitives):
        primitive.primitive_id = f"P{primitive_index:08d}"
    audit = {
        "drawing": drawing,
        "scale": scale,
        "svg_object_count": svg_object_count,
        "xml_matched_object_count": xml_matched_object_count,
        "class_counts": dict(class_counts),
    }
    if include_truth_terminals:
        audit["component_terminals"] = {
            key: value for key, value in component_terminals.items()
        }
    return primitives, audit


def feature_rows(
    primitives: list[Primitive],
    scale: float,
) -> np.ndarray:
    if not primitives:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    lengths = np.asarray(
        [primitive.length for primitive in primitives],
        dtype=float,
    )
    file_median = max(float(np.median(lengths)), 1e-9)
    tolerance = max(scale * 0.02, 1e-6)
    endpoint_counts: Counter[tuple[int, int]] = Counter()
    endpoint_keys: list[list[tuple[int, int]]] = []
    for primitive in primitives:
        keys = []
        for point in (primitive.start, primitive.end):
            if point is None:
                continue
            key = (
                round(point[0] / tolerance),
                round(point[1] / tolerance),
            )
            endpoint_counts[key] += 1
            keys.append(key)
        endpoint_keys.append(keys)
    centers = np.asarray(
        [primitive.center for primitive in primitives],
        dtype=float,
    )
    tree = cKDTree(centers)
    neighbor_1h = np.asarray(
        tree.query_ball_point(centers, r=scale, return_length=True),
        dtype=float,
    ) - 1.0
    neighbor_3h = np.asarray(
        tree.query_ball_point(
            centers,
            r=scale * 3.0,
            return_length=True,
        ),
        dtype=float,
    ) - 1.0
    rows = []
    for index, primitive in enumerate(primitives):
        width = primitive.bbox[2] - primitive.bbox[0]
        height = primitive.bbox[3] - primitive.bbox[1]
        maximum = max(width, height, 1e-9)
        minimum = min(width, height)
        axis = primitive.kind == "line" and minimum <= maximum * 0.02
        horizontal = axis and height <= width * 0.02
        vertical = axis and width <= height * 0.02
        degrees = [
            endpoint_counts[key] - 1 for key in endpoint_keys[index]
        ]
        rows.append(
            [
                primitive.length / scale,
                math.log1p(primitive.length / scale),
                primitive.length / file_median,
                width / scale,
                height / scale,
                abs(math.log((width + 1e-9) / (height + 1e-9))),
                float(axis),
                float(horizontal),
                float(vertical),
                float(primitive.kind == "line" and not axis),
                float(primitive.closed),
                float(primitive.kind == "line"),
                float(primitive.kind == "circle"),
                float(min(degrees, default=0)),
                float(max(degrees, default=0)),
                float(sum(degrees)),
                neighbor_1h[index],
                neighbor_3h[index],
            ]
        )
    return np.asarray(rows, dtype=float)


def stable_fraction(name: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_name(name: str, seed: int) -> str:
    value = stable_fraction(name, seed)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray | None = None,
) -> dict[str, Any]:
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    output = {
        "primitive_count": int(len(truth)),
        "component_primitive_count": int(np.sum(truth == 0)),
        "wire_primitive_count": int(np.sum(truth == 1)),
        "wire_precision": round(
            float(precision_score(truth, prediction, zero_division=0)),
            6,
        ),
        "wire_recall": round(
            float(recall_score(truth, prediction, zero_division=0)),
            6,
        ),
        "wire_f1": round(
            float(f1_score(truth, prediction, zero_division=0)),
            6,
        ),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(truth, prediction)),
            6,
        ),
        "confusion_matrix_component_wire": matrix.tolist(),
    }
    if probability is not None and len(np.unique(truth)) == 2:
        output["roc_auc"] = round(
            float(roc_auc_score(truth, probability)),
            6,
        )
    return output


def best_length_threshold(
    lengths: np.ndarray,
    truth: np.ndarray,
    axis: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(
        np.quantile(lengths, np.linspace(0.02, 0.98, 120))
    )
    best = (-1.0, 0.0, None)
    for threshold in candidates:
        prediction = lengths >= threshold
        if axis is not None:
            prediction &= axis >= 0.5
        current = f1_score(truth, prediction, zero_division=0)
        if current > best[0]:
            best = (
                float(current),
                float(threshold),
                metrics(truth, prediction.astype(int)),
            )
    return best[1], best[2] or {}


def summarize_lengths(
    matrix: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    output = {}
    for label, name in ((0, "component"), (1, "wire")):
        values = matrix[truth == label, 0]
        output[name] = {
            "count": int(len(values)),
            "p10_h": round(float(np.quantile(values, 0.10)), 6),
            "median_h": round(float(np.median(values)), 6),
            "p90_h": round(float(np.quantile(values, 0.90)), 6),
            "p99_h": round(float(np.quantile(values, 0.99)), 6),
            "axis_aligned_fraction": round(
                float(np.mean(matrix[truth == label, 6] >= 0.5)),
                6,
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg-dir", required=True, type=Path)
    parser.add_argument("--xml-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="0 uses every paired drawing",
    )
    args = parser.parse_args()

    svg_paths = sorted(args.svg_dir.glob("*.svg"))
    xml_by_stem = {
        path.stem: path for path in args.xml_dir.glob("*.xml")
    }
    pairs = [
        (path, xml_by_stem[path.stem])
        for path in svg_paths
        if path.stem in xml_by_stem
    ]
    if args.max_files > 0:
        pairs = sorted(
            pairs,
            key=lambda pair: stable_fraction(
                f"sample:{pair[0].stem}",
                args.seed,
            ),
        )[: args.max_files]
    if not pairs:
        raise SystemExit("no paired SVG/XML files found")

    matrices: dict[str, list[np.ndarray]] = defaultdict(list)
    targets: dict[str, list[np.ndarray]] = defaultdict(list)
    manifests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    extraction_warnings = []
    for index, (svg_path, xml_path) in enumerate(pairs, 1):
        try:
            primitives, audit = extract_drawing(svg_path, xml_path)
        except Exception as error:
            extraction_warnings.append(
                {
                    "drawing": svg_path.stem,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        labels = np.asarray(
            [primitive.label for primitive in primitives],
            dtype=int,
        )
        if not len(labels) or len(np.unique(labels)) < 2:
            extraction_warnings.append(
                {
                    "drawing": svg_path.stem,
                    "error": "drawing lacks both component and wire primitives",
                }
            )
            continue
        matrix = feature_rows(primitives, float(audit["scale"]))
        split = split_name(svg_path.stem, args.seed)
        matrices[split].append(matrix)
        targets[split].append(labels)
        manifests[split].append(
            {
                **audit,
                "primitive_count": int(len(labels)),
                "component_primitive_count": int(np.sum(labels == 0)),
                "wire_primitive_count": int(np.sum(labels == 1)),
            }
        )
        if index % 100 == 0:
            print(
                f"processed {index}/{len(pairs)} paired drawings",
                flush=True,
            )

    missing_splits = [
        name
        for name in ("train", "validation", "test")
        if not matrices[name]
    ]
    if missing_splits:
        raise SystemExit(
            "no usable drawings in split(s): "
            + ", ".join(missing_splits)
            + "; increase --max-files"
        )
    x = {
        name: np.vstack(matrices[name])
        for name in ("train", "validation", "test")
    }
    y = {
        name: np.concatenate(targets[name])
        for name in ("train", "validation", "test")
    }
    length_threshold, validation_length = best_length_threshold(
        x["validation"][:, 0],
        y["validation"],
    )
    axis_length_threshold, validation_axis_length = best_length_threshold(
        x["validation"][:, 0],
        y["validation"],
        x["validation"][:, 6],
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=0.5,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(x["train"], y["train"])
    validation_probability = model.predict_proba(x["validation"])[:, 1]
    probability_thresholds = np.linspace(0.20, 0.95, 151)
    best_probability_threshold = max(
        probability_thresholds,
        key=lambda threshold: f1_score(
            y["validation"],
            validation_probability >= threshold,
            zero_division=0,
        ),
    )
    test_probability = model.predict_proba(x["test"])[:, 1]
    test_prediction = (
        test_probability >= best_probability_threshold
    ).astype(int)
    test_length_prediction = (
        x["test"][:, 0] >= length_threshold
    ).astype(int)
    test_axis_length_prediction = (
        (x["test"][:, 0] >= axis_length_threshold)
        & (x["test"][:, 6] >= 0.5)
    ).astype(int)

    permutation = permutation_importance(
        model,
        x["validation"],
        y["validation"],
        n_repeats=3,
        random_state=args.seed,
        scoring="f1",
    )
    importances = sorted(
        [
            {
                "feature": name,
                "importance": round(float(value), 7),
            }
            for name, value in zip(
                FEATURE_NAMES,
                permutation.importances_mean,
            )
        ],
        key=lambda item: -item["importance"],
    )
    xml_object_count = sum(
        item["svg_object_count"]
        for rows in manifests.values()
        for item in rows
    )
    xml_match_count = sum(
        item["xml_matched_object_count"]
        for rows in manifests.values()
        for item in rows
    )
    report = {
        "schema_version": "exploded-svg-conductor-evaluation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "svg_dir": str(args.svg_dir),
            "xml_dir": str(args.xml_dir),
            "paired_file_count": len(pairs),
            "successfully_used_file_count": sum(
                len(rows) for rows in manifests.values()
            ),
            "warning_count": len(extraction_warnings),
            "svg_object_count": xml_object_count,
            "svg_objects_matched_to_xml_fraction": round(
                xml_match_count / max(xml_object_count, 1),
                6,
            ),
        },
        "split": {
            "unit": "whole paired drawing",
            "policy": "70% train, 15% validation, 15% test by stable hash",
            "seed": args.seed,
            "drawing_counts": {
                name: len(manifests[name])
                for name in ("train", "validation", "test")
            },
            "primitive_counts": {
                name: int(len(y[name]))
                for name in ("train", "validation", "test")
            },
        },
        "explosion_policy": {
            "component_use_instances_expanded": True,
            "polylines_polygons_rectangles_paths_split_into_segments": True,
            "semantic_ids_used_as_features": False,
            "labels": {
                "wire": sorted(WIRE_CLASSES),
                "component": "non-ignored equipment SVG classes",
            },
        },
        "feature_names": FEATURE_NAMES,
        "length_feature_analysis": {
            name: summarize_lengths(x[name], y[name])
            for name in ("train", "validation", "test")
        },
        "length_only_baseline": {
            "selected_on": "validation",
            "threshold_h": round(length_threshold, 6),
            "validation": validation_length,
            "test": metrics(y["test"], test_length_prediction),
        },
        "axis_and_length_baseline": {
            "selected_on": "validation",
            "threshold_h": round(axis_length_threshold, 6),
            "validation": validation_axis_length,
            "test": metrics(y["test"], test_axis_length_prediction),
        },
        "learned_classifier": {
            "algorithm": "HistGradientBoostingClassifier",
            "probability_threshold_selected_on_validation": round(
                float(best_probability_threshold),
                6,
            ),
            "validation": metrics(
                y["validation"],
                (
                    validation_probability
                    >= best_probability_threshold
                ).astype(int),
                validation_probability,
            ),
            "test": metrics(
                y["test"],
                test_prediction,
                test_probability,
            ),
            "permutation_importance_on_validation": importances,
        },
        "warnings": extraction_warnings,
        "limitations": [
            "SVG semantic groups provide labels but are never model features.",
            "SVG-to-DXF explosion is simulated and cannot reproduce every CAD export artifact.",
            "ConnectiveLineClass is treated as wire because it represents visible conductive connection geometry.",
            "The test split is not used for threshold or feature selection.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "conductor_primitive_model.joblib"
    joblib.dump(
        {
            "schema_version": "exploded-svg-conductor-classifier-v1",
            "feature_names": FEATURE_NAMES,
            "model": model,
            "wire_probability_threshold": float(
                best_probability_threshold
            ),
            "metadata": {
                "trained_at": report["generated_at"],
                "split_seed": args.seed,
                "training_drawing_count": len(manifests["train"]),
                "validation_drawing_count": len(
                    manifests["validation"]
                ),
                "test_evaluation": report["learned_classifier"]["test"],
                "semantic_fields_used_as_features": False,
            },
        },
        model_path,
    )
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report["learned_classifier"]["artifact"] = {
        "path": str(model_path),
        "sha256": model_sha256,
        "schema_version": "exploded-svg-conductor-classifier-v1",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
