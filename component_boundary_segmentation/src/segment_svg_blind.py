#!/usr/bin/env python3
"""Blind SVG boundary segmentation without reading paired XML truth."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

from evaluate_exploded_svg_conductors import (
    NUMBER_RE,
    XLINK_HREF,
    Primitive,
    element_geometry,
    feature_rows,
    geometry_to_primitives,
    identity,
    local_name,
    number,
    transform_function,
)
from evaluate_graph_message_passing import (
    build_model,
    contact_graph,
    graph_probability,
)
from network_boundary import network_connected_components


def element_visible(element: ET.Element) -> bool:
    """Use visual attributes only; group IDs and semantic classes are ignored."""
    style = str(element.attrib.get("style") or "").replace(" ", "").lower()
    display = str(element.attrib.get("display") or "").lower()
    visibility = str(element.attrib.get("visibility") or "").lower()
    opacity = str(element.attrib.get("opacity") or "1")
    if display == "none" or visibility == "hidden" or opacity == "0":
        return False
    if "display:none" in style or "visibility:hidden" in style:
        return False
    stroke = str(element.attrib.get("stroke") or "").replace(" ", "").lower()
    fill = str(element.attrib.get("fill") or "").replace(" ", "").lower()
    if stroke in {"rgb(255,255,255)", "#fff", "#ffffff"} and fill in {
        "rgb(255,255,255)",
        "#fff",
        "#ffffff",
    }:
        return False
    return True


def symbol_definitions(
    root: ET.Element,
) -> dict[str, tuple[tuple[float, float, float, float], list[dict[str, Any]]]]:
    symbols = {}
    for element in root.iter():
        if local_name(element.tag) != "symbol":
            continue
        symbol_id = str(element.attrib.get("id") or "")
        values = [
            float(item)
            for item in NUMBER_RE.findall(
                str(element.attrib.get("viewBox") or "0 0 1 1")
            )
        ]
        view_box = (
            tuple(values[:4])
            if len(values) >= 4
            else (0.0, 0.0, 1.0, 1.0)
        )
        geometries = []
        for child in element.iter():
            if child is element or not element_visible(child):
                continue
            if local_name(child.tag) == "use":
                continue
            geometries.extend(element_geometry(child))
        symbols[symbol_id] = (view_box, geometries)
    return symbols


def svg_primitives_blind(
    svg_path: Path,
) -> tuple[list[Primitive], dict[str, Any]]:
    """Flatten visible SVG geometry while ignoring all semantic group names."""
    root = ET.parse(svg_path).getroot()
    symbols = symbol_definitions(root)
    primitives: list[Primitive] = []
    use_scales = []
    use_count = 0
    direct_geometry_count = 0

    def visit(element: ET.Element) -> None:
        nonlocal use_count, direct_geometry_count
        tag = local_name(element.tag)
        if tag in {"defs", "symbol", "metadata", "text", "script", "style"}:
            return
        if not element_visible(element):
            return
        if tag == "use":
            href = str(
                element.attrib.get(XLINK_HREF)
                or element.attrib.get("href")
                or ""
            ).lstrip("#")
            if href.startswith("terminal:") or href not in symbols:
                return
            view_box, geometries = symbols[href]
            transform = transform_function(element, view_box)
            primitives.extend(
                geometry_to_primitives(
                    svg_path.stem, 0, geometries, transform
                )
            )
            use_count += 1
            use_scales.append(
                max(
                    number(element.attrib.get("width"), view_box[2]),
                    number(element.attrib.get("height"), view_box[3]),
                )
            )
            return
        geometries = element_geometry(element)
        if geometries:
            primitives.extend(
                geometry_to_primitives(
                    svg_path.stem, 0, geometries, identity
                )
            )
            direct_geometry_count += len(geometries)
        for child in list(element):
            visit(child)

    for child in list(root):
        visit(child)
    scale = (
        statistics.median(use_scales)
        if use_scales
        else statistics.median(
            [primitive.length for primitive in primitives]
        )
        if primitives
        else 1.0
    )
    scale = max(float(scale), 1e-6)
    for index, primitive in enumerate(primitives):
        primitive.primitive_id = f"P{index:08d}"
        primitive.truth_component_id = None
        primitive.truth_component_class = ""
    return primitives, {
        "drawing": svg_path.stem,
        "scale": scale,
        "primitive_count": len(primitives),
        "flattened_use_count": use_count,
        "direct_geometry_count": direct_geometry_count,
        "xml_file_read": False,
        "svg_class_group_id_used": False,
        "svg_object_id_used": False,
        "symbol_href_used_only_for_graphical_expansion": True,
        "truth_field_populated": False,
    }


def component_records(
    primitives: list[Primitive],
    assignment: np.ndarray,
    probability: np.ndarray,
) -> list[dict[str, Any]]:
    members: dict[int, list[int]] = defaultdict(list)
    for index, cluster in enumerate(assignment):
        if cluster >= 0:
            members[int(cluster)].append(index)
    output = []
    for cluster, indices in sorted(members.items()):
        left = min(primitives[index].bbox[0] for index in indices)
        bottom = min(primitives[index].bbox[1] for index in indices)
        right = max(primitives[index].bbox[2] for index in indices)
        top = max(primitives[index].bbox[3] for index in indices)
        roles = probability[indices].mean(axis=0)
        output.append(
            {
                "component_id": f"C{cluster:05d}",
                "primitive_count": len(indices),
                "primitive_ids": [
                    primitives[index].primitive_id for index in indices
                ],
                "bbox": [left, bottom, right, top],
                "center": [(left + right) / 2.0, (bottom + top) / 2.0],
                "mean_role_probability": {
                    "component_body": float(roles[0]),
                    "interface_lead": float(roles[1]),
                    "main_conductor": float(roles[2]),
                },
            }
        )
    return output


def segment_svg(
    svg_path: Path,
    base_model_path: Path,
    graph_model_path: Path,
    device: torch.device,
    conductor_threshold: float,
    interface_threshold: float,
) -> dict[str, Any]:
    primitives, audit = svg_primitives_blind(svg_path)
    base_payload = joblib.load(base_model_path)
    base_model = base_payload["model"]
    graph_payload = torch.load(
        graph_model_path, map_location=device, weights_only=False
    )
    model = build_model(
        graph_payload["architecture"],
        len(graph_payload["node_feature_names"]),
        len(graph_payload["edge_feature_names"]),
        int(graph_payload["hidden"]),
        int(graph_payload["layers"]),
        float(graph_payload["dropout"]),
    ).to(device)
    model.load_state_dict(graph_payload["model_state_dict"])
    model.eval()
    features = feature_rows(primitives, float(audit["scale"]))
    base_probability = base_model.predict_proba(features)[:, 1]
    edge_index, edge_attr = contact_graph(
        primitives, float(audit["scale"])
    )
    graph = {
        "x": np.hstack(
            [features, base_probability.reshape(-1, 1)]
        ).astype(np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
    probability = graph_probability(
        model,
        graph,
        np.asarray(graph_payload["feature_mean"], dtype=np.float32),
        np.asarray(graph_payload["feature_std"], dtype=np.float32),
        device,
    )
    assignment = network_connected_components(
        edge_index,
        probability,
        conductor_threshold,
        interface_threshold,
    )
    components = component_records(primitives, assignment, probability)
    role_count = Counter(
        ["body", "interface", "conductor"][int(value)]
        for value in np.argmax(probability, axis=1)
    )
    return {
        "schema_version": "blind-svg-network-boundary-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_format": "SVG",
        "source_svg": str(svg_path.resolve()),
        "scale_h": float(audit["scale"]),
        "inference_contract": {
            "paired_xml_read": False,
            "truth_labels_read": False,
            "svg_semantic_class_names_read": False,
            "svg_object_ids_read": False,
            "symbol_href_use": "graphical_expansion_only",
        },
        "method": {
            "base_tree_use": "initial_graph_feature_only",
            "final_boundary": "network_roles_plus_graph_connected_components",
            "legacy_boundary_used": False,
            "conductor_probability_threshold": conductor_threshold,
            "interface_probability_threshold": interface_threshold,
        },
        "audit": audit,
        "role_argmax_count": dict(role_count),
        "graph_edge_count_directed": int(edge_index.shape[1]),
        "component_count": len(components),
        "conductor_primitive_count": int(np.sum(assignment < 0)),
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=PACKAGE_ROOT / "models" / "base_conductor_model.joblib",
    )
    parser.add_argument(
        "--graph-model",
        type=Path,
        default=PACKAGE_ROOT / "models" / "edge_gated_residual_gnn.pt",
    )
    parser.add_argument("--conductor-threshold", type=float, default=0.70)
    parser.add_argument("--interface-threshold", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    args = parser.parse_args()
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    result = segment_svg(
        args.svg,
        args.base_model,
        args.graph_model,
        device,
        args.conductor_threshold,
        args.interface_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"primitives: {result['audit']['primitive_count']}")
    print(f"components: {result['component_count']}")
    print(f"conductors: {result['conductor_primitive_count']}")
    print(f"result: {args.output}")


if __name__ == "__main__":
    main()
