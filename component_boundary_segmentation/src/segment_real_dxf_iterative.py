#!/usr/bin/env python3
"""Run stage 2 + iterative stage 3 on one real DXF drawing."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from classify_real_dxf_component_side import load_model
from component_side import component_side_probability, long_axis_aligned_conductor_mask
from evaluate_exploded_svg_conductors import feature_rows
from evaluate_graph_message_passing import contact_graph, graph_probability
from iterative_boundary import IterativeBoundaryConfig, iterative_component_boundaries
from logical_context import (
    LOGIC_FEATURE_NAMES,
    RELATION_LOGIC_FEATURE_NAMES,
    logical_feature_rows,
    relation_logical_feature_rows,
)
from segment_real_dxf import cluster_records, dxf_primitives, render


ROOT = Path(__file__).resolve().parents[1]


def stage2_scores(
    payload: dict,
    probability: np.ndarray,
    primitives,
    scale: float,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> tuple[np.ndarray, float]:
    metadata = payload.get("component_side_binary", {})
    score = component_side_probability(probability)
    logic = logical_feature_rows(primitives, scale, edge_index, edge_attr)
    override = metadata.get("long_axis_aligned_conductor_override", {})
    if override:
        symbol_column = LOGIC_FEATURE_NAMES.index(
            "local_closed_or_circle_count_0_5h"
        )
        long_axis = long_axis_aligned_conductor_mask(
            primitives,
            scale,
            float(override.get("minimum_length_h", 1.5)),
            float(override.get("direction_tolerance_degrees", 1.0)),
        )
        score[long_axis & (logic[:, symbol_column] <= 0.0)] = 0.0
    relation_override = metadata.get("carrier_relation_override", {})
    if relation_override:
        relation = relation_logical_feature_rows(
            primitives, scale, edge_index, edge_attr, logic
        )
        column = RELATION_LOGIC_FEATURE_NAMES.index(
            "short_oblique_component_candidate"
        )
        selected = relation[:, column] >= 0.5
        score[selected] = np.maximum(
            score[selected],
            float(relation_override["short_oblique_symbol_minimum_score"]),
        )
    return score, float(metadata.get("threshold", 0.5))


def load_iterative_runtime(
    base_model_path: Path,
    stage2_model_path: Path,
    edge_model_path: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Load the three production DXF boundary models once for batch use."""

    device = torch.device(
        "cuda" if device_name == "cuda" or (
            device_name == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    base_model = joblib.load(base_model_path)["model"]
    stage2_payload, stage2_model = load_model(stage2_model_path, device)
    edge_payload = (
        joblib.load(edge_model_path) if edge_model_path.exists() else None
    )
    return {
        "device": device,
        "base_model": base_model,
        "stage2_payload": stage2_payload,
        "stage2_model": stage2_model,
        "edge_payload": edge_payload,
        "base_model_path": str(base_model_path.resolve()),
        "stage2_model_path": str(stage2_model_path.resolve()),
        "edge_model_path": (
            str(edge_model_path.resolve()) if edge_payload is not None else None
        ),
    }


def infer_dxf_boundaries(
    dxf_path: Path,
    runtime: dict[str, Any],
    scale: float = 100.0,
) -> tuple[dict[str, Any], np.ndarray, list[Any], list[dict[str, Any]]]:
    """Run the production stage-1/2/3 DXF frontend without writing files.

    The returned evidence keeps both primitive IDs (one DXF entity may yield
    several primitives) and original source handles.  Downstream recognition
    can therefore consume learned component boundaries while topology output
    still points back to the original DXF.
    """

    primitives, evidence, parser_audit = dxf_primitives(dxf_path, scale)
    if not primitives:
        raise RuntimeError(f"No supported visible geometry: {dxf_path}")
    base_features = feature_rows(primitives, scale)
    base_wire_probability = runtime["base_model"].predict_proba(
        base_features
    )[:, 1]
    node_features = np.hstack([
        base_features, base_wire_probability[:, None]
    ]).astype(np.float32)
    role_edge_index, role_edge_attr = contact_graph(primitives, scale)
    role_probability = graph_probability(
        runtime["stage2_model"],
        {
            "x": node_features,
            "edge_index": role_edge_index,
            "edge_attr": role_edge_attr,
        },
        np.asarray(runtime["stage2_payload"]["feature_mean"], dtype=np.float32),
        np.asarray(runtime["stage2_payload"]["feature_std"], dtype=np.float32),
        runtime["device"],
    )
    component_probability, component_threshold = stage2_scores(
        runtime["stage2_payload"],
        role_probability,
        primitives,
        scale,
        role_edge_index,
        role_edge_attr,
    )
    edge_payload = runtime.get("edge_payload")
    boundary_tolerance = float(
        edge_payload.get("contact_tolerance_ratio", 0.08)
        if edge_payload is not None else 0.08
    )
    boundary_edge_index, boundary_edge_attr = contact_graph(
        primitives, scale, boundary_tolerance
    )
    assignment, boundary_audit = iterative_component_boundaries(
        primitives,
        scale,
        boundary_edge_index,
        boundary_edge_attr,
        role_probability,
        component_probability,
        component_threshold,
        base_features,
        base_wire_probability,
        edge_payload=edge_payload,
        config=IterativeBoundaryConfig.dxf_fixed_h_profile(),
        source_group_ids=[
            str(item.get("source_handle") or "") for item in evidence
        ],
    )
    components = cluster_records(
        primitives, evidence, assignment, role_probability
    )
    membership_source = boundary_audit["membership_source"]
    for component in components:
        cluster = int(component["component_id"][1:])
        indices = np.flatnonzero(assignment == cluster)
        source_count = Counter(membership_source[index] for index in indices)
        scores = component_probability[indices]
        component.update({
            "type": "unknown",
            "recognition_status": "pending_stage4",
            "boundary_status": "component_candidate",
            "boundary_confidence": float(np.mean(scores)),
            "minimum_component_side_probability": float(np.min(scores)),
            "maximum_component_side_probability": float(np.max(scores)),
            "seed_primitive_count": int(source_count.get("seed", 0)),
            "reabsorbed_primitive_count": int(source_count.get("reabsorbed", 0)),
        })
    result = {
        "schema_version": "iterative-component-boundary-v1",
        "drawing": dxf_path.stem,
        "source_dxf": str(dxf_path.resolve()),
        "scale_h": scale,
        "pipeline": [
            "DXF primitive normalization",
            "stage2 soft component-side classification",
            "stage3 iterative component boundary growth",
            "stage4 type recognition pending",
        ],
        "stage2": {
            "model": runtime["stage2_model_path"],
            "component_probability": "P(component_body)+P(interface_lead)",
            "threshold": component_threshold,
            "selected_primitive_count": int(
                np.sum(component_probability >= component_threshold)
            ),
        },
        "stage3": boundary_audit,
        "parser_audit": parser_audit,
        "component_count": len(components),
        "components": components,
    }
    return result, assignment, primitives, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf", type=Path)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument(
        "--base-model", type=Path,
        default=ROOT / "models" / "base_conductor_deployment.joblib",
    )
    parser.add_argument(
        "--stage2-model", type=Path,
        default=ROOT / "models" / "component_side_dxf_deployment.pt",
    )
    parser.add_argument(
        "--edge-model", type=Path,
        default=ROOT / "models" / "same_component_edge_dxf_deployment.joblib",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "output" / "iterative_component_boundaries",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    runtime = load_iterative_runtime(
        args.base_model, args.stage2_model, args.edge_model, args.device
    )
    result, assignment, primitives, _ = infer_dxf_boundaries(
        args.dxf, runtime, args.scale
    )
    components = result["components"]
    boundary_audit = result["stage3"]
    output_dir = args.output_dir / args.dxf.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "boundary_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_dir / "components.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "component_id", "type", "recognition_status", "primitive_count",
            "seed_primitive_count", "reabsorbed_primitive_count",
            "boundary_confidence", "center_x", "center_y", "width", "height",
            "source_handles",
        ])
        for item in components:
            writer.writerow([
                item["component_id"], item["type"], item["recognition_status"],
                item["primitive_count"], item["seed_primitive_count"],
                item["reabsorbed_primitive_count"], item["boundary_confidence"],
                *item["center"], item["width"], item["height"],
                ";".join(item["source_handles"]),
            ])
    image_path = output_dir / "boundary_overlay.png"
    render(primitives, assignment, components, image_path)
    print(json.dumps({
        "drawing": args.dxf.name,
        "primitive_count": len(primitives),
        "stage2_selected": result["stage2"]["selected_primitive_count"],
        "stage3_reabsorbed": boundary_audit["reabsorbed_primitive_count"],
        "component_count": len(components),
    }, ensure_ascii=False))
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"image: {image_path}")


if __name__ == "__main__":
    main()
