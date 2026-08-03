#!/usr/bin/env python3
"""Run stage 2 only: classify DXF primitives before boundary grouping."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_exploded_svg_conductors import feature_rows
from component_side import (
    apply_stage2_policy,
    component_side_probability,
    long_axis_aligned_conductor_mask,
)
from logical_context import (
    LOGIC_FEATURE_NAMES,
    RELATION_LOGIC_FEATURE_NAMES,
    logical_feature_rows,
    relation_logical_feature_rows,
)
from evaluate_graph_message_passing import (
    EdgeGatedResidualGNN,
    contact_graph,
    graph_probability,
)
from segment_real_dxf import draw_primitive, dxf_primitives


ROOT = Path(__file__).resolve().parents[1]


def load_model(path: Path, device: torch.device) -> tuple[dict[str, Any], EdgeGatedResidualGNN]:
    payload = torch.load(path, map_location=device, weights_only=False)
    output_classes = int(payload.get("output_classes", len(payload.get("role_names", []))))
    model = EdgeGatedResidualGNN(
        len(payload["node_feature_names"]), len(payload["edge_feature_names"]),
        int(payload["hidden"]), int(payload["layers"]), float(payload["dropout"]),
        output_classes=output_classes,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return payload, model


def render(primitives, labels: np.ndarray, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(16, 10), facecolor="white")
    axis.set_facecolor("white")
    for primitive, is_component in zip(primitives, labels):
        draw_primitive(
            axis, primitive,
            "#d62728" if is_component else "#4c78a8",
            1.5 if is_component else 0.75,
            1.0 if is_component else 0.72,
        )
    axis.autoscale()
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    figure.tight_layout(pad=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf", type=Path)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument(
        "--base-model", type=Path,
        default=ROOT / "models" / "base_conductor_model.joblib",
    )
    parser.add_argument(
        "--model", type=Path,
        default=ROOT / "models" / "hierarchical_component_side_dxf_adapted.pt",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "output" / "component_side_classification",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    primitives, evidence, parser_audit = dxf_primitives(args.dxf, args.scale)
    if not primitives:
        raise RuntimeError(f"No supported visible geometry: {args.dxf}")
    payload, model = load_model(args.model, device)
    features = feature_rows(primitives, args.scale)
    node_features = features
    if "base_wire_probability" in payload["node_feature_names"]:
        base_model = joblib.load(args.base_model)["model"]
        base_probability = base_model.predict_proba(features)[:, 1]
        node_features = np.hstack([features, base_probability[:, None]])
    edge_index, edge_attr = contact_graph(primitives, args.scale)
    graph = {
        "x": node_features.astype(np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
    probability = graph_probability(
        model, graph,
        np.asarray(payload["feature_mean"], dtype=np.float32),
        np.asarray(payload["feature_std"], dtype=np.float32),
        device,
    )
    metadata = payload.get("component_side_binary", {})
    override = metadata.get("long_axis_aligned_conductor_override", {})
    logic = None
    if override.get("preserve_when_closed_or_circle_within_h") is not None:
        score = component_side_probability(probability)
        logic = logical_feature_rows(
            primitives, args.scale, edge_index, edge_attr
        )
        symbol_column = LOGIC_FEATURE_NAMES.index(
            "local_closed_or_circle_count_0_5h"
        )
        long_axis = long_axis_aligned_conductor_mask(
            primitives,
            args.scale,
            float(override.get("minimum_length_h", 1.5)),
            float(override.get("direction_tolerance_degrees", 1.0)),
        )
        score[long_axis & (logic[:, symbol_column] <= 0.0)] = 0.0
    else:
        score = apply_stage2_policy(
            probability,
            primitives,
            args.scale,
            float(override.get("minimum_length_h", 1.5)),
            float(override.get("direction_tolerance_degrees", 1.0)),
        )
    relation_override = metadata.get("carrier_relation_override", {})
    if relation_override:
        if logic is None:
            logic = logical_feature_rows(
                primitives, args.scale, edge_index, edge_attr
            )
        relation = relation_logical_feature_rows(
            primitives, args.scale, edge_index, edge_attr, logic
        )
        oblique_column = RELATION_LOGIC_FEATURE_NAMES.index(
            "short_oblique_component_candidate"
        )
        oblique_symbol = relation[:, oblique_column] >= 0.5
        score[oblique_symbol] = np.maximum(
            score[oblique_symbol],
            float(relation_override["short_oblique_symbol_minimum_score"]),
        )
    threshold = float(
        args.threshold if args.threshold is not None
        else metadata.get("threshold", payload.get("component_side_threshold", 0.5))
    )
    labels = score >= threshold
    rows = []
    handle_scores: dict[str, list[float]] = defaultdict(list)
    for index, (primitive, item) in enumerate(zip(primitives, evidence)):
        handle = str(item.get("source_handle") or "")
        handle_scores[handle].append(float(score[index]))
        rows.append({
            "primitive_index": index,
            "primitive_id": primitive.primitive_id,
            "source_handle": handle,
            "entity_type": item.get("entity_type", ""),
            "layer": item.get("layer", ""),
            "kind": primitive.kind,
            "bbox": [float(value) for value in primitive.bbox],
            "length": float(primitive.length),
            "component_side_probability": float(score[index]),
            "classification": "component_side" if labels[index] else "non_component_side",
        })
    handles = [
        {
            "source_handle": handle,
            "component_side_probability": max(values),
            "classification": "component_side" if max(values) >= threshold else "non_component_side",
            "primitive_count": len(values),
        }
        for handle, values in sorted(handle_scores.items()) if handle
    ]
    output_dir = args.output_dir / args.dxf.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "component-side-classification-v1",
        "drawing": args.dxf.stem,
        "source_dxf": str(args.dxf.resolve()),
        "stage": "2_primitive_binary_classification_before_boundary_segmentation",
        "definition": {
            "component_side": "component body plus interface lead",
            "non_component_side": "main conductor plus remaining frame/annotation geometry",
        },
        "model": str(args.model.resolve()),
        "threshold": threshold,
        "primitive_count": len(primitives),
        "component_side_primitive_count": int(labels.sum()),
        "non_component_side_primitive_count": int((~labels).sum()),
        "uncertain_primitive_count_within_0_10": int(np.sum(np.abs(score - threshold) <= 0.10)),
        "graph_edge_count_directed": int(edge_index.shape[1]),
        "parser_audit": parser_audit,
        "primitive_classifications": rows,
        "handle_classifications": handles,
    }
    json_path = output_dir / "component_side_classification.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_dir / "component_side_primitives.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "primitive_index", "primitive_id", "source_handle", "entity_type",
            "layer", "kind", "length", "component_side_probability", "classification",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    image_path = output_dir / "component_side_classification.png"
    render(primitives, labels, image_path)
    print(json.dumps({
        "drawing": args.dxf.name,
        "threshold": threshold,
        "primitive_count": len(primitives),
        "component_side": int(labels.sum()),
        "non_component_side": int((~labels).sum()),
        "uncertain": result["uncertain_primitive_count_within_0_10"],
    }, ensure_ascii=False), flush=True)
    print(f"json: {json_path}", flush=True)
    print(f"csv: {csv_path}", flush=True)
    print(f"image: {image_path}", flush=True)


if __name__ == "__main__":
    main()
