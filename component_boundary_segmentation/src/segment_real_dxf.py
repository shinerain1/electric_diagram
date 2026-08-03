#!/usr/bin/env python3
"""Apply the packaged graph boundary model to a real DXF drawing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import ezdxf
import joblib
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HOST_ROOT = PACKAGE_ROOT.parent
HOST_SRC = HOST_ROOT / "src"
if str(HOST_SRC) not in sys.path:
    sys.path.insert(0, str(HOST_SRC))

from recognize_topology import (
    annotation_layer,
    extract_shapes,
    flatten_modelspace,
    shape_raw_vertices,
)

from evaluate_exploded_svg_conductors import (
    Primitive,
    feature_rows,
    segment_primitive,
)
from evaluate_graph_message_passing import (
    build_model,
    contact_graph,
    graph_probability,
)
from network_boundary import network_connected_components
from raw_full_boundary import load_runtime, segment_full_primitives
from same_component_edge import learned_edge_connected_components


def dxf_primitives(
    dxf_path: Path,
    scale: float,
) -> tuple[list[Primitive], list[dict[str, Any]], dict[str, Any]]:
    """Recursively flatten DXF entities into anonymous model primitives."""
    document = ezdxf.readfile(dxf_path)
    records, inserts, visibility = flatten_modelspace(document)
    shapes, record_by_id = extract_shapes(records, scale)
    primitives: list[Primitive] = []
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    quantum = max(scale * 0.002, 1e-8)

    def boundary_vertices(shape: Any) -> list[tuple[float, float]]:
        """Return geometry vertices without reusing dense feature samples.

        ``Shape.points`` is intended for shape-feature calculation.  Curves are
        represented there by ``sample_polyline()``, which places 32 samples on
        every already-flattened curve segment.  Treating those samples as new
        DXF segments made one ordinary ARC expand to roughly 3,000 primitives.
        Recover only the original flattened segment endpoints here.
        """
        vertices = shape_raw_vertices(shape)
        if len(vertices) >= 2:
            return vertices
        sampled = np.asarray(shape.points, dtype=float)
        if sampled.ndim != 2 or len(sampled) < 2:
            return []
        if shape.entity_type in {"ARC", "ELLIPSE", "SPLINE"}:
            stride = 32
            vertices = [
                (float(sampled[index, 0]), float(sampled[index, 1]))
                for index in range(0, len(sampled), stride)
            ]
            final = (float(sampled[-1, 0]), float(sampled[-1, 1]))
            if not vertices or math.dist(vertices[-1], final) > 1e-9:
                vertices.append(final)
            return vertices
        return [(float(point[0]), float(point[1])) for point in sampled]

    def append_segment(
        shape: Any,
        segment_index: int,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        primitive = segment_primitive(
            dxf_path.stem,
            0,
            start,
            end,
            kind="line",
            closed=bool(shape.closed),
        )
        if primitive is None:
            return
        ordered = sorted((start, end))
        key = (
            "line",
            bool(shape.closed),
            *(
                round(value / quantum)
                for point in ordered
                for value in point
            ),
        )
        if key in seen:
            return
        seen.add(key)
        primitive.primitive_id = f"{shape.handle}:{segment_index}"
        primitives.append(primitive)
        record = record_by_id.get(shape.handle)
        evidence.append(
            {
                "primitive_id": primitive.primitive_id,
                "handle": shape.handle,
                "source_handle": (
                    record.source_handle if record is not None else shape.handle
                ),
                "insert_path": (
                    list(record.insert_path) if record is not None else []
                ),
                "entity_type": shape.entity_type,
                "layer": shape.layer,
            }
        )

    for shape in shapes:
        if annotation_layer(shape.layer):
            continue
        if shape.entity_type in {"ARC", "ELLIPSE", "SPLINE"}:
            vertices = boundary_vertices(shape)
            if len(vertices) < 2:
                continue
            key = (
                shape.entity_type.lower(),
                *(round(value / quantum) for value in shape.bbox),
                *(round(value / quantum) for point in (vertices[0], vertices[-1]) for value in point),
            )
            if key in seen:
                continue
            seen.add(key)
            primitive = Primitive(
                drawing=dxf_path.stem,
                label=0,
                kind=shape.entity_type.lower(),
                closed=bool(shape.closed),
                start=vertices[0],
                end=vertices[-1],
                center=(
                    (shape.bbox[0] + shape.bbox[2]) / 2.0,
                    (shape.bbox[1] + shape.bbox[3]) / 2.0,
                ),
                bbox=shape.bbox,
                length=shape.length,
                primitive_id=f"{shape.handle}:0",
                geometry_points=tuple(vertices),
            )
            primitives.append(primitive)
            record = record_by_id.get(shape.handle)
            evidence.append(
                {
                    "primitive_id": primitive.primitive_id,
                    "handle": shape.handle,
                    "source_handle": (
                        record.source_handle
                        if record is not None
                        else shape.handle
                    ),
                    "insert_path": (
                        list(record.insert_path) if record is not None else []
                    ),
                    "entity_type": shape.entity_type,
                    "layer": shape.layer,
                }
            )
            continue
        if shape.kind == "circle":
            key = (
                "circle",
                *(round(value / quantum) for value in shape.bbox),
            )
            if key in seen:
                continue
            seen.add(key)
            primitive = Primitive(
                drawing=dxf_path.stem,
                label=0,
                kind="circle",
                closed=True,
                start=None,
                end=None,
                center=(
                    (shape.bbox[0] + shape.bbox[2]) / 2.0,
                    (shape.bbox[1] + shape.bbox[3]) / 2.0,
                ),
                bbox=shape.bbox,
                length=shape.length,
                primitive_id=f"{shape.handle}:0",
            )
            primitives.append(primitive)
            record = record_by_id.get(shape.handle)
            evidence.append(
                {
                    "primitive_id": primitive.primitive_id,
                    "handle": shape.handle,
                    "source_handle": (
                        record.source_handle
                        if record is not None
                        else shape.handle
                    ),
                    "insert_path": (
                        list(record.insert_path) if record is not None else []
                    ),
                    "entity_type": shape.entity_type,
                    "layer": shape.layer,
                }
            )
            continue
        vertices = boundary_vertices(shape)
        pairs = list(zip(vertices, vertices[1:]))
        if shape.closed and len(vertices) >= 3:
            pairs.append((vertices[-1], vertices[0]))
        for segment_index, (start, end) in enumerate(pairs):
            append_segment(shape, segment_index, start, end)

    audit = {
        "source_entity_count": len(document.modelspace()),
        "flattened_entity_count": len(records),
        "shape_count_after_visibility_filter": len(shapes),
        "primitive_count_after_deduplication": len(primitives),
        "expanded_insert_count": len(inserts),
        "visibility": visibility,
    }
    entity_expansion = Counter(
        (
            str(item.get("source_handle") or ""),
            tuple(item.get("insert_path") or []),
            str(item.get("entity_type") or ""),
        )
        for item in evidence
    )
    expansion_rows = sorted(
        (
            {
                "source_handle": source_handle,
                "insert_path": list(insert_path),
                "entity_type": entity_type,
                "primitive_count": count,
            }
            for (source_handle, insert_path, entity_type), count in (
                entity_expansion.items()
            )
        ),
        key=lambda item: int(item["primitive_count"]),
        reverse=True,
    )
    audit["primitive_count_by_entity_type"] = dict(
        Counter(str(item.get("entity_type") or "") for item in evidence)
    )
    audit["max_primitives_from_one_entity"] = (
        int(expansion_rows[0]["primitive_count"]) if expansion_rows else 0
    )
    audit["entities_expanded_over_32_primitives"] = [
        item for item in expansion_rows if int(item["primitive_count"]) > 32
    ]
    audit["largest_entity_expansions"] = expansion_rows[:10]
    return primitives, evidence, audit


def load_models(
    base_path: Path,
    graph_path: Path,
    device: torch.device,
) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    base_payload = joblib.load(base_path)
    graph_payload = torch.load(
        graph_path,
        map_location=device,
        weights_only=False,
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
    return base_payload["model"], model, graph_payload


def dxf_role_thresholds(
    graph_payload: dict[str, Any],
    conductor_override: float | None,
    interface_override: float | None,
) -> tuple[float, float]:
    """Resolve DXF thresholds from CLI overrides or adapted model metadata."""
    adaptation = graph_payload.get("domain_adaptation", {})
    conductor = (
        float(conductor_override)
        if conductor_override is not None
        else float(adaptation.get("dxf_conductor_threshold", 0.70))
    )
    interface = (
        float(interface_override)
        if interface_override is not None
        else float(adaptation.get("dxf_interface_threshold", 0.60))
    )
    return conductor, interface


def cluster_records(
    primitives: list[Primitive],
    evidence: list[dict[str, Any]],
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
                "component_id": f"C{cluster:04d}",
                "primitive_count": len(indices),
                "bbox": [left, bottom, right, top],
                "center": [(left + right) / 2.0, (bottom + top) / 2.0],
                "width": right - left,
                "height": top - bottom,
                "mean_role_probability": {
                    "component_body": float(roles[0]),
                    "interface_lead": float(roles[1]),
                    "main_conductor": float(roles[2]),
                },
                "primitive_ids": [
                    primitives[index].primitive_id for index in indices
                ],
                "source_handles": sorted(
                    {
                        str(evidence[index]["source_handle"])
                        for index in indices
                    }
                ),
                "layers": sorted(
                    {str(evidence[index]["layer"]) for index in indices}
                ),
            }
        )
    return output


def draw_primitive(
    axis: Any,
    primitive: Primitive,
    color: Any,
    linewidth: float,
    alpha: float,
) -> None:
    if primitive.kind == "circle":
        axis.add_patch(
            Ellipse(
                primitive.center,
                primitive.bbox[2] - primitive.bbox[0],
                primitive.bbox[3] - primitive.bbox[1],
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                alpha=alpha,
            )
        )
    elif len(primitive.geometry_points) >= 2:
        axis.plot(
            [point[0] for point in primitive.geometry_points],
            [point[1] for point in primitive.geometry_points],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    elif primitive.start is not None and primitive.end is not None:
        axis.plot(
            [primitive.start[0], primitive.end[0]],
            [primitive.start[1], primitive.end[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def render(
    primitives: list[Primitive],
    assignment: np.ndarray,
    components: list[dict[str, Any]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(16, 10), facecolor="white")
    axis.set_facecolor("white")
    for index, primitive in enumerate(primitives):
        if assignment[index] < 0:
            draw_primitive(axis, primitive, "#c7c7c7", 0.65, 0.85)
    palette = plt.get_cmap("tab20")
    for index, component in enumerate(components):
        cluster = int(component["component_id"][1:])
        color = palette(index % 20)
        for primitive_index, primitive in enumerate(primitives):
            if int(assignment[primitive_index]) == cluster:
                draw_primitive(axis, primitive, color, 1.6, 1.0)
        center = component["center"]
        axis.text(
            center[0],
            center[1],
            component["component_id"],
            fontsize=6,
            color="#111111",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.85,
            },
        )
    axis.autoscale()
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    figure.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf", type=Path)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument(
        "--foreground-model",
        type=Path,
        default=PACKAGE_ROOT / "models" / "raw_foreground_model.joblib",
        help=(
            "SVG-trained raw foreground model, protected by the DXF-safe "
            "component-side rescue pass."
        ),
    )
    parser.add_argument(
        "--no-foreground",
        action="store_true",
        help="Disable the foreground stage for comparison.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=PACKAGE_ROOT / "models" / "base_conductor_model.joblib",
    )
    parser.add_argument(
        "--graph-model",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "models"
            / "edge_gated_residual_gnn_blind_dxf_light_adapted.pt"
        ),
    )
    parser.add_argument(
        "--edge-model",
        type=Path,
        default=PACKAGE_ROOT / "models" / "same_component_edge_model.joblib",
    )
    parser.add_argument(
        "--no-edge-model",
        action="store_true",
        help="Use the old policy that joins every component-side contact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "output" / "real_dxf",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--conductor-threshold",
        type=float,
        default=None,
        help="Override the model's DXF conductor threshold.",
    )
    parser.add_argument(
        "--interface-threshold",
        type=float,
        default=None,
        help="Override the model's DXF interface threshold.",
    )
    args = parser.parse_args()
    primitives, evidence, audit = dxf_primitives(args.dxf, args.scale)
    if not primitives:
        raise RuntimeError(f"No supported visible geometry: {args.dxf}")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    foreground_audit: dict[str, Any] = {}
    if not args.no_foreground and args.foreground_model is not None:
        runtime = load_runtime(
            args.foreground_model,
            args.base_model,
            args.graph_model,
            args.device,
            args.edge_model if not args.no_edge_model else None,
        )
        graph_payload = runtime["graph_payload"]
        conductor_threshold, interface_threshold = dxf_role_thresholds(
            graph_payload,
            args.conductor_threshold,
            args.interface_threshold,
        )
        assignment, foreground_audit = segment_full_primitives(
            primitives,
            args.scale,
            runtime,
            conductor_threshold,
            interface_threshold,
            return_internal=True,
            dxf_safe_rescue=True,
        )
        probability = foreground_audit.pop("_full_role_probability")
        foreground_audit.pop("_foreground_indices")
        edge_count = foreground_audit.pop("_graph_edge_count_directed")
        for key in (
            "_foreground_edge_index",
            "_foreground_edge_attr",
            "_foreground_base_features",
            "_foreground_base_wire_probability",
            "_foreground_electrical_probability",
        ):
            foreground_audit.pop(key)
        foreground_stage = "svg_foreground_with_dxf_component_side_rescue"
    else:
        base_model, model, graph_payload = load_models(
            args.base_model,
            args.graph_model,
            device,
        )
        conductor_threshold, interface_threshold = dxf_role_thresholds(
            graph_payload,
            args.conductor_threshold,
            args.interface_threshold,
        )
        features = feature_rows(primitives, args.scale)
        base_probability = base_model.predict_proba(features)[:, 1]
        edge_index, edge_attr = contact_graph(primitives, args.scale)
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
        if not args.no_edge_model and args.edge_model.exists():
            edge_payload = joblib.load(args.edge_model)
            boundary_tolerance = float(
                edge_payload.get("contact_tolerance_ratio", 0.08)
            )
            boundary_edge_index, boundary_edge_attr = contact_graph(
                primitives, args.scale, boundary_tolerance
            )
            assignment, edge_boundary_audit = (
                learned_edge_connected_components(
                    boundary_edge_index,
                    boundary_edge_attr,
                    probability,
                    primitives,
                    features,
                    base_probability,
                    np.ones(len(primitives), dtype=float),
                    edge_payload,
                    conductor_threshold,
                    interface_threshold,
                )
            )
            foreground_audit["boundary_join_method"] = (
                "svg_trained_same_component_edge_on_dxf"
            )
            foreground_audit["edge_boundary_audit"] = edge_boundary_audit
        else:
            assignment = network_connected_components(
                edge_index,
                probability,
                conductor_threshold,
                interface_threshold,
            )
            foreground_audit["boundary_join_method"] = (
                "all_component_side_contact_edges"
            )
        edge_count = int(edge_index.shape[1])
        foreground_stage = "disabled_for_unvalidated_svg_to_dxf_transfer"
    components = cluster_records(
        primitives,
        evidence,
        assignment,
        probability,
    )
    role_count = Counter(
        ["body", "interface", "conductor"][int(index)]
        for index in np.argmax(probability, axis=1)
    )
    singleton_count = sum(
        item["primitive_count"] == 1 for item in components
    )
    warnings = []
    if role_count.get("interface", 0) == 0:
        warnings.append(
            "No primitive was classified as an interface lead; "
            "SVG/XML-to-DXF domain shift is likely."
        )
    if singleton_count > max(3, len(components) // 3):
        warnings.append(
            "Many singleton components indicate boundary over-segmentation."
        )
    result = {
        "schema_version": "1.0",
        "drawing": args.dxf.stem,
        "source_dxf": str(args.dxf.resolve()),
        "scale_h": args.scale,
        "device": str(device),
        "method": {
            "full_drawing_foreground_stage": foreground_stage,
            "base_tree_use": "initial_graph_feature_only",
            "graph_architecture": graph_payload["architecture"],
            "graph_layers": int(graph_payload["layers"]),
            "final_boundary": (
                foreground_audit.get(
                    "boundary_join_method",
                    "network_roles_plus_graph_connected_components",
                )
            ),
            "legacy_boundary_used": False,
            "forced_geometric_seed_used": False,
            "short_stroke_reclaim_used": False,
            "conductor_probability_threshold": (
                conductor_threshold
            ),
            "interface_probability_threshold": interface_threshold,
        },
        "audit": {**audit, **foreground_audit},
        "role_argmax_count": dict(role_count),
        "graph_edge_count_directed": edge_count,
        "component_count": len(components),
        "conductor_primitive_count": int(np.sum(assignment < 0)),
        "quality_audit": {
            "singleton_component_count": singleton_count,
            "component_with_at_most_two_primitives_count": sum(
                item["primitive_count"] <= 2 for item in components
            ),
            "warnings": warnings,
        },
        "components": components,
    }
    output_dir = args.output_dir / args.dxf.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "boundary_result.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = output_dir / "components.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "component_id",
                "primitive_count",
                "center_x",
                "center_y",
                "width",
                "height",
                "mean_body_probability",
                "mean_interface_probability",
                "mean_conductor_probability",
                "source_handles",
                "layers",
            ]
        )
        for item in components:
            writer.writerow(
                [
                    item["component_id"],
                    item["primitive_count"],
                    *item["center"],
                    item["width"],
                    item["height"],
                    item["mean_role_probability"]["component_body"],
                    item["mean_role_probability"]["interface_lead"],
                    item["mean_role_probability"]["main_conductor"],
                    ";".join(item["source_handles"]),
                    ";".join(item["layers"]),
                ]
            )
    image_path = output_dir / "boundary_overlay.png"
    render(primitives, assignment, components, image_path)
    print(f"primitives: {len(primitives)}")
    print(f"directed graph edges: {edge_count}")
    print(f"components: {len(components)}")
    print(f"conductors: {int(np.sum(assignment < 0))}")
    print(f"result: {json_path}")
    print(f"components: {csv_path}")
    print(f"overlay: {image_path}")


if __name__ == "__main__":
    main()
