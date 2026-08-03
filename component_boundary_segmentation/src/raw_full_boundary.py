#!/usr/bin/env python3
"""Runtime full-drawing foreground filtering and component boundary inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from evaluate_exploded_svg_conductors import Primitive, feature_rows
from evaluate_graph_message_passing import (
    build_model,
    contact_graph,
    graph_probability,
)
from network_boundary import component_node_mask, network_connected_components
from raw_svg_full import outer_frame_rule, raw_foreground_feature_rows
from same_component_edge import learned_edge_connected_components


ROOT = Path(__file__).resolve().parents[1]


def load_runtime(
    foreground_model_path: Path,
    base_model_path: Path = ROOT / "models" / "base_conductor_model.joblib",
    graph_model_path: Path = ROOT / "models" / "edge_gated_residual_gnn.pt",
    device_name: str = "auto",
    edge_model_path: Path | None = ROOT / "models" / "same_component_edge_model.joblib",
) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if device_name == "cuda"
        or (device_name == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    foreground_payload = joblib.load(foreground_model_path)
    base_payload = joblib.load(base_model_path)
    graph_payload = torch.load(
        graph_model_path, map_location=device, weights_only=False
    )
    graph_model = build_model(
        str(graph_payload["architecture"]),
        len(graph_payload["node_feature_names"]),
        len(graph_payload["edge_feature_names"]),
        int(graph_payload["hidden"]),
        int(graph_payload["layers"]),
        float(graph_payload["dropout"]),
        int(graph_payload.get("output_classes", 3)),
    ).to(device)
    graph_model.load_state_dict(graph_payload["model_state_dict"])
    graph_model.eval()
    edge_payload = (
        joblib.load(edge_model_path)
        if edge_model_path is not None and edge_model_path.exists()
        else None
    )
    return {
        "foreground": foreground_payload,
        "base_model": base_payload["model"],
        "graph_payload": graph_payload,
        "graph_model": graph_model,
        "edge_payload": edge_payload,
        "device": device,
    }


def electrical_foreground_probability(
    primitives: list[Primitive],
    scale: float,
    foreground_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Training stores the repeated CAD-ratio features as float32.  Keep the
    # exact same representation at inference because many standardized frame
    # strokes sit directly on learned tree split thresholds.
    features = raw_foreground_feature_rows(primitives, scale).astype(
        np.float32
    )
    model = foreground_payload["model"]
    electrical_class = int(foreground_payload["electrical_class"])
    column = int(np.flatnonzero(model.classes_ == electrical_class)[0])
    probability = model.predict_proba(features)[:, column]
    frame = outer_frame_rule(primitives)
    threshold = float(
        foreground_payload["electrical_probability_threshold"]
    )
    mask = probability >= threshold
    if foreground_payload.get("outer_frame_rule_forced_background", True):
        mask &= ~frame
    return probability, mask, frame


def segment_full_primitives(
    primitives: list[Primitive],
    scale: float,
    runtime: dict[str, Any],
    conductor_threshold: float = 0.70,
    interface_threshold: float = 0.60,
    return_internal: bool = False,
    dxf_safe_rescue: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return full-index boundary assignments; -1 means non-component."""

    probability, foreground_mask, frame = electrical_foreground_probability(
        primitives, scale, runtime["foreground"]
    )
    rescued_component_side = np.zeros(len(primitives), dtype=bool)
    if dxf_safe_rescue and primitives:
        # The SVG foreground score is strongly shifted on engineering DXF and
        # can reject real component strokes.  Run the role network once on the
        # complete DXF and preserve every node that it regards as component
        # body or interface.  The foreground model may still remove geometry
        # only when both stages agree that it is not component-side.
        full_role_features = feature_rows(primitives, scale)
        full_base_probability = runtime["base_model"].predict_proba(
            full_role_features
        )[:, 1]
        full_edge_index, full_edge_attr = contact_graph(primitives, scale)
        full_graph_payload = runtime["graph_payload"]
        full_role_probability = graph_probability(
            runtime["graph_model"],
            {
                "x": np.hstack(
                    [
                        full_role_features,
                        full_base_probability.reshape(-1, 1),
                    ]
                ).astype(np.float32),
                "edge_index": full_edge_index,
                "edge_attr": full_edge_attr,
            },
            np.asarray(
                full_graph_payload["feature_mean"], dtype=np.float32
            ),
            np.asarray(
                full_graph_payload["feature_std"], dtype=np.float32
            ),
            runtime["device"],
        )
        component_side = component_node_mask(
            full_role_probability,
            conductor_threshold,
            interface_threshold,
        )
        rescued_component_side = component_side & ~foreground_mask & ~frame
        foreground_mask |= component_side
        foreground_mask &= ~frame
        # Keep the role probabilities and contact graph computed from the
        # complete drawing.  Re-running the GNN after filtering changes each
        # node's neighbourhood and caused severe artificial over-segmentation
        # on DXF.  The foreground mask is therefore an output/background gate,
        # not a destructive preprocessing step for message passing.
        safe_role_probability = full_role_probability.copy()
        safe_role_probability[frame] = (0.0, 0.0, 1.0)
        edge_audit: dict[str, Any] = {}
        if runtime.get("edge_payload") is not None:
            boundary_tolerance = float(
                runtime["edge_payload"].get("contact_tolerance_ratio", 0.08)
            )
            boundary_edge_index, boundary_edge_attr = contact_graph(
                primitives, scale, boundary_tolerance
            )
            assignment, edge_audit = learned_edge_connected_components(
                boundary_edge_index,
                boundary_edge_attr,
                safe_role_probability,
                primitives,
                full_role_features,
                full_base_probability,
                np.ones(len(primitives), dtype=float),
                runtime["edge_payload"],
                conductor_threshold,
                interface_threshold,
            )
            boundary_join_method = "learned_same_component_contact_edge"
        else:
            assignment = network_connected_components(
                full_edge_index,
                safe_role_probability,
                conductor_threshold,
                interface_threshold,
            )
            boundary_join_method = "all_component_side_contact_edges"
        assignment[~foreground_mask] = -1
        foreground_indices = np.flatnonzero(foreground_mask)
        audit = {
            "raw_primitive_count": len(primitives),
            "electrical_foreground_count": int(np.sum(foreground_mask)),
            "predicted_background_count": int(np.sum(~foreground_mask)),
            "forced_outer_frame_background_count": int(np.sum(frame)),
            "dxf_safe_foreground_rescue": True,
            "rescued_component_side_count": int(
                np.sum(rescued_component_side)
            ),
            "component_candidate_count": len(
                set(int(value) for value in assignment if value >= 0)
            ),
            "conductor_or_background_primitive_count": int(
                np.sum(assignment < 0)
            ),
            "boundary_join_method": boundary_join_method,
            "edge_boundary_audit": edge_audit,
            "foreground_probability_min": float(np.min(probability)),
            "foreground_probability_max": float(np.max(probability)),
            "message_passing_geometry": "complete_unfiltered_dxf",
        }
        if return_internal:
            audit["_full_role_probability"] = safe_role_probability
            audit["_foreground_indices"] = foreground_indices
            audit["_graph_edge_count_directed"] = int(
                full_edge_index.shape[1]
            )
            audit["_foreground_edge_index"] = full_edge_index
            audit["_foreground_edge_attr"] = full_edge_attr
            audit["_foreground_base_features"] = full_role_features
            audit["_foreground_base_wire_probability"] = (
                full_base_probability
            )
            audit["_foreground_electrical_probability"] = probability
        return assignment, audit
    foreground_indices = np.flatnonzero(foreground_mask)
    assignment = np.full(len(primitives), -1, dtype=np.int32)
    if not len(foreground_indices):
        return assignment, {
            "raw_primitive_count": len(primitives),
            "electrical_foreground_count": 0,
            "forced_outer_frame_background_count": int(np.sum(frame)),
            "component_candidate_count": 0,
        }
    foreground_primitives = [
        primitives[int(index)] for index in foreground_indices
    ]
    role_features = feature_rows(foreground_primitives, scale)
    base_probability = runtime["base_model"].predict_proba(role_features)[
        :, 1
    ]
    edge_index, edge_attr = contact_graph(foreground_primitives, scale)
    graph = {
        "x": np.hstack(
            [role_features, base_probability.reshape(-1, 1)]
        ).astype(np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
    graph_payload = runtime["graph_payload"]
    role_probability = graph_probability(
        runtime["graph_model"],
        graph,
        np.asarray(graph_payload["feature_mean"], dtype=np.float32),
        np.asarray(graph_payload["feature_std"], dtype=np.float32),
        runtime["device"],
    )
    edge_audit: dict[str, Any] = {}
    if runtime.get("edge_payload") is not None:
        boundary_tolerance = float(
            runtime["edge_payload"].get("contact_tolerance_ratio", 0.08)
        )
        boundary_edge_index, boundary_edge_attr = contact_graph(
            foreground_primitives, scale, boundary_tolerance
        )
        foreground_assignment, edge_audit = learned_edge_connected_components(
            boundary_edge_index,
            boundary_edge_attr,
            role_probability,
            foreground_primitives,
            role_features,
            base_probability,
            probability[foreground_indices],
            runtime["edge_payload"],
            conductor_threshold,
            interface_threshold,
        )
        boundary_join_method = "learned_same_component_contact_edge"
    else:
        foreground_assignment = network_connected_components(
            edge_index,
            role_probability,
            conductor_threshold,
            interface_threshold,
        )
        boundary_join_method = "all_component_side_contact_edges"
    assignment[foreground_indices] = foreground_assignment
    audit = {
        "raw_primitive_count": len(primitives),
        "electrical_foreground_count": int(np.sum(foreground_mask)),
        "predicted_background_count": int(np.sum(~foreground_mask)),
        "forced_outer_frame_background_count": int(np.sum(frame)),
        "dxf_safe_foreground_rescue": dxf_safe_rescue,
        "rescued_component_side_count": int(
            np.sum(rescued_component_side)
        ),
        "component_candidate_count": len(
            set(int(value) for value in foreground_assignment if value >= 0)
        ),
        "conductor_primitive_count": int(
            np.sum(foreground_assignment < 0)
        ),
        "boundary_join_method": boundary_join_method,
        "edge_boundary_audit": edge_audit,
        "foreground_probability_min": float(np.min(probability)),
        "foreground_probability_max": float(np.max(probability)),
    }
    if return_internal:
        full_role_probability = np.zeros((len(primitives), 3), dtype=float)
        full_role_probability[:, 2] = 1.0
        full_role_probability[foreground_indices] = role_probability
        audit["_full_role_probability"] = full_role_probability
        audit["_foreground_indices"] = foreground_indices
        audit["_graph_edge_count_directed"] = int(edge_index.shape[1])
        audit["_foreground_edge_index"] = edge_index
        audit["_foreground_edge_attr"] = edge_attr
        audit["_foreground_base_features"] = role_features
        audit["_foreground_base_wire_probability"] = base_probability
        audit["_foreground_electrical_probability"] = probability[
            foreground_indices
        ]
    return assignment, audit


__all__ = [
    "electrical_foreground_probability",
    "load_runtime",
    "segment_full_primitives",
]
