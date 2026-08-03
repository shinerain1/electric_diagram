#!/usr/bin/env python3
"""Evaluate graph message passing for exploded component boundaries.

The model sees anonymous geometric primitives and their physical contact graph.
Component class names, template names and truth instance IDs are never model
inputs.  They are used only to create training labels and independent metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_exploded_svg_conductors import (
    FEATURE_NAMES,
    Primitive,
    extract_drawing,
    feature_rows,
)
from evaluate_three_class_context import (
    ROLE_NAMES,
    classification_metrics,
    sample_role_indices,
    three_class_labels,
)
from evaluate_unknown_component_segmentation import (
    HELD_OUT_CLASSES,
    SegmentationConfig,
    candidate_pairs,
    combine_metrics,
    evaluate_segmentation,
    primitive_distance,
    selection_score,
    segment_components,
    segments_intersect,
)


NODE_FEATURE_NAMES = FEATURE_NAMES + ["base_wire_probability"]
EDGE_FEATURE_NAMES = [
    "contact_distance_over_0_08h",
    "endpoint_endpoint_contact",
    "endpoint_to_interior_contact",
    "interior_crossing",
    "direction_parallel",
    "direction_perpendicular",
    "log_length_ratio_abs",
]


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def primitive_direction(primitive: Primitive) -> tuple[float, float] | None:
    if (
        primitive.kind != "line"
        or primitive.start is None
        or primitive.end is None
    ):
        return None
    dx = primitive.end[0] - primitive.start[0]
    dy = primitive.end[1] - primitive.start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        return None
    return dx / length, dy / length


def point_distance(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def endpoint_distance(left: Primitive, right: Primitive) -> float:
    left_points = [
        point for point in (left.start, left.end) if point is not None
    ]
    right_points = [
        point for point in (right.start, right.end) if point is not None
    ]
    if not left_points or not right_points:
        return float("inf")
    return min(
        point_distance(a, b) for a in left_points for b in right_points
    )


def contact_graph(
    primitives: list[Primitive],
    scale: float,
    tolerance_ratio: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Build directed primitive contacts with explicit relation attributes."""
    tolerance = max(float(tolerance_ratio) * scale, 1e-6)
    source: list[int] = []
    target: list[int] = []
    attributes: list[list[float]] = []
    for left_index, right_index in candidate_pairs(
        primitives,
        range(len(primitives)),
        tolerance,
    ):
        left = primitives[left_index]
        right = primitives[right_index]
        distance = primitive_distance(left, right)
        if distance > tolerance:
            continue
        end_distance = endpoint_distance(left, right)
        endpoint_contact = float(end_distance <= tolerance)
        line_contact = bool(
            left.kind == "line"
            and right.kind == "line"
            and left.start is not None
            and right.start is not None
            and segments_intersect(
                left.start,
                left.end,
                right.start,
                right.end,
            )
        )
        interior_crossing = float(line_contact and not endpoint_contact)
        endpoint_to_interior = float(
            distance <= tolerance
            and not endpoint_contact
            and not interior_crossing
        )
        left_direction = primitive_direction(left)
        right_direction = primitive_direction(right)
        if left_direction is None or right_direction is None:
            parallel = 0.0
            perpendicular = 0.0
        else:
            dot = abs(
                left_direction[0] * right_direction[0]
                + left_direction[1] * right_direction[1]
            )
            parallel = dot
            perpendicular = math.sqrt(max(0.0, 1.0 - dot * dot))
        length_ratio = abs(
            math.log(
                max(left.length, 1e-6) / max(right.length, 1e-6)
            )
        )
        row = [
            min(distance / tolerance, 2.0),
            endpoint_contact,
            endpoint_to_interior,
            interior_crossing,
            parallel,
            perpendicular,
            min(length_ratio, 6.0) / 6.0,
        ]
        source.extend([left_index, right_index])
        target.extend([right_index, left_index])
        attributes.extend([row, row])
    if source:
        edge_index = np.asarray([source, target], dtype=np.int64)
        edge_attr = np.asarray(attributes, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    return edge_index, edge_attr


def graph_record(
    drawing: str,
    svg_path: Path,
    xml_path: Path,
    base_model: Any,
    seed: int,
    training: bool,
) -> dict[str, Any]:
    primitives, audit = extract_drawing(
        svg_path,
        xml_path,
        include_truth_terminals=True,
    )
    scale = float(audit["scale"])
    base_features = feature_rows(primitives, scale)
    base_probability = base_model.predict_proba(base_features)[:, 1]
    node_features = np.hstack(
        [base_features, base_probability.reshape(-1, 1)]
    ).astype(np.float32)
    edge_index, edge_attr = contact_graph(primitives, scale)
    roles = three_class_labels(
        primitives,
        scale,
        audit.get("component_terminals", {}),
    )
    if training:
        supervised = np.asarray(
            sample_role_indices(primitives, roles, drawing, seed),
            dtype=np.int64,
        )
    else:
        supervised = np.arange(len(primitives), dtype=np.int64)
    return {
        "drawing": drawing,
        "primitives": primitives,
        "scale": scale,
        "base_features": base_features,
        "base_probability": base_probability,
        "x": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "roles": roles,
        "supervised": supervised,
    }


class MessageLayer(nn.Module):
    def __init__(self, hidden: int, edge_features: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden + edge_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            aggregate = torch.zeros_like(states)
        else:
            source, target = edge_index
            messages = self.message(
                torch.cat([states[source], edge_attr], dim=1)
            )
            aggregate = torch.zeros_like(states)
            aggregate.index_add_(0, target, messages)
            degree = torch.zeros(
                states.shape[0],
                dtype=states.dtype,
                device=states.device,
            )
            degree.index_add_(
                0,
                target,
                torch.ones_like(target, dtype=states.dtype),
            )
            aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(1)
        change = self.update(torch.cat([states, aggregate], dim=1))
        return F.relu(self.norm(states + change))


class PrimitiveMPNN(nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden: int,
        layers: int,
        dropout: float,
        output_classes: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.layers = nn.ModuleList(
            [MessageLayer(hidden, edge_features) for _ in range(layers)]
        )
        self.dropout = dropout
        self.head = nn.Linear(hidden, output_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        states = self.encoder(x)
        for layer in self.layers:
            states = layer(states, edge_index, edge_attr)
            states = F.dropout(states, self.dropout, self.training)
        return self.head(states)


class EdgeGatedMessageLayer(nn.Module):
    """Edge-aware residual layer with learned edge and node update gates."""

    def __init__(self, hidden: int, edge_features: int) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(hidden + edge_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden * 2 + edge_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.node_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(states)
        gate_sum = torch.zeros(
            (states.shape[0], 1),
            dtype=states.dtype,
            device=states.device,
        )
        if edge_index.numel() != 0:
            source, target = edge_index
            values = self.value(
                torch.cat([states[source], edge_attr], dim=1)
            )
            edge_gate = self.edge_gate(
                torch.cat(
                    [states[source], states[target], edge_attr],
                    dim=1,
                )
            )
            aggregate.index_add_(0, target, edge_gate * values)
            gate_sum.index_add_(0, target, edge_gate)
            aggregate = aggregate / gate_sum.clamp_min(1e-6)
        combined = torch.cat([states, aggregate], dim=1)
        change = self.update(combined)
        update_gate = self.node_gate(combined)
        return F.relu(self.norm(states + update_gate * change))


class EdgeGatedResidualGNN(nn.Module):
    """Deeper edge-gated graph network with across-layer feature fusion."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden: int,
        layers: int,
        dropout: float,
        output_classes: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.layers = nn.ModuleList(
            [
                EdgeGatedMessageLayer(hidden, edge_features)
                for _ in range(layers)
            ]
        )
        self.dropout = dropout
        self.fusion = nn.Sequential(
            nn.Linear(hidden * (layers + 1), hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.head = nn.Linear(hidden, output_classes)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        states = self.encoder(x)
        history = [states]
        for layer in self.layers:
            states = layer(states, edge_index, edge_attr)
            states = F.dropout(states, self.dropout, self.training)
            history.append(states)
        return self.fusion(torch.cat(history, dim=1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encode(x, edge_index, edge_attr))


def build_model(
    architecture: str,
    node_features: int,
    edge_features: int,
    hidden: int,
    layers: int,
    dropout: float,
    output_classes: int = 3,
) -> nn.Module:
    if architecture == "mpnn":
        return PrimitiveMPNN(
            node_features,
            edge_features,
            hidden,
            layers,
            dropout,
            output_classes,
        )
    if architecture == "edge_gated_residual":
        return EdgeGatedResidualGNN(
            node_features,
            edge_features,
            hidden,
            layers,
            dropout,
            output_classes,
        )
    raise ValueError(f"unknown architecture: {architecture}")


def tensor_graph(
    graph: dict[str, Any],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = (graph["x"] - feature_mean) / feature_std
    return (
        torch.as_tensor(normalized, dtype=torch.float32, device=device),
        torch.as_tensor(
            graph["edge_index"],
            dtype=torch.long,
            device=device,
        ),
        torch.as_tensor(
            graph["edge_attr"],
            dtype=torch.float32,
            device=device,
        ),
    )


@torch.no_grad()
def graph_probability(
    model: nn.Module,
    graph: dict[str, Any],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    tensors = tensor_graph(graph, feature_mean, feature_std, device)
    return torch.softmax(model(*tensors), dim=1).cpu().numpy()


def load_graphs(
    names: list[str],
    svg_by_stem: dict[str, Path],
    xml_by_stem: dict[str, Path],
    base_model: Any,
    seed: int,
    training: bool,
    label: str,
) -> list[dict[str, Any]]:
    output = []
    for number, drawing in enumerate(names, 1):
        output.append(
            graph_record(
                drawing,
                svg_by_stem[drawing],
                xml_by_stem[drawing],
                base_model,
                seed,
                training,
            )
        )
        if number % 25 == 0 or number == len(names):
            print(f"{label} graph {number}/{len(names)}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--svg-dir",
        type=Path,
        default=Path("data/svg"),
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=Path("data/xml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "config" / "split_manifest.json",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=ROOT / "models" / "base_conductor_model.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "training",
    )
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument(
        "--architecture",
        choices=["mpnn", "edge_gated_residual"],
        default="mpnn",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation_names = [str(row["drawing"]) for row in manifest["validation"]]
    test_names = [str(row["drawing"]) for row in manifest["test"]]
    ordered_validation = sorted(
        validation_names,
        key=lambda name: stable_fraction(f"three:{name}", args.seed),
    )
    midpoint = len(ordered_validation) // 2
    calibration_names = ordered_validation[:midpoint]
    selection_names = ordered_validation[midpoint:]
    svg_by_stem = {path.stem: path for path in args.svg_dir.glob("*.svg")}
    xml_by_stem = {path.stem: path for path in args.xml_dir.glob("*.xml")}
    base_payload = joblib.load(args.base_model)
    base_model = base_payload["model"]
    segmentation_config = SegmentationConfig(
        **base_payload["segmentation_config"]
    )

    calibration_graphs = load_graphs(
        calibration_names,
        svg_by_stem,
        xml_by_stem,
        base_model,
        args.seed,
        True,
        "calibration",
    )
    selection_graphs = load_graphs(
        selection_names,
        svg_by_stem,
        xml_by_stem,
        base_model,
        args.seed,
        False,
        "selection",
    )
    selected_x = np.vstack(
        [graph["x"][graph["supervised"]] for graph in calibration_graphs]
    )
    feature_mean = selected_x.mean(axis=0).astype(np.float32)
    feature_std = selected_x.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-5] = 1.0
    class_count = np.bincount(
        np.concatenate(
            [
                graph["roles"][graph["supervised"]]
                for graph in calibration_graphs
            ]
        ),
        minlength=3,
    )
    class_weight = np.sqrt(class_count.sum() / np.maximum(class_count, 1))
    class_weight = class_weight / class_weight.mean()
    class_weight_tensor = torch.as_tensor(
        class_weight,
        dtype=torch.float32,
        device=device,
    )

    model = build_model(
        args.architecture,
        len(NODE_FEATURE_NAMES),
        len(EDGE_FEATURE_NAMES),
        args.hidden,
        args.layers,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    history = []
    best_state = None
    best_validation = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(calibration_graphs)
        loss_sum = 0.0
        correct = 0
        observed = 0
        for graph in calibration_graphs:
            x, edge_index, edge_attr = tensor_graph(
                graph,
                feature_mean,
                feature_std,
                device,
            )
            index = torch.as_tensor(
                graph["supervised"],
                dtype=torch.long,
                device=device,
            )
            truth = torch.as_tensor(
                graph["roles"][graph["supervised"]],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edge_index, edge_attr)
            loss = F.cross_entropy(
                logits[index],
                truth,
                weight=class_weight_tensor,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(index)
            correct += int((logits[index].argmax(1) == truth).sum().item())
            observed += len(index)
        selection_truth = []
        selection_prediction = []
        for graph in selection_graphs:
            probability = graph_probability(
                model,
                graph,
                feature_mean,
                feature_std,
                device,
            )
            selection_truth.append(graph["roles"])
            selection_prediction.append(probability.argmax(axis=1))
        validation = classification_metrics(
            np.concatenate(selection_truth),
            np.concatenate(selection_prediction),
        )
        macro_f1 = float(validation["macro_f1"])
        if macro_f1 > best_validation:
            best_validation = macro_f1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "train_loss": round(loss_sum / max(observed, 1), 6),
            "train_accuracy": round(correct / max(observed, 1), 6),
            "selection_macro_f1": macro_f1,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)

    lead_thresholds = [0.30, 0.45, 0.60, 0.75]
    threshold_results = []
    known_classes = {
        primitive.truth_component_class
        for graph in selection_graphs
        for primitive in graph["primitives"]
        if primitive.label == 0
        and primitive.truth_component_class not in HELD_OUT_CLASSES
    }
    selection_probabilities = [
        graph_probability(
            model,
            graph,
            feature_mean,
            feature_std,
            device,
        )
        for graph in selection_graphs
    ]
    for threshold in lead_thresholds:
        rows = []
        for graph, probability in zip(
            selection_graphs,
            selection_probabilities,
        ):
            leads = {
                index
                for index in range(len(graph["primitives"]))
                if probability[index, 1] >= threshold
                and probability[index, 1] > probability[index, 0]
            }
            assignment = segment_components(
                graph["primitives"],
                graph["base_features"],
                graph["base_probability"],
                graph["scale"],
                segmentation_config,
                predicted_interface_leads=leads,
            )
            rows.append(
                evaluate_segmentation(
                    graph["primitives"],
                    assignment,
                    include_classes=known_classes,
                    ignore_other_component_classes=True,
                )
            )
        metrics = combine_metrics(rows)
        threshold_results.append(
            {
                "lead_probability_threshold": threshold,
                "metrics": metrics,
                "selection_score": round(selection_score(metrics), 6),
            }
        )
    selected_threshold = float(
        max(
            threshold_results,
            key=lambda row: row["selection_score"],
        )["lead_probability_threshold"]
    )

    del calibration_graphs
    del selection_graphs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_graphs = load_graphs(
        test_names,
        svg_by_stem,
        xml_by_stem,
        base_model,
        args.seed,
        False,
        "test",
    )
    baseline_rows = []
    graph_rows = []
    unknown_baseline_rows = []
    unknown_graph_rows = []
    truth_roles = []
    predicted_roles = []
    heldout_truth_roles = []
    heldout_predicted_roles = []
    for number, graph in enumerate(test_graphs, 1):
        probability = graph_probability(
            model,
            graph,
            feature_mean,
            feature_std,
            device,
        )
        prediction = probability.argmax(axis=1)
        leads = {
            index
            for index in range(len(graph["primitives"]))
            if probability[index, 1] >= selected_threshold
            and probability[index, 1] > probability[index, 0]
        }
        baseline_assignment = segment_components(
            graph["primitives"],
            graph["base_features"],
            graph["base_probability"],
            graph["scale"],
            segmentation_config,
        )
        graph_assignment = segment_components(
            graph["primitives"],
            graph["base_features"],
            graph["base_probability"],
            graph["scale"],
            segmentation_config,
            predicted_interface_leads=leads,
        )
        baseline_rows.append(
            evaluate_segmentation(graph["primitives"], baseline_assignment)
        )
        graph_rows.append(
            evaluate_segmentation(graph["primitives"], graph_assignment)
        )
        unknown_baseline_rows.append(
            evaluate_segmentation(
                graph["primitives"],
                baseline_assignment,
                include_classes=HELD_OUT_CLASSES,
            )
        )
        unknown_graph_rows.append(
            evaluate_segmentation(
                graph["primitives"],
                graph_assignment,
                include_classes=HELD_OUT_CLASSES,
            )
        )
        truth_roles.append(graph["roles"])
        predicted_roles.append(prediction)
        heldout = [
            index
            for index, primitive in enumerate(graph["primitives"])
            if primitive.label == 0
            and primitive.truth_component_class in HELD_OUT_CLASSES
        ]
        heldout_truth_roles.append(graph["roles"][heldout])
        heldout_predicted_roles.append(prediction[heldout])
        if number % 25 == 0 or number == len(test_graphs):
            print(f"test evaluation {number}/{len(test_graphs)}", flush=True)

    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "name": (
                "edge_gated_residual_gnn"
                if args.architecture == "edge_gated_residual"
                else "edge_aware_primitive_mpnn"
            ),
            "architecture": args.architecture,
            "node_feature_names": NODE_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "message_passing_layers": args.layers,
            "hidden_width": args.hidden,
            "component_type_or_template_used_as_model_input": False,
            "truth_instance_id_used_as_model_input": False,
            "held_out_component_classes": sorted(HELD_OUT_CLASSES),
        },
        "data": {
            "training_drawings": len(calibration_names),
            "threshold_selection_drawings": len(selection_names),
            "independent_test_drawings": len(test_names),
            "training_primitive_count": int(class_count.sum()),
            "training_class_count": class_count.tolist(),
        },
        "device": str(device),
        "training_history": history,
        "best_selection_macro_f1": best_validation,
        "lead_threshold_search": threshold_results,
        "selected_lead_probability_threshold": selected_threshold,
        "test_three_class_primitive_classification": classification_metrics(
            np.concatenate(truth_roles),
            np.concatenate(predicted_roles),
        ),
        "test_held_out_component_role_classification": (
            classification_metrics(
                np.concatenate(heldout_truth_roles),
                np.concatenate(heldout_predicted_roles),
            )
        ),
        "test_component_segmentation": {
            "two_class_geometry_baseline": combine_metrics(baseline_rows),
            "base_conductor_plus_graph_interface": combine_metrics(graph_rows),
        },
        "test_unknown_component_segmentation": {
            "two_class_geometry_baseline": combine_metrics(
                unknown_baseline_rows
            ),
            "base_conductor_plus_graph_interface": combine_metrics(
                unknown_graph_rows
            ),
        },
        "segmentation_config": asdict(segmentation_config),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model_path = args.output_dir / "primitive_mpnn.pt"
    torch.save(
        {
            "schema_version": "1.0",
            "architecture": args.architecture,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "node_feature_names": NODE_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "lead_probability_threshold": selected_threshold,
            "segmentation_config": asdict(segmentation_config),
        },
        model_path,
    )
    print(f"report: {report_path}")
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
