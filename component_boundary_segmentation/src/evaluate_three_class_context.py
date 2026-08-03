#!/usr/bin/env python3
"""Train body/interface-lead/main-conductor classification with context."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from evaluate_contextual_conductors import (
    CONTEXT_FEATURE_NAMES,
    contextual_feature_rows,
)
from evaluate_exploded_svg_conductors import (
    FEATURE_NAMES,
    Primitive,
    extract_drawing,
    feature_rows,
)
from evaluate_unknown_component_segmentation import (
    HELD_OUT_CLASSES,
    SegmentationConfig,
    combine_metrics,
    evaluate_segmentation,
    obvious_component_seed,
    point_segment_distance,
    segment_components,
    selection_score,
)


ROLE_NAMES = {
    0: "component_body",
    1: "interface_lead",
    2: "main_conductor",
}

RELATION_FEATURE_NAMES = [
    "endpoint_body_evidence_max",
    "endpoint_body_evidence_min",
    "endpoint_obvious_body_side_count",
    "endpoint_wire_evidence_max",
    "endpoint_wire_evidence_min",
    "endpoint_long_wire_evidence_max",
    "opposite_side_body_wire_bridge_score",
    "one_sided_body_protrusion_score",
    "both_sides_body_score",
    "endpoint_neighbor_count_min",
    "endpoint_neighbor_count_max",
    "endpoint_wire_probability_contrast",
]


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def point_primitive_distance(
    point: tuple[float, float],
    primitive: Primitive,
) -> float:
    if primitive.start is not None:
        return point_segment_distance(
            point,
            primitive.start,
            primitive.end,
        )
    radius = max(
        primitive.bbox[2] - primitive.bbox[0],
        primitive.bbox[3] - primitive.bbox[1],
    ) / 2.0
    return max(
        0.0,
        math.hypot(
            point[0] - primitive.center[0],
            point[1] - primitive.center[1],
        )
        - radius,
    )


def directional_relation_features(
    primitives: list[Primitive],
    base_features: np.ndarray,
    base_wire_probability: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Describe what lies at each end instead of averaging both ends."""
    tolerance = max(0.08 * scale, 1e-6)
    endpoint_map: dict[tuple[int, int], list[tuple[int, tuple[float, float]]]] = (
        defaultdict(list)
    )
    primitive_endpoints = []
    for index, primitive in enumerate(primitives):
        points = [
            point
            for point in (primitive.start, primitive.end)
            if point is not None
        ]
        primitive_endpoints.append(points)
        for point in points:
            key = (
                round(point[0] / tolerance),
                round(point[1] / tolerance),
            )
            endpoint_map[key].append((index, point))
    output = np.zeros(
        (len(primitives), len(RELATION_FEATURE_NAMES)),
        dtype=float,
    )
    for index, points in enumerate(primitive_endpoints):
        if len(points) != 2:
            continue
        side_rows = []
        for point in points:
            key = (
                round(point[0] / tolerance),
                round(point[1] / tolerance),
            )
            neighbors = set()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for neighbor, neighbor_point in endpoint_map.get(
                        (key[0] + dx, key[1] + dy),
                        [],
                    ):
                        if neighbor == index:
                            continue
                        if math.hypot(
                            point[0] - neighbor_point[0],
                            point[1] - neighbor_point[1],
                        ) <= tolerance:
                            neighbors.add(neighbor)
            body_values = [
                max(
                    1.0 - float(base_wire_probability[item]),
                    1.0
                    if obvious_component_seed(
                        primitives[item],
                        base_features[item],
                    )
                    else 0.0,
                )
                for item in neighbors
            ]
            wire_values = [
                float(base_wire_probability[item]) for item in neighbors
            ]
            long_wire_values = [
                float(base_wire_probability[item])
                for item in neighbors
                if base_features[item, 0] >= 1.5
            ]
            side_rows.append(
                {
                    "body": max(body_values, default=0.0),
                    "wire": max(wire_values, default=0.0),
                    "long_wire": max(long_wire_values, default=0.0),
                    "obvious": float(
                        any(
                            obvious_component_seed(
                                primitives[item],
                                base_features[item],
                            )
                            for item in neighbors
                        )
                    ),
                    "count": float(len(neighbors)),
                }
            )
        left, right = side_rows
        body_high = max(left["body"], right["body"])
        body_low = min(left["body"], right["body"])
        wire_high = max(left["wire"], right["wire"])
        wire_low = min(left["wire"], right["wire"])
        output[index] = [
            body_high,
            body_low,
            left["obvious"] + right["obvious"],
            wire_high,
            wire_low,
            max(left["long_wire"], right["long_wire"]),
            max(
                left["body"] * right["wire"],
                right["body"] * left["wire"],
            ),
            max(
                left["body"] * (1.0 - right["body"]),
                right["body"] * (1.0 - left["body"]),
            ),
            min(left["body"], right["body"]),
            min(left["count"], right["count"]),
            max(left["count"], right["count"]),
            abs(left["wire"] - right["wire"]),
        ]
    return output


def three_class_labels(
    primitives: list[Primitive],
    scale: float,
    component_terminals: dict[str, list[list[float]]],
) -> np.ndarray:
    """Create evaluation/training roles from semantic wires and SVG terminals."""
    labels = np.zeros(len(primitives), dtype=np.int8)
    labels[
        np.asarray([primitive.label == 1 for primitive in primitives])
    ] = 2
    component_indices: dict[str, list[int]] = defaultdict(list)
    for index, primitive in enumerate(primitives):
        if primitive.label == 0 and primitive.truth_component_id:
            component_indices[primitive.truth_component_id].append(index)
    tolerance = 0.08 * scale
    for component_id, indices in component_indices.items():
        terminals = [
            (float(point[0]), float(point[1]))
            for point in component_terminals.get(component_id, [])
        ]
        if not terminals:
            continue
        candidates = []
        for index in indices:
            primitive = primitives[index]
            if (
                primitive.kind != "line"
                or primitive.closed
                or primitive.length > 1.5 * scale
            ):
                continue
            if any(
                point_primitive_distance(point, primitive) <= tolerance
                for point in terminals
            ):
                candidates.append(index)
        # Preserve a body seed. A single-stroke symbol is a component body,
        # even when both of its endpoints are terminals.
        if len(candidates) >= len(indices):
            continue
        labels[candidates] = 1
    return labels


def sample_role_indices(
    primitives: list[Primitive],
    roles: np.ndarray,
    drawing: str,
    seed: int,
) -> list[int]:
    limits = {0: 500, 1: 250, 2: 350}
    output = []
    for role, maximum in limits.items():
        candidates = [
            index
            for index, primitive in enumerate(primitives)
            if roles[index] == role
            and (
                role == 2
                or primitive.truth_component_class not in HELD_OUT_CLASSES
            )
        ]
        candidates.sort(
            key=lambda index: stable_fraction(
                f"{drawing}:{primitives[index].primitive_id}:{role}",
                seed,
            )
        )
        output.extend(candidates[:maximum])
    return output


def classification_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=[0, 1, 2],
        zero_division=0,
    )
    return {
        "primitive_count": int(len(truth)),
        "accuracy": round(float(np.mean(truth == prediction)), 6),
        "macro_f1": round(float(np.mean(f1)), 6),
        "confusion_matrix_body_interface_conductor": confusion_matrix(
            truth,
            prediction,
            labels=[0, 1, 2],
        ).tolist(),
        "per_class": {
            ROLE_NAMES[index]: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index in range(3)
        },
    }


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
        default=Path(
            "artifacts/exploded_svg_conductors/split_manifest.json"
        ),
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path(
            "artifacts/unknown_component_segmentation/"
            "unknown_category_conductor_model.joblib"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/three_class_context"),
    )
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation_names = [str(row["drawing"]) for row in manifest["validation"]]
    test_names = [str(row["drawing"]) for row in manifest["test"]]
    ordered_validation = sorted(
        validation_names,
        key=lambda name: stable_fraction(f"three:{name}", args.seed),
    )
    midpoint = len(ordered_validation) // 2
    calibration_names = set(ordered_validation[:midpoint])
    selection_names = set(ordered_validation[midpoint:])
    svg_by_stem = {
        path.stem: path for path in args.svg_dir.glob("*.svg")
    }
    xml_by_stem = {
        path.stem: path for path in args.xml_dir.glob("*.xml")
    }
    base_payload = joblib.load(args.base_model)
    base_model = base_payload["model"]
    segmentation_config = SegmentationConfig(
        **base_payload["segmentation_config"]
    )

    calibration_x = []
    calibration_y = []
    selection_cache = []
    terminal_instance_count = 0
    for number, drawing in enumerate(validation_names, 1):
        primitives, audit = extract_drawing(
            svg_by_stem[drawing],
            xml_by_stem[drawing],
            include_truth_terminals=True,
        )
        scale = float(audit["scale"])
        base_features = feature_rows(primitives, scale)
        base_probability = base_model.predict_proba(base_features)[:, 1]
        context = contextual_feature_rows(
            primitives,
            base_features,
            base_probability,
            scale,
        )
        relation = directional_relation_features(
            primitives,
            base_features,
            base_probability,
            scale,
        )
        combined = np.hstack([base_features, context, relation])
        terminals = audit.get("component_terminals", {})
        roles = three_class_labels(primitives, scale, terminals)
        terminal_instance_count += len(terminals)
        if drawing in calibration_names:
            selected = sample_role_indices(
                primitives,
                roles,
                drawing,
                args.seed,
            )
            calibration_x.append(combined[selected])
            calibration_y.append(roles[selected])
        else:
            selection_cache.append(
                (
                    primitives,
                    base_features,
                    combined,
                    base_probability,
                    roles,
                    scale,
                )
            )
        if number % 50 == 0:
            print(f"validation three-class {number}/{len(validation_names)}")

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.5,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(np.vstack(calibration_x), np.concatenate(calibration_y))
    lead_thresholds = np.asarray([0.30, 0.45, 0.60, 0.65, 0.75])
    threshold_results = []
    known_classes = {
        primitive.truth_component_class
        for primitives, _, _, _, _, _ in selection_cache
        for primitive in primitives
        if primitive.label == 0
        and primitive.truth_component_class not in HELD_OUT_CLASSES
    }
    for threshold in lead_thresholds:
        rows = []
        for (
            primitives,
            base_features,
            combined,
            base_probability,
            _,
            scale,
        ) in selection_cache:
            probability = model.predict_proba(combined)
            leads = {
                index
                for index in range(len(primitives))
                if probability[index, 1] >= threshold
                and probability[index, 1] > probability[index, 0]
            }
            assignment = segment_components(
                primitives,
                base_features,
                base_probability,
                scale,
                segmentation_config,
                predicted_interface_leads=leads,
            )
            rows.append(
                evaluate_segmentation(
                    primitives,
                    assignment,
                    include_classes=known_classes,
                    ignore_other_component_classes=True,
                )
            )
        metrics = combine_metrics(rows)
        threshold_results.append(
            {
                "lead_probability_threshold": round(float(threshold), 6),
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

    baseline_rows = []
    three_class_rows = []
    unknown_baseline_rows = []
    unknown_three_rows = []
    truth_roles = []
    predicted_roles = []
    heldout_truth_roles = []
    heldout_predicted_roles = []
    for number, drawing in enumerate(test_names, 1):
        primitives, audit = extract_drawing(
            svg_by_stem[drawing],
            xml_by_stem[drawing],
            include_truth_terminals=True,
        )
        scale = float(audit["scale"])
        base_features = feature_rows(primitives, scale)
        base_probability = base_model.predict_proba(base_features)[:, 1]
        context = contextual_feature_rows(
            primitives,
            base_features,
            base_probability,
            scale,
        )
        relation = directional_relation_features(
            primitives,
            base_features,
            base_probability,
            scale,
        )
        combined = np.hstack([base_features, context, relation])
        probability = model.predict_proba(combined)
        role_prediction = np.argmax(probability, axis=1)
        roles = three_class_labels(
            primitives,
            scale,
            audit.get("component_terminals", {}),
        )
        leads = {
            index
            for index in range(len(primitives))
            if probability[index, 1] >= selected_threshold
            and probability[index, 1] > probability[index, 0]
        }
        baseline_assignment = segment_components(
            primitives,
            base_features,
            base_probability,
            scale,
            segmentation_config,
        )
        three_assignment = segment_components(
            primitives,
            base_features,
            base_probability,
            scale,
            segmentation_config,
            predicted_interface_leads=leads,
        )
        baseline_rows.append(
            evaluate_segmentation(primitives, baseline_assignment)
        )
        three_class_rows.append(
            evaluate_segmentation(primitives, three_assignment)
        )
        unknown_baseline_rows.append(
            evaluate_segmentation(
                primitives,
                baseline_assignment,
                include_classes=HELD_OUT_CLASSES,
            )
        )
        unknown_three_rows.append(
            evaluate_segmentation(
                primitives,
                three_assignment,
                include_classes=HELD_OUT_CLASSES,
            )
        )
        truth_roles.append(roles)
        predicted_roles.append(role_prediction)
        heldout_indices = [
            index
            for index, primitive in enumerate(primitives)
            if primitive.label == 0
            and primitive.truth_component_class in HELD_OUT_CLASSES
        ]
        heldout_truth_roles.append(roles[heldout_indices])
        heldout_predicted_roles.append(role_prediction[heldout_indices])
        if number % 50 == 0:
            print(f"test three-class {number}/{len(test_names)}")

    report = {
        "schema_version": "1.0",
        "decision": "adopted_as_interface_model_in_hybrid",
        "decision_reason": (
            "Keep the base 18-feature conductor probability and use the "
            "three-class model only to identify interface bridges."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "classes": ROLE_NAMES,
            "feature_count": (
                len(FEATURE_NAMES)
                + len(CONTEXT_FEATURE_NAMES)
                + len(RELATION_FEATURE_NAMES)
            ),
            "base_feature_names": FEATURE_NAMES,
            "context_feature_names": CONTEXT_FEATURE_NAMES,
            "directional_relation_feature_names": RELATION_FEATURE_NAMES,
            "interface_truth_source": (
                "open component stroke touching an SVG symbol terminal; "
                "single-stroke symbols remain component bodies"
            ),
            "component_type_or_template_used_as_model_input": False,
            "held_out_component_classes": sorted(HELD_OUT_CLASSES),
        },
        "data": {
            "context_training_drawings": len(calibration_names),
            "threshold_selection_drawings": len(selection_names),
            "independent_test_drawings": len(test_names),
            "training_primitive_count": int(
                sum(len(row) for row in calibration_y)
            ),
            "validation_component_instances_with_terminals": (
                terminal_instance_count
            ),
        },
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
            "current_two_class_plus_geometry": combine_metrics(baseline_rows),
            "hybrid_base_conductor_plus_three_class_interface": (
                combine_metrics(three_class_rows)
            ),
        },
        "test_unknown_component_segmentation": {
            "current_two_class_plus_geometry": combine_metrics(
                unknown_baseline_rows
            ),
            "hybrid_base_conductor_plus_three_class_interface": (
                combine_metrics(unknown_three_rows)
            ),
        },
        "segmentation_config": asdict(segmentation_config),
        "notes": [
            (
                "The three-class model is trained on 135 validation drawings; "
                "the lead threshold is selected on the other 135 drawings."
            ),
            (
                "The 273 test drawings are not used for training or parameter "
                "selection."
            ),
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model_path = args.output_dir / "three_class_context_model.joblib"
    joblib.dump(
        {
            "schema_version": "1.0",
            "base_feature_names": FEATURE_NAMES,
            "context_feature_names": CONTEXT_FEATURE_NAMES,
            "directional_relation_feature_names": RELATION_FEATURE_NAMES,
            "role_names": ROLE_NAMES,
            "base_model": base_model,
            "three_class_model": model,
            "lead_probability_threshold": selected_threshold,
            "segmentation_config": asdict(segmentation_config),
        },
        model_path,
    )
    print(f"report: {report_path}")
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
