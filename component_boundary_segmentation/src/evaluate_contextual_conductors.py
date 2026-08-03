#!/usr/bin/env python3
"""Evaluate whether neighboring primitive evidence improves wire separation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from evaluate_exploded_svg_conductors import (
    FEATURE_NAMES,
    Primitive,
    extract_drawing,
    feature_rows,
)
from evaluate_unknown_component_segmentation import (
    CONFIGS,
    HELD_OUT_CLASSES,
    SegmentationConfig,
    combine_metrics,
    evaluate_segmentation,
    selection_score,
    segment_components,
)


CONTEXT_FEATURE_NAMES = [
    "neighbor_wire_probability_mean_1h",
    "neighbor_wire_probability_max_1h",
    "neighbor_high_wire_fraction_1h",
    "neighbor_wire_probability_mean_3h",
    "neighbor_wire_probability_max_3h",
    "endpoint_neighbor_wire_probability_mean",
    "endpoint_neighbor_wire_probability_max",
    "endpoint_neighbor_high_wire_fraction",
    "collinear_neighbor_wire_probability_mean_1h",
    "collinear_neighbor_wire_probability_max_1h",
    "perpendicular_neighbor_wire_probability_mean_1h",
    "long_neighbor_wire_probability_max_1h",
]


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def probability_summary(
    values: list[float],
    high_threshold: float = 0.80,
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=float)
    return (
        float(np.mean(array)),
        float(np.max(array)),
        float(np.mean(array >= high_threshold)),
    )


def contextual_feature_rows(
    primitives: list[Primitive],
    base_features: np.ndarray,
    base_wire_probability: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Summarize preliminary wire evidence around each primitive."""
    count = len(primitives)
    if count == 0:
        return np.empty((0, len(CONTEXT_FEATURE_NAMES)), dtype=float)
    centers = np.asarray(
        [primitive.center for primitive in primitives],
        dtype=float,
    )
    tree = cKDTree(centers)
    neighbors_1h = tree.query_ball_point(centers, r=scale)
    neighbors_3h = tree.query_ball_point(centers, r=3.0 * scale)

    endpoint_tolerance = max(0.05 * scale, 1e-6)
    endpoint_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    endpoint_keys: list[list[tuple[int, int]]] = []
    for index, primitive in enumerate(primitives):
        keys = []
        for point in (primitive.start, primitive.end):
            if point is None:
                continue
            key = (
                round(point[0] / endpoint_tolerance),
                round(point[1] / endpoint_tolerance),
            )
            endpoint_map[key].append(index)
            keys.append(key)
        endpoint_keys.append(keys)

    output = np.zeros((count, len(CONTEXT_FEATURE_NAMES)), dtype=float)
    for index in range(count):
        local_1h = [
            neighbor
            for neighbor in neighbors_1h[index]
            if neighbor != index
        ]
        local_3h = [
            neighbor
            for neighbor in neighbors_3h[index]
            if neighbor != index
        ]
        mean_1h, max_1h, high_1h = probability_summary(
            [float(base_wire_probability[item]) for item in local_1h]
        )
        mean_3h, max_3h, _ = probability_summary(
            [float(base_wire_probability[item]) for item in local_3h]
        )
        endpoint_neighbors = {
            neighbor
            for key in endpoint_keys[index]
            for neighbor in endpoint_map[key]
            if neighbor != index
        }
        endpoint_mean, endpoint_max, endpoint_high = probability_summary(
            [
                float(base_wire_probability[item])
                for item in endpoint_neighbors
            ]
        )
        horizontal = base_features[index, 7] >= 0.5
        vertical = base_features[index, 8] >= 0.5
        collinear = [
            item
            for item in local_1h
            if (
                (horizontal and base_features[item, 7] >= 0.5)
                or (vertical and base_features[item, 8] >= 0.5)
            )
        ]
        perpendicular = [
            item
            for item in local_1h
            if (
                (horizontal and base_features[item, 8] >= 0.5)
                or (vertical and base_features[item, 7] >= 0.5)
            )
        ]
        collinear_mean, collinear_max, _ = probability_summary(
            [float(base_wire_probability[item]) for item in collinear]
        )
        perpendicular_mean, _, _ = probability_summary(
            [float(base_wire_probability[item]) for item in perpendicular]
        )
        long_neighbors = [
            item for item in local_1h if base_features[item, 0] >= 1.5
        ]
        long_max = (
            max(float(base_wire_probability[item]) for item in long_neighbors)
            if long_neighbors
            else 0.0
        )
        output[index] = [
            mean_1h,
            max_1h,
            high_1h,
            mean_3h,
            max_3h,
            endpoint_mean,
            endpoint_max,
            endpoint_high,
            collinear_mean,
            collinear_max,
            perpendicular_mean,
            long_max,
        ]
    return output


def sample_indices(
    primitives: list[Primitive],
    maximum_component: int,
    maximum_wire: int,
    drawing: str,
    seed: int,
) -> list[int]:
    component = [
        index
        for index, primitive in enumerate(primitives)
        if primitive.label == 0
        and primitive.truth_component_class not in HELD_OUT_CLASSES
    ]
    wire = [
        index
        for index, primitive in enumerate(primitives)
        if primitive.label == 1
    ]

    def select(indices: list[int], maximum: int, label: str) -> list[int]:
        return sorted(
            indices,
            key=lambda index: stable_fraction(
                f"{drawing}:{primitives[index].primitive_id}:{label}",
                seed,
            ),
        )[:maximum]

    return select(component, maximum_component, "component") + select(
        wire, maximum_wire, "wire"
    )


def classification_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = probability >= threshold
    return {
        "primitive_count": int(len(truth)),
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
        default=Path("artifacts/contextual_conductors"),
    )
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation_names = [str(row["drawing"]) for row in manifest["validation"]]
    test_names = [str(row["drawing"]) for row in manifest["test"]]
    ordered_validation = sorted(
        validation_names,
        key=lambda name: stable_fraction(f"context:{name}", args.seed),
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
    for number, drawing in enumerate(validation_names, 1):
        primitives, audit = extract_drawing(
            svg_by_stem[drawing],
            xml_by_stem[drawing],
        )
        base_features = feature_rows(primitives, float(audit["scale"]))
        base_probability = base_model.predict_proba(base_features)[:, 1]
        context = contextual_feature_rows(
            primitives,
            base_features,
            base_probability,
            float(audit["scale"]),
        )
        combined = np.hstack([base_features, context])
        if drawing in calibration_names:
            chosen = sample_indices(
                primitives,
                500,
                300,
                drawing,
                args.seed,
            )
            calibration_x.append(combined[chosen])
            calibration_y.append(
                np.asarray([primitives[index].label for index in chosen])
            )
        elif drawing in selection_names:
            selection_cache.append(
                (
                    primitives,
                    base_features,
                    combined,
                    base_probability,
                    float(audit["scale"]),
                )
            )
        if number % 50 == 0:
            print(f"validation context {number}/{len(validation_names)}")

    context_model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=0.5,
        class_weight="balanced",
        random_state=args.seed,
    )
    context_model.fit(
        np.vstack(calibration_x),
        np.concatenate(calibration_y),
    )
    selection_truth = []
    selection_base_probability = []
    selection_context_probability = []
    for primitives, _, combined, base_probability, _ in selection_cache:
        eligible = [
            index
            for index, primitive in enumerate(primitives)
            if primitive.label == 1
            or primitive.truth_component_class not in HELD_OUT_CLASSES
        ]
        selection_truth.append(
            np.asarray([primitives[index].label for index in eligible])
        )
        selection_base_probability.append(base_probability[eligible])
        selection_context_probability.append(
            context_model.predict_proba(combined[eligible])[:, 1]
        )
    selection_truth_array = np.concatenate(selection_truth)
    selection_base_array = np.concatenate(selection_base_probability)
    selection_context_array = np.concatenate(selection_context_probability)
    thresholds = np.linspace(0.20, 0.99, 160)
    base_threshold = float(
        max(
            thresholds,
            key=lambda threshold: f1_score(
                selection_truth_array,
                selection_base_array >= threshold,
                zero_division=0,
            ),
        )
    )
    context_threshold = float(
        max(
            thresholds,
            key=lambda threshold: f1_score(
                selection_truth_array,
                selection_context_array >= threshold,
                zero_division=0,
            ),
        )
    )
    known_classes = {
        primitive.truth_component_class
        for primitives, _, _, _, _ in selection_cache
        for primitive in primitives
        if primitive.label == 0
        and primitive.truth_component_class not in HELD_OUT_CLASSES
    }
    segmentation_search = []
    for config in CONFIGS:
        rows = []
        for primitives, base_features, combined, _, scale in selection_cache:
            probability = context_model.predict_proba(combined)[:, 1]
            assignment = segment_components(
                primitives,
                base_features,
                probability,
                scale,
                config,
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
        segmentation_search.append(
            {
                "config": asdict(config),
                "selection_metrics": metrics,
                "selection_score": round(selection_score(metrics), 6),
            }
        )
    context_segmentation_config = SegmentationConfig(
        **max(
            segmentation_search,
            key=lambda row: row["selection_score"],
        )["config"]
    )

    baseline_rows = []
    context_rows = []
    unknown_baseline_rows = []
    unknown_context_rows = []
    test_truth = []
    test_base_probability = []
    test_context_probability = []
    for number, drawing in enumerate(test_names, 1):
        primitives, audit = extract_drawing(
            svg_by_stem[drawing],
            xml_by_stem[drawing],
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
        combined = np.hstack([base_features, context])
        context_probability = context_model.predict_proba(combined)[:, 1]
        baseline_assignment = segment_components(
            primitives,
            base_features,
            base_probability,
            scale,
            segmentation_config,
        )
        context_assignment = segment_components(
            primitives,
            base_features,
            context_probability,
            scale,
            context_segmentation_config,
        )
        baseline_rows.append(
            evaluate_segmentation(primitives, baseline_assignment)
        )
        context_rows.append(
            evaluate_segmentation(primitives, context_assignment)
        )
        unknown_baseline_rows.append(
            evaluate_segmentation(
                primitives,
                baseline_assignment,
                include_classes=HELD_OUT_CLASSES,
            )
        )
        unknown_context_rows.append(
            evaluate_segmentation(
                primitives,
                context_assignment,
                include_classes=HELD_OUT_CLASSES,
            )
        )
        test_truth.append(
            np.asarray([primitive.label for primitive in primitives])
        )
        test_base_probability.append(base_probability)
        test_context_probability.append(context_probability)
        if number % 50 == 0:
            print(f"test context {number}/{len(test_names)}")

    truth_array = np.concatenate(test_truth)
    base_probability_array = np.concatenate(test_base_probability)
    context_probability_array = np.concatenate(test_context_probability)
    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "base_feature_count": len(FEATURE_NAMES),
            "context_feature_count": len(CONTEXT_FEATURE_NAMES),
            "combined_feature_count": (
                len(FEATURE_NAMES) + len(CONTEXT_FEATURE_NAMES)
            ),
            "base_feature_names": FEATURE_NAMES,
            "context_feature_names": CONTEXT_FEATURE_NAMES,
            "context_uses_only_geometry_and_base_model_probabilities": True,
            "held_out_component_classes": sorted(HELD_OUT_CLASSES),
        },
        "data": {
            "context_calibration_drawings": len(calibration_names),
            "threshold_selection_drawings": len(selection_names),
            "independent_test_drawings": len(test_names),
            "context_training_primitives": int(
                sum(len(row) for row in calibration_y)
            ),
        },
        "thresholds_selected_without_test": {
            "base": round(base_threshold, 6),
            "context": round(context_threshold, 6),
        },
        "test_conductor_classification": {
            "base_18_features": classification_metrics(
                truth_array,
                base_probability_array,
                base_threshold,
            ),
            "context_30_features": classification_metrics(
                truth_array,
                context_probability_array,
                context_threshold,
            ),
        },
        "test_component_segmentation": {
            "base_18_features": combine_metrics(baseline_rows),
            "context_30_features": combine_metrics(context_rows),
        },
        "test_unknown_component_segmentation": {
            "base_18_features": combine_metrics(unknown_baseline_rows),
            "context_30_features": combine_metrics(unknown_context_rows),
        },
        "base_segmentation_config": asdict(segmentation_config),
        "context_segmentation_parameter_search": segmentation_search,
        "selected_context_segmentation_config": asdict(
            context_segmentation_config
        ),
        "notes": [
            (
                "The context model is trained on one half of validation "
                "drawings; thresholds are selected on the other half."
            ),
            (
                "The 273 test drawings are not used for training, feature "
                "selection, threshold selection, or segmentation parameters."
            ),
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model_path = args.output_dir / "contextual_conductor_model.joblib"
    joblib.dump(
        {
            "schema_version": "1.0",
            "base_feature_names": FEATURE_NAMES,
            "context_feature_names": CONTEXT_FEATURE_NAMES,
            "base_model": base_model,
            "context_model": context_model,
            "base_probability_threshold": base_threshold,
            "context_probability_threshold": context_threshold,
            "base_segmentation_config": asdict(segmentation_config),
            "context_segmentation_config": asdict(
                context_segmentation_config
            ),
        },
        model_path,
    )
    print(f"report: {report_path}")
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
