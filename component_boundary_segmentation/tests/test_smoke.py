from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import ezdxf
import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_graph_message_passing import build_model, contact_graph
from network_boundary import network_connected_components
from segment_real_dxf import dxf_primitives, dxf_role_thresholds


def test_packaged_models_load() -> None:
    base = joblib.load(ROOT / "models" / "base_conductor_model.joblib")
    assert base["model"] is not None

    payload = torch.load(
        ROOT / "models" / "edge_gated_residual_gnn.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(
        payload["architecture"],
        len(payload["node_feature_names"]),
        len(payload["edge_feature_names"]),
        int(payload["hidden"]),
        int(payload["layers"]),
        float(payload["dropout"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    assert payload["architecture"] == "edge_gated_residual"
    assert int(payload["layers"]) == 4

    adapted = torch.load(
        ROOT / "models" / "edge_gated_residual_gnn_dxf_adapted.pt",
        map_location="cpu",
        weights_only=False,
    )
    adapted_model = build_model(
        adapted["architecture"],
        len(adapted["node_feature_names"]),
        len(adapted["edge_feature_names"]),
        int(adapted["hidden"]),
        int(adapted["layers"]),
        float(adapted["dropout"]),
    )
    adapted_model.load_state_dict(adapted["model_state_dict"])
    conductor, interface = dxf_role_thresholds(adapted, None, None)
    assert (conductor, interface) == (0.5, 0.7)

    edge_payload = joblib.load(
        ROOT / "models" / "same_component_edge_model_dxf_adapted.joblib"
    )
    assert edge_payload["contact_tolerance_ratio"] == 0.12
    assert edge_payload["same_component_probability_threshold"] == 0.55

    strict = torch.load(
        ROOT / "models" / "edge_gated_residual_gnn_blind_dxf_light_adapted.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert strict["training_contract"][
        "full_visible_svg_geometry_used_as_input"
    ]
    assert not strict["training_contract"][
        "svg_class_or_object_id_used_as_input"
    ]
    assert dxf_role_thresholds(strict, None, None) == (0.3, 0.7)


def test_network_boundary_has_no_geometric_seed_override() -> None:
    probability = np.asarray(
        [
            [0.002, 0.003, 0.995],  # Even a diagonal/circle stays conductor.
            [0.900, 0.050, 0.050],
        ],
        dtype=np.float32,
    )
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    assignment = network_connected_components(
        edge_index,
        probability,
        conductor_threshold=0.60,
        interface_threshold=0.60,
    )
    assert assignment[0] == -1
    assert assignment[1] == 0


def test_network_boundary_uses_graph_connectivity() -> None:
    probability = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.85, 0.05, 0.10],
            [0.90, 0.05, 0.05],
        ],
        dtype=np.float32,
    )
    edge_index = np.asarray(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=np.int64,
    )
    assignment = network_connected_components(
        edge_index,
        probability,
        conductor_threshold=0.60,
        interface_threshold=0.60,
    )
    assert assignment.tolist() == [0, 0, 0, 1]


def test_dxf_arc_is_not_sampled_twice() -> None:
    document = ezdxf.new("R2010")
    document.modelspace().add_arc(
        center=(0.0, 0.0),
        radius=500.0,
        start_angle=0.0,
        end_angle=180.0,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "arc.dxf"
        document.saveas(path)
        primitives, evidence, audit = dxf_primitives(path, scale=100.0)
    assert len(primitives) == 1
    assert primitives[0].kind == "arc"
    assert len(primitives[0].geometry_points) >= 8
    assert len(evidence) == len(primitives)
    assert audit["primitive_count_after_deduplication"] == len(primitives)


def test_dxf_spline_is_one_graph_primitive() -> None:
    document = ezdxf.new("R2010")
    document.modelspace().add_open_spline(
        [
            (0.0, 0.0),
            (70.0, 50.0),
            (140.0, 50.0),
            (210.0, 0.0),
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "spline.dxf"
        document.saveas(path)
        primitives, _, _ = dxf_primitives(path, scale=100.0)
    assert len(primitives) == 1
    assert primitives[0].kind == "spline"
    assert len(primitives[0].geometry_points) >= 2


def test_boundary_graph_can_use_wider_tolerance_than_role_graph() -> None:
    document = ezdxf.new("R2010")
    modelspace = document.modelspace()
    modelspace.add_line((0.0, 0.0), (100.0, 0.0))
    modelspace.add_line((111.0, 0.0), (211.0, 0.0))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "small_gap.dxf"
        document.saveas(path)
        primitives, _, _ = dxf_primitives(path, scale=100.0)
    role_edges, _ = contact_graph(primitives, 100.0)
    boundary_edges, _ = contact_graph(primitives, 100.0, 0.12)
    assert role_edges.shape[1] == 0
    assert boundary_edges.shape[1] == 2


if __name__ == "__main__":
    test_packaged_models_load()
    test_network_boundary_has_no_geometric_seed_override()
    test_network_boundary_uses_graph_connectivity()
    test_dxf_arc_is_not_sampled_twice()
    test_dxf_spline_is_one_graph_primitive()
    test_boundary_graph_can_use_wider_tolerance_than_role_graph()
    print("component boundary package smoke test passed")
