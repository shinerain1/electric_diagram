from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate_exploded_svg_conductors import feature_rows, segment_primitive  # noqa: E402
from evaluate_graph_message_passing import contact_graph  # noqa: E402
from iterative_boundary import (  # noqa: E402
    IterativeBoundaryConfig,
    iterative_component_boundaries,
)


def primitives(points):
    return [
        segment_primitive("test", 0, start, end)
        for start, end in points
    ]


class IterativeBoundaryTests(unittest.TestCase):
    def run_boundary(
        self, items, component_probability, threshold=0.60, source_group_ids=None
    ):
        graph_edges, graph_attributes = contact_graph(items, 100.0)
        features = feature_rows(items, 100.0)
        probability = np.column_stack([
            component_probability,
            np.zeros(len(items)),
            1.0 - np.asarray(component_probability),
        ])
        return iterative_component_boundaries(
            items,
            100.0,
            graph_edges,
            graph_attributes,
            probability,
            np.asarray(component_probability),
            threshold,
            features,
            1.0 - np.asarray(component_probability),
            config=IterativeBoundaryConfig(rescue_edge_probability=0.55),
            source_group_ids=source_group_ids,
        )

    def test_uncertain_attached_stroke_is_reabsorbed(self):
        items = primitives([
            ((0, 0), (50, 50)),
            ((50, 50), (100, 50)),
        ])
        assignment, audit = self.run_boundary(items, [0.95, 0.40])
        self.assertEqual(assignment.tolist(), [0, 0])
        self.assertEqual(audit["reabsorbed_primitive_count"], 1)
        self.assertEqual(audit["membership_source"][1], "reabsorbed")

    def test_long_axis_carrier_is_not_reabsorbed(self):
        items = primitives([
            ((0, 0), (50, 50)),
            ((50, 50), (350, 50)),
            ((200, 50), (200, 100)),
        ])
        assignment, audit = self.run_boundary(items, [0.95, 0.40, 0.10])
        self.assertEqual(assignment[0], 0)
        self.assertEqual(assignment[1], -1)
        self.assertTrue(audit["hard_carrier_mask"][1])

    def test_bridge_between_two_boundaries_is_not_absorbed(self):
        items = primitives([
            ((0, 0), (50, 0)),
            ((50, 0), (100, 0)),
            ((100, 0), (150, 50)),
        ])
        assignment, audit = self.run_boundary(items, [0.95, 0.40, 0.95])
        self.assertNotEqual(assignment[0], assignment[2])
        self.assertEqual(assignment[1], -1)
        self.assertGreaterEqual(audit["rejected_multi_cluster_candidate_count"], 1)

    def test_selected_sides_of_one_closed_source_entity_stay_together(self):
        items = primitives([
            ((0, 0), (50, 0)),
            ((300, 0), (350, 0)),
        ])
        for item in items:
            item.closed = True
        assignment, audit = self.run_boundary(
            items, [0.95, 0.95], source_group_ids=["A", "A"]
        )
        self.assertEqual(assignment.tolist(), [0, 0])
        self.assertEqual(audit["closed_source_entity_union_count"], 1)


if __name__ == "__main__":
    unittest.main()
