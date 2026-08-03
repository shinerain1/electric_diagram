from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from component_matching_core import Shape
from recognize_topology import (
    candidate_acceptable,
    learned_boundary_shape_groups,
)


def line(handle: str, y: float = 0.0) -> Shape:
    return Shape(
        handle=handle,
        entity=None,
        entity_type="LINE",
        layer="0",
        kind="line",
        bbox=(0.0, y, 100.0, y),
        points=np.asarray([[0.0, y], [100.0, y]], dtype=float),
        length=100.0,
        closed=False,
    )


class LearnedFrontendIntegrationTests(unittest.TestCase):
    def test_primitive_votes_assign_one_handle_to_one_boundary(self) -> None:
        shapes = [line("A"), line("B", 10.0)]

        def infer(*_args):
            return (
                {
                    "component_count": 2,
                    "components": [
                        {
                            "component_id": "C0001",
                            "primitive_ids": ["A:0", "A:1", "B:0"],
                            "boundary_confidence": 0.8,
                        },
                        {
                            "component_id": "C0002",
                            "primitive_ids": ["A:2"],
                            "boundary_confidence": 0.9,
                        },
                    ],
                    "stage2": {},
                    "stage3": {"membership_source": []},
                    "parser_audit": {},
                },
                np.asarray([1, 1, 1, 2]),
                [],
                [],
            )

        groups, _, audit = learned_boundary_shape_groups(
            Path("drawing.dxf"), shapes, set(), {"infer": infer}
        )
        handles = [{shape.handle for shape in group} for group in groups]
        self.assertIn({"A", "B"}, handles)
        self.assertEqual(audit["conflicting_handle_count"], 1)

    def test_stage3_boundary_is_not_deleted_by_stage4_template_rejection(self) -> None:
        shape = line("A")
        result = {
            "candidate": {
                "owned_handles": ["A"],
                "primitive_counts": {"line": 1},
            },
            "top": {"family": "UnknownFamily", "combined_score": 1.0},
            "family_margin": 0.0,
            "learned_frontend_boundary": {"accepted": True},
        }
        accepted, reason = candidate_acceptable([shape], result, 100.0)
        self.assertTrue(accepted)
        self.assertEqual(reason, "learned_iterative_boundary_accepted")


if __name__ == "__main__":
    unittest.main()
