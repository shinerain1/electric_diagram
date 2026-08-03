from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_exploded_svg_conductors import segment_primitive
from same_component_edge import edge_join_labels, structural_carrier_mask


class SameComponentEdgeTests(unittest.TestCase):
    def test_truth_ids_create_edge_labels_but_are_not_features(self) -> None:
        primitives = [
            segment_primitive("x", 0, (0, 0), (1, 0)),
            segment_primitive("x", 0, (1, 0), (2, 0)),
            segment_primitive("x", 0, (2, 0), (3, 0)),
        ]
        items = [item for item in primitives if item is not None]
        items[0].truth_component_id = "A"
        items[1].truth_component_id = "A"
        items[2].truth_component_id = "B"
        labels = edge_join_labels(
            items, np.asarray([[0, 1], [1, 2]], dtype=np.int64)
        )
        self.assertEqual(labels.tolist(), [1, 0])

    def test_long_line_with_two_short_taps_is_a_carrier_candidate(self) -> None:
        primitives = [
            segment_primitive("x", 0, (0, 0), (10, 0)),
            segment_primitive("x", 0, (2, 0), (2, 1)),
            segment_primitive("x", 0, (8, 0), (8, 1)),
        ]
        items = [item for item in primitives if item is not None]
        features = np.zeros((3, 18), dtype=float)
        features[:, 6] = 1.0
        features[:, 0] = [2.0, 0.2, 0.2]
        mask = structural_carrier_mask(
            items,
            np.asarray([[0, 1], [0, 2]], dtype=np.int64),
            features,
        )
        self.assertEqual(mask.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
