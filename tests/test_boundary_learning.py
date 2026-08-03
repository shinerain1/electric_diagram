from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boundary_learning import FEATURE_NAMES, candidate_feature_vector


class BoundaryFeatureTests(unittest.TestCase):
    def test_feature_vector_is_finite_and_category_free(self) -> None:
        group = [
            {
                "handle": "A",
                "kind": "circle",
                "bbox": (0.0, 0.0, 1.0, 1.0),
                "length": 3.14,
                "closed": True,
            },
            {
                "handle": "B",
                "kind": "line",
                "bbox": (1.0, 0.5, 4.0, 0.5),
                "length": 3.0,
                "closed": False,
            },
        ]
        vector = candidate_feature_vector(group, {"A"})
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(vector).all())
        self.assertFalse(
            any(
                token in name
                for name in FEATURE_NAMES
                for token in ("family", "type", "name", "xml", "template")
            )
        )


if __name__ == "__main__":
    unittest.main()
