from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from component_side import (  # noqa: E402
    apply_stage2_policy,
    component_side_probability,
)
from evaluate_exploded_svg_conductors import Primitive  # noqa: E402


def line(start, end):
    return Primitive(
        drawing="test", label=0, kind="line", closed=False,
        start=start, end=end,
        center=((start[0] + end[0]) / 2, (start[1] + end[1]) / 2),
        bbox=(min(start[0], end[0]), min(start[1], end[1]),
              max(start[0], end[0]), max(start[1], end[1])),
        length=((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5,
    )


class ComponentSideTests(unittest.TestCase):
    def test_three_role_probability_is_body_plus_interface(self):
        probability = np.asarray([[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]])
        np.testing.assert_allclose(
            component_side_probability(probability), [0.5, 0.8]
        )

    def test_long_axis_aligned_stroke_is_fixed_as_conductor(self):
        primitives = [line((0, 0), (200, 0)), line((0, 0), (50, 0))]
        probability = np.asarray([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
        score = apply_stage2_policy(probability, primitives, scale=100.0)
        np.testing.assert_allclose(score, [0.0, 0.9])


if __name__ == "__main__":
    unittest.main()
