from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segment_svg_blind import svg_primitives_blind
from evaluate_exploded_svg_conductors import segment_primitive
from raw_svg_full import outer_frame_rule


class BlindSvgInferenceTests(unittest.TestCase):
    def test_no_xml_or_semantic_truth_is_read(self) -> None:
        svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 100 100">
  <defs>
    <symbol id="Breaker:semantic-name" viewBox="0 0 10 10">
      <line x1="0" y1="0" x2="10" y2="10" />
    </symbol>
  </defs>
  <g id="BreakerClass">
    <use xlink:href="#Breaker:semantic-name" x="20" y="20"
         width="10" height="10" />
    <line x1="30" y1="30" x2="80" y2="30" />
  </g>
</svg>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind.svg"
            path.write_text(svg, encoding="utf-8")
            primitives, audit = svg_primitives_blind(path)
        self.assertEqual(len(primitives), 2)
        self.assertFalse(audit["xml_file_read"])
        self.assertFalse(audit["svg_class_group_id_used"])
        self.assertFalse(audit["svg_object_id_used"])
        self.assertTrue(all(item.truth_component_id is None for item in primitives))
        self.assertTrue(all(not item.truth_component_class for item in primitives))

    def test_outer_frame_requires_rectangular_evidence(self) -> None:
        lines = [
            segment_primitive("x", 0, (0, 0), (100, 0)),
            segment_primitive("x", 0, (0, 80), (100, 80)),
            segment_primitive("x", 0, (0, 0), (0, 80)),
            segment_primitive("x", 0, (100, 0), (100, 80)),
            segment_primitive("x", 0, (20, 40), (80, 40)),
        ]
        primitives = [item for item in lines if item is not None]
        mask = outer_frame_rule(primitives)
        self.assertEqual(mask[:4].tolist(), [True, True, True, True])
        self.assertFalse(bool(mask[4]))

    def test_single_long_line_is_not_forced_to_background(self) -> None:
        line = segment_primitive("x", 0, (0, 0), (100, 0))
        self.assertIsNotNone(line)
        mask = outer_frame_rule([line])
        self.assertFalse(bool(mask[0]))


if __name__ == "__main__":
    unittest.main()
