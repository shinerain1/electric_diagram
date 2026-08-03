from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate_exploded_svg_conductors import Primitive  # noqa: E402
from evaluate_graph_message_passing import contact_graph  # noqa: E402
from logical_context import (  # noqa: E402
    LOGIC_FEATURE_NAMES,
    RELATION_LOGIC_FEATURE_NAMES,
    collinear_chains,
    logical_feature_rows,
    relation_logical_feature_rows,
)


def line(start, end):
    return Primitive(
        drawing="test", label=0, kind="line", closed=False,
        start=start, end=end,
        center=((start[0] + end[0]) / 2, (start[1] + end[1]) / 2),
        bbox=(min(start[0], end[0]), min(start[1], end[1]),
              max(start[0], end[0]), max(start[1], end[1])),
        length=((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5,
    )


class LogicalContextTests(unittest.TestCase):
    def test_collinear_segments_form_one_chain(self):
        primitives = [
            line((0, 0), (100, 0)),
            line((105, 0), (250, 0)),
            line((120, 0), (120, 80)),
        ]
        ids, chains = collinear_chains(primitives, scale=100.0)
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[0], ids[2])
        self.assertEqual(len(chains), 2)

    def test_long_branched_chain_gets_carrier_feature(self):
        primitives = [
            line((0, 0), (100, 0)),
            line((100, 0), (250, 0)),
            line((100, 0), (100, 80)),
        ]
        edge_index, edge_attr = contact_graph(primitives, scale=100.0)
        features = logical_feature_rows(
            primitives, 100.0, edge_index, edge_attr
        )
        carrier_index = LOGIC_FEATURE_NAMES.index("chain_long_carrier_candidate")
        branch_index = LOGIC_FEATURE_NAMES.index("chain_branch_contact_count")
        self.assertEqual(float(features[0, carrier_index]), 1.0)
        self.assertGreater(float(features[0, branch_index]), 0.0)
        self.assertEqual(float(features[2, carrier_index]), 0.0)

    def test_short_perpendicular_t_branch_is_explicit_evidence(self):
        primitives = [
            line((0, 0), (300, 0)),
            line((150, 0), (150, 100)),
        ]
        edge_index, edge_attr = contact_graph(primitives, scale=100.0)
        features = relation_logical_feature_rows(
            primitives, 100.0, edge_index, edge_attr
        )
        column = RELATION_LOGIC_FEATURE_NAMES.index(
            "short_perpendicular_branch_candidate"
        )
        self.assertEqual(float(features[1, column]), 1.0)

    def test_oblique_stroke_near_symbol_is_component_evidence(self):
        primitives = [
            line((0, 0), (300, 0)),
            line((150, 0), (200, 50)),
            line((200, 50), (250, 50)),
        ]
        edge_index, edge_attr = contact_graph(primitives, scale=100.0)
        features = relation_logical_feature_rows(
            primitives, 100.0, edge_index, edge_attr
        )
        column = RELATION_LOGIC_FEATURE_NAMES.index(
            "short_oblique_component_candidate"
        )
        self.assertEqual(float(features[1, column]), 1.0)


if __name__ == "__main__":
    unittest.main()
