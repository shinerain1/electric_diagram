#!/usr/bin/env python3
"""Add non-breaking composite recognition profiles to the component library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    library = read_json(args.source)
    library["profile_extension_version"] = "composite-recognition-profiles-v1"
    profiles = library.setdefault("recognition_profiles", {})
    profiles["PowerTransformerComposite"] = {
        "family": "PowerTransformer",
        "purpose": "DXF炸散图元的完整变压器组合识别",
        "required_geometry": {
            "large_winding_circles": 2,
            "vertical_alignment": True,
        },
        "optional_geometry": {
            "small_phase_circles": [0, 3],
            "vertical_lead": True,
            "aligned_triangle_or_polygon": True,
        },
        "normalization": {
            "rotation": True,
            "mirror": True,
            "triangle_position": ["above_windings", "below_windings"],
        },
        "association_limits_at_h_100": {
            "triangle_axis_offset_max": 100.0,
            "triangle_winding_center_distance_max": 3000.0,
        },
        "terminal_definition": {
            "expected_external_terminal_count": 2,
            "ports": ["high_voltage_side", "low_voltage_side"],
            "position_rule": "topmost_and_bottommost_component_boundary",
        },
        "exclusion_rule":
            "a triangle absorbed by this composite cannot also be emitted as CableTermination",
        "preferred_complete_template_keys": [
            "PowerTransformer:公变1@0",
            "PowerTransformer:专变5@0",
            "PowerTransformer:公变3@0",
        ],
        "fallback_template_keys": [
            "PowerTransformer:专变1@0",
            "PowerTransformer:专变2@0",
            "PowerTransformer:专变3@0",
        ],
    }
    write_json(args.output, library)


if __name__ == "__main__":
    main()
