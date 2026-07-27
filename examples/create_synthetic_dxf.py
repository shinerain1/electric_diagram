#!/usr/bin/env python3
"""Create a small, non-project DXF for an end-to-end smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import ezdxf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # A simple vertical conductor and a two-winding transformer-like motif.
    msp.add_line((0, 500), (0, 100))
    msp.add_circle((0, 25), 50)
    msp.add_circle((0, -75), 50)
    msp.add_line((0, -125), (0, -300))
    msp.add_text("SYNTHETIC TRANSFORMER", height=100).set_placement(
        (150, -75)
    )
    doc.saveas(args.output)


if __name__ == "__main__":
    main()
