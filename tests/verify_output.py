#!/usr/bin/env python3
"""Verify the minimum output contract produced by the smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    required = {
        "equipment",
        "terminals",
        "connectivity_nodes",
        "crossings",
        "detailed_topology",
        "engineering_topology",
    }
    missing = sorted(required - set(result))
    if missing:
        raise SystemExit(f"missing output keys: {missing}")
    if result.get("truth_used_during_recognition") is not False:
        raise SystemExit("recognizer did not declare truth independence")
    print("output contract verified")


if __name__ == "__main__":
    main()
