#!/usr/bin/env python3
"""Create deployment model copies without local training drawing names."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import joblib
import torch


def sanitize_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(payload)
    adaptation = output.get("domain_adaptation")
    if isinstance(adaptation, dict):
        for key in ("training_drawings", "source_replay_drawings"):
            values = adaptation.pop(key, None)
            if isinstance(values, list):
                adaptation[f"{key}_count"] = len(values)
        adaptation["local_source_names_removed_for_release"] = True
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = joblib.load(args.source_dir / "base_conductor_model.joblib")
    joblib.dump(
        sanitize_metadata(base),
        args.output_dir / "base_conductor_deployment.joblib",
    )
    roles = torch.load(
        args.source_dir / "hierarchical_component_side_dxf_adapted.pt",
        map_location="cpu",
        weights_only=False,
    )
    torch.save(
        sanitize_metadata(roles),
        args.output_dir / "component_side_dxf_deployment.pt",
    )
    edges = joblib.load(
        args.source_dir / "same_component_edge_model_dxf_adapted.joblib"
    )
    joblib.dump(
        sanitize_metadata(edges),
        args.output_dir / "same_component_edge_dxf_deployment.joblib",
    )


if __name__ == "__main__":
    main()
