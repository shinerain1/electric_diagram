#!/usr/bin/env python3
"""Create public knowledge-library copies without local paths or file names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIVATE_EVIDENCE_KEYS = {
    "examples",
    "source_svg_examples",
    "source_xml_examples",
}
LOCAL_DIRECTORY_KEYS = {
    "svg_directory",
    "xml_directory",
}


def sanitize(value: Any) -> Any:
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if not isinstance(value, dict):
        return value

    output: dict[str, Any] = {}
    for key, item in value.items():
        if key in PRIVATE_EVIDENCE_KEYS:
            continue
        if key in LOCAL_DIRECTORY_KEYS:
            output[key] = "<source dataset not bundled>"
            continue
        output[key] = sanitize(item)
    return output


def convert(source: Path, destination: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(sanitize(data), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-source", required=True, type=Path)
    parser.add_argument("--logic-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    convert(
        args.component_source,
        args.output_dir / "standard_component_library.json",
    )
    convert(
        args.logic_source,
        args.output_dir / "electrical_logic_library.json",
    )


if __name__ == "__main__":
    main()
