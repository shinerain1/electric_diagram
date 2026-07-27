#!/usr/bin/env python3
"""Evaluate two-level automatic topology against an already sealed visual label.

This module is intentionally separate from the recognizer.  It reads automatic
output only after recognition has finished and never feeds truth back into any
recognition rule.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


TYPE_MAP = {
    "公用智能柜": "SmartPublicCabinet",
    "公用柜": "PublicCabinet",
    "公用DTU柜": "DTUCabinet",
    "专用环网柜": "RingMainUnit",
    "专用变压器": "PowerTransformer",
    "变压器": "PowerTransformer",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_name(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or "")).upper()
    # The same cabinet may be labeled with or without its long asset number,
    # while the visual truth may append internal bay labels such as K1/K3/K2.
    compact = re.sub(r"^\d{6,}", "", compact)
    compact = re.sub(r"K\d+(?:/K\d+)+$", "", compact)
    return compact


def transformer_number(value: str) -> str | None:
    match = re.search(r"新建\s*(\d+)\s*#", str(value or ""))
    return match.group(1) if match else None


def identity_key(type_name: str, name: str) -> str:
    if type_name in {"PowerTransformer", "RingMainUnit"}:
        number = transformer_number(name)
        if number is not None:
            return f"{type_name}:新建{number}#"
    return f"{type_name}:{normalize_name(name)}"


def metrics(
    predicted: set[Any],
    expected: set[Any],
) -> dict[str, Any]:
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "predicted": len(predicted),
        "expected": len(expected),
        "true_positive": true_positive,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "missing": sorted(expected - predicted, key=str),
        "extra": sorted(predicted - expected, key=str),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--automatic", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    seal = read_json(args.seal)
    if not seal.get("sealed"):
        raise RuntimeError("visual truth is not sealed")
    automatic = read_json(args.automatic)
    truth = read_json(args.truth)
    engineering = automatic["engineering_topology"]

    truth_id_to_key = {}
    truth_types = Counter()
    for item in truth.get("equipment", []):
        canonical_type = TYPE_MAP.get(str(item.get("type")), str(item.get("type")))
        name = str(item.get("visible_label") or item.get("type") or "")
        key = identity_key(canonical_type, name)
        truth_id_to_key[str(item["id"])] = key
        truth_types[canonical_type] += 1

    auto_id_to_key = {}
    automatic_types = Counter()
    for item in engineering.get("equipment", []):
        canonical_type = str(item.get("type") or "")
        key = identity_key(canonical_type, str(item.get("name") or ""))
        auto_id_to_key[str(item["engineering_equipment_id"])] = key
        automatic_types[canonical_type] += 1

    truth_equipment = set(truth_id_to_key.values())
    auto_equipment = set(auto_id_to_key.values())

    truth_relations = {
        (
            truth_id_to_key[str(item["from"])],
            truth_id_to_key[str(item["to"])],
            str(item.get("relation") or ""),
        )
        for item in truth.get("device_relations", [])
    }
    auto_relations = {
        (
            auto_id_to_key[str(item["from_equipment"])],
            auto_id_to_key[str(item["to_equipment"])],
            str(item.get("relation") or ""),
        )
        for item in engineering.get("device_relations", [])
    }

    truth_nodes = {
        tuple(
            sorted(
                truth_id_to_key[str(reference)]
                for reference in item.get("connected_equipment_refs", [])
            )
        )
        for item in truth.get("connectivity_nodes", [])
    }
    auto_nodes = {
        tuple(
            sorted(
                auto_id_to_key[str(reference)]
                for reference in item.get("connected_equipment_ids", [])
            )
        )
        for item in engineering.get("connectivity_nodes", [])
    }

    payload = {
        "schema_version": "sealed-two-level-topology-evaluation-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "recognition_and_evaluation_code_separate": True,
            "truth_used_during_recognition": automatic.get(
                "truth_used_during_recognition"
            ),
            "sealed_before_comparison": True,
        },
        "type_counts": {
            "truth": dict(truth_types),
            "automatic": dict(automatic_types),
            "exact_match": truth_types == automatic_types,
        },
        "equipment": metrics(auto_equipment, truth_equipment),
        "device_relations": metrics(auto_relations, truth_relations),
        "connectivity_nodes": metrics(auto_nodes, truth_nodes),
        "unavailable_metrics": [
            "terminal F1: sealed truth has no terminal coordinates",
            "crossing F1: sealed truth has no crossing annotations",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "系统接线图_两级拓扑独立评价.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 系统接线图两级拓扑独立评价",
        "",
        "识别程序先独立生成结果；本评价程序随后才读取封存视觉标注。",
        "",
        "|评价项|预测|真值|准确率|召回率|F1|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("工程设备", "equipment"),
        ("柜体—变压器关系", "device_relations"),
        ("工程连接节点", "connectivity_nodes"),
    ):
        item = payload[key]
        lines.append(
            f"|{label}|{item['predicted']}|{item['expected']}|"
            f"{item['precision']:.2%}|{item['recall']:.2%}|"
            f"{item['f1']:.2%}|"
        )
    lines += [
        "",
        f"- 设备类型数量完全一致："
        f"{'是' if payload['type_counts']['exact_match'] else '否'}",
        "- 接口和交叉点没有坐标真值，因此不计算相应F1。",
    ]
    (args.output_dir / "系统接线图_两级拓扑独立评价.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
