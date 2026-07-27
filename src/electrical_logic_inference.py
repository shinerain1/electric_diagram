#!/usr/bin/env python3
"""Pure electrical-logic enrichment with no annotation/report dependency."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logic_class(equipment: dict[str, Any]) -> str:
    value = equipment.get("new_type") or equipment.get("type") or "Unknown"
    return {
        "VoltageTransformerFunctionalBranch": "PotentialTransformer",
        "VoltageTransformer": "PotentialTransformer",
        "CableTermination": "CableSegment",
        "CableTerminal": "CableSegment",
        "SwitchCombination": "SwitchCombination",
    }.get(str(value), str(value))


def build_context(
    result: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    equipment = {
        item["equipment_id"]: item for item in result.get("equipment", [])
    }
    node_members: dict[str, set[str]] = defaultdict(set)
    equipment_nodes: dict[str, set[str]] = defaultdict(set)
    for terminal in result.get("terminals", []):
        node = terminal.get("connectivity_node")
        equipment_id = terminal.get("equipment_id")
        if not node or equipment_id not in equipment:
            continue
        node_members[str(node)].add(str(equipment_id))
        equipment_nodes[str(equipment_id)].add(str(node))
    contexts: dict[str, dict[str, Any]] = {}
    for node, members in node_members.items():
        classes = [logic_class(equipment[item]) for item in members]
        switch_count = sum(value == "SwitchCombination" for value in classes)
        contexts[node] = {
            "member_equipment_ids": sorted(members),
            "member_logic_classes": dict(Counter(classes)),
            "busbar_like": len(members) >= 3 or switch_count >= 2,
        }
    return (
        {key: sorted(value) for key, value in equipment_nodes.items()},
        contexts,
    )


def physical_switch_candidates(
    equipment: dict[str, Any],
    patterns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    maximum = max(
        (int(item.get("occurrences", 0)) for item in patterns),
        default=1,
    )
    template_text = " ".join(equipment.get("matched_templates") or [])
    output = []
    for item in patterns:
        classes = sorted(
            [
                str(item.get("switch_class_a", "")),
                str(item.get("switch_class_b", "")),
            ]
        )
        geometry = (
            1.0
            if "Breaker" in template_text and "Disconnector" in template_text
            else 0.35
            if {"Breaker", "Disconnector"} & set(classes)
            else 0.05
        )
        occurrences = int(item.get("occurrences", 0))
        frequency = math.log1p(occurrences) / math.log1p(maximum)
        output.append(
            {
                "classes": classes,
                "name": "+".join(classes),
                "combined_score": round(
                    0.65 * geometry + 0.35 * frequency,
                    6,
                ),
                "geometry_score": round(geometry, 6),
                "xml_logic_score": round(frequency, 6),
                "occurrences": occurrences,
                "file_count": int(item.get("file_count", 0)),
                "evidence_grade": item.get("evidence_grade"),
            }
        )
    return sorted(
        output,
        key=lambda item: (
            -item["combined_score"],
            -item["occurrences"],
            item["name"],
        ),
    )[:5]


def infer_switch(
    equipment: dict[str, Any],
    equipment_by_id: dict[str, dict[str, Any]],
    equipment_nodes: dict[str, list[str]],
    node_context: dict[str, dict[str, Any]],
    patterns: list[dict[str, Any]],
) -> dict[str, Any]:
    sides = []
    for node in equipment_nodes.get(equipment["equipment_id"], []):
        context = node_context[node]
        neighbors = [
            item
            for item in context["member_equipment_ids"]
            if item != equipment["equipment_id"]
        ]
        classes = [
            logic_class(equipment_by_id[item])
            for item in neighbors
            if item in equipment_by_id
        ]
        sides.append(
            {
                "connectivity_node": node,
                "neighbor_equipment_ids": neighbors,
                "neighbor_logic_classes": dict(Counter(classes)),
                "busbar_like": context["busbar_like"],
            }
        )
    bus_sides = sum(bool(side["busbar_like"]) for side in sides)

    def has_non_bus(*classes: str) -> bool:
        return any(
            not side["busbar_like"]
            and any(side["neighbor_logic_classes"].get(value, 0) for value in classes)
            for side in sides
        )

    candidates = []
    if has_non_bus("PowerTransformer", "TransformerWinding"):
        candidates.append(
            ("TransformerHVFeederOrProtectionSwitch", 0.90)
        )
    if has_non_bus("PotentialTransformer", "VoltageTransformer"):
        candidates.append(("PTIsolationOrProtectionSwitch", 0.86))
    if bus_sides >= 2:
        candidates.append(("BusCouplerOrSectionSwitch", 0.78))
    if bus_sides >= 1 and has_non_bus("CableSegment"):
        candidates.append(("IncomingOrOutgoingFeederSwitch", 0.80))
    if not candidates:
        candidates.append(("SerialSwitchCombination", 0.62))
    candidates.sort(key=lambda item: -item[1])
    physical = physical_switch_candidates(equipment, patterns)
    return {
        "functional_role": candidates[0][0],
        "role_confidence": candidates[0][1],
        "functional_role_candidates": [
            {"functional_role": role, "confidence": confidence}
            for role, confidence in candidates
        ],
        "physical_type_candidates": physical,
        "selected_physical_composition": (
            physical[0]["name"] if physical else ""
        ),
        "physical_type_confidence": (
            physical[0]["combined_score"] if physical else 0.0
        ),
        "connected_equipment": {"side_contexts": sides},
        "conflicts": (
            ["multiple electrically plausible roles retained"]
            if len(candidates) > 1
            else []
        ),
    }


def enhance_one(
    base: dict[str, Any],
    component_library_path: Path,
    component_library: dict[str, Any],
    logic_library_path: Path,
    logic_library: dict[str, Any],
) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    equipment_by_id = {
        item["equipment_id"]: item for item in result.get("equipment", [])
    }
    equipment_nodes, node_context = build_context(result)
    patterns = logic_library.get("serial_switch_patterns", [])
    role_counts: Counter[str] = Counter()
    switch_count = 0
    for item in result.get("equipment", []):
        item["component_library_inference"] = {
            "type": item.get("new_type"),
            "match_mode": item.get("match_mode"),
            "matched_templates": item.get("matched_templates", []),
            "confidence": item.get("confidence"),
            "basis": item.get("basis"),
        }
        if item.get("new_type") != "SwitchCombination":
            item["logic_inference"] = None
            continue
        switch_count += 1
        inference = infer_switch(
            item,
            equipment_by_id,
            equipment_nodes,
            node_context,
            patterns,
        )
        item["logic_inference"] = inference
        role_counts[inference["functional_role"]] += 1
    result["schema_version"] = "independent-dual-library-recognition-v2"
    result["truth_used_during_recognition"] = False
    result["knowledge_libraries"] = {
        "component_library": {
            "path": str(component_library_path),
            "sha256": sha256(component_library_path),
            "schema_version": component_library.get("schema_version"),
            "use": "component geometry and template evidence",
        },
        "electrical_logic_library": {
            "path": str(logic_library_path),
            "sha256": sha256(logic_library_path),
            "schema_version": logic_library.get("schema_version"),
            "use": "connected-object context and switch role inference",
            "serial_switch_patterns_used": len(patterns),
        },
    }
    result["logic_inference_summary"] = {
        "equipment_count": len(result.get("equipment", [])),
        "switch_combination_count": switch_count,
        "functional_role_counts": dict(role_counts),
        "component_template_count": len(
            component_library.get("templates", [])
        ),
        "logic_serial_pattern_count": len(patterns),
        "logic_engineering_rule_count": len(
            logic_library.get("engineering_logic_rules", [])
        ),
    }
    return result
