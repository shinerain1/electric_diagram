from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RDF_ID = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}ID"
RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"

SWITCH_CLASSES = {
    "Breaker",
    "LoadBreakSwitch",
    "Disconnector",
    "GroundDisconnector",
    "Fuse",
    "Switch",
    "Recloser",
    "Sectionaliser",
}

# These CIM objects carry current between ConnectivityNodes. Contracting them
# yields the equipment that is connected through wires/busbars rather than
# merely sharing one XML node.
CONNECTOR_CLASSES = {
    "ACLineSegment",
    "DCLineSegment",
    "BusbarSection",
    "Junction",
    "CableSegment",
    "Conductor",
    "SeriesCompensator",
}

# Terminal-bearing drawing/location helpers are evidence about installation
# structure, not electrical functional equipment.
STRUCTURAL_CLASSES = {
    "Pole",
    "PoleSite",
    "Tower",
    "Bay",
    "VoltageLevel",
    "Substation",
    "EquipmentContainer",
}

ROLE_KEYWORDS = [
    ("bus_coupler", re.compile(r"母联|母线联络")),
    ("tie_switch", re.compile(r"联络")),
    ("incoming_switch", re.compile(r"进线|主进|进柜")),
    ("outgoing_switch", re.compile(r"出线|主出|出柜")),
    ("sectionalizing_switch", re.compile(r"分段")),
    ("transformer_switch", re.compile(r"配变|变压器|台变|专变")),
    ("grounding_switch", re.compile(r"地刀|接地")),
]

CIM_TO_TEMPLATE_FAMILY = {
    "Breaker": "Breaker",
    "LoadBreakSwitch": "LoadSwitch",
    "Disconnector": "Disconnector",
    "GroundDisconnector": "GroundDisconnector",
    "Fuse": "Fuse",
    "PowerTransformer": "PowerTransformer",
    "TransformerWinding": "PowerTransformer",
    "PotentialTransformer": "PT",
    "VoltageTransformer": "PT",
    "CurrentTransformer": "CT",
    "Junction": "ConnectivePoint",
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def ref_id(value: str | None) -> str:
    if not value:
        return ""
    return value[1:] if value.startswith("#") else value


def normalized_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def evidence_grade(occurrences: int, file_count: int) -> str:
    if occurrences >= 100 and file_count >= 20:
        return "high"
    if occurrences >= 20 and file_count >= 5:
        return "medium"
    return "exploratory"


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[item] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass
class ObjectRecord:
    object_id: str
    class_name: str
    name: str
    subtype: str
    base_voltage: str
    container: str
    circuit_section: str
    normal_open: str
    references: dict[str, str]


@dataclass
class TerminalRecord:
    terminal_id: str
    equipment_id: str
    node_id: str
    sequence: str


def parse_xml(path: Path) -> tuple[
    dict[str, ObjectRecord],
    list[TerminalRecord],
    list[dict[str, str]],
]:
    objects: dict[str, ObjectRecord] = {}
    terminals: list[TerminalRecord] = []
    parse_warnings: list[dict[str, str]] = []
    try:
        iterator = ET.iterparse(path, events=("end",))
        for _, element in iterator:
            object_id = element.attrib.get(RDF_ID)
            if not object_id:
                continue
            class_name = local_name(element.tag)
            properties: dict[str, str] = {}
            references: dict[str, str] = {}
            for child in element:
                property_name = local_name(child.tag)
                text = normalized_text(child.text)
                resource = ref_id(child.attrib.get(RDF_RESOURCE))
                if text:
                    properties[property_name] = text
                if resource:
                    references[property_name] = resource
            if class_name == "Terminal":
                equipment_id = references.get(
                    "Terminal.ConductingEquipment", ""
                )
                node_id = references.get("Terminal.ConnectivityNode", "")
                if equipment_id and node_id:
                    terminals.append(
                        TerminalRecord(
                            object_id,
                            equipment_id,
                            node_id,
                            properties.get("Terminal.sequenceNumber", ""),
                        )
                    )
                else:
                    parse_warnings.append(
                        {
                            "file": path.name,
                            "object_id": object_id,
                            "warning": "terminal_missing_equipment_or_node",
                        }
                    )
            elif class_name != "ConnectivityNode":
                objects[object_id] = ObjectRecord(
                    object_id=object_id,
                    class_name=class_name,
                    name=properties.get("Naming.name", ""),
                    subtype=properties.get(
                        "PowerSystemResource.subType", ""
                    ),
                    base_voltage=references.get(
                        "PowerSystemResource.BaseVoltage", ""
                    ),
                    container=references.get(
                        "Equipment.MemberOf_EquipmentContainer", ""
                    ),
                    circuit_section=references.get(
                        "Equipment.MemberOf_CircuitSection", ""
                    ),
                    normal_open=properties.get("Switch.normalOpen", ""),
                    references=references,
                )
            element.clear()
    except Exception as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
    return objects, terminals, parse_warnings


class Aggregate:
    def __init__(self) -> None:
        self.identity = hex(id(self))
        self.occurrences: Counter[str] = Counter()
        self.files: Counter[str] = Counter()
        self.examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(
        self,
        key: str,
        file_name: str,
        example: dict[str, Any] | None = None,
        seen_in_file: set[str] | None = None,
    ) -> None:
        self.occurrences[key] += 1
        file_key = f"{self.identity}\0{file_name}\0{key}"
        if seen_in_file is None or file_key not in seen_in_file:
            self.files[key] += 1
            if seen_in_file is not None:
                seen_in_file.add(file_key)
        if example is not None and len(self.examples[key]) < 3:
            self.examples[key].append(example)

    def records(
        self,
        decode,
        minimum_occurrences: int = 1,
        minimum_files: int = 1,
    ) -> list[dict[str, Any]]:
        result = []
        for key, occurrences in self.occurrences.items():
            file_count = self.files[key]
            if (
                occurrences < minimum_occurrences
                or file_count < minimum_files
            ):
                continue
            result.append(
                {
                    **decode(key),
                    "occurrences": occurrences,
                    "file_count": file_count,
                    "evidence_grade": evidence_grade(
                        occurrences, file_count
                    ),
                    "examples": self.examples[key],
                }
            )
        result.sort(
            key=lambda item: (
                -item["file_count"],
                -item["occurrences"],
                json.dumps(item, ensure_ascii=False, sort_keys=True),
            )
        )
        return result


def pair_key(left: str, right: str) -> str:
    return "\0".join(sorted((left, right)))


def decode_pair(key: str) -> dict[str, Any]:
    left, right = key.split("\0")
    return {"class_a": left, "class_b": right}


def switch_neighbor_key(switch_class: str, neighbor_class: str) -> str:
    return f"{switch_class}\0{neighbor_class}"


def decode_switch_neighbor(key: str) -> dict[str, Any]:
    switch_class, neighbor_class = key.split("\0")
    return {
        "switch_class": switch_class,
        "neighbor_class": neighbor_class,
    }


def serial_key(left: str, right: str) -> str:
    return pair_key(left, right)


def decode_serial(key: str) -> dict[str, Any]:
    left, right = key.split("\0")
    return {"switch_class_a": left, "switch_class_b": right}


def role_key(class_name: str, role: str, source: str) -> str:
    return f"{class_name}\0{role}\0{source}"


def decode_role(key: str) -> dict[str, Any]:
    class_name, role, source = key.split("\0")
    return {
        "equipment_class": class_name,
        "role": role,
        "label_source": source,
    }


def context_side_descriptor(
    root: str,
    equipment_id: str,
    boundaries: dict[str, list[str]],
    connector_counts: dict[str, Counter[str]],
    objects: dict[str, ObjectRecord],
) -> dict[str, Any]:
    neighbor_counts = Counter(
        objects[item].class_name
        for item in boundaries.get(root, [])
        if item != equipment_id and item in objects
    )
    connectors = connector_counts.get(root, Counter())
    return {
        "neighbor_classes": dict(sorted(neighbor_counts.items())),
        "connector_classes": dict(sorted(connectors.items())),
        "has_busbar": bool(connectors.get("BusbarSection")),
        "has_line": any(
            connectors.get(item)
            for item in ("ACLineSegment", "DCLineSegment", "CableSegment")
        ),
    }


def context_key(class_name: str, sides: list[dict[str, Any]]) -> str:
    normalized_sides = sorted(
        sides,
        key=lambda item: json.dumps(
            item, ensure_ascii=False, sort_keys=True
        ),
    )
    return class_name + "\0" + json.dumps(
        normalized_sides,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def decode_context(key: str) -> dict[str, Any]:
    class_name, encoded = key.split("\0", 1)
    return {
        "switch_class": class_name,
        "terminal_sides": json.loads(encoded),
    }


def serial_context_key(members: list[dict[str, Any]]) -> str:
    normalized = sorted(
        members,
        key=lambda item: json.dumps(
            item, ensure_ascii=False, sort_keys=True
        ),
    )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def decode_serial_context(key: str) -> dict[str, Any]:
    members = json.loads(key)
    return {
        "switch_members": members,
        "switch_pair": sorted(
            member["switch_class"] for member in members
        ),
    }


def csv_rows(records: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = {}
        for field in fields:
            value = record.get(field, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            row[field] = value
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_engineering_rules(
    switch_neighbor_records: list[dict[str, Any]],
    serial_records: list[dict[str, Any]],
    explicit_role_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    neighbor_lookup = {
        (item["switch_class"], item["neighbor_class"]): item
        for item in switch_neighbor_records
    }
    rules = []

    def evidence_for_neighbors(
        neighbors: set[str],
        switch_classes: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "switch_class": switch_class,
                "neighbor_class": neighbor,
                "occurrences": item["occurrences"],
                "file_count": item["file_count"],
                "evidence_grade": item["evidence_grade"],
            }
            for (switch_class, neighbor), item in neighbor_lookup.items()
            if neighbor in neighbors
            and (
                switch_classes is None
                or switch_class in switch_classes
            )
        ]

    interrupting_or_isolating = {
        "Breaker",
        "LoadBreakSwitch",
        "Fuse",
        "Disconnector",
    }
    rules.append(
        {
            "rule_id": "LOGIC_TRANSFORMER_HV_SWITCH",
            "conditions": {
                "candidate_family": sorted(interrupting_or_isolating),
                "connected_equipment_any": [
                    "PowerTransformer",
                    "TransformerWinding",
                ],
            },
            "conclusion": {
                "functional_role": "TransformerHVFeederOrProtectionSwitch",
                "role_cn": "变压器高压侧馈线/保护开关",
                "physical_type_policy": (
                    "保留Breaker、LoadBreakSwitch、Fuse、Disconnector候选，"
                    "根据几何与串联模式进一步区分"
                ),
            },
            "constraints": [
                "Disconnector不应在缺少其他开断/保护设备时单独承担故障保护。",
                "若同时出现Fuse或Breaker，应优先解释为保护组合。",
            ],
            "provenance": "engineering_interpretation_plus_xml_topology",
            "xml_evidence": evidence_for_neighbors(
                {"PowerTransformer", "TransformerWinding"},
                interrupting_or_isolating,
            ),
        }
    )
    rules.append(
        {
            "rule_id": "LOGIC_PT_BRANCH_SWITCH",
            "conditions": {
                "candidate_family": sorted(interrupting_or_isolating),
                "connected_equipment_any": [
                    "PotentialTransformer",
                    "VoltageTransformer",
                ],
            },
            "conclusion": {
                "functional_role": "PTIsolationOrProtectionSwitch",
                "role_cn": "PT隔离/保护开关",
            },
            "constraints": [
                "结合Fuse与Disconnector串联关系判断熔断保护或隔离作用。"
            ],
            "provenance": "engineering_interpretation_plus_xml_topology",
            "xml_evidence": evidence_for_neighbors(
                {"PotentialTransformer", "VoltageTransformer"},
                interrupting_or_isolating,
            ),
        }
    )
    rules.append(
        {
            "rule_id": "LOGIC_BUS_TO_CABLE_FEEDER_SWITCH",
            "conditions": {
                "candidate_family": sorted(interrupting_or_isolating),
                "one_side_connector": "BusbarSection",
                "other_side_connector_any": [
                    "ACLineSegment",
                    "CableSegment",
                    "Junction",
                ],
            },
            "conclusion": {
                "functional_role": "IncomingOrOutgoingFeederSwitch",
                "role_cn": "进线或出线馈线开关",
            },
            "constraints": [
                "进线/出线方向需结合Circuit.SourceBreaker、柜体文字或电源方向确定。"
            ],
            "provenance": "engineering_interpretation_plus_xml_topology",
            "xml_evidence": evidence_for_neighbors(
                {"ACLineSegment", "Junction", "CableSegment"}
            ),
        }
    )
    rules.append(
        {
            "rule_id": "LOGIC_BUS_COUPLER",
            "conditions": {
                "candidate_family": ["Breaker", "LoadBreakSwitch"],
                "both_terminal_sides_have_busbar": True,
            },
            "conclusion": {
                "functional_role": "BusCouplerOrSectionSwitch",
                "role_cn": "母联或母线分段开关",
            },
            "constraints": [
                "应结合名称中的“母联/分段”及两侧母线身份确认。"
            ],
            "provenance": "engineering_interpretation_plus_xml_topology",
            "xml_evidence": [],
        }
    )
    rules.append(
        {
            "rule_id": "LOGIC_GROUND_DISCONNECTOR",
            "conditions": {"candidate_family": ["GroundDisconnector"]},
            "conclusion": {
                "functional_role": "GroundingIsolation",
                "role_cn": "接地隔离",
            },
            "constraints": [
                "不得把GroundDisconnector解释为正常负荷开断设备。"
            ],
            "provenance": "xml_class_semantics",
            "xml_evidence": [],
        }
    )
    source_breaker_evidence = [
        item
        for item in explicit_role_records
        if item["role"] == "circuit_source_breaker"
    ]
    rules.append(
        {
            "rule_id": "LOGIC_CIRCUIT_SOURCE_BREAKER",
            "conditions": {
                "xml_reference": "Circuit.SourceBreaker",
            },
            "conclusion": {
                "functional_role": "CircuitSourceBreaker",
                "role_cn": "线路电源侧断路器",
            },
            "constraints": [],
            "provenance": "explicit_xml_reference",
            "xml_evidence": source_breaker_evidence,
        }
    )
    rules.append(
        {
            "rule_id": "LOGIC_SERIAL_SWITCH_COMBINATION",
            "conditions": {
                "two_switching_devices_share_exclusive_conductive_component": True
            },
            "conclusion": {
                "functional_role": "SerialSwitchCombination",
                "role_cn": "串联开关组合",
                "physical_combination_candidates": [
                    {
                        "classes": [
                            item["switch_class_a"],
                            item["switch_class_b"],
                        ],
                        "occurrences": item["occurrences"],
                        "file_count": item["file_count"],
                        "evidence_grade": item["evidence_grade"],
                    }
                    for item in serial_records[:30]
                ],
            },
            "constraints": [
                "只有当中间导电分量不连接第三个功能设备时，才判定为直接串联。"
            ],
            "provenance": "mined_xml_topology_pattern",
            "xml_evidence": serial_records[:30],
        }
    )
    return rules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-occurrences", type=int, default=5)
    parser.add_argument("--minimum-files", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--component-template-library", type=Path)
    args = parser.parse_args()

    xml_paths = sorted(args.xml_dir.glob("*.xml"))
    if args.limit > 0:
        xml_paths = xml_paths[: args.limit]
    class_instances: Counter[str] = Counter()
    terminal_count_by_class: dict[str, Counter[int]] = defaultdict(Counter)
    direct_pairs = Aggregate()
    contracted_pairs = Aggregate()
    switch_neighbors = Aggregate()
    serial_switches = Aggregate()
    serial_switch_contexts = Aggregate()
    switch_contexts = Aggregate()
    explicit_roles = Aggregate()
    connector_class_instances: Counter[str] = Counter()
    switch_instance_counts: Counter[str] = Counter()
    parse_errors: list[dict[str, str]] = []
    warning_count = 0
    total_objects = 0
    total_terminals = 0
    total_nodes = 0
    total_conductive_components = 0

    for file_index, xml_path in enumerate(xml_paths, 1):
        try:
            objects, terminals, warnings = parse_xml(xml_path)
            warning_count += len(warnings)
        except Exception as exc:
            parse_errors.append(
                {
                    "file": str(xml_path),
                    "error": str(exc),
                }
            )
            continue
        total_objects += len(objects)
        total_terminals += len(terminals)
        terminals_by_equipment: dict[str, list[TerminalRecord]] = defaultdict(list)
        node_members: dict[str, list[str]] = defaultdict(list)
        for terminal in terminals:
            terminals_by_equipment[terminal.equipment_id].append(terminal)
            node_members[terminal.node_id].append(terminal.equipment_id)
        total_nodes += len(node_members)
        seen_file: set[str] = set()

        for object_id, record in objects.items():
            class_instances[record.class_name] += 1
            terminal_count_by_class[record.class_name][
                len(terminals_by_equipment.get(object_id, []))
            ] += 1
            if record.class_name in CONNECTOR_CLASSES:
                connector_class_instances[record.class_name] += 1
            if record.class_name in SWITCH_CLASSES:
                switch_instance_counts[record.class_name] += 1
                for role, pattern in ROLE_KEYWORDS:
                    if pattern.search(record.name):
                        explicit_roles.add(
                            role_key(
                                record.class_name,
                                role,
                                "equipment_name_keyword",
                            ),
                            xml_path.name,
                            {
                                "file": xml_path.name,
                                "equipment_id": object_id,
                                "name": record.name,
                            },
                            seen_file,
                        )

        for record in objects.values():
            source_breaker = record.references.get(
                "Circuit.SourceBreaker", ""
            )
            if source_breaker and source_breaker in objects:
                breaker = objects[source_breaker]
                explicit_roles.add(
                    role_key(
                        breaker.class_name,
                        "circuit_source_breaker",
                        "Circuit.SourceBreaker",
                    ),
                    xml_path.name,
                    {
                        "file": xml_path.name,
                        "equipment_id": source_breaker,
                        "name": breaker.name,
                    },
                    seen_file,
                )

        for node_id, member_ids in node_members.items():
            functional = [
                item
                for item in sorted(set(member_ids))
                if item in objects
                and objects[item].class_name not in STRUCTURAL_CLASSES
            ]
            for left_index, left_id in enumerate(functional):
                for right_id in functional[left_index + 1 :]:
                    left = objects[left_id]
                    right = objects[right_id]
                    direct_pairs.add(
                        pair_key(left.class_name, right.class_name),
                        xml_path.name,
                        {
                            "file": xml_path.name,
                            "node": node_id,
                            "equipment_a": left_id,
                            "name_a": left.name,
                            "equipment_b": right_id,
                            "name_b": right.name,
                        },
                        seen_file,
                    )

        union = UnionFind(node_members)
        for object_id, record in objects.items():
            if record.class_name not in CONNECTOR_CLASSES:
                continue
            nodes = sorted(
                {
                    terminal.node_id
                    for terminal in terminals_by_equipment.get(object_id, [])
                    if terminal.node_id in union.parent
                }
            )
            for node in nodes[1:]:
                union.union(nodes[0], node)

        component_nodes: dict[str, list[str]] = defaultdict(list)
        for node in node_members:
            component_nodes[union.find(node)].append(node)
        total_conductive_components += len(component_nodes)
        boundaries: dict[str, list[str]] = defaultdict(list)
        connector_counts: dict[str, Counter[str]] = defaultdict(Counter)
        equipment_roots: dict[str, list[str]] = defaultdict(list)

        for root, nodes in component_nodes.items():
            member_ids = {
                equipment_id
                for node in nodes
                for equipment_id in node_members[node]
                if equipment_id in objects
            }
            for equipment_id in member_ids:
                record = objects[equipment_id]
                if record.class_name in CONNECTOR_CLASSES:
                    connector_counts[root][record.class_name] += 1
                elif record.class_name not in STRUCTURAL_CLASSES:
                    boundaries[root].append(equipment_id)
            boundaries[root] = sorted(set(boundaries[root]))

        for equipment_id, equipment_terminals in terminals_by_equipment.items():
            if equipment_id not in objects:
                continue
            roots = []
            for terminal in sorted(
                equipment_terminals,
                key=lambda item: (
                    int(item.sequence)
                    if item.sequence.isdigit()
                    else 9999,
                    item.terminal_id,
                ),
            ):
                if terminal.node_id in union.parent:
                    roots.append(union.find(terminal.node_id))
            equipment_roots[equipment_id] = roots

        for root, member_ids in boundaries.items():
            functional = [
                item for item in member_ids if item in objects
            ]
            for left_index, left_id in enumerate(functional):
                for right_id in functional[left_index + 1 :]:
                    left = objects[left_id]
                    right = objects[right_id]
                    contracted_pairs.add(
                        pair_key(left.class_name, right.class_name),
                        xml_path.name,
                        {
                            "file": xml_path.name,
                            "equipment_a": left_id,
                            "name_a": left.name,
                            "equipment_b": right_id,
                            "name_b": right.name,
                            "connector_classes": dict(
                                connector_counts.get(root, Counter())
                            ),
                        },
                        seen_file,
                    )
            switch_members = [
                item
                for item in functional
                if objects[item].class_name in SWITCH_CLASSES
            ]
            if len(functional) == 2 and len(switch_members) == 2:
                left = objects[switch_members[0]]
                right = objects[switch_members[1]]
                serial_switches.add(
                    serial_key(left.class_name, right.class_name),
                    xml_path.name,
                    {
                        "file": xml_path.name,
                        "equipment_a": left.object_id,
                        "name_a": left.name,
                        "equipment_b": right.object_id,
                        "name_b": right.name,
                        "connector_classes": dict(
                            connector_counts.get(root, Counter())
                        ),
                    },
                    seen_file,
                )
                context_members = []
                for switch_id in switch_members:
                    switch_record = objects[switch_id]
                    external_sides = [
                        context_side_descriptor(
                            other_root,
                            switch_id,
                            boundaries,
                            connector_counts,
                            objects,
                        )
                        for other_root in equipment_roots.get(
                            switch_id, []
                        )
                        if other_root != root
                    ]
                    context_members.append(
                        {
                            "switch_class": switch_record.class_name,
                            "external_sides": external_sides,
                        }
                    )
                serial_switch_contexts.add(
                    serial_context_key(context_members),
                    xml_path.name,
                    {
                        "file": xml_path.name,
                        "equipment_a": left.object_id,
                        "name_a": left.name,
                        "equipment_b": right.object_id,
                        "name_b": right.name,
                    },
                    seen_file,
                )

        for equipment_id, record in objects.items():
            if record.class_name not in SWITCH_CLASSES:
                continue
            roots = equipment_roots.get(equipment_id, [])
            if not roots:
                continue
            sides = []
            neighbor_classes: set[str] = set()
            for root in roots:
                descriptor = context_side_descriptor(
                    root,
                    equipment_id,
                    boundaries,
                    connector_counts,
                    objects,
                )
                sides.append(descriptor)
                neighbor_classes.update(descriptor["neighbor_classes"])
            switch_contexts.add(
                context_key(record.class_name, sides),
                xml_path.name,
                {
                    "file": xml_path.name,
                    "equipment_id": equipment_id,
                    "name": record.name,
                },
                seen_file,
            )
            for neighbor_class in neighbor_classes:
                switch_neighbors.add(
                    switch_neighbor_key(
                        record.class_name, neighbor_class
                    ),
                    xml_path.name,
                    {
                        "file": xml_path.name,
                        "equipment_id": equipment_id,
                        "name": record.name,
                    },
                    seen_file,
                )

        if file_index % 50 == 0:
            print(
                json.dumps(
                    {
                        "progress": f"{file_index}/{len(xml_paths)}",
                        "objects": total_objects,
                        "terminals": total_terminals,
                        "errors": len(parse_errors),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    minimum_occurrences = args.minimum_occurrences
    minimum_files = args.minimum_files
    direct_records = direct_pairs.records(
        decode_pair, minimum_occurrences, minimum_files
    )
    contracted_records = contracted_pairs.records(
        decode_pair, minimum_occurrences, minimum_files
    )
    switch_neighbor_records = switch_neighbors.records(
        decode_switch_neighbor, minimum_occurrences, minimum_files
    )
    for record in switch_neighbor_records:
        denominator = switch_instance_counts.get(
            record["switch_class"], 0
        )
        record["switch_instance_count"] = denominator
        record["conditional_support"] = round(
            record["occurrences"] / denominator, 6
        ) if denominator else 0.0
    serial_records = serial_switches.records(
        decode_serial, minimum_occurrences, minimum_files
    )
    context_records = switch_contexts.records(
        decode_context, minimum_occurrences, minimum_files
    )
    serial_context_records = serial_switch_contexts.records(
        decode_serial_context,
        minimum_occurrences,
        minimum_files,
    )
    explicit_role_records = explicit_roles.records(
        decode_role, 1, 1
    )
    engineering_rules = build_engineering_rules(
        switch_neighbor_records,
        serial_records,
        explicit_role_records,
    )

    class_records = [
        {
            "class_name": class_name,
            "instances": count,
            "terminal_count_distribution": dict(
                sorted(terminal_count_by_class[class_name].items())
            ),
            "is_switching_class": class_name in SWITCH_CLASSES,
            "is_conductive_connector": class_name in CONNECTOR_CLASSES,
            "is_structural_class": class_name in STRUCTURAL_CLASSES,
        }
        for class_name, count in class_instances.most_common()
    ]

    template_integration: dict[str, Any] = {
        "cim_to_template_family": CIM_TO_TEMPLATE_FAMILY,
        "component_template_library": "",
        "component_template_schema": "",
        "equipment_template_variants": 0,
        "family_summary": [],
    }
    if args.component_template_library:
        component_library = json.loads(
            args.component_template_library.read_text(
                encoding="utf-8-sig"
            )
        )
        template_integration.update(
            {
                "component_template_library": str(
                    args.component_template_library
                ),
                "component_template_schema": component_library.get(
                    "schema_version", ""
                ),
                "equipment_template_variants": component_library.get(
                    "statistics", {}
                ).get("equipment_template_variants", 0),
                "family_summary": component_library.get(
                    "family_summary", []
                ),
            }
        )

    library = {
        "schema_version": "electrical-logic-knowledge-library-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "source": {
            "xml_directory": str(args.xml_dir),
            "xml_files": len(xml_paths),
            "data_model": "CIM Equipment—Terminal—ConnectivityNode",
            "paired_svg_available": True,
        },
        "method": {
            "direct_relation": (
                "两个设备的Terminal直接指向同一个ConnectivityNode"
            ),
            "through_conductor_relation": (
                "收缩ACLineSegment、BusbarSection、Junction等导电对象后，"
                "位于同一导电分量边界的设备视为通过导线相连"
            ),
            "serial_switch_pattern": (
                "同一导电分量的功能设备边界恰好只有两个，且两者均为开关类"
            ),
            "switch_context": (
                "按开关每个端子所在导电分量，记录相邻设备类型、母线和线路上下文"
            ),
            "rule_publication_threshold": {
                "minimum_occurrences": minimum_occurrences,
                "minimum_files": minimum_files,
            },
            "evidence_grade": {
                "high": "occurrences>=100且file_count>=20",
                "medium": "occurrences>=20且file_count>=5",
                "exploratory": "低于上述阈值，仅作探索性证据",
            },
        },
        "statistics": {
            "parsed_files": len(xml_paths) - len(parse_errors),
            "parse_errors": len(parse_errors),
            "incomplete_terminal_records": warning_count,
            "objects": total_objects,
            "valid_terminals": total_terminals,
            "terminal_records_observed": total_terminals + warning_count,
            "connectivity_nodes": total_nodes,
            "conductive_components": total_conductive_components,
            "equipment_classes": len(class_instances),
            "published_direct_relation_types": len(direct_records),
            "published_contracted_relation_types": len(
                contracted_records
            ),
            "published_switch_neighbor_rules": len(
                switch_neighbor_records
            ),
            "published_serial_switch_patterns": len(serial_records),
            "published_serial_switch_context_patterns": len(
                serial_context_records
            ),
            "published_switch_context_patterns": len(context_records),
            "engineering_logic_rules": len(engineering_rules),
        },
        "class_taxonomy": {
            "switch_classes": sorted(SWITCH_CLASSES),
            "connector_classes": sorted(CONNECTOR_CLASSES),
            "structural_classes": sorted(STRUCTURAL_CLASSES),
            "observed_classes": class_records,
        },
        "component_template_integration": template_integration,
        "direct_node_relations": direct_records,
        "through_conductor_relations": contracted_records,
        "switch_neighbor_rules": switch_neighbor_records,
        "serial_switch_patterns": serial_records,
        "serial_switch_context_patterns": serial_context_records,
        "switch_terminal_context_patterns": context_records,
        "explicit_role_evidence": explicit_role_records,
        "engineering_logic_rules": engineering_rules,
        "inference_output_contract": {
            "required_fields": [
                "component_id",
                "geometry_type_candidates",
                "connected_equipment",
                "connector_context",
            ],
            "result_fields": [
                "physical_type_candidates",
                "functional_role",
                "role_confidence",
                "physical_type_confidence",
                "matched_logic_rules",
                "evidence_trace",
                "conflicts",
            ],
            "policy": (
                "拓扑逻辑优先确定功能作用；几何模板与XML经验模式共同排序"
                "物理类型；存在多种合法设备配置时保留候选，不强行唯一化。"
            ),
        },
        "parse_errors": parse_errors,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "电气逻辑知识库.json"
    json_path.write_text(
        json.dumps(library, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(
        args.output_dir / "设备经导线连接关系.csv",
        csv_rows(
            contracted_records,
            [
                "class_a",
                "class_b",
                "occurrences",
                "file_count",
                "evidence_grade",
                "examples",
            ],
        ),
        [
            "class_a",
            "class_b",
            "occurrences",
            "file_count",
            "evidence_grade",
            "examples",
        ],
    )
    write_csv(
        args.output_dir / "开关相邻设备规则.csv",
        csv_rows(
            switch_neighbor_records,
            [
                "switch_class",
                "neighbor_class",
                "occurrences",
                "file_count",
                "conditional_support",
                "evidence_grade",
                "examples",
            ],
        ),
        [
            "switch_class",
            "neighbor_class",
            "occurrences",
            "file_count",
            "conditional_support",
            "evidence_grade",
            "examples",
        ],
    )
    write_csv(
        args.output_dir / "串联开关组合模式.csv",
        csv_rows(
            serial_records,
            [
                "switch_class_a",
                "switch_class_b",
                "occurrences",
                "file_count",
                "evidence_grade",
                "examples",
            ],
        ),
        [
            "switch_class_a",
            "switch_class_b",
            "occurrences",
            "file_count",
            "evidence_grade",
            "examples",
        ],
    )
    write_csv(
        args.output_dir / "串联开关外部关系模式.csv",
        csv_rows(
            serial_context_records,
            [
                "switch_pair",
                "switch_members",
                "occurrences",
                "file_count",
                "evidence_grade",
                "examples",
            ],
        ),
        [
            "switch_pair",
            "switch_members",
            "occurrences",
            "file_count",
            "evidence_grade",
            "examples",
        ],
    )
    write_csv(
        args.output_dir / "XML明确功能角色.csv",
        csv_rows(
            explicit_role_records,
            [
                "equipment_class",
                "role",
                "label_source",
                "occurrences",
                "file_count",
                "evidence_grade",
                "examples",
            ],
        ),
        [
            "equipment_class",
            "role",
            "label_source",
            "occurrences",
            "file_count",
            "evidence_grade",
            "examples",
        ],
    )

    top_relations = "\n".join(
        f"| {item['class_a']} | {item['class_b']} | "
        f"{item['occurrences']} | {item['file_count']} | "
        f"{item['evidence_grade']} |"
        for item in contracted_records[:30]
    )
    top_switch_neighbors = "\n".join(
        f"| {item['switch_class']} | {item['neighbor_class']} | "
        f"{item['occurrences']} | {item['file_count']} | "
        f"{item['conditional_support']:.2%} | {item['evidence_grade']} |"
        for item in switch_neighbor_records[:30]
    )
    top_serial = "\n".join(
        f"| {item['switch_class_a']} | {item['switch_class_b']} | "
        f"{item['occurrences']} | {item['file_count']} | "
        f"{item['evidence_grade']} |"
        for item in serial_records[:30]
    )
    top_serial_context = "\n".join(
        f"| {' + '.join(item['switch_pair'])} | "
        f"`{json.dumps(item['switch_members'], ensure_ascii=False)}` | "
        f"{item['occurrences']} | {item['file_count']} | "
        f"{item['evidence_grade']} |"
        for item in serial_context_records[:20]
    )
    report = f"""# SVG/XML电气逻辑知识库构建报告

## 数据与方法

- XML文件：{len(xml_paths)}
- 成功解析：{len(xml_paths) - len(parse_errors)}
- 设备及模型对象：{total_objects}
- 有效端子：{total_terminals}
- 缺少设备或连接节点引用而未入图的端子记录：{warning_count}
- 连接节点：{total_nodes}
- 收缩线路/母线后导电分量：{total_conductive_components}

XML采用Equipment—Terminal—ConnectivityNode结构。本次先恢复原始节点关系，
再收缩ACLineSegment、BusbarSection、Junction等导电对象，得到“设备通过导线相连”
的关系。只有在至少{minimum_files}张图、累计至少{minimum_occurrences}次出现的模式
才进入发布规则；低频事实仍可从原XML重新追溯，不直接当作稳定规则。
所有XML文件均完成解析；缺少设备或连接节点引用的端子被保留为质量统计，
但不用于生成拓扑关系。

## 知识库成果

- 设备类别：{len(class_instances)}
- 经导线连接关系类型：{len(contracted_records)}
- 开关—相邻设备规则：{len(switch_neighbor_records)}
- 直接串联开关组合：{len(serial_records)}
- 串联组合及两端外部关系：{len(serial_context_records)}
- 开关两侧上下文模式：{len(context_records)}
- 可执行工程逻辑规则：{len(engineering_rules)}

## 高频设备经导线连接关系

| 设备A | 设备B | 出现次数 | 文件数 | 证据等级 |
|---|---|---:|---:|---|
{top_relations}

## 高频开关相邻设备

| 开关类型 | 相邻设备 | 出现次数 | 文件数 | 条件支持度 | 证据等级 |
|---|---|---:|---:|---:|---|
{top_switch_neighbors}

条件支持度表示：具有该类相邻设备的开关实例数，占该开关类全部实例的比例。
一个开关可连接多种设备，因此各项支持度之和不要求等于100%。

## 高频直接串联开关组合

| 开关A | 开关B | 出现次数 | 文件数 | 证据等级 |
|---|---|---:|---:|---|
{top_serial}

“直接串联”要求两者之间的导电分量没有第三个功能设备，避免把同一母线上多个
并列开关误判为组合。

## 串联开关及其两端外部关系

| 开关组合 | 两端外部上下文 | 出现次数 | 文件数 | 证据等级 |
|---|---|---:|---:|---|
{top_serial_context}

该模式进一步保存组合两端连接的设备类型以及是否包含母线、线路，可用于判断
“母线—开关组合—变压器”“线路—开关组合—母线”等整体功能。

## 使用方式

对DXF先生成元件候选和拓扑图，再把每个候选的几何类型、相邻设备、母线/线路
上下文送入逻辑知识库。逻辑规则优先确定“进线、出线、联络、变压器保护”等
功能作用；SVG几何模板与XML经验模式共同排序负荷开关、断路器、隔离开关等
物理类型。多种配置均符合电气逻辑时保留候选，不能仅凭拓扑强行唯一化。

## 限制

- XML关系反映数据集中的实际建模习惯，不等同于强制性设计规范。
- 设备在同一导电分量中相连，不表示开关运行状态下必然导通。
- 名称关键词只能提供弱监督，不能替代设备类和端子连接。
- 组合模式的应用仍需DXF候选边界和端口识别正确。
"""
    report_path = args.output_dir / "电气逻辑知识库构建报告.md"
    report_path.write_text(report, encoding="utf-8")

    checksum_paths = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and not path.name.endswith(".zip")
    )
    checksum_lines = []
    for path in checksum_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.name}")
    checksum_path = args.output_dir / "SHA256SUMS.txt"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    package_path = args.output_dir / "电气逻辑知识库_可分享.zip"
    with zipfile.ZipFile(
        package_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(args.output_dir.iterdir()):
            if path == package_path or not path.is_file():
                continue
            archive.write(path, arcname=path.name)

    print(
        json.dumps(
            {
                "xml_files": len(xml_paths),
                "parsed_files": len(xml_paths) - len(parse_errors),
                "objects": total_objects,
                "terminals": total_terminals,
                "contracted_relation_types": len(contracted_records),
                "switch_neighbor_rules": len(switch_neighbor_records),
                "serial_switch_patterns": len(serial_records),
                "serial_switch_context_patterns": len(
                    serial_context_records
                ),
                "engineering_rules": len(engineering_rules),
                "parse_errors": len(parse_errors),
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
