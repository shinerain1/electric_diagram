from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from create_topology_annotation_workbook import make_xlsx, stringify, worksheet_xml


SYMBOL_RE = re.compile(r"<symbol\b(?P<attrs>[^>]*)>(?P<body>.*?)</symbol>", re.S)
ATTR_RE = re.compile(r'([\w:-]+)="([^"]*)"')
USE_WITH_METADATA_RE = re.compile(
    r'<use\b(?P<attrs>[^>]*\bxlink:href="#(?P<symbol_id>[^"]+)"[^>]*)/?>'
    r'(?:(?!<use\b).){0,1200}?'
    r'<metadata>.*?<cge:PSR_Ref\b[^>]*\bObjectID="(?P<object_id>[^"]+)"[^>]*/?>.*?</metadata>',
    re.S,
)
XML_OBJECT_RE = re.compile(
    r'<cim:(?P<class_name>[A-Za-z0-9_]+)\b[^>]*\brdf:ID="(?P<object_id>[^"]+)"[^>]*>'
    r'(?P<body>.*?)</cim:(?P=class_name)>',
    re.S,
)
XML_TERMINAL_RE = re.compile(r"<cim:Terminal\b[^>]*>(?P<body>.*?)</cim:Terminal>", re.S)
NAME_RE = re.compile(r"<cim:Naming\.name>(.*?)</cim:Naming\.name>", re.S)
EQUIPMENT_REF_RE = re.compile(
    r'<cim:Terminal\.ConductingEquipment\b[^>]*\brdf:resource="#([^"]+)"'
)


NON_EQUIPMENT_FAMILIES = {
    "terminal",
    "flow",
    "Text",
    "PrintFrame",
    "ReviewInfo",
    "LinkPoint",
    "ACLineSegment",
    "BusbarSection",
}


def attrs(text: str) -> dict[str, str]:
    return dict(ATTR_RE.findall(text))


def number(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def rounded(value: float) -> float:
    return round(value, 8)


def viewbox_values(value: str) -> tuple[float, float, float, float]:
    values = [number(item) for item in re.split(r"[\s,]+", value.strip()) if item]
    if len(values) != 4:
        return (0.0, 0.0, 1.0, 1.0)
    x, y, width, height = values
    return (x, y, width or 1.0, height or 1.0)


def normalize_xy(
    x: float,
    y: float,
    viewbox: tuple[float, float, float, float],
) -> list[float]:
    origin_x, origin_y, width, height = viewbox
    return [rounded((x - origin_x) / width), rounded((y - origin_y) / height)]


def parse_points(value: str, viewbox: tuple[float, float, float, float]) -> list[list[float]]:
    values = [number(item) for item in re.split(r"[\s,]+", value.strip()) if item]
    return [
        normalize_xy(values[index], values[index + 1], viewbox)
        for index in range(0, len(values) - 1, 2)
    ]


def canonical_path(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def extract_geometry(
    symbol_attrs: dict[str, str],
    body: str,
) -> tuple[dict[str, Any], dict[str, int], list[dict[str, Any]]]:
    viewbox_text = symbol_attrs.get("viewBox", "0 0 1 1")
    viewbox = viewbox_values(viewbox_text)
    terminals = []
    for match in re.finditer(r"<use\b([^>]*)/?>", body):
        item_attrs = attrs(match.group(1))
        if "terminal-index" not in item_attrs:
            continue
        terminals.append({
            "index": int(number(item_attrs["terminal-index"])),
            "point": normalize_xy(number(item_attrs.get("x")), number(item_attrs.get("y")), viewbox),
        })
    terminals.sort(key=lambda item: item["index"])

    primitives: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for tag in ("line", "circle", "polygon", "polyline", "rect", "ellipse", "path"):
        for match in re.finditer(fr"<{tag}\b([^>]*)/?>", body):
            item_attrs = attrs(match.group(1))
            counts[tag] += 1
            if tag == "line":
                primitives.append({
                    "type": tag,
                    "start": normalize_xy(number(item_attrs.get("x1")), number(item_attrs.get("y1")), viewbox),
                    "end": normalize_xy(number(item_attrs.get("x2")), number(item_attrs.get("y2")), viewbox),
                })
            elif tag == "circle":
                primitives.append({
                    "type": tag,
                    "center": normalize_xy(number(item_attrs.get("cx")), number(item_attrs.get("cy")), viewbox),
                    "radius": rounded(number(item_attrs.get("r")) / max(viewbox[2], viewbox[3])),
                })
            elif tag in {"polygon", "polyline"}:
                primitives.append({
                    "type": tag,
                    "points": parse_points(item_attrs.get("points", ""), viewbox),
                })
            elif tag == "rect":
                primitives.append({
                    "type": tag,
                    "origin": normalize_xy(number(item_attrs.get("x")), number(item_attrs.get("y")), viewbox),
                    "size": [
                        rounded(number(item_attrs.get("width")) / viewbox[2]),
                        rounded(number(item_attrs.get("height")) / viewbox[3]),
                    ],
                })
            elif tag == "ellipse":
                primitives.append({
                    "type": tag,
                    "center": normalize_xy(number(item_attrs.get("cx")), number(item_attrs.get("cy")), viewbox),
                    "radius": [
                        rounded(number(item_attrs.get("rx")) / viewbox[2]),
                        rounded(number(item_attrs.get("ry")) / viewbox[3]),
                    ],
                })
            else:
                primitives.append({"type": tag, "d": canonical_path(item_attrs.get("d", ""))})
    primitives.sort(key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    geometry = {
        "view_box": [rounded(item) for item in viewbox],
        "terminals": terminals,
        "primitives": primitives,
    }
    return geometry, dict(sorted(counts.items())), terminals


def geometry_signature(geometry: dict[str, Any]) -> str:
    payload = json.dumps(geometry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_symbol_id(symbol_id: str) -> tuple[str, str, str]:
    family, _, remainder = symbol_id.partition(":")
    if not remainder:
        return (family, family, "")
    name, separator, state = remainder.rpartition("@")
    return (family, name if separator else remainder, state if separator else "")


def xml_semantics(xml_path: Path) -> tuple[dict[str, dict[str, str]], Counter[str]]:
    if not xml_path.exists():
        return {}, Counter()
    text = xml_path.read_text(encoding="utf-8", errors="ignore")
    objects: dict[str, dict[str, str]] = {}
    for match in XML_OBJECT_RE.finditer(text):
        body = match.group("body")
        name_match = NAME_RE.search(body)
        objects[match.group("object_id")] = {
            "class": match.group("class_name"),
            "name": re.sub(r"\s+", " ", name_match.group(1).strip()) if name_match else "",
        }
    terminal_counts: Counter[str] = Counter()
    for match in XML_TERMINAL_RE.finditer(text):
        reference = EQUIPMENT_REF_RE.search(match.group("body"))
        if reference:
            terminal_counts[reference.group(1)] += 1
    return objects, terminal_counts


def is_equipment_template(record: dict[str, Any]) -> bool:
    family = record["family"]
    xml_classes = set(record["xml_semantic_classes"])
    has_terminal = record["terminal_count"] > 0
    semantic_equipment = bool(xml_classes - {
        "Terminal",
        "ConnectivityNode",
        "ACLineSegment",
        "BusbarSection",
        "Text",
    })
    return family not in NON_EQUIPMENT_FAMILIES and (has_terminal or semantic_equipment)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deduplicated standard component template library.")
    parser.add_argument("svg_dir", type=Path)
    parser.add_argument("xml_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_xlsx", type=Path)
    parser.add_argument("output_md", type=Path)
    args = parser.parse_args()

    svg_paths = sorted(args.svg_dir.glob("*.svg"))
    xml_paths = list(args.xml_dir.glob("*.xml"))
    xml_by_stem = {path.stem: path for path in xml_paths}

    variants: dict[tuple[str, str], dict[str, Any]] = {}
    file_symbol_keys: dict[str, dict[str, tuple[str, str]]] = {}
    parse_errors: list[dict[str, str]] = []
    total_symbol_definitions = 0
    total_instance_uses = 0
    paired_files = 0

    for index, svg_path in enumerate(svg_paths, 1):
        try:
            svg_text = svg_path.read_text(encoding="utf-8", errors="ignore")
            xml_path = xml_by_stem.get(svg_path.stem)
            objects, xml_terminal_counts = xml_semantics(xml_path) if xml_path else ({}, Counter())
            if xml_path:
                paired_files += 1

            symbol_keys: dict[str, tuple[str, str]] = {}
            for symbol_match in SYMBOL_RE.finditer(svg_text):
                symbol_attributes = attrs(symbol_match.group("attrs"))
                symbol_id = symbol_attributes.get("id")
                if not symbol_id:
                    continue
                total_symbol_definitions += 1
                geometry, primitive_counts, terminals = extract_geometry(
                    symbol_attributes,
                    symbol_match.group("body"),
                )
                signature = geometry_signature(geometry)
                key = (symbol_id, signature)
                symbol_keys[symbol_id] = key
                family, name, state = parse_symbol_id(symbol_id)
                if key not in variants:
                    variants[key] = {
                        "symbol_id": symbol_id,
                        "geometry_signature": signature,
                        "family": family,
                        "name": name,
                        "state": state,
                        "view_box": geometry["view_box"],
                        "terminal_count": len(terminals),
                        "terminals": terminals,
                        "primitive_counts": primitive_counts,
                        "normalized_primitives": geometry["primitives"],
                        "definition_occurrences": 0,
                        "usage_count": 0,
                        "xml_semantic_classes": Counter(),
                        "xml_semantic_names": Counter(),
                        "xml_terminal_count_evidence": Counter(),
                        "source_svg_examples": [],
                        "source_xml_examples": [],
                    }
                record = variants[key]
                record["definition_occurrences"] += 1
                if len(record["source_svg_examples"]) < 5:
                    record["source_svg_examples"].append(str(svg_path))
                if xml_path and len(record["source_xml_examples"]) < 5:
                    record["source_xml_examples"].append(str(xml_path))

            file_symbol_keys[svg_path.stem] = symbol_keys
            for use_match in USE_WITH_METADATA_RE.finditer(svg_text):
                symbol_id = use_match.group("symbol_id")
                key = symbol_keys.get(symbol_id)
                if not key:
                    continue
                total_instance_uses += 1
                record = variants[key]
                record["usage_count"] += 1
                object_id = use_match.group("object_id")
                semantic = objects.get(object_id)
                if semantic:
                    record["xml_semantic_classes"][semantic["class"]] += 1
                    if semantic["name"]:
                        record["xml_semantic_names"][semantic["name"]] += 1
                    if xml_terminal_counts[object_id]:
                        record["xml_terminal_count_evidence"][str(xml_terminal_counts[object_id])] += 1
        except Exception as exc:
            parse_errors.append({"file": str(svg_path), "error": f"{type(exc).__name__}: {exc}"})
        if index % 100 == 0:
            print(f"PROGRESS {index}/{len(svg_paths)} variants={len(variants)} errors={len(parse_errors)}", flush=True)

    symbol_variant_counts = Counter(symbol_id for symbol_id, _ in variants)
    records = []
    for (symbol_id, signature), raw_record in sorted(
        variants.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        variant_number = 1 + sum(
            1
            for candidate_id, candidate_signature in variants
            if candidate_id == symbol_id and candidate_signature < signature
        )
        template_key = (
            symbol_id
            if symbol_variant_counts[symbol_id] == 1
            else f"{symbol_id}#v{variant_number}"
        )
        record = dict(raw_record)
        record["template_key"] = template_key
        record["xml_semantic_classes"] = dict(record["xml_semantic_classes"].most_common())
        record["xml_semantic_names"] = dict(record["xml_semantic_names"].most_common(20))
        record["xml_terminal_count_evidence"] = dict(
            record["xml_terminal_count_evidence"].most_common()
        )
        record["is_equipment_template"] = is_equipment_template(record)
        record["semantic_status"] = (
            "xml_confirmed"
            if record["xml_semantic_classes"]
            else "id_and_geometry_only"
        )
        records.append(record)

    equipment_records = [record for record in records if record["is_equipment_template"]]
    family_counts = Counter(record["family"] for record in equipment_records)
    library = {
        "schema_version": "standard-component-template-library-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "svg_directory": str(args.svg_dir),
            "xml_directory": str(args.xml_dir),
            "svg_files": len(svg_paths),
            "xml_files": len(xml_paths),
            "paired_files": paired_files,
        },
        "statistics": {
            "symbol_definitions_scanned": total_symbol_definitions,
            "unique_symbol_ids": len(symbol_variant_counts),
            "unique_geometry_variants": len(records),
            "equipment_template_variants": len(equipment_records),
            "observed_svg_instances": total_instance_uses,
            "families": len(family_counts),
            "parse_errors": len(parse_errors),
        },
        "family_summary": [
            {"family": family, "template_variants": count}
            for family, count in family_counts.most_common()
        ],
        "templates": records,
        "parse_errors": parse_errors,
    }

    summary_rows = [
        ["项目", "数量/内容"],
        ["SVG文件", len(svg_paths)],
        ["XML文件", len(xml_paths)],
        ["同名配对文件", paired_files],
        ["扫描到的符号定义", total_symbol_definitions],
        ["不同符号名称", len(symbol_variant_counts)],
        ["去重后的几何版本", len(records)],
        ["可用于元件识别的模板版本", len(equipment_records)],
        ["SVG中观察到的元件实例", total_instance_uses],
        ["元件家族数", len(family_counts)],
        ["解析失败文件", len(parse_errors)],
        ["JSON文件", str(args.output_json)],
    ]
    summary_xml = worksheet_xml(summary_rows, [34, 95], autofilter=False)

    template_rows = [[
        "模板键", "SVG符号ID", "家族", "中文名称", "状态版本", "接口数",
        "图形组成", "XML设备类别", "XML名称示例", "定义出现次数", "实例使用次数",
        "是否元件模板", "语义状态", "SVG来源示例",
    ]]
    for record in records:
        template_rows.append([
            record["template_key"], record["symbol_id"], record["family"], record["name"],
            record["state"], record["terminal_count"], stringify(record["primitive_counts"]),
            stringify(record["xml_semantic_classes"]), stringify(record["xml_semantic_names"]),
            record["definition_occurrences"], record["usage_count"],
            "是" if record["is_equipment_template"] else "否", record["semantic_status"],
            stringify(record["source_svg_examples"]),
        ])
    template_xml = worksheet_xml(
        template_rows,
        [55, 50, 28, 35, 14, 12, 42, 48, 55, 16, 16, 16, 22, 85],
    )

    family_rows = [["元件家族", "模板版本数"]]
    family_rows.extend([[family, count] for family, count in family_counts.most_common()])
    family_xml = worksheet_xml(family_rows, [38, 18])

    error_rows = [["文件", "错误"]]
    error_rows.extend([[item["file"], item["error"]] for item in parse_errors])
    if len(error_rows) == 1:
        error_rows.append(["", "无"])
    error_xml = worksheet_xml(error_rows, [100, 100])

    make_xlsx(args.output_xlsx, [
        ("模板库概况", summary_xml),
        ("全部模板", template_xml),
        ("元件家族", family_xml),
        ("解析问题", error_xml),
    ])

    report = f"""# 标准元件模板库构建报告

## 数据来源

- SVG文件：{len(svg_paths)}份
- XML文件：{len(xml_paths)}份
- 同名SVG/XML配对：{paired_files}组

## 构建结果

- 扫描符号定义：{total_symbol_definitions}次
- 不同SVG符号名称：{len(symbol_variant_counts)}个
- 去重后几何版本：{len(records)}个
- 可用于DXF元件识别的模板版本：{len(equipment_records)}个
- 元件家族：{len(family_counts)}类
- 解析失败：{len(parse_errors)}份

## 每个模板保存的内容

- SVG符号ID、元件家族和中文名称
- 归一化后的直线、圆、多边形、折线和路径
- 接口数量及接口相对坐标
- 配对XML确认的CIM设备类别和名称
- 模板出现次数、实际使用次数及来源文件
- 几何签名，用于识别同名但画法不同的版本

## 使用方式

DXF元件先被转换为相同的几何特征，再与本JSON中的模板做直接匹配、家族匹配或组合匹配。
"""

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(library, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")

    print(json.dumps({
        "output_json": str(args.output_json),
        "output_xlsx": str(args.output_xlsx),
        "output_md": str(args.output_md),
        **library["statistics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
