from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import ezdxf
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False


DRAWING_NAME = "02系统接线图.dxf"
VISUAL_TRUTH_ID = "EQ03"
VISUAL_LABEL = "原有变压器 / 原有配变250kVA"
REFERENCE_IMAGE = "04_02系统接线图/高清局部_002.png"

# The transformer was selected visually first.  These handles are the DXF
# evidence inside that visually selected region.
CORE_CIRCLE_HANDLES = ["3A7", "3A8"]
UPPER_LEAD_HANDLE = "3AD"
FULL_BRANCH_HANDLES = [
    "3A2",  # lower triangle
    "3A3",
    "3A4",
    "3A5",  # three small phase circles
    "3A6",  # lower lead
    "3A7",
    "3A8",  # transformer core circles
    "3AD",  # upper lead
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sample_line(start: tuple[float, float], end: tuple[float, float], n: int = 48) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack(
        (start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1]))
    )


def sample_circle(center: tuple[float, float], radius: float, n: int = 96) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack(
        (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
    )


def sample_polyline(points: list[tuple[float, float]], closed: bool) -> np.ndarray:
    if closed and points and points[0] != points[-1]:
        points = points + [points[0]]
    samples = [
        sample_line(start, end, 32) for start, end in zip(points, points[1:])
    ]
    return np.vstack(samples) if samples else np.empty((0, 2))


def normalize_points(points: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    scale = max(float((maximum - minimum).max()), 1e-12)
    return (points - center) / scale


def dihedral_variants(points: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    for mirrored in (False, True):
        base = points.copy()
        if mirrored:
            base[:, 0] *= -1.0
        for turns in range(4):
            angle = turns * math.pi / 2.0
            rotation = np.array(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ]
            )
            variants.append(
                (
                    f"{'mirror_x+' if mirrored else ''}rotate_{turns * 90}",
                    base @ rotation.T,
                )
            )
    return variants


def symmetric_chamfer(left: np.ndarray, right: np.ndarray) -> float:
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2.0)


def count_similarity(
    candidate_counts: dict[str, int], template_counts: dict[str, int]
) -> float:
    names = set(candidate_counts) | set(template_counts)
    intersection = sum(
        min(candidate_counts.get(name, 0), template_counts.get(name, 0))
        for name in names
    )
    union = sum(
        max(candidate_counts.get(name, 0), template_counts.get(name, 0))
        for name in names
    )
    return intersection / union if union else 0.0


def sample_template(record: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    samples: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for primitive in record.get("normalized_primitives", []):
        kind = primitive["type"]
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "line":
            samples.append(
                sample_line(tuple(primitive["start"]), tuple(primitive["end"]))
            )
        elif kind == "circle":
            samples.append(
                sample_circle(tuple(primitive["center"]), float(primitive["radius"]))
            )
        elif kind in {"polygon", "polyline"}:
            points = [tuple(point) for point in primitive["points"]]
            samples.append(sample_polyline(points, closed=kind == "polygon"))
    if not samples:
        raise RuntimeError(f"模板没有可采样几何：{record['symbol_id']}")
    return normalize_points(np.vstack(samples)), counts


def entity_map(doc: Any) -> dict[str, Any]:
    return {
        entity.dxf.handle: entity
        for entity in doc.modelspace()
        if entity.dxf.handle
    }


def circle_samples(entity: Any) -> np.ndarray:
    return sample_circle(
        (float(entity.dxf.center.x), float(entity.dxf.center.y)),
        float(entity.dxf.radius),
    )


def line_samples(entity: Any) -> np.ndarray:
    return sample_line(
        (float(entity.dxf.start.x), float(entity.dxf.start.y)),
        (float(entity.dxf.end.x), float(entity.dxf.end.y)),
    )


def candidate_variants(doc: Any) -> dict[str, dict[str, Any]]:
    by_handle = entity_map(doc)
    core_circles = [by_handle[handle] for handle in CORE_CIRCLE_HANDLES]
    circle_points = np.vstack([circle_samples(entity) for entity in core_circles])
    core_counts = {"circle": 2}

    radii = [float(entity.dxf.radius) for entity in core_circles]
    top_y = max(
        float(entity.dxf.center.y) + float(entity.dxf.radius)
        for entity in core_circles
    )
    lead = by_handle[UPPER_LEAD_HANDLE]
    lead_x = float(lead.dxf.start.x)
    local_lead_length = statistics_median(radii) * 1.125
    clipped_lead = sample_line(
        (lead_x, top_y), (lead_x, top_y + local_lead_length)
    )

    full_samples: list[np.ndarray] = []
    full_counts: dict[str, int] = {}
    for handle in FULL_BRANCH_HANDLES:
        entity = by_handle[handle]
        kind = entity.dxftype()
        if kind == "CIRCLE":
            full_samples.append(circle_samples(entity))
            full_counts["circle"] = full_counts.get("circle", 0) + 1
        elif kind == "LINE":
            full_samples.append(line_samples(entity))
            full_counts["line"] = full_counts.get("line", 0) + 1
        elif kind == "LWPOLYLINE":
            points = [
                (float(x), float(y)) for x, y in entity.get_points("xy")
            ]
            full_samples.append(sample_polyline(points, bool(entity.closed)))
            full_counts["polygon" if entity.closed else "polyline"] = (
                full_counts.get("polygon" if entity.closed else "polyline", 0)
                + 1
            )

    return {
        "current_auto_core": {
            "description": "当前程序候选：仅两个变压器主绕组圆",
            "handles": CORE_CIRCLE_HANDLES,
            "points": normalize_points(circle_points),
            "primitive_counts": core_counts,
        },
        "core_plus_recovered_lead": {
            "description": "从相邻导线中补回模板尺度的局部上引线",
            "handles": CORE_CIRCLE_HANDLES + [UPPER_LEAD_HANDLE],
            "points": normalize_points(np.vstack([circle_points, clipped_lead])),
            "primitive_counts": {"circle": 2, "line": 1},
        },
        "full_visual_branch": {
            "description": "完整可见支路：主圆、小圆、上下长引线和三角端头",
            "handles": FULL_BRANCH_HANDLES,
            "points": normalize_points(np.vstack(full_samples)),
            "primitive_counts": full_counts,
        },
    }


def statistics_median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def rank_templates(
    variant: dict[str, Any], templates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidate = variant["points"]
    results: list[dict[str, Any]] = []
    for template in templates:
        template_points, template_counts = sample_template(template)
        best_transform = ""
        best_distance = float("inf")
        best_points = template_points
        for transform_name, transformed in dihedral_variants(template_points):
            distance = symmetric_chamfer(candidate, transformed)
            if distance < best_distance:
                best_distance = distance
                best_transform = transform_name
                best_points = transformed
        primitive_score = count_similarity(
            variant["primitive_counts"], template_counts
        )
        geometry_score = math.exp(-8.0 * best_distance)
        total_score = 100.0 * (0.85 * geometry_score + 0.15 * primitive_score)
        results.append(
            {
                "template_id": template["symbol_id"],
                "template_name": template["name"],
                "usage_count": template.get("usage_count", 0),
                "semantic_status": template.get("semantic_status", ""),
                "template_primitive_counts": template.get(
                    "primitive_counts", {}
                ),
                "best_transform": best_transform,
                "chamfer_distance": round(best_distance, 6),
                "geometry_score": round(geometry_score * 100.0, 2),
                "primitive_count_score": round(primitive_score * 100.0, 2),
                "combined_score": round(total_score, 2),
                "_best_points": best_points,
            }
        )
    results.sort(
        key=lambda item: (
            -item["combined_score"],
            -item["usage_count"],
            item["template_id"],
        )
    )
    for index, result in enumerate(results, 1):
        result["rank"] = index
    return results


def render_comparison(
    output: Path,
    variant: dict[str, Any],
    ranked: list[dict[str, Any]],
) -> None:
    top = ranked[:3]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=180)
    axes[0].scatter(
        variant["points"][:, 0],
        variant["points"][:, 1],
        s=1.6,
        color="black",
    )
    axes[0].set_title("DXF候选\n补回局部引线")
    for axis, item in zip(axes[1:], top):
        points = item["_best_points"]
        axis.scatter(points[:, 0], points[:, 1], s=1.6, color="black")
        axis.set_title(
            f"第{item['rank']}名 {item['template_name']}\n"
            f"综合分 {item['combined_score']:.2f}"
        )
    for axis in axes:
        axis.set_aspect("equal")
        axis.axis("off")
    figure.patch.set_facecolor("white")
    figure.tight_layout()
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", type=Path, required=True)
    parser.add_argument("--template-library", type=Path, required=True)
    parser.add_argument("--blind-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    truth = load_json(args.blind_truth)
    truth_item = next(
        item for item in truth["equipment"] if item["id"] == VISUAL_TRUTH_ID
    )
    if truth_item["type"] != "配电变压器":
        raise RuntimeError("视觉真值对象类型不是配电变压器")

    doc = ezdxf.readfile(args.dxf)
    library = load_json(args.template_library)
    transformer_templates = [
        item
        for item in library["templates"]
        if item.get("is_equipment_template") and item["family"] == "PowerTransformer"
    ]
    if len(transformer_templates) != 15:
        raise RuntimeError(
            f"预期15个变压器模板，实际{len(transformer_templates)}个"
        )

    variants = candidate_variants(doc)
    rankings: dict[str, list[dict[str, Any]]] = {}
    for name, variant in variants.items():
        rankings[name] = rank_templates(variant, transformer_templates)

    serializable_rankings: dict[str, list[dict[str, Any]]] = {}
    for name, ranked in rankings.items():
        serializable_rankings[name] = [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in ranked
        ]

    primary = rankings["core_plus_recovered_lead"]
    result = {
        "schema_version": "single-transformer-full-template-ranking-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "experiment_question": (
            "视觉确认的一台DXF变压器能否在知识库15个PowerTransformer模板中"
            "获得稳定的几何相似度排序"
        ),
        "selection": {
            "drawing": DRAWING_NAME,
            "visual_truth_id": VISUAL_TRUTH_ID,
            "visual_label": VISUAL_LABEL,
            "reference_image": REFERENCE_IMAGE,
            "selection_method": (
                "先依据纯视觉盲标和高清图选定左侧原有配变，再用图像—DXF"
                "坐标关系定位其实际DXF圆和引线"
            ),
            "automatic_result_used_for_selection": False,
            "automatic_candidate_checked_after_selection": True,
            "truth_record": truth_item,
        },
        "candidate_variants": {
            name: {
                key: value
                for key, value in variant.items()
                if key != "points"
            }
            for name, variant in variants.items()
        },
        "matching_method": {
            "geometry": (
                "对圆、线和多边形等距采样，保持宽高比归一化；允许90度旋转"
                "和镜像；使用对称Chamfer距离"
            ),
            "primitive_counts": "图元类型多重集合Jaccard",
            "combined_score": "85%几何分+15%图元计数分",
            "note": "本实验只在15个变压器模板内部排序，不等同于全类别识别准确率。",
        },
        "template_count": len(transformer_templates),
        "rankings": serializable_rankings,
        "primary_variant": "core_plus_recovered_lead",
        "primary_top5": serializable_rankings["core_plus_recovered_lead"][:5],
        "findings": [
            "当前自动候选只保留两个圆，缺少模板中的上引线，但已具有明显的变压器核心几何。",
            "补回局部引线后，候选图元组成与多种标准变压器模板的2圆+1线结构一致。",
            "完整支路含小圆、长引线和三角端头，不应作为一个单体变压器直接匹配；应先拆成变压器核心及相邻附件。",
            "若内部排序得分较高，说明模板库可用于变压器识别，当前失败主要来自候选拆分及分类器未调用模板。",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "单台变压器_15模板全量相似度实验.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = args.output_dir / "单台变压器_15模板排序.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "candidate_variant",
            "rank",
            "template_id",
            "template_name",
            "combined_score",
            "geometry_score",
            "primitive_count_score",
            "chamfer_distance",
            "best_transform",
            "usage_count",
            "semantic_status",
            "template_primitive_counts",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant_name, ranked in serializable_rankings.items():
            for item in ranked:
                writer.writerow(
                    {
                        "candidate_variant": variant_name,
                        **{field: item.get(field, "") for field in fields[1:]},
                    }
                )

    image_path = args.output_dir / "单台变压器_候选与Top3模板.png"
    render_comparison(
        image_path, variants["core_plus_recovered_lead"], primary
    )

    top = serializable_rankings["core_plus_recovered_lead"]
    core_only_top = serializable_rankings["current_auto_core"][0]
    full_top = serializable_rankings["full_visual_branch"][0]
    table = "\n".join(
        f"| {item['rank']} | {item['template_name']} | "
        f"{item['combined_score']:.2f} | {item['geometry_score']:.2f} | "
        f"{item['primitive_count_score']:.2f} |"
        for item in top
    )
    report = f"""# 单台变压器与15个标准模板全量相似度实验

## 对象

- 图纸：{DRAWING_NAME}
- 视觉真值：{VISUAL_TRUTH_ID}，{VISUAL_LABEL}
- 图像证据：{REFERENCE_IMAGE}
- DXF核心HANDLE：{", ".join(CORE_CIRCLE_HANDLES)}

该对象先通过纯视觉图选定，再映射回DXF；没有依据自动结果选择“容易匹配”的对象。

## 现有程序为什么漏识别

当前程序把两个主绕组圆分为一个候选，但把相邻竖引线划入导线，因此该候选只有2个圆。硬编码规则要求“至少5个圆、1个三角形和1条竖线”，最终得到Unknown。

## 三种候选的结果

- 当前两圆候选的最高分：{core_only_top['combined_score']:.2f}，模板“{core_only_top['template_name']}”。
- 补回模板尺度局部引线后的最高分：{top[0]['combined_score']:.2f}，模板“{top[0]['template_name']}”。
- 把整条支路全部合并后的最高分：{full_top['combined_score']:.2f}，模板“{full_top['template_name']}”。

## 补回局部引线后的15模板排序

| 排名 | 模板 | 综合分 | 几何分 | 图元计数分 |
|---:|---|---:|---:|---:|
{table}

## 结论

如果“两个大圆＋局部引线”作为单体候选，候选与知识库变压器模板能够直接比较；当前失败不是因为知识库中没有变压器，而是完整核心被拆开、随后又没有执行模板相似度排序。

整条支路还包含三相小圆、长引线和三角端头，应分别识别为变压器核心及相邻连接/附件，不能全部合成一个变压器。

本实验只验证15个变压器模板内部排序。正式识别还需要把该方法扩展至146个设备模板，并加入非变压器负样本和拒识阈值。
"""
    report_path = args.output_dir / "单台变压器_15模板全量相似度实验报告.md"
    report_path.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "templates_ranked": len(transformer_templates),
                "current_core_top_score": core_only_top["combined_score"],
                "recovered_lead_top_score": top[0]["combined_score"],
                "full_branch_top_score": full_top["combined_score"],
                "top_template": top[0]["template_id"],
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
