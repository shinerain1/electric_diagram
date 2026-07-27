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
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from template_geometry import (
    count_similarity,
    dihedral_variants,
    normalize_points,
    sample_circle,
    sample_line,
    sample_polyline,
    sample_template,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False


ROI = (43000.0, 6500.0, 45200.0, 12100.0)
STUB_LENGTH_RATIOS = [0.50, 0.75, 1.00, 1.125, 1.25, 1.50, 2.00]
NEIGHBOR_SHELL_RATIOS = [0.50, 1.00, 2.00, 3.00]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bbox_distance(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def circle_bbox(entity: Any) -> tuple[float, float, float, float]:
    x = float(entity.dxf.center.x)
    y = float(entity.dxf.center.y)
    radius = float(entity.dxf.radius)
    return x - radius, y - radius, x + radius, y + radius


def line_bbox(entity: Any) -> tuple[float, float, float, float]:
    xs = [float(entity.dxf.start.x), float(entity.dxf.end.x)]
    ys = [float(entity.dxf.start.y), float(entity.dxf.end.y)]
    return min(xs), min(ys), max(xs), max(ys)


def polyline_bbox(entity: Any) -> tuple[float, float, float, float]:
    points = [(float(x), float(y)) for x, y in entity.get_points("xy")]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def inside_roi(box: tuple[float, float, float, float]) -> bool:
    return not (
        box[2] < ROI[0]
        or box[0] > ROI[2]
        or box[3] < ROI[1]
        or box[1] > ROI[3]
    )


def find_circle_pair(circles: list[Any]) -> tuple[Any, Any]:
    ranked: list[tuple[float, float, Any, Any]] = []
    for index, left in enumerate(circles):
        for right in circles[index + 1 :]:
            left_radius = float(left.dxf.radius)
            right_radius = float(right.dxf.radius)
            radius_ratio = abs(left_radius - right_radius) / max(
                left_radius, right_radius
            )
            if radius_ratio > 0.15:
                continue
            center_distance = math.hypot(
                float(left.dxf.center.x) - float(right.dxf.center.x),
                float(left.dxf.center.y) - float(right.dxf.center.y),
            )
            mean_radius = (left_radius + right_radius) / 2.0
            overlap_error = abs(center_distance / mean_radius - 1.0)
            if overlap_error > 0.35:
                continue
            # Prefer the largest clean intersecting equal-radius pair.
            ranked.append(
                (overlap_error + radius_ratio, -mean_radius, left, right)
            )
    if not ranked:
        raise RuntimeError("局部区域内没有找到相交等半径圆对")
    _, _, left, right = min(
        ranked, key=lambda item: (item[0], item[1])
    )
    return left, right


def closest_line_endpoint_to_circles(
    line: Any, circles: list[Any]
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    endpoints = [
        (float(line.dxf.start.x), float(line.dxf.start.y)),
        (float(line.dxf.end.x), float(line.dxf.end.y)),
    ]
    ranked = []
    for endpoint_index, endpoint in enumerate(endpoints):
        boundary_gap = min(
            abs(
                math.hypot(
                    endpoint[0] - float(circle.dxf.center.x),
                    endpoint[1] - float(circle.dxf.center.y),
                )
                - float(circle.dxf.radius)
            )
            for circle in circles
        )
        ranked.append((boundary_gap, endpoint_index))
    gap, near_index = min(ranked)
    return gap, endpoints[near_index], endpoints[1 - near_index]


def clipped_stub(
    near: tuple[float, float],
    far: tuple[float, float],
    length: float,
) -> np.ndarray:
    dx = far[0] - near[0]
    dy = far[1] - near[1]
    norm = max(math.hypot(dx, dy), 1e-12)
    end = (near[0] + dx / norm * length, near[1] + dy / norm * length)
    return sample_line(near, end)


def primitive_points(entity: Any) -> tuple[np.ndarray, str]:
    kind = entity.dxftype()
    if kind == "CIRCLE":
        return (
            sample_circle(
                (float(entity.dxf.center.x), float(entity.dxf.center.y)),
                float(entity.dxf.radius),
            ),
            "circle",
        )
    if kind == "LINE":
        return (
            sample_line(
                (float(entity.dxf.start.x), float(entity.dxf.start.y)),
                (float(entity.dxf.end.x), float(entity.dxf.end.y)),
            ),
            "line",
        )
    if kind == "LWPOLYLINE":
        points = [(float(x), float(y)) for x, y in entity.get_points("xy")]
        return (
            sample_polyline(points, bool(entity.closed)),
            "polygon" if entity.closed else "polyline",
        )
    raise RuntimeError(f"不支持的局部图元：{kind}")


def add_variant(
    variants: list[dict[str, Any]],
    seen: set[str],
    name: str,
    description: str,
    arrays: list[np.ndarray],
    counts: dict[str, int],
    handles: list[str],
    growth_level: float,
) -> None:
    signature = json.dumps(
        {
            "counts": counts,
            "handles": sorted(handles),
            "description": description,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if signature in seen:
        return
    seen.add(signature)
    variants.append(
        {
            "variant_id": name,
            "description": description,
            "points": normalize_points(np.vstack(arrays)),
            "primitive_counts": counts,
            "source_handles": sorted(set(handles)),
            "growth_level": growth_level,
        }
    )


def generate_variants(doc: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    circles = [
        entity
        for entity in doc.modelspace()
        if entity.dxftype() == "CIRCLE" and inside_roi(circle_bbox(entity))
    ]
    lines = [
        entity
        for entity in doc.modelspace()
        if entity.dxftype() == "LINE" and inside_roi(line_bbox(entity))
    ]
    polylines = [
        entity
        for entity in doc.modelspace()
        if entity.dxftype() == "LWPOLYLINE"
        and inside_roi(polyline_bbox(entity))
    ]
    seed_pair = list(find_circle_pair(circles))
    seed_radius = sum(float(circle.dxf.radius) for circle in seed_pair) / 2.0
    seed_box = (
        min(circle_bbox(circle)[0] for circle in seed_pair),
        min(circle_bbox(circle)[1] for circle in seed_pair),
        max(circle_bbox(circle)[2] for circle in seed_pair),
        max(circle_bbox(circle)[3] for circle in seed_pair),
    )
    seed_arrays = [primitive_points(circle)[0] for circle in seed_pair]
    seed_handles = [circle.dxf.handle for circle in seed_pair]

    touching_lines = []
    for line in lines:
        gap, near, far = closest_line_endpoint_to_circles(line, seed_pair)
        if gap <= seed_radius * 0.20:
            touching_lines.append(
                {
                    "entity": line,
                    "gap": gap,
                    "near": near,
                    "far": far,
                }
            )
    touching_lines.sort(key=lambda item: (item["gap"], item["entity"].dxf.handle))

    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    add_variant(
        variants,
        seen,
        "V000_seed",
        "相交等半径圆对种子",
        seed_arrays,
        {"circle": 2},
        seed_handles,
        0.0,
    )

    for line_index, line_info in enumerate(touching_lines, 1):
        for ratio in STUB_LENGTH_RATIOS:
            add_variant(
                variants,
                seen,
                f"V_line{line_index}_{ratio:.3f}",
                f"种子+邻接线{line_info['entity'].dxf.handle}截取{ratio:.3f}倍半径",
                seed_arrays
                + [
                    clipped_stub(
                        line_info["near"],
                        line_info["far"],
                        seed_radius * ratio,
                    )
                ],
                {"circle": 2, "line": 1},
                seed_handles + [line_info["entity"].dxf.handle],
                ratio,
            )

    if len(touching_lines) >= 2:
        for ratio in STUB_LENGTH_RATIOS:
            arrays = list(seed_arrays)
            handles = list(seed_handles)
            for line_info in touching_lines[:2]:
                arrays.append(
                    clipped_stub(
                        line_info["near"],
                        line_info["far"],
                        seed_radius * ratio,
                    )
                )
                handles.append(line_info["entity"].dxf.handle)
            add_variant(
                variants,
                seen,
                f"V_both_{ratio:.3f}",
                f"种子+上下两条邻接线各截取{ratio:.3f}倍半径",
                arrays,
                {"circle": 2, "line": 2},
                handles,
                ratio,
            )

    for shell in NEIGHBOR_SHELL_RATIOS:
        neighbors: list[Any] = []
        for entity in circles:
            if entity not in seed_pair and bbox_distance(
                seed_box, circle_bbox(entity)
            ) <= seed_radius * shell:
                neighbors.append(entity)
        for entity in polylines:
            if bbox_distance(seed_box, polyline_bbox(entity)) <= seed_radius * shell:
                neighbors.append(entity)
        arrays = list(seed_arrays)
        handles = list(seed_handles)
        counts: dict[str, int] = {"circle": 2}
        for entity in neighbors:
            points, kind = primitive_points(entity)
            arrays.append(points)
            handles.append(entity.dxf.handle)
            counts[kind] = counts.get(kind, 0) + 1
        for mode, selected_lines in [
            ("none", []),
            ("one", touching_lines[:1]),
            ("both", touching_lines[:2]),
        ]:
            mode_arrays = list(arrays)
            mode_handles = list(handles)
            mode_counts = dict(counts)
            for line_info in selected_lines:
                mode_arrays.append(
                    clipped_stub(
                        line_info["near"],
                        line_info["far"],
                        seed_radius * 1.125,
                    )
                )
                mode_handles.append(line_info["entity"].dxf.handle)
                mode_counts["line"] = mode_counts.get("line", 0) + 1
            add_variant(
                variants,
                seen,
                f"V_shell_{shell:.2f}_{mode}",
                f"{shell:.2f}倍半径邻域图元+{mode}条局部引线",
                mode_arrays,
                mode_counts,
                mode_handles,
                shell,
            )

    metadata = {
        "roi": ROI,
        "seed_circle_handles": seed_handles,
        "seed_radius": seed_radius,
        "touching_line_handles": [
            item["entity"].dxf.handle for item in touching_lines
        ],
        "variant_count": len(variants),
    }
    return variants, metadata


def fast_chamfer(left: np.ndarray, right: np.ndarray) -> float:
    left_tree = cKDTree(left)
    right_tree = cKDTree(right)
    left_to_right = right_tree.query(left, k=1)[0].mean()
    right_to_left = left_tree.query(right, k=1)[0].mean()
    return float((left_to_right + right_to_left) / 2.0)


def prepare_templates(library: dict[str, Any]) -> list[dict[str, Any]]:
    prepared = []
    for record in library["templates"]:
        if not record.get("is_equipment_template"):
            continue
        try:
            points, sampled_counts = sample_template(record)
            transforms = dihedral_variants(points)
            supported = True
        except RuntimeError:
            sampled_counts = {}
            transforms = []
            supported = False
        prepared.append(
            {
                "record": record,
                "sampled_counts": sampled_counts,
                "transforms": transforms,
                "supported": supported,
            }
        )
    return prepared


def score_variant(
    variant: dict[str, Any],
    templates: list[dict[str, Any]],
    text_prior_family: str,
) -> list[dict[str, Any]]:
    results = []
    for prepared in templates:
        record = prepared["record"]
        if not prepared["supported"]:
            results.append(
                {
                    "template_id": record["symbol_id"],
                    "template_name": record["name"],
                    "family": record["family"],
                    "supported_geometry": False,
                    "combined_score": 0.0,
                    "geometry_score": 0.0,
                    "primitive_count_score": 0.0,
                    "text_prior_bonus": 0.0,
                    "best_transform": "",
                    "chamfer_distance": None,
                }
            )
            continue
        best_distance = float("inf")
        best_transform = ""
        for transform_name, transformed in prepared["transforms"]:
            distance = fast_chamfer(variant["points"], transformed)
            if distance < best_distance:
                best_distance = distance
                best_transform = transform_name
        geometry_score = math.exp(-8.0 * best_distance)
        primitive_score = count_similarity(
            variant["primitive_counts"],
            record.get("primitive_counts", {}),
        )
        base_score = 100.0 * (0.85 * geometry_score + 0.15 * primitive_score)
        text_bonus = 5.0 if record["family"] == text_prior_family else 0.0
        results.append(
            {
                "template_id": record["symbol_id"],
                "template_name": record["name"],
                "family": record["family"],
                "supported_geometry": True,
                "combined_score": round(min(100.0, base_score + text_bonus), 2),
                "base_geometry_and_count_score": round(base_score, 2),
                "geometry_score": round(geometry_score * 100.0, 2),
                "primitive_count_score": round(primitive_score * 100.0, 2),
                "text_prior_bonus": text_bonus,
                "best_transform": best_transform,
                "chamfer_distance": round(best_distance, 6),
            }
        )
    results.sort(
        key=lambda item: (
            -item["combined_score"],
            item["template_id"],
        )
    )
    for index, item in enumerate(results, 1):
        item["rank"] = index
    return results


def render_search_plot(
    output: Path,
    evaluations: list[dict[str, Any]],
    best_variant: dict[str, Any],
    best_ranked: list[dict[str, Any]],
    templates: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=180)
    ordered = sorted(
        evaluations,
        key=lambda item: (
            len(item["variant"]["primitive_counts"]),
            item["variant"]["growth_level"],
            item["variant"]["variant_id"],
        ),
    )
    x = list(range(len(ordered)))
    y = [item["ranked"][0]["combined_score"] for item in ordered]
    colors = [
        "#d62728"
        if item["variant"]["variant_id"] == best_variant["variant_id"]
        else "#4c78a8"
        for item in ordered
    ]
    axes[0].scatter(x, y, c=colors, s=24)
    axes[0].set_xlabel("候选组合序号")
    axes[0].set_ylabel("全146模板最高分")
    axes[0].set_title("候选增长过程中的最高模板分")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        best_variant["points"][:, 0],
        best_variant["points"][:, 1],
        s=2,
        color="black",
        label="最佳DXF候选",
    )
    top_template_id = best_ranked[0]["template_id"]
    prepared = next(
        item for item in templates if item["record"]["symbol_id"] == top_template_id
    )
    transform_name = best_ranked[0]["best_transform"]
    template_points = next(
        points
        for name, points in prepared["transforms"]
        if name == transform_name
    )
    axes[1].scatter(
        template_points[:, 0],
        template_points[:, 1],
        s=1,
        color="#d62728",
        alpha=0.65,
        label=f"Top1：{best_ranked[0]['template_name']}",
    )
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    axes[1].legend(loc="lower right")
    axes[1].set_title(
        f"自动选择：{best_variant['description']}\n"
        f"{best_ranked[0]['family']} / {best_ranked[0]['combined_score']:.2f}"
    )
    figure.tight_layout()
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", type=Path, required=True)
    parser.add_argument("--template-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    doc = ezdxf.readfile(args.dxf)
    variants, generation = generate_variants(doc)
    library = load_json(args.template_library)
    templates = prepare_templates(library)
    if len(templates) != 146:
        raise RuntimeError(f"预期146个设备模板，实际{len(templates)}")

    # The nearest visible text in the selected ROI is “原有配变250kVA”.
    # Text is only a small prior; geometry and primitive structure dominate.
    text_prior_family = "PowerTransformer"
    evaluations = []
    for variant in variants:
        ranked = score_variant(variant, templates, text_prior_family)
        evaluations.append({"variant": variant, "ranked": ranked})

    evaluations.sort(
        key=lambda item: (
            -item["ranked"][0]["combined_score"],
            len(item["variant"]["source_handles"]),
            item["variant"]["variant_id"],
        )
    )
    best = evaluations[0]
    best_variant = best["variant"]
    best_ranked = best["ranked"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for evaluation in evaluations:
        variant = evaluation["variant"]
        for item in evaluation["ranked"][:10]:
            rows.append(
                {
                    "variant_id": variant["variant_id"],
                    "variant_description": variant["description"],
                    "variant_primitive_counts": json.dumps(
                        variant["primitive_counts"], ensure_ascii=False
                    ),
                    "variant_handles": ",".join(variant["source_handles"]),
                    **item,
                }
            )
    csv_path = args.output_dir / "多尺度候选_146模板Top10.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema_version": "multiscale-candidate-full-template-search-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "drawing": args.dxf.name,
        "experiment_scope": (
            "视觉选定变压器附近区域内，自动寻找圆对种子、生成多尺度组合，"
            "并与146个设备模板全量比较"
        ),
        "candidate_generation": generation,
        "template_count": len(templates),
        "templates_with_supported_sampled_geometry": sum(
            item["supported"] for item in templates
        ),
        "templates_with_path_only_geometry": [
            item["record"]["symbol_id"]
            for item in templates
            if not item["supported"]
        ],
        "text_prior": {
            "visible_text": "原有配变250kVA",
            "family": text_prior_family,
            "bonus_points": 5.0,
        },
        "best_candidate": {
            key: value
            for key, value in best_variant.items()
            if key != "points"
        },
        "best_top10": best_ranked[:10],
        "all_variant_summaries": [
            {
                **{
                    key: value
                    for key, value in evaluation["variant"].items()
                    if key != "points"
                },
                "top1": evaluation["ranked"][0],
                "top5": evaluation["ranked"][:5],
            }
            for evaluation in evaluations
        ],
        "conclusion": {
            "selected_family": best_ranked[0]["family"],
            "selected_template": best_ranked[0]["template_id"],
            "selected_score": best_ranked[0]["combined_score"],
            "candidate_boundary_found_by_search": True,
            "fixed_merge_distance_used_as_final_boundary": False,
            "interpretation": (
                "程序通过多种局部组合的模板得分自动选择边界；固定距离只用于"
                "生成候选，不直接决定最终元件范围。"
            ),
        },
        "limitations": [
            "实验仍在一个视觉选定的局部区域内运行，尚未覆盖整张图。",
            "26个含SVG path的模板目前只采样其线、圆和多边形部分；其中1个纯path模板记为不支持。",
            "5分文字先验用于区分几何相似的PT与变压器，阈值仍需更多真值标定。",
        ],
    }
    json_path = args.output_dir / "多尺度候选_146模板全量搜索.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    image_path = args.output_dir / "多尺度候选_得分与最佳模板.png"
    render_search_plot(
        image_path, evaluations, best_variant, best_ranked, templates
    )

    top_rows = "\n".join(
        f"| {item['rank']} | {item['family']} | {item['template_name']} | "
        f"{item['combined_score']:.2f} | {item['geometry_score']:.2f} | "
        f"{item['primitive_count_score']:.2f} | {item['text_prior_bonus']:.2f} |"
        for item in best_ranked[:10]
    )
    variant_rows = "\n".join(
        f"| {index} | {item['variant']['description']} | "
        f"{item['ranked'][0]['family']} | {item['ranked'][0]['template_name']} | "
        f"{item['ranked'][0]['combined_score']:.2f} |"
        for index, item in enumerate(evaluations[:12], 1)
    )
    report = f"""# 多尺度候选与146模板全量搜索试验

## 方法

在“02系统接线图”的原有配变附近，不预先指定最终合并范围：

1. 自动寻找相交且半径相近的圆对作为种子；
2. 逐步加入相邻线段的不同长度截取、附近圆和闭合多段线；
3. 共生成{len(variants)}个候选组合；
4. 每个组合与146个设备模板比较；
5. 几何和图元计数决定主要得分，“原有配变250kVA”只提供5分的小幅类别先验；
6. 选择全局最高分候选作为元件边界。

## 自动选择结果

- 种子圆HANDLE：{", ".join(generation['seed_circle_handles'])}
- 邻接线HANDLE：{", ".join(generation['touching_line_handles'])}
- 最佳候选：{best_variant['description']}
- 最佳候选图元：{best_variant['primitive_counts']}
- Top1类别：{best_ranked[0]['family']}
- Top1模板：{best_ranked[0]['template_name']}
- 综合得分：{best_ranked[0]['combined_score']:.2f}

## 最佳候选的全类别Top10

| 排名 | 家族 | 模板 | 综合分 | 几何分 | 图元计数分 | 文字先验 |
|---:|---|---|---:|---:|---:|---:|
{top_rows}

## 得分最高的候选组合

| 排名 | 候选组合 | Top1家族 | Top1模板 | 得分 |
|---:|---|---|---|---:|
{variant_rows}

## 结论

试验不再用固定距离直接决定元件范围。距离只负责提出候选，最终边界由146模板的最高匹配分决定。

如果最高分落在“两圆＋一段局部引线”附近，而继续加入小圆、长引线和三角端头后得分下降，就说明算法能够自动找到较合理的单体边界。

这只是单个局部区域验证。下一步需要把圆对、矩形、斜刀闸等多种种子扩展到整张图，并通过全局不重叠选择解决候选冲突。
"""
    report_path = args.output_dir / "多尺度候选_146模板全量搜索报告.md"
    report_path.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "variants": len(variants),
                "templates": len(templates),
                "best_variant": best_variant["description"],
                "best_counts": best_variant["primitive_counts"],
                "top_family": best_ranked[0]["family"],
                "top_template": best_ranked[0]["template_id"],
                "top_score": best_ranked[0]["combined_score"],
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
