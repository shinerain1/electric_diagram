"""Prepared template transforms and fast symmetric Chamfer distance."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from template_geometry import dihedral_variants, sample_template


def fast_chamfer(left: np.ndarray, right: np.ndarray) -> float:
    left_tree = cKDTree(left)
    right_tree = cKDTree(right)
    left_to_right = right_tree.query(left, k=1)[0].mean()
    right_to_left = left_tree.query(right, k=1)[0].mean()
    return float((left_to_right + right_to_left) / 2.0)


def prepare_templates(
    library: dict[str, Any],
) -> list[dict[str, Any]]:
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
