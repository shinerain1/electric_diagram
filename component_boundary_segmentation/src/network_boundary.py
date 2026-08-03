#!/usr/bin/env python3
"""Form component boundaries only from graph-network role probabilities."""

from __future__ import annotations

import numpy as np


class UnionFind:
    """Small disjoint-set implementation for graph connected components."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def component_node_mask(
    role_probability: np.ndarray,
    conductor_threshold: float,
    interface_threshold: float,
) -> np.ndarray:
    """Select component-side nodes without geometric seed overrides.

    Role columns are component body, interface lead and main conductor.
    The interface test is explicit because an interface lead belongs to the
    component boundary even when its body/conductor conditional score is
    ambiguous.
    """
    if role_probability.ndim != 2 or role_probability.shape[1] != 3:
        raise ValueError("role_probability must have shape [N, 3]")
    denominator = role_probability[:, 0] + role_probability[:, 2]
    conditional_conductor = (
        role_probability[:, 2] / np.maximum(denominator, 1e-8)
    )
    interface = (
        (role_probability[:, 1] >= interface_threshold)
        & (role_probability[:, 1] > role_probability[:, 0])
        & (role_probability[:, 1] > role_probability[:, 2])
    )
    return (conditional_conductor < conductor_threshold) | interface


def network_connected_components(
    edge_index: np.ndarray,
    role_probability: np.ndarray,
    conductor_threshold: float,
    interface_threshold: float,
) -> np.ndarray:
    """Return cluster IDs, or -1 for conductors, using only network output.

    The graph edges are the same local-contact edges presented to the GNN.
    No diagonal/closed/circle seed, short-stroke reclaim, template rule, or
    legacy component boundary is consulted.
    """
    count = len(role_probability)
    assignment = np.full(count, -1, dtype=np.int32)
    is_component = component_node_mask(
        role_probability,
        conductor_threshold,
        interface_threshold,
    )
    union_find = UnionFind(count)
    if edge_index.size:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        for left, right in edge_index.T:
            left_index = int(left)
            right_index = int(right)
            if is_component[left_index] and is_component[right_index]:
                union_find.union(left_index, right_index)

    root_to_cluster: dict[int, int] = {}
    for index in np.flatnonzero(is_component):
        root = union_find.find(int(index))
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        assignment[index] = root_to_cluster[root]
    return assignment
