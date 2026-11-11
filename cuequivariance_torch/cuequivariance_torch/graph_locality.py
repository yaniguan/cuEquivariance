# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""3D Morton space-filling curve and edge locality reordering for equivariant GNNs.

Reordering molecular and materials graphs along a 3D Morton (Z-order) curve
within atomic species groups simultaneously enables:

1. Zero-overhead ``cuet.Linear(..., method="indexed_linear")`` execution because
   ``species_reordered`` is monotonically non-decreasing (so ``species_counts``
   from ``torch.bincount`` directly defines contiguous per-species slices
   without runtime ``argsort`` or inverse-permutation HBM roundtrips).
2. High L2 cache hit rates on A100 (40 MB) and H100 (50 MB) GPUs during edge
   ``src`` feature gather and reduced ``atomicAdd`` L2 cache-line contention
   during ``dst`` feature scatter when edges are sorted lexicographically by
   ``(dst, src)``.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass(frozen=True)
class GraphLocalityPermutation:
    """Bijective node and edge permutations produced by locality reordering.

    Attributes:
        node_perm: Index tensor of shape ``(N,)`` mapping new node order from
            original nodes (``x_reordered = x[node_perm]``).
        inv_node_perm: Inverse permutation of shape ``(N,)`` mapping original
            node indices to reordered indices (``x = x_reordered[inv_node_perm]``).
        edge_perm: Index tensor of shape ``(E,)`` mapping new edge order from
            original edges (``edge_attr_reordered = edge_attr[edge_perm]``).
        inv_edge_perm: Inverse permutation of shape ``(E,)`` mapping original
            edge indices to reordered indices.
        species_counts: Optional ``int64`` tensor from ``torch.bincount`` giving
            the count of each species when ``species`` is provided, otherwise
            ``None``.
    """

    node_perm: torch.Tensor
    inv_node_perm: torch.Tensor
    edge_perm: torch.Tensor
    inv_edge_perm: torch.Tensor
    species_counts: Optional[torch.Tensor] = None


def _spread_bits_3d(coords_int64: torch.Tensor) -> torch.Tensor:
    """Dilate 21-bit non-negative integers so bit ``k`` moves to position ``3 * k``."""
    x = coords_int64 & 0x1FFFFF
    x = (x | (x << 32)) & 0x001F00000000FFFF
    x = (x | (x << 16)) & 0x001F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


def morton_code_3d(positions: torch.Tensor, bits: int = 21) -> torch.Tensor:
    """Compute 63-bit 3D Morton (Z-order) codes for 3D atomic coordinates.

    Coordinates are affinely normalized per axis into ``[0, 2**bits - 1]`` and
    bit-interleaved into non-negative signed ``torch.int64`` integers using pure
    vectorized tensor operations (CPU and CUDA compatible, zero Python loops).

    Args:
        positions: Coordinate tensor of shape ``(N, 3)``.
        bits: Quantization bits per spatial axis (``1 <= bits <= 21``). Defaults
            to ``21`` so ``3 * 21 = 63`` bits fit in non-negative ``torch.int64``.

    Returns:
        1D ``torch.int64`` tensor of shape ``(N,)`` containing non-negative
        Morton codes.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape (N, 3), got {tuple(positions.shape)}."
        )
    if not (1 <= bits <= 21):
        raise ValueError(
            f"bits must be in [1, 21] so 3 * bits <= 63 fits in non-negative int64, got {bits}."
        )

    num_nodes = positions.shape[0]
    device = positions.device
    if num_nodes == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    if num_nodes == 1:
        return torch.zeros((1,), dtype=torch.int64, device=device)

    # Detach and promote to float64 on CPU/CUDA so 21-bit quantization retains full mantissa precision.
    compute_dtype = torch.float64 if device.type in ("cpu", "cuda") else torch.float32
    pos = positions.detach().to(dtype=compute_dtype)

    p_min = pos.amin(dim=0, keepdim=True)
    p_max = pos.amax(dim=0, keepdim=True)
    extent = p_max - p_min
    non_degenerate = extent > 0.0
    safe_extent = torch.where(non_degenerate, extent, torch.ones_like(extent))

    max_grid = float((1 << bits) - 1)
    normalized = torch.where(
        non_degenerate, (pos - p_min) / safe_extent, torch.zeros_like(pos)
    )
    quantized = torch.clamp(
        torch.round(normalized * max_grid), min=0.0, max=max_grid
    ).to(torch.int64)

    sx = _spread_bits_3d(quantized[:, 0])
    sy = _spread_bits_3d(quantized[:, 1])
    sz = _spread_bits_3d(quantized[:, 2])
    return (sx << 2) | (sy << 1) | sz


def reorder_graph_for_locality(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    species: Optional[torch.Tensor] = None,
    *,
    sort_by_species_first: bool = True,
    primary_edge_key: str = "dst",
    bits: int = 21,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    GraphLocalityPermutation,
]:
    """Reorder graph nodes and edges for GPU L2 locality and ``indexed_linear``.

    Nodes are ordered by ``(species, morton_code_3d(positions))`` when
    ``species`` is provided and ``sort_by_species_first=True`` (guaranteeing
    ``species_reordered`` is monotonically non-decreasing for ``indexed_linear``),
    or purely by ``morton_code_3d(positions)`` otherwise. Edges are remapped to
    the new node numbering and sorted lexicographically by ``(dst, src)`` (or
    ``(src, dst)`` when ``primary_edge_key="src"``).

    Args:
        positions: Node 3D coordinates of shape ``(N, 3)``.
        edge_index: Directed edge list ``[src, dst]`` of shape ``(2, E)``.
        species: Optional integer species/element indices of shape ``(N,)``.
        sort_by_species_first: If ``True`` and ``species`` is given, sort nodes
            primarily by ``species`` and secondarily by 3D Morton code.
        primary_edge_key: Primary lexicographic key for sorting edges, either
            ``"dst"`` (default, optimal for scatter-add accumulation) or ``"src"``.
        bits: Bits per coordinate axis passed to :func:`morton_code_3d`.

    Returns:
        Tuple ``(positions_reordered, edge_index_reordered, species_reordered, perm)``.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape (N, 3), got {tuple(positions.shape)}."
        )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}."
        )
    if (
        edge_index.dtype.is_floating_point
        or edge_index.dtype.is_complex
        or edge_index.dtype == torch.bool
    ):
        raise ValueError(
            f"edge_index must be an integer tensor, got {edge_index.dtype}."
        )
    if primary_edge_key not in ("dst", "src"):
        raise ValueError(
            f"primary_edge_key must be 'dst' or 'src', got {primary_edge_key!r}."
        )

    num_nodes = positions.shape[0]
    num_edges = edge_index.shape[1]
    device = positions.device

    if species is not None:
        if species.ndim != 1 or species.shape[0] != num_nodes:
            raise ValueError(
                f"species must have shape ({num_nodes},), got {tuple(species.shape)}."
            )
        if (
            species.dtype.is_floating_point
            or species.dtype.is_complex
            or species.dtype == torch.bool
        ):
            raise ValueError(f"species must be an integer tensor, got {species.dtype}.")
        if num_nodes > 0 and bool((species < 0).any().item()):
            raise ValueError("species indices must be non-negative.")

    morton = morton_code_3d(positions, bits=bits)

    if num_nodes == 0:
        node_perm = torch.empty((0,), dtype=torch.int64, device=device)
        inv_node_perm = torch.empty((0,), dtype=torch.int64, device=device)
    else:
        order_morton = torch.argsort(morton, stable=True)
        if species is not None and sort_by_species_first:
            # Two-pass stable radix sort preserves all 63 bits of Morton code without int64 overflow.
            order_species = torch.argsort(species[order_morton], stable=True)
            node_perm = order_morton[order_species]
        else:
            node_perm = order_morton

        inv_node_perm = torch.empty((num_nodes,), dtype=torch.int64, device=device)
        inv_node_perm[node_perm] = torch.arange(
            num_nodes, dtype=torch.int64, device=device
        )

    positions_reordered = positions[node_perm]
    if species is not None:
        species_reordered: Optional[torch.Tensor] = species[node_perm]
        species_counts: Optional[torch.Tensor] = torch.bincount(
            species_reordered.to(torch.int64)
        )
    else:
        species_reordered = None
        species_counts = None

    if num_edges == 0:
        edge_perm = torch.empty((0,), dtype=torch.int64, device=device)
        inv_edge_perm = torch.empty((0,), dtype=torch.int64, device=device)
        edge_index_reordered = edge_index.clone()
    else:
        remapped = inv_node_perm[edge_index.to(torch.int64)]
        src_remapped = remapped[0]
        dst_remapped = remapped[1]
        primary, secondary = (
            (dst_remapped, src_remapped)
            if primary_edge_key == "dst"
            else (src_remapped, dst_remapped)
        )

        if num_nodes <= (1 << 31):
            composite_key = primary * max(num_nodes, 1) + secondary
            edge_perm = torch.argsort(composite_key, stable=True)
        else:
            sec_order = torch.argsort(secondary, stable=True)
            pri_order = torch.argsort(primary[sec_order], stable=True)
            edge_perm = sec_order[pri_order]

        inv_edge_perm = torch.empty((num_edges,), dtype=torch.int64, device=device)
        inv_edge_perm[edge_perm] = torch.arange(
            num_edges, dtype=torch.int64, device=device
        )
        edge_index_reordered = remapped[:, edge_perm].to(dtype=edge_index.dtype)

    perm = GraphLocalityPermutation(
        node_perm=node_perm,
        inv_node_perm=inv_node_perm,
        edge_perm=edge_perm,
        inv_edge_perm=inv_edge_perm,
        species_counts=species_counts,
    )
    return positions_reordered, edge_index_reordered, species_reordered, perm


def restore_node_order(
    x_reordered: torch.Tensor, perm: GraphLocalityPermutation
) -> torch.Tensor:
    """Restore a node-indexed tensor (e.g., atomic forces) to the original node order."""
    return x_reordered[perm.inv_node_perm]


def restore_edge_order(
    edge_attr_reordered: torch.Tensor, perm: GraphLocalityPermutation
) -> torch.Tensor:
    """Restore an edge-indexed tensor to the original edge order."""
    return edge_attr_reordered[perm.inv_edge_perm]


def _mean_unique_per_window(indices_1d: torch.Tensor, window_size: int) -> float:
    """Vectorized mean count of unique integers per contiguous ``window_size`` chunk."""
    numel = indices_1d.numel()
    if numel == 0:
        return 0.0
    rem = numel % window_size
    if rem != 0:
        # Pad partial tail window with its own final element so unique count is unchanged.
        pad = indices_1d[-1:].expand(window_size - rem)
        padded = torch.cat([indices_1d, pad], dim=0)
    else:
        padded = indices_1d

    windows = padded.reshape(-1, window_size)
    sorted_w, _ = torch.sort(windows, dim=1)
    unique_counts = 1 + (sorted_w[:, 1:] != sorted_w[:, :-1]).sum(dim=1)
    return float(unique_counts.to(torch.float64).mean().item())


def estimate_edge_bandwidth_metrics(
    edge_index: torch.Tensor,
    num_nodes: int,
    window_size: int = 32,
) -> dict[str, float]:
    """Estimate GPU warp gather/scatter cache-line locality metrics on an edge list.

    Partitions ``edge_index`` into contiguous windows of ``window_size`` edges
    (32 threads = 1 GPU warp by default) using pure vectorized PyTorch ops and
    measures unique node touches, 4-node cache-line block touches, edge index
    span, and consecutive-edge step sizes.

    Args:
        edge_index: Edge index tensor ``[src, dst]`` of shape ``(2, E)``.
        num_nodes: Total number of nodes in the graph.
        window_size: Number of consecutive edges processed per warp window
            (must be ``>= 1``; default ``32``).

    Returns:
        Dictionary of locality metrics (lower unique touches and smaller spans
        indicate better L2 cache locality and less ``atomicAdd`` contention).
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}."
        )
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}.")

    num_edges = edge_index.shape[1]
    if num_edges == 0 or num_nodes <= 0:
        return {
            "unique_src_per_window": 0.0,
            "unique_dst_per_window": 0.0,
            "unique_src_cache_lines_per_window": 0.0,
            "unique_dst_cache_lines_per_window": 0.0,
            "mean_edge_span": 0.0,
            "mean_src_step": 0.0,
            "mean_dst_step": 0.0,
            "l2_reuse_factor_src": 0.0,
            "l2_reuse_factor_dst": 0.0,
        }

    src = edge_index[0].to(torch.int64)
    dst = edge_index[1].to(torch.int64)

    unique_src = _mean_unique_per_window(src, window_size)
    unique_dst = _mean_unique_per_window(dst, window_size)

    # Model 4 consecutive nodes sharing a 128-byte cache line (e.g., 32B node blocks/coords).
    nodes_per_cache_line = 4
    unique_src_cl = _mean_unique_per_window(src // nodes_per_cache_line, window_size)
    unique_dst_cl = _mean_unique_per_window(dst // nodes_per_cache_line, window_size)

    mean_span = float(
        (src - dst).abs().to(torch.float64).mean().item() / max(num_nodes, 1)
    )
    if num_edges > 1:
        mean_src_step = float(
            (src[1:] - src[:-1]).abs().to(torch.float64).mean().item()
        )
        mean_dst_step = float(
            (dst[1:] - dst[:-1]).abs().to(torch.float64).mean().item()
        )
    else:
        mean_src_step = 0.0
        mean_dst_step = 0.0

    num_windows = (num_edges + window_size - 1) // window_size
    avg_edges_per_window = num_edges / num_windows

    return {
        "unique_src_per_window": unique_src,
        "unique_dst_per_window": unique_dst,
        "unique_src_cache_lines_per_window": unique_src_cl,
        "unique_dst_cache_lines_per_window": unique_dst_cl,
        "mean_edge_span": mean_span,
        "mean_src_step": mean_src_step,
        "mean_dst_step": mean_dst_step,
        "l2_reuse_factor_src": avg_edges_per_window / max(unique_src, 1e-12),
        "l2_reuse_factor_dst": avg_edges_per_window / max(unique_dst, 1e-12),
    }
