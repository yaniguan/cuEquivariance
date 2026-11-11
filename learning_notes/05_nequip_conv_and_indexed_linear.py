# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Architecture Note 05: NequIP Convolution & Species-Grouped Indexed Linear Audit.

Demonstrates:
1. NequIP / DiffDock / Allegro radial-scalar edge MLP psi_ij = MLP(e_ij, h_i^0, h_j^0):
   Why `FullyConnectedTensorProductConv` splits the first linear weight into
   `(w_edge, w_src, w_dst)` BEFORE edge gathering (O(N) node GEMMs vs O(E) edge GEMM),
   quantifying exact FLOP & HBM savings on a z=30 liquid-electrolyte MD graph.
2. `SegmentedPolynomialIndexedLinear` & `SUBDICT` canonical lowering:
   How forward and backward (`grad_x`, `grad_W`) equivariant linear layers fold
   magnetic dimension `w = 2*l + 1` into `counts * w` species-grouped GEMMs.
3. Exact forward + backward numerical equivalence between `cuet.Linear(weight_classes=Z)`
   (`input_indices` gather) and sorted-species `bincount` Grouped GEMM.
"""

from __future__ import annotations

import logging
import warnings

import torch
from cuequivariance_torch.primitives.segmented_polynomial_indexed_linear import (
    SUBDICT,
    SegmentedPolynomialIndexedLinear,
)

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors

logging.getLogger(
    "cuequivariance_torch.primitives.segmented_polynomial_naive"
).setLevel(logging.ERROR)


def section(title: str) -> None:
    print(f"\n{'=' * 82}\n{title}\n{'=' * 82}")


def audit_nequip_conv_split_mlp() -> None:
    section(
        "1. NequIP / Allegro Edge MLP Split: O(N) Node GEMM vs O(E) Concatenated GEMM"
    )
    torch.manual_seed(42)
    num_nodes = 40
    avg_degree = 30
    num_edges = num_nodes * avg_degree

    d_edge, d_src, d_dst, h_mlp = 16, 64, 64, 64
    in_irreps = cue.Irreps("O3", "16x0e + 16x1o + 8x2e")
    sh_irreps = cue.Irreps("O3", "0e + 1o + 2e")
    out_irreps = cue.Irreps("O3", "16x0e + 16x1o + 8x2e")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        conv = cuet.layers.FullyConnectedTensorProductConv(
            in_irreps=in_irreps,
            sh_irreps=sh_irreps,
            out_irreps=out_irreps,
            batch_norm=False,
            mlp_channels=[d_edge + d_src + d_dst, h_mlp, h_mlp],
            mlp_activation=torch.nn.SiLU(),
            layout=cue.ir_mul,
            use_fallback=True,
        ).to(dtype=torch.float64)

    src = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int64)
    graph = (torch.stack([src, dst], dim=0), (num_nodes, num_nodes))

    src_feat_a = torch.randn(
        num_nodes, in_irreps.dim, dtype=torch.float64, requires_grad=True
    )
    edge_sh = torch.randn(num_edges, sh_irreps.dim, dtype=torch.float64)
    edge_emb_a = torch.randn(num_edges, d_edge, dtype=torch.float64, requires_grad=True)
    src_scal_a = torch.randn(num_nodes, d_src, dtype=torch.float64, requires_grad=True)
    dst_scal_a = torch.randn(num_nodes, d_dst, dtype=torch.float64, requires_grad=True)
    # Cosine cutoff envelope smoothly attenuating outer-shell neighbors
    envelope = 0.5 * (
        1.0 + torch.cos(torch.linspace(0.1, 3.0, num_edges, dtype=torch.float64))
    )

    # Pathway A: Naive O(E) concatenation before MLP
    edge_cat = torch.hstack([edge_emb_a, src_scal_a[src], dst_scal_a[dst]])
    out_concat = conv(
        src_feat_a,
        edge_sh,
        edge_cat,
        graph,
        reduce="sum",
        edge_envelope=envelope,
    )
    cotangent = torch.randn_like(out_concat)
    grads_a = torch.autograd.grad(
        out_concat,
        (src_feat_a, edge_emb_a, src_scal_a, dst_scal_a, conv.mlp[0].weight),
        grad_outputs=cotangent,
    )

    # Pathway B: Pre-gather split GEMM inside FullyConnectedTensorProductConv
    src_feat_b = src_feat_a.detach().clone().requires_grad_(True)
    edge_emb_b = edge_emb_a.detach().clone().requires_grad_(True)
    src_scal_b = src_scal_a.detach().clone().requires_grad_(True)
    dst_scal_b = dst_scal_a.detach().clone().requires_grad_(True)

    out_split = conv(
        src_feat_b,
        edge_sh,
        edge_emb_b,
        graph,
        src_scalars=src_scal_b,
        dst_scalars=dst_scal_b,
        reduce="sum",
        edge_envelope=envelope,
    )
    grads_b = torch.autograd.grad(
        out_split,
        (src_feat_b, edge_emb_b, src_scal_b, dst_scal_b, conv.mlp[0].weight),
        grad_outputs=cotangent,
    )

    fwd_err = float((out_concat - out_split).detach().abs().max())
    bwd_err = max(
        float((ga - gb).detach().abs().max()) for ga, gb in zip(grads_a, grads_b)
    )
    print(f"  TP path weight_numel   : {conv.tp.weight_numel}")
    print(f"  Forward max |diff|     : {fwd_err:.3e}")
    print(f"  Backward max |diff|    : {bwd_err:.3e} (all 5 parameter/input tensors)")
    assert fwd_err < 1e-13 and bwd_err < 1e-12

    # Analytical Electrolyte MD Scaling (N = 8,192 atoms, coordination z = 30 -> E = 245,760)
    n_md = 8192
    z_md = 30
    e_md = n_md * z_md
    de_md, ds_md, dd_md, h_md = 16, 128, 128, 128
    elem_bytes = 4  # FP32

    flops_naive = 2 * e_md * (de_md + ds_md + dd_md) * h_md + e_md * h_md
    flops_split = (
        2 * e_md * de_md * h_md
        + 2 * n_md * (ds_md + dd_md) * h_md
        + 3 * e_md * h_md  # bias + src gather-add + dst gather-add
    )
    # Materializing hstack([e_ij, h_src[src], h_dst[dst]]) writes (E, de+ds+dd) and reads it back
    extra_concat_bytes = 2 * e_md * (ds_md + dd_md) * elem_bytes

    print(
        f"\n  [Electrolyte MD Graph Scaling: N={n_md:,}, z={z_md}, E={e_md:,}, "
        f"dims=({de_md}+{ds_md}+{dd_md})->{h_md}]"
    )
    print(f"    Naive O(E) concat 1st-layer FLOPs : {flops_naive / 1e9:6.3f} GFLOPs")
    print(
        f"    Split O(N)+O(E) 1st-layer FLOPs   : {flops_split / 1e9:6.3f} GFLOPs "
        f"({flops_naive / flops_split:.2f}x speedup, {100.0 * (1.0 - flops_split / flops_naive):.1f}% saved!)"
    )
    print(
        f"    Avoided HBM scalar gather-concat  : {extra_concat_bytes / 1e6:6.1f} MB per interaction block"
    )
    assert flops_naive / flops_split > 10.0


def audit_indexed_linear_subdict_and_memory() -> None:
    section(
        "2. SegmentedPolynomialIndexedLinear: SUBDICT Canonical Subscripts & HBM Footprint"
    )
    irreps_in = cue.Irreps("O3", "32x0e + 32x1o + 32x2e")
    irreps_out = cue.Irreps("O3", "32x0e + 32x1o + 32x2e")
    desc = descriptors.linear(irreps_in, irreps_out)
    fwd_poly = desc.polynomial
    bwd_poly, _ = fwd_poly.backward([True, True], [True])

    sp_fwd = SegmentedPolynomialIndexedLinear(fwd_poly)
    sp_bwd = SegmentedPolynomialIndexedLinear(bwd_poly)

    print(
        f"  Supported SUBDICT patterns ({len(SUBDICT)} total): {list(SUBDICT.keys())}"
    )
    print(
        f"  Forward Linear STP  : raw='{fwd_poly.operations[0][1].subscripts}' -> "
        f"IndexedLinear canonical={sp_fwd.tps[0].subscripts}, "
        f"indexed_buffers={sp_fwd.indexed_input}"
    )
    for idx, (op, stp) in enumerate(bwd_poly.operations):
        tp = sp_bwd.tps[idx]
        role = (
            "grad_W (species-binned outer product)"
            if tp.Cind[0] == 2
            else "grad_x (grouped GEMM)"
        )
        print(
            f"  Backward Op {idx}       : buffers={op.buffers}, raw='{stp.subscripts}' -> "
            f"canonical={tp.subscripts}, Cind={tp.Cind} [{role}]"
        )

    assert sp_fwd.indexed_input == [True, False, None]
    assert sp_bwd.indexed_input == [True, False, False, True, None]

    # Quantify HBM weight-gather explosion for N=16,384 atoms, Z=6 species, C=128 (l_max=2)
    ir_big = cue.Irreps("O3", "128x0e + 128x1o + 128x2e")
    w_numel = descriptors.linear(ir_big, ir_big).inputs[0].dim
    n_atoms, z_species = 16384, 6
    compact_mb = z_species * w_numel * 4 / (1024**2)
    gathered_mb = n_atoms * w_numel * 4 / (1024**2)
    print(
        f"\n  [Species Weight HBM Audit: N={n_atoms:,}, Z={z_species} (LiPF6/EC/DMC), irreps='{ir_big}']"
    )
    print(
        f"    Compact species table (Z, {w_numel:,}) : {compact_mb:7.2f} MB (fits in L2 cache!)"
    )
    print(
        f"    Naive W[species] gather (N, {w_numel:,}): {gathered_mb:7.2f} MB "
        f"({gathered_mb / compact_mb:.0f}x memory blowup eliminated by IndexedLinear!)"
    )


def species_grouped_linear_ir_mul(
    x: torch.Tensor,
    weight: torch.Tensor,
    species_idx: torch.Tensor,
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
) -> torch.Tensor:
    """Pure-PyTorch reference matching `SegmentedPolynomialIndexedLinear` (`counts * w`).

    1. Sorts atoms by species so each chemical element forms a contiguous slice.
    2. Computes `counts = bincount(sorted_species, minlength=Z)`.
    3. Executes one dense GEMM `(count_z * (2l + 1), mul_in) @ (mul_in, mul_out)` per
       `(species z, irrep block)`, mirroring `apply_linear(..., counts * w, ...)`.
    4. Un-permutes back to original atom order.
    """
    num_classes = weight.shape[0]
    perm = torch.argsort(species_idx, stable=True)
    sorted_species = species_idx[perm]
    assert bool(torch.all(sorted_species[1:] - sorted_species[:-1] >= 0).item())
    counts = torch.bincount(sorted_species, minlength=num_classes).tolist()

    x_sorted = x[perm]
    desc = descriptors.linear(irreps_in, irreps_out)
    stp = desc.polynomial.operations[0][1]
    w_slices = stp.operands[0].segment_slices()
    x_slices = stp.operands[1].segment_slices()
    y_slices = stp.operands[2].segment_slices()

    out_segments: list[torch.Tensor] = [
        torch.zeros(x.shape[0], y_sl.stop - y_sl.start, dtype=x.dtype, device=x.device)
        for y_sl in y_slices
    ]

    for path in stp.paths:
        iw, ix, iy = path.indices
        u, v = stp.get_segment_shape(0, path)
        ir_dim, u_check = stp.get_segment_shape(1, path)
        assert u == u_check
        coef = float(path.coefficients.item())

        x_seg = x_sorted[:, x_slices[ix]].reshape(x.shape[0], ir_dim, u)
        w_seg = weight[:, w_slices[iw]].reshape(num_classes, u, v)

        # Grouped GEMM across species bins of row length `count_z * ir_dim`
        y_chunks = []
        row_offset = 0
        for z_id, count_z in enumerate(counts):
            if count_z == 0:
                continue
            x_bin = x_seg[row_offset : row_offset + count_z].reshape(
                count_z * ir_dim, u
            )
            y_bin = (x_bin @ (coef * w_seg[z_id])).reshape(count_z, ir_dim * v)
            y_chunks.append(y_bin)
            row_offset += count_z

        out_segments[iy] = out_segments[iy] + torch.cat(y_chunks, dim=0)

    y_sorted = torch.cat(out_segments, dim=-1)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
    return y_sorted[inv_perm]


def audit_grouped_gemm_fwd_bwd_equivalence() -> None:
    section(
        "3. Species-Grouped GEMM vs Linear(weight_classes=Z) Forward & Backward Gate"
    )
    torch.manual_seed(77)
    num_atoms = 64
    num_species = 6
    irreps_in = cue.Irreps("O3", "24x0e + 16x1o + 8x2e")
    irreps_out = cue.Irreps("O3", "16x0e + 24x1o + 12x2e")

    # Verify SegmentedPolynomialIndexedLinear enforces sorted species indices before kernel call
    sp_idx = SegmentedPolynomialIndexedLinear(
        descriptors.linear(irreps_in, irreps_out).polynomial
    )
    unsorted_species = torch.randint(0, num_species, (num_atoms,), dtype=torch.int64)
    if bool(torch.all(unsorted_species[1:] >= unsorted_species[:-1]).item()):
        unsorted_species[0], unsorted_species[1] = 5, 0

    caught_unsorted = False
    try:
        sp_idx(
            [
                torch.zeros(num_species, sp_idx.input_sizes[0]),
                torch.zeros(num_atoms, sp_idx.input_sizes[1]),
            ],
            input_indices={0: unsorted_species},
            output_shapes={},
            output_indices={},
        )
    except AssertionError as exc:
        caught_unsorted = "non-sorted indices" in str(exc)
    print(f"  Unsorted species guard enforced : {caught_unsorted}")
    assert caught_unsorted

    # Compare cuet.Linear(weight_classes=Z) vs sorted-species bincount Grouped GEMM
    lin_mod = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=cue.ir_mul,
        weight_classes=num_species,
        method="naive",
        dtype=torch.float64,
        math_dtype=torch.float64,
    )

    x_ref = torch.randn(
        num_atoms, irreps_in.dim, dtype=torch.float64, requires_grad=True
    )
    y_ref = lin_mod(x_ref, weight_indices=unsorted_species)
    cotangent = torch.randn_like(y_ref)
    gx_ref, gw_ref = torch.autograd.grad(
        y_ref, (x_ref, lin_mod.weight), grad_outputs=cotangent
    )

    x_grp = x_ref.detach().clone().requires_grad_(True)
    w_grp = lin_mod.weight.detach().clone().requires_grad_(True)
    y_grp = species_grouped_linear_ir_mul(
        x_grp, w_grp, unsorted_species, irreps_in, irreps_out
    )
    gx_grp, gw_grp = torch.autograd.grad(y_grp, (x_grp, w_grp), grad_outputs=cotangent)

    err_y = float((y_ref - y_grp).detach().abs().max())
    err_gx = float((gx_ref - gx_grp).detach().abs().max())
    err_gw = float((gw_ref - gw_grp).detach().abs().max())
    print(f"  Forward  max |y_naive  - y_grouped | : {err_y:.3e}")
    print(f"  Backward max |gx_naive - gx_grouped| : {err_gx:.3e}")
    print(f"  Backward max |gw_naive - gw_grouped| : {err_gw:.3e}")
    assert err_y < 1e-13 and err_gx < 1e-13 and err_gw < 1e-13


if __name__ == "__main__":
    audit_nequip_conv_split_mlp()
    audit_indexed_linear_subdict_and_memory()
    audit_grouped_gemm_fwd_bwd_equivalence()
    print(
        "\n[PASS] 05_nequip_conv_and_indexed_linear.py completed all checks successfully."
    )
