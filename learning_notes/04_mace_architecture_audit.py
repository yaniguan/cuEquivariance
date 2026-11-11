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
"""Architecture Note 04: MACE Multi-Body Pipeline & Symmetric Contraction Audit.

Demonstrates:
1. Stage A (2-Body Atomic Density Basis A_{i, c l m}):
   Edge `SphericalHarmonics` + `ChannelWiseTensorProduct` with fused neighbor
   gather/scatter (`indices_1=src`, `indices_out=dst`) + `Linear` channel mixing.
2. Stage B (Multi-Body Product Basis, correlation orders nu = 1, 2, 3 -> up to 4-body):
   How `SegmentedPolynomial.stack` fuses nu = 1, 2, 3 into one polynomial, and why
   `cuEquivariance` compresses `original_mace=True` (`e3nn` CG U-matrix basis) via
   projection matrix P without losing any equivariant expressivity.
3. Exact numerical equivalence (`original_mace=True` vs offline-folded `original_mace=False`
   vs raw `U_matrix_real` polynomial) and end-to-end O(3) rotation + inversion equivariance.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.spatial.transform
import torch
from cuequivariance.group_theory.experimental.mace import (
    symmetric_contractions as mace_sc,
)

import cuequivariance as cue
import cuequivariance_torch as cuet

logging.getLogger(
    "cuequivariance_torch.primitives.segmented_polynomial_naive"
).setLevel(logging.ERROR)


def section(title: str) -> None:
    print(f"\n{'=' * 82}\n{title}\n{'=' * 82}")


def build_cluster_graph(
    num_atoms: int = 12, cutoff: float = 2.6, seed: int = 7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    pos = torch.tensor(rng.uniform(-1.5, 1.5, size=(num_atoms, 3)), dtype=torch.float64)
    species = torch.tensor(rng.integers(0, 4, size=(num_atoms,)), dtype=torch.int64)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # r_ij = r_j - r_i
    dist = torch.linalg.norm(diff, dim=-1)
    mask = (dist > 1e-6) & (dist < cutoff)
    dst, src = torch.where(mask)
    return pos, species, src, dst


def build_mace_u_matrix_poly(
    irreps_in: cue.Irreps, irreps_out: cue.Irreps, degrees: tuple[int, ...]
) -> cue.SegmentedPolynomial:
    """Reconstructs the un-compressed original MACE CG U-matrix polynomial (`poly1`)."""
    return cue.SegmentedPolynomial.stack(
        [
            cue.SegmentedPolynomial.stack(
                [
                    mace_sc._symmetric_contraction_poly(
                        irreps_in, irreps_out[i : i + 1], deg
                    )
                    for deg in reversed(degrees)
                ],
                [True, False, False],
            )
            for i in range(len(irreps_out))
        ],
        [True, False, True],
    )


def audit_stage_a_atomic_basis() -> None:
    section("1. Stage A: 2-Body Atomic Basis A_{i,clm} & Fused Neighbor Gather/Scatter")
    pos, _, src, dst = build_cluster_graph()
    num_atoms = pos.shape[0]
    num_edges = src.shape[0]

    mul = 16
    irreps_node = cue.Irreps("O3", f"{mul}x0e + {mul}x1o + {mul}x2e")
    sh_irreps = cue.Irreps("O3", "0e + 1o + 2e")
    irreps_a = cue.Irreps("O3", f"{mul}x0e + {mul}x1o + {mul}x2e")

    sh_mod = cuet.SphericalHarmonics(
        ls=[0, 1, 2], normalize=True, method="naive", math_dtype=torch.float64
    )
    tp_mod = cuet.ChannelWiseTensorProduct(
        irreps_node,
        sh_irreps,
        filter_irreps_out=[cue.O3(0, 1), cue.O3(1, -1), cue.O3(2, 1)],
        layout=cue.ir_mul,
        shared_weights=False,
        internal_weights=False,
        method="naive",
        dtype=torch.float64,
        math_dtype=torch.float64,
    )
    mix_lin = cuet.Linear(
        tp_mod.irreps_out,
        irreps_a,
        layout=cue.ir_mul,
        method="naive",
        dtype=torch.float64,
        math_dtype=torch.float64,
    )

    torch.manual_seed(11)
    h_nodes = torch.randn(num_atoms, irreps_node.dim, dtype=torch.float64)
    radial_w = torch.randn(num_edges, tp_mod.weight_numel, dtype=torch.float64)
    edge_vec = pos[src] - pos[dst]
    edge_sh = sh_mod(edge_vec)

    # Fused gather + scatter inside ChannelWiseTensorProduct
    agg_fused = tp_mod(
        h_nodes,
        edge_sh,
        radial_w,
        indices_1=src,
        indices_out=dst,
        size_out=num_atoms,
    )
    # Explicit unfused gather -> edge TP -> scatter_add reference
    edge_messages = tp_mod(h_nodes[src], edge_sh, radial_w)
    agg_unfused = torch.zeros(num_atoms, tp_mod.irreps_out.dim, dtype=torch.float64)
    agg_unfused.index_add_(0, dst, edge_messages)

    a_basis = mix_lin(agg_fused)
    diff = float((agg_fused - agg_unfused).detach().abs().max())

    print(
        f"  Cluster graph          : N={num_atoms} atoms, E={num_edges} directed edges"
    )
    print(
        f"  ChannelWiseTP irreps   : {irreps_node} x {sh_irreps} -> {tp_mod.irreps_out}"
    )
    print(
        f"  Edge message buffer    : ({num_edges}, {tp_mod.irreps_out.dim}) eliminated in HBM via fused gather/scatter"
    )
    print(
        f"  Atomic basis A_i shape : {tuple(a_basis.shape)} ({irreps_a}, layout=ir_mul)"
    )
    print(f"  Max |fused - unfused|  : {diff:.3e}")
    assert diff < 1e-14


def audit_stage_b_symmetric_contraction() -> None:
    section(
        "2. Stage B: Multi-Body Symmetric Contraction (nu=1,2,3) & Basis Compression"
    )
    degrees = (1, 2, 3)  # nu=1 (2-body), nu=2 (3-body), nu=3 (4-body)

    for group_name, ir_in_str, ir_out_str in [
        ("SO3", "16x0 + 16x1 + 16x2", "16x0 + 16x1"),
        ("O3", "16x0e + 16x1o + 16x2e", "16x0e + 16x1o"),
    ]:
        irreps_in = cue.Irreps(group_name, ir_in_str)
        irreps_out = cue.Irreps(group_name, ir_out_str)
        mul = irreps_in.muls[0]

        etp_mace, proj = mace_sc.symmetric_contraction(irreps_in, irreps_out, degrees)
        etp_std = cue.descriptors.symmetric_contraction(irreps_in, irreps_out, degrees)
        assert etp_mace == etp_std

        poly_u = build_mace_u_matrix_poly(irreps_in, irreps_out, degrees)
        poly_c = etp_std.polynomial

        paths_u = sum(stp.num_paths for _, stp in poly_u.operations)
        paths_c = sum(stp.num_paths for _, stp in poly_c.operations)
        flops_u = poly_u.flop(1)
        flops_c = poly_c.flop(1)
        rank_p = int(np.linalg.matrix_rank(proj))

        print(f"  [{group_name}] {irreps_in} -> {irreps_out} (degrees={degrees}):")
        print(
            f"    Projection P shape (MACE CG -> cuEquivariance) : {proj.shape}, rank={rank_p} "
            f"({proj.shape[0] / proj.shape[1]:.2f}x fewer basis weights!)"
        )
        print(
            f"    Total STP paths (U_matrix_real vs compressed)  : {paths_u:4d} -> {paths_c:4d} "
            f"({paths_u / paths_c:.2f}x path reduction)"
        )
        print(
            f"    FLOPs per atom  (U_matrix_real vs compressed)  : {flops_u:6d} -> {flops_c:6d} "
            f"({100.0 * (1.0 - flops_c / flops_u):.1f}% FLOP saving)"
        )
        for op, stp in poly_c.operations:
            nu = len(op.buffers) - 2
            print(
                f"      nu={nu} ({nu + 1}-body term): buffers={op.buffers}, "
                f"subscripts={stp.subscripts}, num_paths={stp.num_paths:3d}, segment_u={mul}"
            )
        assert rank_p == proj.shape[1]
        assert flops_c < flops_u

    # Verify exact numerical equivalence between original_mace=True, folded original_mace=False, and raw poly_u
    torch.manual_seed(23)
    irreps_in = cue.Irreps("O3", "16x0e + 16x1o + 16x2e")
    irreps_out = cue.Irreps("O3", "16x0e + 16x1o")
    num_elements = 4
    num_atoms = 12

    sc_orig = cuet.SymmetricContraction(
        irreps_in,
        irreps_out,
        contraction_degree=3,
        num_elements=num_elements,
        layout=cue.ir_mul,
        original_mace=True,
        method="naive",
        dtype=torch.float64,
        math_dtype=torch.float64,
    )
    sc_comp = cuet.SymmetricContraction(
        irreps_in,
        irreps_out,
        contraction_degree=3,
        num_elements=num_elements,
        layout=cue.ir_mul,
        original_mace=False,
        method="naive",
        dtype=torch.float64,
        math_dtype=torch.float64,
    )
    with torch.no_grad():
        folded_w = torch.einsum("zau,ab->zbu", sc_orig.weight, sc_orig.projection)
        sc_comp.weight.copy_(folded_w)

    x_a = torch.randn(num_atoms, irreps_in.dim, dtype=torch.float64)
    elem_idx = torch.randint(0, num_elements, (num_atoms,), dtype=torch.int64)

    y_orig = sc_orig(x_a, elem_idx)
    y_comp = sc_comp(x_a, elem_idx)

    poly_u = build_mace_u_matrix_poly(irreps_in, irreps_out, degrees)
    sp_u = cuet.SegmentedPolynomial(poly_u, method="naive", math_dtype=torch.float64)
    [y_raw_u] = sp_u(
        [sc_orig.weight.detach().flatten(1), x_a], input_indices={0: elem_idx}
    )

    err_fold = float((y_orig - y_comp).detach().abs().max())
    err_raw = float((y_orig - y_raw_u).detach().abs().max())
    print(
        f"  Max |original_mace=True - offline-folded False| : {err_fold:.3e} (zero runtime projection cost!)"
    )
    print(f"  Max |original_mace=True - raw U_matrix_real IR| : {err_raw:.3e}")
    assert err_fold < 1e-13 and err_raw < 1e-12


def audit_end_to_end_mace_o3_equivariance() -> None:
    section("3. End-to-End 2-Stage MACE Pipeline O(3) Rotation + Inversion Audit")
    pos, species, src, dst = build_cluster_graph()
    num_atoms = pos.shape[0]
    num_edges = src.shape[0]

    mul = 8
    irreps_in = cue.Irreps("O3", f"{mul}x0e + {mul}x1o")
    sh_irreps = cue.Irreps("O3", "0e + 1o + 2e")
    irreps_a = cue.Irreps("O3", f"{mul}x0e + {mul}x1o + {mul}x2e")
    irreps_out = cue.Irreps("O3", f"{mul}x0e + {mul}x1o + {mul}x2e")

    torch.manual_seed(99)
    sh_mod = cuet.SphericalHarmonics(
        ls=[0, 1, 2], normalize=True, method="naive", math_dtype=torch.float64
    )
    tp_mod = cuet.ChannelWiseTensorProduct(
        irreps_in,
        sh_irreps,
        filter_irreps_out=[cue.O3(0, 1), cue.O3(1, -1), cue.O3(2, 1)],
        layout=cue.ir_mul,
        shared_weights=False,
        internal_weights=False,
        method="naive",
        dtype=torch.float64,
        math_dtype=torch.float64,
    )
    mix_lin = cuet.Linear(
        tp_mod.irreps_out,
        irreps_a,
        layout=cue.ir_mul,
        method="naive",
        dtype=torch.float64,
        math_dtype=torch.float64,
    )
    sym_ctr = cuet.SymmetricContraction(
        irreps_a,
        irreps_out,
        contraction_degree=3,
        num_elements=4,
        layout=cue.ir_mul,
        original_mace=False,
        method="naive",
        dtype=torch.float64,
        math_dtype=torch.float64,
    )

    h0 = torch.randn(num_atoms, irreps_in.dim, dtype=torch.float64)
    # Radial weights depend strictly on O(3)-invariant pair distances ||r_ij||
    dists = torch.linalg.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
    w_rad_proj = torch.randn(1, tp_mod.weight_numel, dtype=torch.float64)
    radial_w = torch.exp(-0.5 * dists**2) @ w_rad_proj
    assert radial_w.shape == (num_edges, tp_mod.weight_numel)

    def forward_mace_block(
        coords: torch.Tensor, node_feats: torch.Tensor
    ) -> torch.Tensor:
        y_ij = sh_mod(coords[src] - coords[dst])
        m_i = tp_mod(
            node_feats,
            y_ij,
            radial_w,
            indices_1=src,
            indices_out=dst,
            size_out=num_atoms,
        )
        a_i = mix_lin(m_i)
        return sym_ctr(a_i, species)

    rng = np.random.default_rng(2025)
    rotvec = scipy.spatial.transform.Rotation.random(random_state=rng).as_rotvec()
    vec_ir = cue.O3(1, -1)
    ir_in_layout = cue.IrrepsAndLayout(irreps_in, cue.ir_mul)
    ir_out_layout = cue.IrrepsAndLayout(irreps_out, cue.ir_mul)

    out_ref = forward_mace_block(pos, h0)
    for parity_p, label in [
        (0, "Proper SO(3) rotation (det=+1)"),
        (1, "Improper O(3) roto-inversion (det=-1)"),
    ]:
        d_vec = torch.tensor(
            vec_ir.exp_map(rotvec, np.array([parity_p])), dtype=torch.float64
        )
        d_in = torch.tensor(
            ir_in_layout.exp_map(rotvec, np.array([parity_p])), dtype=torch.float64
        )
        d_out = torch.tensor(
            ir_out_layout.exp_map(rotvec, np.array([parity_p])), dtype=torch.float64
        )

        out_lhs = forward_mace_block(pos @ d_vec.T, h0 @ d_in.T)
        out_rhs = out_ref @ d_out.T
        max_err = float((out_lhs - out_rhs).detach().abs().max())
        rel_err = float((out_lhs - out_rhs).detach().norm() / out_rhs.detach().norm())
        print(f"  {label:<38s}: max|diff|={max_err:.3e}, rel_err={rel_err:.3e}")
        assert max_err < 1e-11 and rel_err < 1e-14


if __name__ == "__main__":
    audit_stage_a_atomic_basis()
    audit_stage_b_symmetric_contraction()
    audit_end_to_end_mace_o3_equivariance()
    print("\n[PASS] 04_mace_architecture_audit.py completed all checks successfully.")
