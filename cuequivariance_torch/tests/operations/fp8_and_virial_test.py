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
"""Tests for H100 O(3)-equivariant FP8 quantization and PBC conservative virial stress."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet


def _build_wigner_matrix(
    irreps: cue.Irreps,
    axis: np.ndarray,
    angle: float,
    invert: bool,
    layout: cue.IrrepsLayout,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Construct the exact block-diagonal O(3) representation matrix D(g) in the given layout."""
    blocks: list[torch.Tensor] = []
    for mul, ir in irreps:
        if mul == 0:
            continue
        d_mat = torch.from_numpy(ir.rotation(axis, angle)).to(
            dtype=dtype, device=device
        )
        if invert and ir.p == -1:
            d_mat = -d_mat
        eye_mul = torch.eye(mul, dtype=dtype, device=device)
        if layout == cue.mul_ir:
            blocks.append(torch.kron(eye_mul, d_mat))
        else:
            blocks.append(torch.kron(d_mat, eye_mul))
    if not blocks:
        return torch.empty((0, 0), dtype=dtype, device=device)
    return torch.block_diag(*blocks)


def _naive_linf_scales(
    x: torch.Tensor,
    irreps: cue.Irreps,
    layout: cue.IrrepsLayout,
    fp8_dtype: torch.dtype,
) -> torch.Tensor:
    """Naive non-equivariant L_inf (amax) per-channel scale across m in [-l, l]."""
    fp8_max = float(torch.finfo(fp8_dtype).max)
    batch_shape = x.shape[:-1]
    parts = []
    for sl, (mul, ir) in zip(irreps.slices(), irreps):
        if mul == 0:
            continue
        seg = x[..., sl]
        if layout == cue.mul_ir:
            amax = seg.reshape(*batch_shape, mul, ir.dim).abs().amax(dim=-1)
        else:
            amax = seg.reshape(*batch_shape, ir.dim, mul).abs().amax(dim=-2)
        parts.append(amax.clamp(min=1e-12) / fp8_max)
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# 1. Exports & Module Surface
# ---------------------------------------------------------------------------


def test_public_exports():
    expected = [
        "irrep_fp8_quantize",
        "irrep_fp8_dequantize",
        "irrep_fp8_ste",
        "EquivariantFP8Linear",
        "apply_cell_strain",
        "compute_forces_and_virial",
        "decompose_stress_irreps",
        "ConservativeVirialStress",
    ]
    for name in expected:
        assert hasattr(cuet, name), f"Missing {name} in cuequivariance_torch"
        assert hasattr(cuet.operations, name), (
            f"Missing {name} in cuequivariance_torch.operations"
        )
        assert getattr(cuet, name) is getattr(cuet.operations, name)


# ---------------------------------------------------------------------------
# 2. FP8 O(3) Scale Invariance, Zero Clipping & Equivariance Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize(
    "granularity", ["per_irrep", "per_channel", "per_atom", "per_tensor"]
)
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("invert", [False, True])
def test_fp8_o3_scale_invariance_and_zero_clipping(
    layout: cue.IrrepsLayout,
    granularity: str,
    fp8_dtype: torch.dtype,
    invert: bool,
):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    irreps = cue.Irreps("O3", "4x0e + 3x1o + 2x2e + 2x3o")
    x = torch.randn(16, irreps.dim, dtype=torch.float64) * 37.5
    # Include an exact zero row and an extreme axis-aligned row to test saturation & eps safety
    x[0] = 0.0
    x[1] = 0.0
    x[1, irreps.slices()[1].start] = 1e4  # Pure m-axis vector in 1o

    q_orig, scale_orig = cuet.irrep_fp8_quantize(
        x,
        irreps,
        layout=layout,
        fp8_dtype=fp8_dtype,
        granularity=granularity,
    )
    _, scale_orig_f64 = cuet.irrep_fp8_quantize(
        x,
        irreps,
        layout=layout,
        fp8_dtype=fp8_dtype,
        granularity=granularity,
        scale_dtype=torch.float64,
    )
    assert q_orig.dtype == fp8_dtype
    assert scale_orig.dtype == torch.float32
    assert torch.isfinite(q_orig.to(torch.float32)).all()

    fp8_max = float(torch.finfo(fp8_dtype).max)

    for _ in range(5):
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = float(rng.uniform(0.1, 3.0))
        d_g = _build_wigner_matrix(
            irreps, axis, angle, invert, layout, torch.float64, x.device
        )
        x_rot = x @ d_g.T

        q_rot, scale_rot = cuet.irrep_fp8_quantize(
            x_rot,
            irreps,
            layout=layout,
            fp8_dtype=fp8_dtype,
            granularity=granularity,
        )
        _, scale_rot_f64 = cuet.irrep_fp8_quantize(
            x_rot,
            irreps,
            layout=layout,
            fp8_dtype=fp8_dtype,
            granularity=granularity,
            scale_dtype=torch.float64,
        )

        # 100% O(3)-invariant scales to < 1e-12 in both FP64 and FP32!
        assert (scale_rot_f64 - scale_orig_f64).abs().max().item() < 1e-12
        assert (scale_rot - scale_orig).abs().max().item() < 1e-12
        assert torch.isfinite(q_rot.to(torch.float32)).all()
        assert q_rot.to(torch.float32).abs().max().item() <= fp8_max

        # Verify bounded FP8 mantissa equivariance error
        deq_orig_rot = (
            cuet.irrep_fp8_dequantize(
                q_orig,
                scale_orig,
                irreps,
                layout=layout,
                granularity=granularity,
                out_dtype=torch.float64,
            )
            @ d_g.T
        )
        deq_rot = cuet.irrep_fp8_dequantize(
            q_rot,
            scale_rot,
            irreps,
            layout=layout,
            granularity=granularity,
            out_dtype=torch.float64,
        )
        # Relative equivariance error on non-zero rows bounded by FP8 mantissa step
        rel_err = torch.linalg.vector_norm(deq_rot[2:] - deq_orig_rot[2:], dim=-1) / (
            torch.linalg.vector_norm(x_rot[2:], dim=-1)
        )
        max_rel_tol = 0.08 if fp8_dtype == torch.float8_e4m3fn else 0.16
        assert rel_err.max().item() < max_rel_tol


def test_fp8_l2_scale_beats_naive_linf_scale_drift():
    """Prove naive L_inf scale drifts by up to sqrt(2l+1) under rotation while L2 scale is exact."""
    irreps = cue.Irreps("O3", "1x1o + 1x2e + 1x3o")
    # Axis-aligned vector in 1o (1, 0, 0) and single-component in 3o (1, 0, 0, 0, 0, 0, 0)
    x = torch.zeros(1, irreps.dim, dtype=torch.float64)
    x[0, 0] = 1.0  # 1o
    x[0, 3] = 1.0  # 2e
    x[0, 8] = 1.0  # 3o

    # Rotate (1, 0, 0) -> (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
    # Rotate around axis (1, 1, 1) won't move (1,1,1); choose rotation sending e_x -> (1,1,1)/sqrt(3)
    v_target = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    e_x = np.array([1.0, 0.0, 0.0])
    axis = np.cross(e_x, v_target)
    axis = axis / np.linalg.norm(axis)
    angle = float(np.arccos(np.dot(e_x, v_target)))

    d_g = _build_wigner_matrix(
        irreps, axis, angle, False, cue.ir_mul, torch.float64, x.device
    )
    x_rot = x @ d_g.T

    linf_orig = _naive_linf_scales(x, irreps, cue.ir_mul, torch.float8_e4m3fn)
    linf_rot = _naive_linf_scales(x_rot, irreps, cue.ir_mul, torch.float8_e4m3fn)
    # For 1o, L_inf drops from 1.0 to 1/sqrt(3) = 0.57735 (> 42% scale drift!)
    linf_rel_drift = ((linf_rot - linf_orig).abs() / linf_orig).max().item()
    assert linf_rel_drift > 0.40

    _, l2_orig = cuet.irrep_fp8_quantize(
        x,
        irreps,
        layout=cue.ir_mul,
        fp8_dtype=torch.float8_e4m3fn,
        granularity="per_channel",
        scale_dtype=torch.float64,
    )
    _, l2_rot = cuet.irrep_fp8_quantize(
        x_rot,
        irreps,
        layout=cue.ir_mul,
        fp8_dtype=torch.float8_e4m3fn,
        granularity="per_channel",
        scale_dtype=torch.float64,
    )
    l2_rel_drift = ((l2_rot - l2_orig).abs() / l2_orig).max().item()
    assert l2_rel_drift < 1e-15


@pytest.mark.parametrize("granularity", ["per_irrep", "per_channel"])
def test_fp8_layout_consistency_without_transpose(granularity: str):
    """Verify ir_mul and mul_ir produce identical scales and consistent quantized tensors."""
    torch.manual_seed(7)
    irreps = cue.Irreps("O3", "3x0e + 4x1o + 2x2e + 3x3o")
    x_ir_mul = torch.randn(8, irreps.dim, dtype=torch.float64)
    t_to_mul_ir = cuet.TransposeIrrepsLayout(
        irreps, source=cue.ir_mul, target=cue.mul_ir, use_fallback=True
    )
    x_mul_ir = t_to_mul_ir(x_ir_mul)

    q_im, s_im = cuet.irrep_fp8_quantize(
        x_ir_mul, irreps, layout=cue.ir_mul, granularity=granularity
    )
    q_mi, s_mi = cuet.irrep_fp8_quantize(
        x_mul_ir, irreps, layout=cue.mul_ir, granularity=granularity
    )

    torch.testing.assert_close(s_im, s_mi, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        t_to_mul_ir(q_im.to(torch.float32)),
        q_mi.to(torch.float32),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("fp8_act", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("fp8_weight", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("scale_grad", [False, True])
def test_equivariant_fp8_linear_fwd_bwd_and_2nd_bwd_force_qat(
    fp8_act: torch.dtype,
    fp8_weight: torch.dtype,
    layout: cue.IrrepsLayout,
    scale_grad: bool,
):
    """Test EquivariantFP8Linear forward, equivariance, 1st backward, and 2nd backward Force-QAT."""
    torch.manual_seed(123)
    irreps_in = cue.Irreps("O3", "4x0e + 4x1o + 2x2e")
    irreps_out = cue.Irreps("O3", "4x0e + 3x1o + 2x2e")

    fp8_lin = cuet.EquivariantFP8Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        fp8_dtype_act=fp8_act,
        fp8_dtype_weight=fp8_weight,
        scale_grad=scale_grad,
        dtype=torch.float64,
    )
    ref_lin = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        dtype=torch.float64,
        method="naive",
    )
    with torch.no_grad():
        ref_lin.weight.copy_(fp8_lin.weight)

    # 1. Forward consistency against full-precision cuet.Linear & O(3) equivariance
    x = torch.randn(16, irreps_in.dim, dtype=torch.float64)
    y_fp8 = fp8_lin(x)
    y_ref = ref_lin(x)
    assert y_fp8.shape == (16, irreps_out.dim)
    rel_diff = torch.linalg.vector_norm(y_fp8 - y_ref) / torch.linalg.vector_norm(y_ref)
    tol = (
        0.09
        if (fp8_act == torch.float8_e4m3fn and fp8_weight == torch.float8_e4m3fn)
        else 0.18
    )
    assert rel_diff.item() < tol

    axis = np.array([0.5, -0.3, 0.8])
    axis = axis / np.linalg.norm(axis)
    angle = 1.25
    d_in = _build_wigner_matrix(
        irreps_in, axis, angle, True, layout, torch.float64, x.device
    )
    d_out = _build_wigner_matrix(
        irreps_out, axis, angle, True, layout, torch.float64, x.device
    )
    y_rot_in = fp8_lin(x @ d_in.T)
    y_rot_out = y_fp8 @ d_out.T
    equiv_err = torch.linalg.vector_norm(
        y_rot_in - y_rot_out
    ) / torch.linalg.vector_norm(y_rot_out)
    assert equiv_err.item() < tol

    # 2. 1st backward & 2nd backward (Force-QAT d/dW of -dE/dR!)
    pos = torch.randn(16, 3, dtype=torch.float64, requires_grad=True)
    sh = cuet.SphericalHarmonics([0, 1, 2], normalize=False, method="naive")(pos)
    # Lift spherical harmonics (0e + 1o + 2e) into irreps_in cleanly
    lift = cuet.Linear(
        cue.Irreps("O3", "0e + 1o + 2e"),
        irreps_in,
        layout_in=cue.ir_mul,
        layout_out=layout,
        dtype=torch.float64,
        method="naive",
    )
    feats = lift(sh) * (pos.pow(2).sum(dim=-1, keepdim=True) + 0.5)
    out = fp8_lin(feats)
    readout = cuet.Linear(
        irreps_out,
        cue.Irreps("O3", "1x0e"),
        layout_in=layout,
        layout_out=cue.ir_mul,
        dtype=torch.float64,
        method="naive",
    )
    energy = readout(out).sum()

    # 1st derivative: conservative atomic forces F = -dE/dR
    (grad_pos,) = torch.autograd.grad(energy, [pos], create_graph=True)
    forces = -grad_pos
    assert torch.isfinite(forces).all()
    assert forces.abs().max().item() > 0.0

    # 2nd derivative (Force-QAT): backpropagate force-matching loss into FP8 linear weights!
    forces_target = torch.randn_like(forces)
    force_loss = torch.nn.functional.mse_loss(
        forces, forces_target
    ) + 0.01 * energy.pow(2)
    (grad_w_fp8, grad_pos_2nd) = torch.autograd.grad(force_loss, [fp8_lin.weight, pos])
    assert torch.isfinite(grad_w_fp8).all()
    assert grad_w_fp8.abs().max().item() > 1e-6
    assert torch.isfinite(grad_pos_2nd).all()


def test_equivariant_fp8_linear_unsimplified_and_external_weights():
    """Verify unsimplified irreps, external non-shared weights, and 16-byte alignment check."""
    torch.manual_seed(99)
    irreps_in = "16x0e + 16x1o + 4x0e"
    irreps_out = "16x0e + 16x1o + 2x3o"
    fp8_lin = cuet.EquivariantFP8Linear(
        irreps_in,
        irreps_out,
        layout_in=cue.mul_ir,
        layout_out=cue.ir_mul,
        shared_weights=False,
        internal_weights=False,
    )
    x = torch.randn(16, fp8_lin.irreps_in.dim, dtype=torch.float32, requires_grad=True)
    w = torch.randn(16, fp8_lin.weight_numel, dtype=torch.float32, requires_grad=True)
    y = fp8_lin(x, w)
    assert y.shape == (16, fp8_lin.irreps_out.dim)
    # Trailing 2x3o has no matching input 3o so must be identically zero
    assert y[:, -14:].abs().max().item() == 0.0
    loss = y.pow(2).sum()
    gx, gw = torch.autograd.grad(loss, [x, w])
    assert torch.isfinite(gx).all() and torch.isfinite(gw).all()


# ---------------------------------------------------------------------------
# 3. PBC Conservative Virial Stress Physics & Conservation Law Tests
# ---------------------------------------------------------------------------


def _build_triclinic_pbc_crystal(
    dtype: torch.dtype = torch.float64,
    r_cut: float = 4.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a 4-atom periodic triclinic crystal with full [-1, 0, 1]^3 neighbor shifts."""
    cell = torch.tensor(
        [
            [3.20, 0.35, 0.15],
            [0.25, 3.35, -0.20],
            [-0.10, 0.30, 3.50],
        ],
        dtype=dtype,
    )
    frac_coords = torch.tensor(
        [
            [0.10, 0.15, 0.20],
            [0.62, 0.18, 0.71],
            [0.21, 0.68, 0.64],
            [0.73, 0.72, 0.19],
        ],
        dtype=dtype,
    )
    positions = frac_coords @ cell
    species = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    num_atoms = positions.shape[0]
    src_list: list[int] = []
    dst_list: list[int] = []
    shift_list: list[list[int]] = []

    for s0 in (-1, 0, 1):
        for s1 in (-1, 0, 1):
            for s2 in (-1, 0, 1):
                shift_vec = s0 * cell[0] + s1 * cell[1] + s2 * cell[2]
                for i in range(num_atoms):
                    for j in range(num_atoms):
                        if i == j and s0 == 0 and s1 == 0 and s2 == 0:
                            continue
                        rij = positions[j] - positions[i] + shift_vec
                        dist = float(torch.linalg.vector_norm(rij).item())
                        if dist < r_cut:
                            src_list.append(i)
                            dst_list.append(j)
                            shift_list.append([s0, s1, s2])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_shifts = torch.tensor(shift_list, dtype=torch.long)
    return positions, cell, species, edge_index, edge_shifts


class _CuEquivariancePBCEnergyModel(torch.nn.Module):
    """True O(3)-invariant periodic interatomic potential using cuEquivariance ops."""

    def __init__(
        self,
        num_species: int = 2,
        num_radial: int = 6,
        r_cut: float = 4.2,
        use_fp8: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.r_cut = r_cut
        self.num_radial = num_radial
        self.node_Irreps = cue.Irreps("O3", "4x0e + 4x1o + 2x2e")
        self.sh_irreps = cue.Irreps("O3", "0e + 1o + 2e")

        self.species_emb = torch.nn.Embedding(num_species, 4, dtype=dtype)
        self.sh = cuet.SphericalHarmonics([0, 1, 2], normalize=True, method="naive")
        self.radial_mlp = torch.nn.Linear(num_radial, 4, dtype=dtype)

        # First message passing: node (4x0e) x edge_sh (0e + 1o + 2e) -> 4x0e + 4x1o + 4x2e
        self.tp1 = cuet.ChannelWiseTensorProduct(
            cue.Irreps("O3", "4x0e"),
            self.sh_irreps,
            filter_irreps_out=self.sh_irreps,
            layout=cue.ir_mul,
            dtype=dtype,
            method="naive",
        )
        # Second interaction: (4x0e + 4x1o + 4x2e) x edge_sh -> 0e + 1o + 2e
        self.tp2 = cuet.ChannelWiseTensorProduct(
            self.tp1.irreps_out,
            self.sh_irreps,
            filter_irreps_out=[cue.O3(0, 1), cue.O3(1, -1), cue.O3(2, 1)],
            layout=cue.ir_mul,
            dtype=dtype,
            method="naive",
        )
        if use_fp8:
            self.mix = cuet.EquivariantFP8Linear(
                self.tp2.irreps_out,
                self.tp2.irreps_out,
                layout=cue.ir_mul,
                dtype=dtype,
            )
        else:
            self.mix = cuet.Linear(
                self.tp2.irreps_out,
                self.tp2.irreps_out,
                layout=cue.ir_mul,
                dtype=dtype,
                method="naive",
            )
        self.readout = cuet.Linear(
            self.tp2.irreps_out,
            cue.Irreps("O3", "1x0e"),
            layout=cue.ir_mul,
            dtype=dtype,
            method="naive",
        )

        centers = torch.linspace(0.8, r_cut * 0.9, num_radial, dtype=dtype)
        self.register_buffer("rbf_centers", centers)

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        edge_vectors: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        species: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cell
        src, dst = edge_index[0], edge_index[1]
        num_atoms = positions.shape[0]

        dist = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True)
        # C^3-smooth polynomial cutoff envelope vanishing smoothly at r_cut
        u = (dist / self.r_cut).clamp(max=1.0)
        envelope = (1.0 - u * u).pow(3) * (dist < self.r_cut).to(dist.dtype)

        rbf = torch.exp(-1.5 * (dist - self.rbf_centers).pow(2)) * envelope
        rad_feat = torch.tanh(self.radial_mlp(rbf)) * envelope  # (E, 4) -> 4x0e

        y_lm = self.sh(edge_vectors)  # (E, 9) in ir_mul
        h0 = self.species_emb(species)  # (N, 4) -> 4x0e
        edge_scalar = h0[src] * rad_feat  # (E, 4)

        m1 = self.tp1(edge_scalar, y_lm)  # (E, 36)
        node_m1 = positions.new_zeros((num_atoms, m1.shape[-1])).index_add(0, dst, m1)

        m2_edge = self.tp2(node_m1[src], y_lm) * envelope
        node_m2 = positions.new_zeros((num_atoms, m2_edge.shape[-1])).index_add(
            0, dst, m2_edge
        )
        node_mixed = self.mix(node_m2)
        atom_energies = self.readout(node_mixed).squeeze(-1)

        if batch is None:
            return atom_energies.sum()
        num_graphs = int(batch.max().item()) + 1
        return atom_energies.new_zeros((num_graphs,)).index_add(0, batch, atom_energies)


def test_pbc_strain_vs_pairwise_virial_and_finite_difference_fp64():
    """Prove Strain Stress == Pairwise Virial Stress (< 1e-11) == Finite-Difference (< 1e-8) in FP64."""
    torch.manual_seed(2026)
    positions, cell, species, edge_index, edge_shifts = _build_triclinic_pbc_crystal(
        dtype=torch.float64
    )
    model = _CuEquivariancePBCEnergyModel(use_fp8=False, dtype=torch.float64)

    # 1. Strain-derivative formulation
    pos_s = positions.clone().requires_grad_(True)
    strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    s_pos, s_cell, s_edge_vecs = cuet.apply_cell_strain(
        pos_s, cell, strain, edge_index=edge_index, edge_shifts=edge_shifts
    )
    assert s_edge_vecs is not None
    e_strain = model(s_pos, s_cell, s_edge_vecs, edge_index=edge_index, species=species)
    res_strain = cuet.compute_forces_and_virial(
        e_strain, pos_s, strain=strain, cell=cell
    )

    # 2. Pairwise / edge-vector formulation (NO cell deformation in forward!)
    pos_p = positions.clone().requires_grad_(True)
    src, dst = edge_index[0], edge_index[1]
    p_edge_vecs = pos_p[dst] - pos_p[src] + edge_shifts.to(torch.float64) @ cell
    e_pair = model(pos_p, cell, p_edge_vecs, edge_index=edge_index, species=species)
    res_pair = cuet.compute_forces_and_virial(
        e_pair, pos_p, edge_vectors=p_edge_vecs, cell=cell
    )

    # Prove Strain-Derivative == Edge-Vector Pairwise Virial Stress to < 1e-11 in FP64!
    assert (e_strain - e_pair).abs().item() < 1e-14
    assert (res_strain["forces"] - res_pair["forces"]).abs().max().item() < 1e-12
    assert (res_strain["virial"] - res_pair["virial"]).abs().max().item() < 1e-11
    assert (res_strain["stress"] - res_pair["stress"]).abs().max().item() < 1e-11
    assert (res_strain["pressure"] - res_pair["pressure"]).abs().max().item() < 1e-12
    assert (
        res_strain["shear_stress"] - res_pair["shear_stress"]
    ).abs().max().item() < 1e-11

    # 3. Prove Analytical Virial & Stress == Central Finite-Difference Cell Strain Derivative (< 1e-8)!
    h = 1e-6
    omega = float(torch.abs(torch.linalg.det(cell)).item())
    virial_fd = torch.zeros((3, 3), dtype=torch.float64)
    stress_fd = torch.zeros((3, 3), dtype=torch.float64)

    with torch.no_grad():
        for a in range(3):
            for b in range(3):
                eps_plus = torch.zeros((3, 3), dtype=torch.float64)
                eps_minus = torch.zeros((3, 3), dtype=torch.float64)
                eps_plus[a, b] = h
                eps_minus[a, b] = -h

                p_hi, c_hi, ev_hi = cuet.apply_cell_strain(
                    positions,
                    cell,
                    eps_plus,
                    edge_index=edge_index,
                    edge_shifts=edge_shifts,
                )
                p_lo, c_lo, ev_lo = cuet.apply_cell_strain(
                    positions,
                    cell,
                    eps_minus,
                    edge_index=edge_index,
                    edge_shifts=edge_shifts,
                )
                assert ev_hi is not None and ev_lo is not None
                e_hi = model(p_hi, c_hi, ev_hi, edge_index=edge_index, species=species)
                e_lo = model(p_lo, c_lo, ev_lo, edge_index=edge_index, species=species)
                de_deps = (e_hi - e_lo) / (2.0 * h)
                virial_fd[a, b] = -de_deps
                stress_fd[a, b] = de_deps / omega

    assert (res_strain["virial"] - virial_fd).abs().max().item() < 1e-8
    assert (res_pair["virial"] - virial_fd).abs().max().item() < 1e-8
    assert (res_strain["stress"] - stress_fd).abs().max().item() < 1e-8
    assert (res_pair["stress"] - stress_fd).abs().max().item() < 1e-8


@pytest.mark.parametrize("invert", [False, True])
def test_pbc_angular_momentum_conservation_and_o3_stress_equivariance(invert: bool):
    """Prove 1e antisymmetric torque vanishes (< 1e-12) and stress transforms as R sigma R^T."""
    torch.manual_seed(314)
    positions, cell, species, edge_index, edge_shifts = _build_triclinic_pbc_crystal(
        dtype=torch.float64
    )
    model = _CuEquivariancePBCEnergyModel(use_fp8=False, dtype=torch.float64)
    wrapper = cuet.ConservativeVirialStress(model, mode="edge_vectors")

    res_orig = wrapper(
        positions,
        cell,
        edge_index,
        edge_shifts,
        create_graph=False,
        species=species,
    )
    virial_orig = res_orig["virial"]
    stress_orig = res_orig["stress"]

    iso_0e, torque_1e, shear_2e = cuet.decompose_stress_irreps(virial_orig)
    # Angular momentum conservation law: 1e antisymmetric torque MUST vanish identically!
    assert torque_1e.abs().max().item() < 1e-12

    # Rotate BOTH positions and triclinic lattice cell by g in O(3) (proper or improper!)
    axis = np.array([-0.4, 0.7, 0.55])
    axis = axis / np.linalg.norm(axis)
    angle = 0.93
    r_mat = torch.from_numpy(cue.O3(1, -1).rotation(axis, angle)).to(torch.float64)
    if invert:
        r_mat = -r_mat  # Improper reflection/rotoinversion det(R) = -1!

    pos_rot = positions @ r_mat.T
    cell_rot = cell @ r_mat.T

    res_rot = wrapper(
        pos_rot,
        cell_rot,
        edge_index,
        edge_shifts,
        create_graph=False,
        species=species,
    )

    # Energy & hydrostatic pressure (0e) are strictly O(3)-invariant
    assert (res_rot["energy"] - res_orig["energy"]).abs().item() < 1e-12
    assert (res_rot["pressure"] - res_orig["pressure"]).abs().item() < 1e-12

    # Atomic forces rotate as 1o vector: F' = F @ R^T
    assert (res_rot["forces"] - res_orig["forces"] @ r_mat.T).abs().max().item() < 1e-12

    # Cauchy stress and deviatoric 2e shear stress rotate as rank-2 even tensor: sigma' = R sigma R^T
    expected_stress_rot = r_mat @ stress_orig @ r_mat.T
    expected_shear_rot = r_mat @ res_orig["shear_stress"] @ r_mat.T
    assert (res_rot["stress"] - expected_stress_rot).abs().max().item() < 1e-12
    assert (res_rot["shear_stress"] - expected_shear_rot).abs().max().item() < 1e-12

    # Verify decompose_stress_irreps Wigner-D equivariance on BOTH crystal stress AND general asymmetric tensors
    d_0e = torch.from_numpy(cue.O3(0, 1).rotation(axis, angle)).to(torch.float64)
    d_1e = torch.from_numpy(cue.O3(1, 1).rotation(axis, angle)).to(torch.float64)
    d_2e = torch.from_numpy(cue.O3(2, 1).rotation(axis, angle)).to(torch.float64)
    # Note: 0e, 1e, 2e all have even parity p = +1 under inversion!

    iso_s, torque_s, shear_s = cuet.decompose_stress_irreps(stress_orig)
    iso_s_rot, torque_s_rot, shear_s_rot = cuet.decompose_stress_irreps(
        res_rot["stress"]
    )
    assert torque_s_rot.abs().max().item() < 1e-12
    assert (iso_s_rot - iso_s @ d_0e.T).abs().max().item() < 1e-12
    assert (shear_s_rot - shear_s @ d_2e.T).abs().max().item() < 1e-12

    # General asymmetric batch tensor check (non-zero 1e torque + isometric norm preservation)
    gen_t = torch.randn(5, 3, 3, dtype=torch.float64)
    g0, g1, g2 = cuet.decompose_stress_irreps(gen_t)
    assert g1.abs().max().item() > 0.1
    frob_sq = gen_t.pow(2).sum(dim=(-2, -1))
    irrep_sq = g0.pow(2).sum(dim=-1) + g1.pow(2).sum(dim=-1) + g2.pow(2).sum(dim=-1)
    torch.testing.assert_close(frob_sq, irrep_sq, atol=1e-13, rtol=1e-13)

    gen_t_rot = r_mat @ gen_t @ r_mat.T
    g0_r, g1_r, g2_r = cuet.decompose_stress_irreps(gen_t_rot)
    torch.testing.assert_close(g0_r, g0 @ d_0e.T, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(g1_r, g1 @ d_1e.T, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(g2_r, g2 @ d_2e.T, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("mode", ["strain", "edge_vectors"])
@pytest.mark.parametrize("use_fp8", [False, True])
def test_second_order_force_and_stress_qat_differentiability(mode: str, use_fp8: bool):
    """Prove 2nd-order Force+Stress QAT loss backpropagates cleanly into all model parameters."""
    torch.manual_seed(777)
    positions, cell, species, edge_index, edge_shifts = _build_triclinic_pbc_crystal(
        dtype=torch.float64
    )
    model = _CuEquivariancePBCEnergyModel(use_fp8=use_fp8, dtype=torch.float64)
    wrapper = cuet.ConservativeVirialStress(model, mode=mode)
    wrapper.train()

    forces_ref = torch.randn_like(positions) * 0.1
    stress_ref = torch.randn((3, 3), dtype=torch.float64) * 0.05
    stress_ref = 0.5 * (stress_ref + stress_ref.T)

    out = wrapper(
        positions,
        cell,
        edge_index,
        edge_shifts,
        create_graph=True,
        species=species,
    )
    loss = (out["forces"] - forces_ref).pow(2).sum() + (out["stress"] - stress_ref).pow(
        2
    ).sum()
    loss.backward()

    grad_norms = []
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} received None grad"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
        grad_norms.append(param.grad.norm().item())
    assert max(grad_norms) > 1e-6


def test_batched_pbc_virial_and_edge_masking():
    """Verify batched (B, 3, 3) cells and padded edge masking in compute_forces_and_virial."""
    torch.manual_seed(55)
    pos1, cell1, sp1, ei1, es1 = _build_triclinic_pbc_crystal(dtype=torch.float64)
    pos2 = pos1 * 1.03
    cell2 = cell1 * 1.03
    sp2 = sp1.clone()
    ei2 = ei1 + pos1.shape[0]
    es2 = es1.clone()

    positions = torch.cat([pos1, pos2], dim=0)
    cells = torch.stack([cell1, cell2], dim=0)  # (2, 3, 3)
    species = torch.cat([sp1, sp2], dim=0)
    batch = torch.tensor([0] * pos1.shape[0] + [1] * pos2.shape[0], dtype=torch.long)
    edge_index = torch.cat([ei1, ei2], dim=1)
    edge_shifts = torch.cat([es1, es2], dim=0)

    model = _CuEquivariancePBCEnergyModel(use_fp8=False, dtype=torch.float64)

    # Strain batched vs Pairwise batched
    w_strain = cuet.ConservativeVirialStress(model, mode="strain")
    w_pair = cuet.ConservativeVirialStress(model, mode="edge_vectors")
    out_s = w_strain(
        positions,
        cells,
        edge_index,
        edge_shifts,
        batch=batch,
        create_graph=False,
        species=species,
    )
    out_p = w_pair(
        positions,
        cells,
        edge_index,
        edge_shifts,
        batch=batch,
        create_graph=False,
        species=species,
    )

    assert out_s["stress"].shape == (2, 3, 3)
    assert out_s["pressure"].shape == (2,)
    assert out_s["shear_stress"].shape == (2, 3, 3)
    torch.testing.assert_close(out_s["stress"], out_p["stress"], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(out_s["forces"], out_p["forces"], atol=1e-12, rtol=1e-12)

    # Verify padded dummy edges with mask=0 do not contaminate pairwise virial
    pos_pad = pos1.clone().requires_grad_(True)
    src, dst = ei1[0], ei1[1]
    real_ev = pos_pad[dst] - pos_pad[src] + es1.to(torch.float64) @ cell1
    dummy_ev = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    padded_ev = torch.cat([real_ev, dummy_ev], dim=0)
    edge_mask = torch.cat([torch.ones(real_ev.shape[0]), torch.zeros(5)], dim=0).to(
        torch.bool
    )
    # Energy computed from padded_ev; mask=edge_mask zeroes out padded dummy edges in virial
    e_real = (padded_ev.pow(2).sum(dim=-1) * 0.5 * edge_mask.to(torch.float64)).sum()
    res_masked = cuet.compute_forces_and_virial(
        e_real, pos_pad, edge_vectors=padded_ev, cell=cell1, mask=edge_mask
    )
    pos_unpad = pos1.clone().requires_grad_(True)
    real_ev2 = pos_unpad[dst] - pos_unpad[src] + es1.to(torch.float64) @ cell1
    res_unpadded = cuet.compute_forces_and_virial(
        (real_ev2.pow(2).sum(dim=-1) * 0.5).sum(),
        pos_unpad,
        edge_vectors=real_ev2,
        cell=cell1,
    )
    torch.testing.assert_close(
        res_masked["virial"], res_unpadded["virial"], atol=1e-14, rtol=1e-14
    )
    torch.testing.assert_close(
        res_masked["forces"], res_unpadded["forces"], atol=1e-14, rtol=1e-14
    )


def test_equivariant_fp8_linear_scaled_mm_dispatch_contract(monkeypatch):
    """Verify H100 _scaled_mm 16-byte alignment & column-major B contract matches CPU FP8 path."""
    torch.manual_seed(2025)
    # 16x0e (d=1 -> M=16*1=16, K=16, N=16) and 16x1o (d=3 -> M=16*3=48, K=16, N=16): both 16-aligned!
    irreps = cue.Irreps("O3", "16x0e + 16x1o")
    lin_ref = cuet.EquivariantFP8Linear(
        irreps, irreps, layout=cue.ir_mul, use_scaled_mm=False, dtype=torch.float32
    )
    lin_hw = cuet.EquivariantFP8Linear(
        irreps, irreps, layout=cue.ir_mul, use_scaled_mm=True, dtype=torch.float32
    )
    with torch.no_grad():
        lin_hw.weight.copy_(lin_ref.weight)

    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def _mock_scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32):
        assert a.dtype == torch.float8_e4m3fn
        assert b.dtype == torch.float8_e4m3fn
        assert a.is_contiguous()  # Row-major A
        assert b.stride(0) == 1  # Column-major B required by cuBLASLt!
        assert a.shape[0] % 16 == 0 and a.shape[1] % 16 == 0 and b.shape[1] % 16 == 0
        assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
        calls.append((tuple(a.shape), tuple(b.shape)))
        return (a.to(torch.float32) * scale_a) @ (b.to(torch.float32) * scale_b)

    monkeypatch.setattr(
        lin_hw,
        "_can_dispatch_scaled_mm",
        lambda _x, m, k, n: (m % 16 == 0 and k % 16 == 0 and n % 16 == 0),
    )
    monkeypatch.setattr(torch, "_scaled_mm", _mock_scaled_mm)

    x = torch.randn(16, irreps.dim, dtype=torch.float32)
    y_ref = lin_ref(x)
    y_hw = lin_hw(x)
    assert calls == [((16, 16), (16, 16)), ((48, 16), (16, 16))]
    torch.testing.assert_close(y_hw, y_ref, atol=1e-6, rtol=1e-6)

    # Also verify training / 2nd-order STE path when _scaled_mm is active
    x_req = x.clone().requires_grad_(True)
    y_train = lin_hw(x_req)
    torch.testing.assert_close(y_train, y_ref, atol=1e-6, rtol=1e-6)
    (gx,) = torch.autograd.grad(y_train.sum(), [x_req], create_graph=True)
    (gw,) = torch.autograd.grad(gx.pow(2).sum(), [lin_hw.weight])
    assert torch.isfinite(gw).all() and gw.abs().max().item() > 0.0
