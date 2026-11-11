# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test combining all 5 SES research enhancements:
1. Pole- & BF16-safe 2-angle `EdgeRotation` with fused `indices_in` / `indices_out` + `O3` `escn_tp`
2. Zero-safe 2nd-order `SphericalHarmonics` with Verlet `mask` isolation
3. Tapered multiplicities (`64x0e + 32x1o + 16x2e`) auto-uniformized via `uniformize_segments()`
4. Fixed-capacity Verlet neighbor list (`-1` sentinel indices + `mask`) & 3D Replica Batching
5. Orthogonal `IrrepsToCartesian` / `CartesianToIrreps` polarizability & traceless quadrupole heads
"""

from __future__ import annotations

import warnings

import cuequivariance as cue
import cuequivariance_torch as cuet
import numpy as np
import scipy.spatial.transform
import torch
from cuequivariance.group_theory.experimental import escn


class _MiniPolarizableESCN(torch.nn.Module):
    def __init__(self, layout: cue.IrrepsLayout = cue.ir_mul) -> None:
        super().__init__()
        self.node_irreps = cue.Irreps("O3", "64x0e + 32x1o + 16x2e")
        self.sh_irreps = cue.Irreps("O3", "1x0e + 1x1o + 1x2e")

        self.sh = cuet.SphericalHarmonics(
            ls=[0, 1, 2], normalize=True, method="naive", math_dtype=torch.float64
        )
        # Verify tapered irreps auto-select uniform_1d without shape-fallback warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.rot_fwd = cuet.EdgeRotation(
                self.node_irreps, inverse=False, layout=layout, math_dtype=torch.float64
            )
            self.global_tp = cuet.ChannelWiseTensorProduct(
                self.node_irreps,
                self.sh_irreps,
                ["0e", "1o", "2e"],
                layout=layout,
                shared_weights=True,
                internal_weights=True,
                math_dtype=torch.float64,
            )
            self.rot_inv = cuet.EdgeRotation(
                self.node_irreps, inverse=True, layout=layout, math_dtype=torch.float64
            )
        shape_warnings = [
            w for w in caught if "Segments are not the same shape" in str(w.message)
        ]
        assert len(shape_warnings) == 0

        # Local aligned-frame O(3) eSCN SO(2) convolution (gauge-invariant around +y!)
        escn_desc = escn.escn_tp(self.node_irreps, self.node_irreps, m_max=2)
        self.escn_weights = torch.nn.Parameter(
            torch.randn(1, escn_desc.inputs[0].dim, dtype=torch.float64) * 0.05
        )
        self.escn_poly = cuet.SegmentedPolynomial(
            escn_desc.polynomial, method="naive", math_dtype=torch.float64
        )
        self.lin_tp = cuet.Linear(
            self.global_tp.irreps_out,
            self.node_irreps,
            layout=layout,
            shared_weights=True,
            internal_weights=True,
            method="naive",
            math_dtype=torch.float64,
        )

        self.readout_pol = cuet.Linear(
            self.node_irreps,
            cue.Irreps("O3", "1x0e + 1x2e"),
            layout=layout,
            shared_weights=True,
            internal_weights=True,
            method="naive",
            math_dtype=torch.float64,
        )
        self.readout_quad = cuet.Linear(
            self.node_irreps,
            cue.Irreps("O3", "1x2e"),
            layout=layout,
            shared_weights=True,
            internal_weights=True,
            method="naive",
            math_dtype=torch.float64,
        )
        self.to_alpha = cuet.IrrepsToCartesian(
            "ij=ji", layout=layout, method="naive", math_dtype=torch.float64
        )
        self.to_quad = cuet.IrrepsToCartesian(
            "ij=ji",
            traceless=True,
            layout=layout,
            method="naive",
            math_dtype=torch.float64,
        )

    def forward(
        self,
        pos: torch.Tensor,
        node_feats: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_atoms = pos.shape[0]
        safe_src = torch.clamp_min(src, 0)
        safe_dst = torch.clamp_min(dst, 0)
        edge_vecs = pos[safe_dst] - pos[safe_src]
        if mask is not None:
            edge_vecs = torch.where(
                mask.unsqueeze(-1), edge_vecs, torch.zeros_like(edge_vecs)
            )

        # Branch 1: Fused 2-angle EdgeRotation(node->edge) + O(3) eSCN SO(2) conv + EdgeRotation(edge->node)
        x_aligned = self.rot_fwd(edge_vecs, node_feats, indices_in=src, mask=mask)
        # Weight radial modulation by smooth squared norm so positions couple into eSCN branch too
        r2 = (edge_vecs * edge_vecs).sum(dim=-1, keepdim=True)
        w_edge = (1.0 / (1.0 + r2)) * self.escn_weights
        [m_aligned] = self.escn_poly([w_edge, x_aligned], mask=mask)
        agg_escn = self.rot_inv(
            edge_vecs, m_aligned, indices_out=dst, size_out=n_atoms, mask=mask
        )

        # Branch 2: Zero-safe SphericalHarmonics + tapered ChannelWiseTensorProduct with Verlet sentinels
        y_edge = self.sh(edge_vecs, mask=mask)
        tp_node = self.global_tp(
            node_feats,
            y_edge,
            indices_1=src,
            indices_out=dst,
            size_out=n_atoms,
            mask=mask,
        )
        agg_total = agg_escn + self.lin_tp(tp_node)

        alpha = self.to_alpha(self.readout_pol(agg_total))  # (N, 3, 3)
        quad = self.to_quad(self.readout_quad(agg_total))  # (N, 3, 3) traceless
        return alpha, quad


def test_ses_polarizable_verlet_equivariance_and_force_qat() -> None:
    torch.manual_seed(7)
    model = _MiniPolarizableESCN(layout=cue.ir_mul).to(dtype=torch.float64)

    n_atoms = 6
    pos_real = torch.randn(n_atoms, 3, dtype=torch.float64)
    # Place atom 1 nearly directly above atom 0 along +y to stress near-pole Wigner alignment!
    pos_real[1] = pos_real[0] + torch.tensor([1e-4, 1.2, 0.0], dtype=torch.float64)

    node_feats_base = torch.randn(n_atoms, model.node_irreps.dim, dtype=torch.float64)
    src_real = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 2], dtype=torch.int64)
    dst_real = torch.tensor([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 5], dtype=torch.int64)

    # Pad to fixed capacity E_max=20 with -1 sentinels & mask=False
    e_pad = 8
    src_padded = torch.cat([src_real, torch.full((e_pad,), -1, dtype=torch.int64)])
    dst_padded = torch.cat([dst_real, torch.full((e_pad,), -1, dtype=torch.int64)])
    mask = torch.cat(
        [
            torch.ones(len(src_real), dtype=torch.bool),
            torch.zeros(e_pad, dtype=torch.bool),
        ]
    )

    e_field = torch.tensor([0.3, -0.5, 0.2], dtype=torch.float64)

    # --- (A) Unpadded vs Padded Forward, Forces (1st deriv), and Force-QAT (2nd deriv) ---
    pos_a = pos_real.clone().requires_grad_(True)
    alpha_a, quad_a = model(pos_a, node_feats_base, src_real, dst_real, mask=None)
    epol_a = -0.5 * torch.einsum("i,nij,j->", e_field, alpha_a, e_field)
    (forces_a,) = torch.autograd.grad(-epol_a, pos_a, create_graph=True)
    qat_grads_a = torch.autograd.grad(
        (forces_a**2).sum(),
        (model.escn_weights, next(iter(model.global_tp.parameters()))),
    )

    pos_b = pos_real.clone().requires_grad_(True)
    alpha_b, quad_b = model(pos_b, node_feats_base, src_padded, dst_padded, mask=mask)
    epol_b = -0.5 * torch.einsum("i,nij,j->", e_field, alpha_b, e_field)
    (forces_b,) = torch.autograd.grad(-epol_b, pos_b, create_graph=True)
    qat_grads_b = torch.autograd.grad(
        (forces_b**2).sum(),
        (model.escn_weights, next(iter(model.global_tp.parameters()))),
    )

    torch.testing.assert_close(alpha_a, alpha_b, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(quad_a, quad_b, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(forces_a, forces_b, atol=1e-12, rtol=1e-12)
    for ga, gb in zip(qat_grads_a, qat_grads_b):
        torch.testing.assert_close(ga, gb, atol=1e-12, rtol=1e-12)

    # Verify symmetry & zero-trace conservation on quadrupole and polarizability
    torch.testing.assert_close(
        alpha_b, alpha_b.transpose(-1, -2), atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(quad_b, quad_b.transpose(-1, -2), atol=1e-13, rtol=1e-13)
    assert (
        float(torch.diagonal(quad_b.detach(), dim1=-2, dim2=-1).sum(-1).abs().max())
        < 1e-13
    )

    # --- (B) Full O(3) Proper Rotation + Improper Roto-Inversion Equivariance Gate ---
    rng = np.random.default_rng(99)
    rotvec = scipy.spatial.transform.Rotation.random(random_state=rng).as_rotvec()
    for parity_val in (0, 1):
        parity = np.array([parity_val])
        r_mat = torch.tensor(cue.O3(1, -1).exp_map(rotvec, parity), dtype=torch.float64)
        d_node = torch.tensor(
            cue.IrrepsAndLayout(model.node_irreps, cue.ir_mul).exp_map(rotvec, parity),
            dtype=torch.float64,
        )

        pos_rot = (pos_real @ r_mat.T).detach().requires_grad_(True)
        feats_rot = node_feats_base @ d_node.T
        e_rot = e_field @ r_mat.T

        alpha_rot, quad_rot = model(
            pos_rot, feats_rot, src_padded, dst_padded, mask=mask
        )
        epol_rot = -0.5 * torch.einsum("i,nij,j->", e_rot, alpha_rot, e_rot)
        (forces_rot,) = torch.autograd.grad(-epol_rot, pos_rot)

        alpha_expected = torch.einsum("ia,nab,jb->nij", r_mat, alpha_b.detach(), r_mat)
        quad_expected = torch.einsum("ia,nab,jb->nij", r_mat, quad_b.detach(), r_mat)
        forces_expected = forces_b.detach() @ r_mat.T

        torch.testing.assert_close(alpha_rot, alpha_expected, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(quad_rot, quad_expected, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(epol_rot, epol_b.detach(), atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(forces_rot, forces_expected, atol=1e-11, rtol=1e-11)
