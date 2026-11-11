# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture Note 02: Spherical Harmonics Recursion, O(3) Equivariance, and Zero-Vector AD Audit.

Demonstrates:
1. Exact algebraic SymPy + CG tree recursion (`sympy_spherical_harmonics`) and symmetric STP path counts.
2. Full O(3) equivariance verification (proper rotations + improper spatial inversion):
   Y_l(g * r) == D^{(l, (-1)^l)}(g) @ Y_l(r).
3. Boundary-condition audit at r = [0, 0, 0] (padded Verlet neighbor slots):
   comparing standard `F.normalize` (1st-grad BF16 overflow & 2nd-grad NaN) against
   the double-`torch.where` safe normalization pattern.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.spatial.transform
import torch
from cuequivariance.group_theory.descriptors.spherical_harmonics_ import (
    sympy_spherical_harmonics,
)

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors

logging.getLogger(
    "cuequivariance_torch.primitives.segmented_polynomial_naive"
).setLevel(logging.ERROR)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def audit_sympy_recursion_and_paths() -> None:
    section("1. Algebraic CG Recursion & Degree-ell Path Scaling")
    ls = list(range(0, 7))
    desc = descriptors.spherical_harmonics(cue.O3(1, -1), ls)
    for ell, (_, stp) in zip(ls, desc.polynomial.operations):
        ir, expr = sympy_spherical_harmonics(cue.O3(1, -1), ell)
        sample_term = str(expr[0]) if ell <= 2 else f"{str(expr[0])[:48]}..."
        print(
            f"  ell={ell} ({str(ir):>2s}, dim={ir.dim:2d}): "
            f"STP degree={ell}, symmetric paths={stp.num_paths:3d} | Y_{ell}[0] = {sample_term}"
        )


def audit_o3_equivariance() -> None:
    section("2. O(3) Proper Rotation + Inversion Equivariance Gate")
    ls = [0, 1, 2, 3, 4]
    vec_ir = cue.O3(1, -1)
    desc = descriptors.spherical_harmonics(vec_ir, ls)
    sh_mod = cuet.SphericalHarmonics(
        ls=ls, normalize=True, method="naive", math_dtype=torch.float64
    )

    rng = np.random.default_rng(123)
    vecs = rng.standard_normal((16, 3))
    rot = scipy.spatial.transform.Rotation.random(random_state=rng)
    rotvec = rot.as_rotvec()

    # Test both proper (det=+1) and improper (det=-1, spatial inversion) O(3) elements
    for parity_power, label in [
        (0, "Proper SO(3) rotation"),
        (1, "Improper O(3) roto-inversion"),
    ]:
        d_vec = vec_ir.exp_map(rotvec, np.array([parity_power]))
        d_out = desc.outputs[0].exp_map(rotvec, np.array([parity_power]))

        vecs_transformed = vecs @ d_vec.T
        y_lhs = sh_mod(torch.tensor(vecs_transformed, dtype=torch.float64)).numpy()
        y_rhs = sh_mod(torch.tensor(vecs, dtype=torch.float64)).numpy() @ d_out.T
        max_err = float(np.max(np.abs(y_lhs - y_rhs)))
        print(f"  {label:<30s}: max |Y(g*r) - D(g)Y(r)| = {max_err:.3e}")
        assert max_err < 1e-13


def safe_normalize_double_where(vectors: torch.Tensor) -> torch.Tensor:
    sq_norm = torch.sum(vectors * vectors, dim=-1, keepdim=True)
    is_zero = sq_norm == 0.0
    safe_sq = torch.where(is_zero, torch.ones_like(sq_norm), sq_norm)
    return torch.where(
        is_zero, torch.zeros_like(vectors), vectors * torch.rsqrt(safe_sq)
    )


def audit_zero_vector_derivatives() -> None:
    section("3. Zero-Vector (Verlet Padding r=[0,0,0]) 1st & 2nd Derivative Audit")
    raw_poly_mod = cuet.SphericalHarmonics(
        ls=[0, 1, 2], normalize=False, method="naive", math_dtype=torch.float64
    )

    # (A) Standard F.normalize behavior (used in baseline cuet.SphericalHarmonics)
    v_std_fp64 = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    y_std = raw_poly_mod(torch.nn.functional.normalize(v_std_fp64, dim=-1))
    (g1_std,) = torch.autograd.grad(y_std.sum(), v_std_fp64, create_graph=True)
    (g2_std,) = torch.autograd.grad(g1_std.sum(), v_std_fp64)

    v_std_bf16 = torch.zeros(1, 3, dtype=torch.bfloat16, requires_grad=True)
    y_std_bf16 = cuet.SphericalHarmonics(ls=[0, 1, 2], normalize=True, method="naive")(
        v_std_bf16
    )
    (g1_std_bf16,) = torch.autograd.grad(y_std_bf16.sum(), v_std_bf16)

    print(
        f"  [Baseline F.normalize] FP64 1st grad at r=0 : {g1_std.detach().numpy().ravel()} (eps^-1 spike!)"
    )
    print(
        f"  [Baseline F.normalize] BF16 1st grad at r=0 : {g1_std_bf16.float().numpy().ravel()}"
    )
    print(
        f"  [Baseline F.normalize] FP64 2nd grad at r=0 : {g2_std.detach().numpy().ravel()} (NaN poisons Force-QAT!)"
    )

    # (B) Double-where zero-safe normalization
    for dtype in (torch.float64, torch.bfloat16):
        v_safe = torch.zeros(1, 3, dtype=dtype, requires_grad=True)
        mod = cuet.SphericalHarmonics(ls=[0, 1, 2], normalize=False, method="naive")
        y_safe = mod(safe_normalize_double_where(v_safe))
        (g1_safe,) = torch.autograd.grad(y_safe.sum(), v_safe, create_graph=True)
        (g2_safe,) = torch.autograd.grad(g1_safe.sum(), v_safe)
        print(
            f"  [Double-where safe]    {str(dtype):<14s}: "
            f"Y(0)[0]={float(y_safe[0, 0].detach()):.1f}, "
            f"||Y_{{l>0}}(0)||={float(y_safe[0, 1:].detach().abs().max()):.1f}, "
            f"max|g1|={float(g1_safe.detach().float().abs().max()):.1f}, "
            f"max|g2|={float(g2_safe.detach().float().abs().max()):.1f}"
        )
        assert torch.isfinite(g1_safe).all() and torch.isfinite(g2_safe).all()
        assert float(g1_safe.detach().float().abs().max()) == 0.0
        assert float(g2_safe.detach().float().abs().max()) == 0.0


if __name__ == "__main__":
    audit_sympy_recursion_and_paths()
    audit_o3_equivariance()
    audit_zero_vector_derivatives()
    print("\n[PASS] 02_spherical_harmonics_audit.py completed all checks successfully.")
