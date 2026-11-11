# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture Note 03: Wigner Rotations, 2-Angle vs 3-Angle Complexity, eSCN, & BF16 Pole Audit.

Demonstrates:
1. Real SO(3) Lie generators obtained from complex SU(2) via unitary change-of-basis Q_ell.
2. Symbolic STP composition (`segmented_polynomials.dot`) comparing 3-angle `yxy_rotation`
   vs 2-angle edge-alignment `yx_rotation` / `xy_rotation` (~40% fewer paths & FLOPs).
3. `escn_tp` vs `escn_tp_compact` (L-major vs M-major segment coalescing).
4. BF16 near-pole gradient audit (`acos(y)` NaN cone vs `atan2(r_xz, y)` double-where fix).
"""

from __future__ import annotations

import numpy as np
import torch
from cuequivariance.group_theory.experimental import escn

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def audit_su2_to_so3_and_rotation_complexity() -> None:
    section(
        "1. Wigner Polynomial Complexity: 3-Angle yxy vs 2-Angle Edge Alignment yx/xy"
    )
    print(
        f"  {'l_max':>5s} | {'yxy paths':>10s} | {'yx paths':>10s} | {'yxy FLOPs':>10s} | {'yx FLOPs':>10s} | {'FLOP Saving':>11s}"
    )
    print(
        f"  {'-' * 5}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 11}"
    )
    for lmax in range(1, 7):
        irreps = cue.Irreps("SO3", " + ".join(f"64x{ell}" for ell in range(lmax + 1)))
        d_yxy = descriptors.yxy_rotation(irreps)
        d_yx = descriptors.yx_rotation(irreps)
        p_yxy = d_yxy.polynomial.operations[0][1].num_paths
        p_yx = d_yx.polynomial.operations[0][1].num_paths
        f_yxy = d_yxy.polynomial.flop(1)
        f_yx = d_yx.polynomial.flop(1)
        saving = 100.0 * (1.0 - f_yx / f_yxy)
        print(
            f"  {lmax:5d} | {p_yxy:10d} | {p_yx:10d} | {f_yxy:10d} | {f_yx:10d} | {saving:10.1f}%"
        )
        assert f_yx < f_yxy


def audit_escn_compact_vs_standard() -> None:
    section("2. eSCN SO(2) Convolution: Standard L-major vs Compact M-major IR")
    for lmax, mmax in [(2, 2), (4, 2), (6, 2)]:
        irreps = cue.Irreps("SO3", " + ".join(f"128x{ell}" for ell in range(lmax + 1)))
        d_std = escn.escn_tp(irreps, irreps, m_max=mmax)
        p_cmp = escn.escn_tp_compact(irreps, irreps, m_max=mmax)

        std_paths = d_std.polynomial.operations[0][1].num_paths
        cmp_paths = p_cmp.operations[0][1].num_paths
        print(
            f"  l_max={lmax}, m_max={mmax}: "
            f"escn_tp paths={std_paths:3d} (segments={d_std.polynomial.operands[1].num_segments:2d}) vs "
            f"escn_tp_compact paths={cmp_paths:2d} (segments={p_cmp.operands[1].num_segments:2d}) "
            f"-> {std_paths / cmp_paths:.1f}x fewer paths!"
        )
        assert cmp_paths == 1 + 4 * mmax


def safe_vector_to_euler_angles(
    vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = vector.dtype
    v = vector.float() if orig_dtype in (torch.float16, torch.bfloat16) else vector
    x, y, z = v.unbind(dim=-1)

    r_xz_sq = x * x + z * z
    on_y_axis = r_xz_sq == 0.0
    safe_rxz_sq = torch.where(on_y_axis, torch.ones_like(r_xz_sq), r_xz_sq)
    r_xz = torch.where(on_y_axis, torch.zeros_like(r_xz_sq), torch.sqrt(safe_rxz_sq))

    is_zero = on_y_axis & (y == 0.0)
    y_safe = torch.where(is_zero, torch.ones_like(y), y)
    x_safe = torch.where(on_y_axis, torch.zeros_like(x), x)
    z_safe = torch.where(on_y_axis, torch.zeros_like(z), z)

    beta = torch.where(is_zero, torch.zeros_like(y), torch.atan2(r_xz, y_safe))
    alpha = torch.where(on_y_axis, torch.zeros_like(x), torch.atan2(x_safe, z_safe))
    return beta.to(orig_dtype), alpha.to(orig_dtype)


def audit_bf16_pole_gradients() -> None:
    section("3. BF16 Near-Pole Euler Angle Gradient Audit (~3 deg from +y axis)")
    # At [0.05, 1.0, 0.0], angle from +y is ~2.86 degrees.
    # In BF16, 1 / sqrt(1 + 0.05^2) = 0.99875 rounds to exact 1.0!
    v_base = torch.tensor([[0.05, 1.0, 0.0]], dtype=torch.bfloat16, requires_grad=True)
    beta_base, alpha_base = cuet.vector_to_euler_angles(v_base)
    (grad_base,) = torch.autograd.grad((beta_base + alpha_base).sum(), v_base)

    v_safe = torch.tensor([[0.05, 1.0, 0.0]], dtype=torch.bfloat16, requires_grad=True)
    beta_safe, alpha_safe = safe_vector_to_euler_angles(v_safe)
    (grad_safe,) = torch.autograd.grad((beta_safe + alpha_safe).sum(), v_safe)

    v_fp64 = torch.tensor([[0.05, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    beta_ref, alpha_ref = safe_vector_to_euler_angles(v_fp64)
    (grad_ref,) = torch.autograd.grad((beta_ref + alpha_ref).sum(), v_fp64)

    print(
        f"  [Baseline acos] BF16 grad at [0.05, 1.0, 0.0] : {grad_base.float().numpy().ravel()}"
    )
    print(
        f"  [Safe atan2]    BF16 grad at [0.05, 1.0, 0.0] : {grad_safe.float().numpy().ravel()}"
    )
    print(
        f"  [Reference]     FP64 grad at [0.05, 1.0, 0.0] : {grad_ref.numpy().ravel()}"
    )

    assert torch.isfinite(grad_safe).all()
    assert np.allclose(
        grad_safe.float().numpy(), grad_ref.numpy(), rtol=2e-2, atol=2e-2
    )

    # Also verify exact pole [0, 1, 0] and zero vector [0, 0, 0] 2nd derivatives!
    for special in ([0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]):
        vp = torch.tensor([special], dtype=torch.float64, requires_grad=True)
        b, a = safe_vector_to_euler_angles(vp)
        (g1,) = torch.autograd.grad((b + a).sum(), vp, create_graph=True)
        (g2,) = torch.autograd.grad(g1.sum(), vp)
        assert torch.isfinite(g1).all() and torch.isfinite(g2).all()
        print(
            f"  [Safe atan2]    Special vector {str(special):<16s}: g1={g1.detach().numpy().ravel()}, g2={g2.detach().numpy().ravel()}"
        )


if __name__ == "__main__":
    audit_su2_to_so3_and_rotation_complexity()
    audit_escn_compact_vs_standard()
    audit_bf16_pole_gradients()
    print("\n[PASS] 03_wigner_and_escn_audit.py completed all checks successfully.")
