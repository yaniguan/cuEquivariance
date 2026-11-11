# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture Note 01: Segmented Polynomials IR, Layout Uniformity, and Symbolic AD.

Demonstrates:
1. How `EquivariantPolynomial` lowers Clebsch-Gordan selection rules into `SegmentedPolynomial`.
2. Why `cue.ir_mul` yields uniform 1D segments `(mul,)` after `flatten_coefficient_modes()`,
   and how `split_mode("u", gcd)` rescues tapered irreps (e.g. "128x0e + 64x1o + 32x2e").
3. How symbolic `jvp` and `backward` close higher-order derivatives (MD forces & Hessians)
   inside the exact same `SegmentedPolynomial` IR.
"""

from __future__ import annotations

import math

import cuequivariance as cue
import cuequivariance_torch as cuet
import numpy as np
import torch
from cuequivariance import descriptors


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def audit_channelwise_tp_and_layouts() -> None:
    section("1. ChannelWise TP Lowering & Layout Impact (ir_mul vs mul_ir)")
    irreps_1 = cue.Irreps("O3", "32x0e + 32x1o + 32x2e")
    irreps_2 = cue.Irreps("O3", "1x0e + 1x1o + 1x2e")
    irreps_out = cue.Irreps("O3", "32x0e + 32x1o + 32x2e")

    desc = descriptors.channelwise_tensor_product(irreps_1, irreps_2, irreps_out)
    poly = desc.polynomial
    stp = poly.operations[0][1]

    print("Equivariant descriptor:")
    print(desc)
    print(f"Raw STP subscripts       : {stp.subscripts}")
    print(
        f"Raw STP num_paths        : {stp.num_paths}, flop_cost(batch=1): {poly.flop(1)}"
    )

    flat_poly = poly.flatten_coefficient_modes().squeeze_modes()
    flat_stp = flat_poly.operations[0][1]
    print(f"Flattened 1D subscripts  : {flat_stp.subscripts}")
    print(f"Flattened num_paths      : {flat_stp.num_paths}")
    print(f"All same segment shape?  : {flat_poly.all_same_segment_shape()}")
    for idx, op in enumerate(flat_poly.operands):
        print(
            f"  Operand {idx}: size={op.size:4d}, num_segments={op.num_segments:3d}, "
            f"segment_shape={op.segment_shape}"
        )

    assert flat_poly.all_same_segment_shape()


def audit_heterogeneous_multiplicity_gcd_split() -> None:
    section("2. Tapered Multiplicities ('128x0e + 64x1o + 32x2e') & GCD Mode Splitting")
    tapered = cue.Irreps("O3", "128x0e + 64x1o + 32x2e")
    sh_irreps = cue.Irreps("O3", "1x0e + 1x1o + 1x2e")
    desc = descriptors.channelwise_tensor_product(tapered, sh_irreps, tapered)

    raw_poly = desc.polynomial.flatten_coefficient_modes().squeeze_modes()
    extents = sorted(
        {s[0] for op in raw_poly.operands if op.ndim == 1 for s in op.segments}
    )
    gcd_extent = math.gcd(*extents)
    split_poly = raw_poly.split_mode("u", gcd_extent)

    print(f"Tapered irreps           : {tapered}")
    print(
        f"1D segment extents       : {extents} ->all_same_shape={raw_poly.all_same_segment_shape()}"
    )
    print(
        f"GCD split ('u', {gcd_extent})     : all_same_shape={split_poly.all_same_segment_shape()}"
    )
    print(
        f"Path count before/after  : {raw_poly.operations[0][1].num_paths} -> "
        f"{split_poly.operations[0][1].num_paths} (uniform segment={split_poly.operands[1].segment_shape})"
    )

    # Verify zero-transpose flat-buffer equivalence in FP64
    rng = np.random.default_rng(42)
    inputs = [rng.standard_normal((5, op.size)) for op in raw_poly.inputs]
    [out_raw] = raw_poly(*inputs)
    [out_split] = split_poly(*inputs)
    max_diff = float(np.max(np.abs(out_raw - out_split)))
    print(
        f"Max |out_raw - out_split|: {max_diff:.3e} (zero memory transpose required!)"
    )
    assert split_poly.all_same_segment_shape()
    assert max_diff < 1e-13


def audit_symbolic_ad_closure() -> None:
    section(
        "3. Symbolic AD Closure: Forward, 1st Backward (Forces), 2nd Backward (Force-QAT)"
    )
    # Degree-3 symmetric polynomial (e.g. MACE 3-body symmetric contraction)
    desc = descriptors.symmetric_contraction(
        cue.Irreps("O3", "16x0e + 16x1o"),
        cue.Irreps("O3", "16x0e + 16x1o"),
        [0, 1, 2, 3],
    )
    poly = desc.polynomial
    bwd1_poly, _ = poly.backward([True, True], [True])
    # Differentiate the backward polynomial AGAIN -> closed-form 2nd-order polynomial!
    bwd2_poly, _ = bwd1_poly.backward(
        [True] * bwd1_poly.num_inputs, [True] * bwd1_poly.num_outputs
    )

    print(
        f"Forward polynomial       : inputs={poly.num_inputs}, outputs={poly.num_outputs}, ops={len(poly.operations)}"
    )
    print(
        f"1st Backward polynomial  : inputs={bwd1_poly.num_inputs}, outputs={bwd1_poly.num_outputs}, ops={len(bwd1_poly.operations)}"
    )
    print(
        f"2nd Backward polynomial  : inputs={bwd2_poly.num_inputs}, outputs={bwd2_poly.num_outputs}, ops={len(bwd2_poly.operations)}"
    )

    # Compare symbolic backward IR against PyTorch autograd to machine precision
    torch.manual_seed(0)
    sp_fwd = cuet.SegmentedPolynomial(poly, method="naive", math_dtype=torch.float64)
    sp_bwd1 = cuet.SegmentedPolynomial(
        bwd1_poly, method="naive", math_dtype=torch.float64
    )

    w = torch.randn(8, poly.inputs[0].size, dtype=torch.float64, requires_grad=True)
    x = torch.randn(8, poly.inputs[1].size, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn(8, poly.outputs[0].size, dtype=torch.float64)

    [y] = sp_fwd([w, x])
    gw_auto, gx_auto = torch.autograd.grad(y, (w, x), grad_outputs=cotangent)
    gw_sym, gx_sym = sp_bwd1([w.detach(), x.detach(), cotangent])

    err_w = float((gw_auto - gw_sym).abs().max())
    err_x = float((gx_auto - gx_sym).abs().max())
    print(f"Symbolic vs Autograd err : grad_w={err_w:.3e}, grad_x={err_x:.3e}")
    assert err_w < 1e-12 and err_x < 1e-12


if __name__ == "__main__":
    audit_channelwise_tp_and_layouts()
    audit_heterogeneous_multiplicity_gcd_split()
    audit_symbolic_ad_closure()
    print("\n[PASS] 01_segmented_polynomials_ir.py completed all checks successfully.")
