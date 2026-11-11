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
import numpy as np
import pytest

import cuequivariance as cue


def _make_tapered_symmetric_contraction(irreps_str: str) -> cue.EquivariantPolynomial:
    """Build a multi-degree tapered symmetric contraction by stacking uniform channel slices."""
    irreps = cue.Irreps("O3", irreps_str)
    # Group irreps by multiplicity tiers so each tier is a valid uniform symmetric contraction
    # For e.g. 128x0e + 64x1o + 32x2e:
    #   32 channels of (0e + 1o + 2e), 32 channels of (0e + 1o), 64 channels of (0e)
    unique_muls = sorted(set(irreps.muls))
    polys = []
    prev_mul = 0
    for m in unique_muls:
        delta = m - prev_mul
        tier_irs = cue.Irreps("O3", [(delta, ir) for mul, ir in irreps if mul >= m])
        polys.append(
            cue.descriptors.symmetric_contraction(tier_irs, tier_irs, (1, 2, 3))
        )
        prev_mul = m
    return cue.EquivariantPolynomial.stack(polys, [True, True, True])


@pytest.mark.parametrize(
    "irreps_str",
    [
        "128x0e + 64x1o + 32x2e",
        "256x0e + 128x1o + 64x2e + 32x3o",
    ],
)
def test_uniformize_segments_equivariant_descriptors(irreps_str: str):
    """Test uniformize_segments on tapered descriptors from MACE/NequIP/UMA architectures."""
    rng = np.random.default_rng(1234)
    ir1 = cue.Irreps("O3", irreps_str)
    ir_sh = cue.Irreps("O3", "0e + 1o + 2e")

    descriptors_to_test = [
        ("cw_tp_lhs", cue.descriptors.channelwise_tensor_product(ir1, ir_sh, ir1)),
        ("cw_tp_rhs", cue.descriptors.channelwise_tensor_product(ir_sh, ir1, ir1)),
        ("elem_tp", cue.descriptors.elementwise_tensor_product(ir1, ir1)),
        ("yxy_rot", cue.descriptors.yxy_rotation(ir1)),
        ("yx_rot", cue.descriptors.yx_rotation(ir1)),
        (
            "fixed_rot",
            cue.descriptors.fixed_axis_angle_rotation(
                ir1, np.array([0.3, -0.7, 0.5]), 0.9
            ),
        ),
        ("inversion", cue.descriptors.inversion(ir1)),
        ("sym_contraction", _make_tapered_symmetric_contraction(irreps_str)),
    ]

    for name, e in descriptors_to_test:
        assert not e.all_same_segment_shape(), f"{name} should start non-uniform"
        eu = e.uniformize_segments()

        assert eu.all_same_segment_shape(), f"{name} should become uniform"
        assert eu.inputs == e.inputs
        assert eu.outputs == e.outputs
        assert [o.size for o in eu.polynomial.operands] == [
            o.size for o in e.polynomial.operands
        ]

        for o in eu.polynomial.operands:
            assert o.ndim in (0, 1)
            if o.ndim == 1:
                assert o.segment_shape == (32,), f"{name} expected (32,) segments"

        # Idempotency: calling uniformize_segments again returns the exact same instance
        assert eu.uniformize_segments() is eu

        # Bit-level / 1e-14 numerical equivalence in FP64 on raw flat buffers without transpose
        inputs = [
            rng.standard_normal((6, op.size), dtype=np.float64)
            for op in e.polynomial.inputs
        ]
        out_ref = e(*inputs)
        out_uni = eu(*inputs)
        for y_ref, y_uni in zip(out_ref, out_uni):
            np.testing.assert_allclose(y_uni, y_ref, atol=1e-14, rtol=1e-14)


def test_uniformize_segments_multi_operation_and_mismatched_operands():
    """Test multi-STP polynomials with distinct mode symbols and mismatched per-operand extents."""
    rng = np.random.default_rng(7)

    # Case 1: Two STPs using different mode symbols ('u' vs 'v') writing to a shared output buffer
    stp0 = cue.SegmentedTensorProduct.from_subscripts("u,,u")
    stp0.add_segment(0, (64,))
    stp0.add_segment(0, (32,))
    stp0.add_segment(1, ())
    stp0.add_segment(2, (64,))
    stp0.add_segment(2, (32,))
    stp0.add_path(0, 0, 0, c=1.25)
    stp0.add_path(1, 0, 1, c=-0.75)

    stp1 = cue.SegmentedTensorProduct.from_subscripts("v,v")
    stp1.add_segment(0, (64,))
    stp1.add_segment(0, (32,))
    stp1.add_segment(1, (64,))
    stp1.add_segment(1, (32,))
    stp1.add_path(0, 0, c=0.5)
    stp1.add_path(1, 1, c=2.0)

    poly = cue.SegmentedPolynomial(
        [stp0.operands[0], stp0.operands[1], stp1.operands[0]],
        [stp0.operands[2]],
        [(cue.Operation((0, 1, 3)), stp0), (cue.Operation((2, 3)), stp1)],
    )
    assert not poly.all_same_segment_shape()
    poly_u = poly.uniformize_segments()
    assert poly_u.all_same_segment_shape()
    for o in poly_u.operands:
        if o.ndim == 1:
            assert o.segment_shape == (32,)

    inps = [rng.standard_normal((4, o.size)) for o in poly.inputs]
    np.testing.assert_allclose(poly(*inps)[0], poly_u(*inps)[0], atol=1e-14, rtol=1e-14)

    # Case 2: Each operand is individually uniform ((64,) vs (32,)), so poly.all_same_segment_shape()
    # is True, yet uniform_1d requires a single shared extent across all 1D operands!
    stp_a = cue.SegmentedTensorProduct.from_subscripts("u,")
    stp_a.add_segment(0, (64,))
    stp_a.add_segment(1, ())
    stp_a.add_path(0, 0, c=1.5)

    stp_b = cue.SegmentedTensorProduct.from_subscripts("v,v")
    stp_b.add_segment(0, (32,))
    stp_b.add_segment(0, (32,))
    stp_b.add_segment(1, (32,))
    stp_b.add_segment(1, (32,))
    stp_b.add_path(0, 0, c=0.8)
    stp_b.add_path(1, 1, c=-1.1)

    poly_mismatch = cue.SegmentedPolynomial(
        [stp_a.operands[0], stp_b.operands[0]],
        [stp_a.operands[1], stp_b.operands[1]],
        [(cue.Operation((0, 2)), stp_a), (cue.Operation((1, 3)), stp_b)],
    )
    assert poly_mismatch.all_same_segment_shape() is True
    poly_mismatch_u = poly_mismatch.uniformize_segments()
    assert poly_mismatch_u.all_same_segment_shape() is True
    for o in poly_mismatch_u.operands:
        if o.ndim == 1:
            assert o.segment_shape == (32,)

    inps2 = [rng.standard_normal((4, o.size)) for o in poly_mismatch.inputs]
    for y_ref, y_uni in zip(poly_mismatch(*inps2), poly_mismatch_u(*inps2)):
        np.testing.assert_allclose(y_uni, y_ref, atol=1e-14, rtol=1e-14)

    # Case 3: Unused 1D output operand via compute_only
    poly_co = poly_mismatch.compute_only([True, False])
    poly_co_u = poly_co.uniformize_segments()
    assert poly_co_u.all_same_segment_shape()
    for o in poly_co_u.operands:
        if o.ndim == 1:
            assert o.segment_shape == (32,)
    for y_ref, y_uni in zip(poly_co(*inps2), poly_co_u(*inps2)):
        np.testing.assert_allclose(y_uni, y_ref, atol=1e-14, rtol=1e-14)


def test_uniformize_segments_cost_model_guardrails():
    """Verify min_extent, max_path_factor, and structural guardrails."""
    # 1. Prime/tiny GCD: "64x0e + 3x1o" has gcd(64, 3) = 1 < min_extent=16
    e_tiny_gcd = cue.descriptors.inversion(cue.Irreps("O3", "64x0e + 3x1o"))
    assert e_tiny_gcd.uniformize_segments() is e_tiny_gcd
    assert e_tiny_gcd.uniformize_segments().all_same_segment_shape() is False

    e_cw_tiny = cue.descriptors.channelwise_tensor_product(
        cue.Irreps("O3", "64x0e + 3x1o"),
        cue.Irreps("O3", "0e + 1o"),
    )
    assert e_cw_tiny.uniformize_segments() is e_cw_tiny
    assert e_cw_tiny.uniformize_segments().all_same_segment_shape() is False

    # 2. GCD = 2 ([64, 128, 2]): rejected at default min_extent=16, accepted at min_extent=2
    e_gcd2 = cue.descriptors.inversion(cue.Irreps("O3", "64x0e + 128x1o + 2x2e"))
    assert e_gcd2.uniformize_segments() is e_gcd2
    e_gcd2_forced = e_gcd2.uniformize_segments(min_extent=2, max_path_factor=64.0)
    assert e_gcd2_forced.all_same_segment_shape() is True
    assert e_gcd2_forced.polynomial.operands[0].segment_shape == (2,)

    # 3. max_path_factor guardrail: extents [512, 16] has gcd=16, splitting 512->32x paths
    stp = cue.SegmentedTensorProduct.from_subscripts("u,u")
    stp.add_segment(0, (512,))
    stp.add_segment(0, (16,))
    stp.add_segment(1, (512,))
    stp.add_segment(1, (16,))
    stp.add_path(0, 0, c=1.0)
    stp.add_path(1, 1, c=1.0)
    poly_blowup = cue.SegmentedPolynomial.eval_last_operand(stp)
    # 2 paths -> 32 + 1 = 33 paths (factor = 16.5x > 4.0)
    assert poly_blowup.uniformize_segments(max_path_factor=4.0) is poly_blowup
    assert poly_blowup.uniformize_segments(max_path_factor=8.0) is poly_blowup
    poly_blowup_ok = poly_blowup.uniformize_segments(max_path_factor=20.0)
    assert poly_blowup_ok.all_same_segment_shape() is True
    assert poly_blowup_ok.operands[0].segment_shape == (16,)

    # 4. Already uniform small extent (e.g. mul=3 < min_extent=16) needs NO splitting -> succeeds!
    e_small_uniform = cue.descriptors.channelwise_tensor_product(
        cue.Irreps("O3", "3x0e + 3x1o"),
        cue.Irreps("O3", "0e + 1o"),
    )
    eu_small = e_small_uniform.uniformize_segments()
    assert eu_small.all_same_segment_shape() is True
    assert eu_small.polynomial.operands[0].segment_shape == (3,)

    # 5. Non-1D polynomials (FCTP, Linear, multi-mode CWTP) safely return self unchanged
    e_fctp = cue.descriptors.fully_connected_tensor_product(
        cue.Irreps("O3", "32x0e + 16x1o"),
        cue.Irreps("O3", "32x0e + 16x1o"),
        cue.Irreps("O3", "32x0e + 16x1o"),
    )
    assert e_fctp.uniformize_segments() is e_fctp

    e_lin = cue.descriptors.linear(
        cue.Irreps("O3", "32x0e + 16x1o"),
        cue.Irreps("O3", "32x0e + 16x1o"),
    )
    assert e_lin.uniformize_segments() is e_lin

    # 6. Invalid guardrail arguments raise ValueError
    with pytest.raises(ValueError, match="min_extent"):
        poly_blowup.uniformize_segments(min_extent=0)
    with pytest.raises(ValueError, match="max_path_factor"):
        poly_blowup.uniformize_segments(max_path_factor=0.5)


@pytest.mark.parametrize(
    "kind",
    ["channelwise_tp", "rotation", "inversion", "symmetric_contraction"],
)
def test_uniformize_segments_symbolic_differentiation(kind: str):
    """Verify jvp, backward, and transpose on uniformized polynomials match original to 1e-12."""
    rng = np.random.default_rng(99)
    ir = cue.Irreps("O3", "128x0e + 64x1o + 32x2e")

    if kind == "channelwise_tp":
        e = cue.descriptors.channelwise_tensor_product(
            ir, cue.Irreps("O3", "0e + 1o + 2e"), ir
        )
        transpose_undef = [False, True, False]
    elif kind == "rotation":
        e = cue.descriptors.yxy_rotation(ir)
        transpose_undef = [False, False, False, True]
    elif kind == "inversion":
        e = cue.descriptors.inversion(ir)
        transpose_undef = [True]
    elif kind == "symmetric_contraction":
        e = _make_tapered_symmetric_contraction("128x0e + 64x1o + 32x2e")
        # Linear in weights (input 0) when input 1 is fixed!
        transpose_undef = [True, False]
    else:
        raise ValueError(kind)

    eu = e.uniformize_segments()
    assert eu.all_same_segment_shape() is True

    primals = [
        rng.standard_normal((5, op.size), dtype=np.float64)
        for op in e.polynomial.inputs
    ]
    tangents = [
        rng.standard_normal((5, op.size), dtype=np.float64)
        for op in e.polynomial.inputs
    ]
    cotangents = [
        rng.standard_normal((5, op.size), dtype=np.float64)
        for op in e.polynomial.outputs
    ]

    # 1. JVP
    has_tangent = [True] * e.num_inputs
    jvp_ref, _ = e.jvp(has_tangent)
    jvp_uni, _ = eu.jvp(has_tangent)
    assert jvp_uni.all_same_segment_shape() is True
    for y_ref, y_uni in zip(jvp_ref(*primals, *tangents), jvp_uni(*primals, *tangents)):
        np.testing.assert_allclose(y_uni, y_ref, atol=1e-12, rtol=1e-12)

    # 2. Backward (1st order)
    req_grad = [True] * e.num_inputs
    has_cot = [True] * e.num_outputs
    bwd_ref, _ = e.backward(req_grad, has_cot)
    bwd_uni, _ = eu.backward(req_grad, has_cot)
    assert bwd_uni.all_same_segment_shape() is True
    for g_ref, g_uni in zip(
        bwd_ref(*primals, *cotangents), bwd_uni(*primals, *cotangents)
    ):
        np.testing.assert_allclose(g_uni, g_ref, atol=1e-12, rtol=1e-12)

    # 3. Second-order symbolic backward (backward of backward)
    req_grad2 = [True] * bwd_ref.num_inputs
    has_cot2 = [True] * bwd_ref.num_outputs
    bwd2_ref, _ = bwd_ref.backward(req_grad2, has_cot2)
    bwd2_uni, _ = bwd_uni.backward(req_grad2, has_cot2)
    assert bwd2_uni.all_same_segment_shape() is True
    bwd2_primals = primals + cotangents
    bwd2_cotangents = [
        rng.standard_normal((5, op.size), dtype=np.float64)
        for op in bwd_ref.polynomial.outputs
    ]
    for g2_ref, g2_uni in zip(
        bwd2_ref(*bwd2_primals, *bwd2_cotangents),
        bwd2_uni(*bwd2_primals, *bwd2_cotangents),
    ):
        np.testing.assert_allclose(g2_uni, g2_ref, atol=1e-12, rtol=1e-12)

    # 4. Transpose wrt linear input
    tr_ref, _ = e.transpose(transpose_undef, has_cot)
    tr_uni, _ = eu.transpose(transpose_undef, has_cot)
    assert tr_uni.all_same_segment_shape() is True
    known_primals = [p for undef, p in zip(transpose_undef, primals) if not undef]
    for t_ref, t_uni in zip(
        tr_ref(*known_primals, *cotangents),
        tr_uni(*known_primals, *cotangents),
    ):
        np.testing.assert_allclose(t_uni, t_ref, atol=1e-12, rtol=1e-12)
