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
import warnings

import numpy as np
import pytest
import torch
from cuequivariance_torch.primitives import segmented_polynomial_uniform_1d as u1d_mod

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TAPERED_IRREPS = [
    "128x0e + 64x1o + 32x2e",
    "256x0e + 128x1o + 64x2e + 32x3o",
]


def _assert_no_segment_shape_warning(caught_warnings: list[warnings.WarningMessage]):
    bad = [
        str(w.message)
        for w in caught_warnings
        if "Segments are not the same shape" in str(w.message)
    ]
    assert not bad, f"Unexpected segment shape warning(s): {bad}"


def _check_fwd_bwd_double_bwd(
    fn_uni,
    fn_ref,
    tensors_uni: list[torch.Tensor],
    tensors_ref: list[torch.Tensor],
    cotangent: torch.Tensor,
    second_cotangents: list[torch.Tensor],
    atol: float = 1e-12,
    rtol: float = 1e-12,
):
    cot_uni = cotangent.detach().clone().requires_grad_(True)
    cot_ref = cotangent.detach().clone().requires_grad_(True)

    out_uni = fn_uni(*tensors_uni)
    out_ref = fn_ref(*tensors_ref)
    torch.testing.assert_close(out_uni, out_ref, atol=atol, rtol=rtol)

    grads_uni = torch.autograd.grad(
        (out_uni * cot_uni).sum(), tensors_uni, create_graph=True
    )
    grads_ref = torch.autograd.grad(
        (out_ref * cot_ref).sum(), tensors_ref, create_graph=True
    )
    for gu, gr in zip(grads_uni, grads_ref):
        torch.testing.assert_close(gu, gr, atol=atol, rtol=rtol)

    loss2_uni = sum((g * v).sum() for g, v in zip(grads_uni, second_cotangents))
    loss2_ref = sum((g * v).sum() for g, v in zip(grads_ref, second_cotangents))
    ggrads_uni = torch.autograd.grad(
        loss2_uni, [*tensors_uni, cot_uni], allow_unused=True
    )
    ggrads_ref = torch.autograd.grad(
        loss2_ref, [*tensors_ref, cot_ref], allow_unused=True
    )
    for ggu, ggr in zip(ggrads_uni, ggrads_ref):
        assert (ggu is None) == (ggr is None)
        if ggu is not None:
            torch.testing.assert_close(ggu, ggr, atol=atol, rtol=rtol)


@pytest.mark.parametrize("irreps_str", TAPERED_IRREPS)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_channelwise_tp_auto_uniform_segments(
    irreps_str: str, layout: cue.IrrepsLayout
):
    torch.manual_seed(0)
    ir1 = cue.Irreps("O3", irreps_str)
    ir2 = cue.Irreps("O3", "0e + 1o + 2e")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m_auto = cuet.ChannelWiseTensorProduct(
            ir1,
            ir2,
            ir1,
            layout=layout,
            shared_weights=True,
            internal_weights=False,
            device=device,
            dtype=torch.float64,
            math_dtype=torch.float64,
        )
    _assert_no_segment_shape_warning(caught)
    assert m_auto.method == "uniform_1d"

    # Explicit method="uniform_1d" succeeds without raising ValueError
    m_explicit = cuet.ChannelWiseTensorProduct(
        ir1,
        ir2,
        ir1,
        layout=layout,
        shared_weights=True,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        math_dtype=torch.float64,
        method="uniform_1d",
    )
    assert m_explicit.method == "uniform_1d"

    # Non-uniformized reference (auto_uniform_segments=False, method="naive")
    m_ref = cuet.ChannelWiseTensorProduct(
        ir1,
        ir2,
        ir1,
        layout=layout,
        shared_weights=True,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        math_dtype=torch.float64,
        method="naive",
        auto_uniform_segments=False,
    )
    assert m_ref.method == "naive"

    batch = 4
    x1 = torch.randn(batch, ir1.dim, device=device, dtype=torch.float64)
    x2 = torch.randn(batch, ir2.dim, device=device, dtype=torch.float64)
    w = torch.randn(1, m_auto.weight_numel, device=device, dtype=torch.float64)

    t_uni = [t.clone().requires_grad_(True) for t in (x1, x2, w)]
    t_ref = [t.clone().requires_grad_(True) for t in (x1, x2, w)]
    cot = torch.randn(batch, m_auto.irreps_out.dim, device=device, dtype=torch.float64)
    cot2 = [torch.randn_like(t) for t in t_uni]

    _check_fwd_bwd_double_bwd(
        lambda a, b, wt: m_auto(a, b, wt),
        lambda a, b, wt: m_ref(a, b, wt),
        t_uni,
        t_ref,
        cot,
        cot2,
    )

    # Also verify indexed gather/scatter message passing (unshared per-edge weights)
    m_edge_auto = cuet.ChannelWiseTensorProduct(
        ir1,
        ir2,
        ir1,
        layout=layout,
        shared_weights=False,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        math_dtype=torch.float64,
    )
    assert m_edge_auto.method == "uniform_1d"
    m_edge_ref = cuet.ChannelWiseTensorProduct(
        ir1,
        ir2,
        ir1,
        layout=layout,
        shared_weights=False,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        math_dtype=torch.float64,
        method="naive",
        auto_uniform_segments=False,
    )
    num_nodes, num_edges = 3, 5
    idx1 = torch.tensor([0, 1, 2, 0, 2], device=device, dtype=torch.int64)
    idx_out = torch.tensor([1, 2, 0, 2, 1], device=device, dtype=torch.int64)
    x1_nodes = torch.randn(num_nodes, ir1.dim, device=device, dtype=torch.float64)
    x2_edges = torch.randn(num_edges, ir2.dim, device=device, dtype=torch.float64)
    w_edges = torch.randn(
        num_edges, m_edge_auto.weight_numel, device=device, dtype=torch.float64
    )
    t_edge_uni = [t.clone().requires_grad_(True) for t in (x1_nodes, x2_edges, w_edges)]
    t_edge_ref = [t.clone().requires_grad_(True) for t in (x1_nodes, x2_edges, w_edges)]
    cot_nodes = torch.randn(
        num_nodes, m_edge_auto.irreps_out.dim, device=device, dtype=torch.float64
    )
    cot2_edge = [torch.randn_like(t) for t in t_edge_uni]

    _check_fwd_bwd_double_bwd(
        lambda a, b, wt: m_edge_auto(
            a, b, wt, indices_1=idx1, indices_out=idx_out, size_out=num_nodes
        ),
        lambda a, b, wt: m_edge_ref(
            a, b, wt, indices_1=idx1, indices_out=idx_out, size_out=num_nodes
        ),
        t_edge_uni,
        t_edge_ref,
        cot_nodes,
        cot2_edge,
    )


@pytest.mark.parametrize("irreps_str", TAPERED_IRREPS)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_rotation_auto_uniform_segments(irreps_str: str, layout: cue.IrrepsLayout):
    torch.manual_seed(1)
    ir = cue.Irreps("O3", irreps_str)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m_auto = cuet.Rotation(
            ir,
            layout=layout,
            device=device,
            math_dtype=torch.float64,
        )
    _assert_no_segment_shape_warning(caught)
    assert m_auto.method == "uniform_1d"

    m_explicit = cuet.Rotation(
        ir,
        layout=layout,
        device=device,
        math_dtype=torch.float64,
        method="uniform_1d",
    )
    assert m_explicit.method == "uniform_1d"

    m_ref = cuet.Rotation(
        ir,
        layout=layout,
        device=device,
        math_dtype=torch.float64,
        method="naive",
        auto_uniform_segments=False,
    )

    batch = 4
    gamma = torch.randn(batch, device=device, dtype=torch.float64)
    beta = torch.randn(batch, device=device, dtype=torch.float64)
    alpha = torch.randn(batch, device=device, dtype=torch.float64)
    x = torch.randn(batch, ir.dim, device=device, dtype=torch.float64)

    t_uni = [t.clone().requires_grad_(True) for t in (gamma, beta, alpha, x)]
    t_ref = [t.clone().requires_grad_(True) for t in (gamma, beta, alpha, x)]
    cot = torch.randn(batch, ir.dim, device=device, dtype=torch.float64)
    cot2 = [torch.randn_like(t) for t in t_uni]

    _check_fwd_bwd_double_bwd(m_auto, m_ref, t_uni, t_ref, cot, cot2)


@pytest.mark.parametrize("irreps_str", TAPERED_IRREPS)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_inversion_auto_uniform_segments(irreps_str: str, layout: cue.IrrepsLayout):
    torch.manual_seed(2)
    ir = cue.Irreps("O3", irreps_str)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m_auto = cuet.Inversion(
            ir,
            layout=layout,
            device=device,
            math_dtype=torch.float64,
        )
    _assert_no_segment_shape_warning(caught)
    assert m_auto.method == "uniform_1d"

    m_explicit = cuet.Inversion(
        ir,
        layout=layout,
        device=device,
        math_dtype=torch.float64,
        method="uniform_1d",
    )
    assert m_explicit.method == "uniform_1d"

    m_ref = cuet.Inversion(
        ir,
        layout=layout,
        device=device,
        math_dtype=torch.float64,
        method="naive",
        auto_uniform_segments=False,
    )

    batch = 4
    x = torch.randn(batch, ir.dim, device=device, dtype=torch.float64)
    t_uni = [x.clone().requires_grad_(True)]
    t_ref = [x.clone().requires_grad_(True)]
    cot = torch.randn(batch, ir.dim, device=device, dtype=torch.float64)
    cot2 = [torch.randn_like(t_uni[0])]

    _check_fwd_bwd_double_bwd(m_auto, m_ref, t_uni, t_ref, cot, cot2)


@pytest.mark.parametrize("irreps_str", TAPERED_IRREPS)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_symmetric_contraction_auto_uniform_segments(
    irreps_str: str, layout: cue.IrrepsLayout, monkeypatch: pytest.MonkeyPatch
):
    """Test SymmetricContraction auto_uniform_segments on tapered stacked descriptors and module."""
    torch.manual_seed(3)
    ir_tapered = cue.Irreps("O3", irreps_str)

    # 1. Build tapered symmetric contraction descriptor across multiplicity tiers
    unique_muls = sorted(set(ir_tapered.muls))
    polys = []
    prev_mul = 0
    for m in unique_muls:
        delta = m - prev_mul
        tier_irs = cue.Irreps("O3", [(delta, ir) for mul, ir in ir_tapered if mul >= m])
        polys.append(cue.descriptors.symmetric_contraction(tier_irs, tier_irs, (1, 2)))
        prev_mul = m
    etp_tapered = cue.EquivariantPolynomial.stack(polys, [True, True, True])
    assert not etp_tapered.all_same_segment_shape()
    etp_uni = etp_tapered.uniformize_segments()
    assert etp_uni.all_same_segment_shape()
    for o in etp_uni.polynomial.operands:
        assert o.segment_shape == (32,)

    # 2. Exercise cuet.SymmetricContraction's auto_uniform_segments codepath directly
    # using a non-uniform descriptor (matching 64-mul total dims via 32+64 or 32+32*2 tiers)
    mul = 64
    ir_uniform = mul * cue.Irreps("O3", "0e + 1o")
    e_hi = cue.descriptors.symmetric_contraction(
        64 * cue.Irreps("O3", "0e"), 64 * cue.Irreps("O3", "0e"), (1, 2)
    )
    e_lo1 = cue.descriptors.symmetric_contraction(
        32 * cue.Irreps("O3", "1o"), 32 * cue.Irreps("O3", "1o"), (1, 2)
    )
    e_lo2 = cue.descriptors.symmetric_contraction(
        32 * cue.Irreps("O3", "1o"), 32 * cue.Irreps("O3", "1o"), (1, 2)
    )
    etp_nonuniform = cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(
                ir_uniform.new_scalars(
                    e_hi.inputs[0].dim + e_lo1.inputs[0].dim + e_lo2.inputs[0].dim
                ),
                cue.ir_mul,
            ),
            cue.IrrepsAndLayout(ir_uniform, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(ir_uniform, cue.ir_mul)],
        cue.SegmentedPolynomial.stack(
            [e_hi.polynomial, e_lo1.polynomial, e_lo2.polynomial],
            [True, True, True],
        ),
    )
    assert not etp_nonuniform.all_same_segment_shape()
    proj_dummy = np.eye(etp_nonuniform.inputs[0].dim // mul)

    monkeypatch.setattr(
        cuet.operations.symmetric_contraction,
        "symmetric_contraction",
        lambda *_args, **_kwargs: (etp_nonuniform, proj_dummy),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m_auto = cuet.SymmetricContraction(
            ir_uniform,
            ir_uniform,
            contraction_degree=2,
            num_elements=3,
            layout=layout,
            device=device,
            dtype=torch.float64,
            math_dtype=torch.float64,
        )
    _assert_no_segment_shape_warning(caught)
    assert m_auto.method == "uniform_1d"
    assert m_auto.etp.all_same_segment_shape() is True

    m_explicit = cuet.SymmetricContraction(
        ir_uniform,
        ir_uniform,
        contraction_degree=2,
        num_elements=3,
        layout=layout,
        device=device,
        dtype=torch.float64,
        math_dtype=torch.float64,
        method="uniform_1d",
    )
    assert m_explicit.method == "uniform_1d"

    m_ref = cuet.SymmetricContraction(
        ir_uniform,
        ir_uniform,
        contraction_degree=2,
        num_elements=3,
        layout=layout,
        device=device,
        dtype=torch.float64,
        math_dtype=torch.float64,
        method="naive",
        auto_uniform_segments=False,
    )
    with torch.no_grad():
        m_ref.weight.copy_(m_auto.weight)

    batch = 4
    indices = torch.tensor([0, 2, 1, 0], device=device, dtype=torch.int64)
    x = torch.randn(batch, ir_uniform.dim, device=device, dtype=torch.float64)
    w_init = m_auto.weight.detach().clone()

    def run_sc(mod, x_in, w_in):
        w_flat = w_in.flatten(1)
        out = mod.f([w_flat, mod.transpose_in(x_in)], input_indices={0: indices})
        return mod.transpose_out(out[0])

    t_uni = [x.clone().requires_grad_(True), w_init.clone().requires_grad_(True)]
    t_ref = [x.clone().requires_grad_(True), w_init.clone().requires_grad_(True)]
    cot = torch.randn(batch, ir_uniform.dim, device=device, dtype=torch.float64)
    cot2 = [torch.randn_like(t) for t in t_uni]

    _check_fwd_bwd_double_bwd(
        lambda xin, win: run_sc(m_auto, xin, win),
        lambda xin, win: run_sc(m_ref, xin, win),
        t_uni,
        t_ref,
        cot,
        cot2,
    )


def test_guardrails_and_disabled_auto_uniform_behavior():
    """Verify pathological GCDs emit warnings / raise ValueError, and auto_uniform_segments=False works."""
    ir_bad = cue.Irreps("O3", "64x0e + 3x1o")
    ir_sh = cue.Irreps("O3", "0e + 1o")

    # 1. Pathological GCD=1 with method=None emits warning and falls back to naive
    for factory in [
        lambda **kw: cuet.ChannelWiseTensorProduct(ir_bad, ir_sh, ir_bad, **kw),
        lambda **kw: cuet.Rotation(ir_bad, **kw),
        lambda **kw: cuet.Inversion(ir_bad, **kw),
    ]:
        with pytest.warns(UserWarning, match="Segments are not the same shape"):
            m = factory(layout=cue.ir_mul)
        assert m.method == "naive"

        # Explicit method="uniform_1d" raises ValueError
        with pytest.raises(ValueError, match="same shape"):
            factory(layout=cue.ir_mul, method="uniform_1d")

    # 2. Tapered irreps with auto_uniform_segments=False preserves old behavior
    ir_good = cue.Irreps("O3", "128x0e + 64x1o + 32x2e")
    with pytest.warns(UserWarning, match="Segments are not the same shape"):
        m_disabled = cuet.Rotation(
            ir_good, layout=cue.ir_mul, auto_uniform_segments=False
        )
    assert m_disabled.method == "naive"

    with pytest.raises(ValueError, match="same shape"):
        cuet.Rotation(
            ir_good,
            layout=cue.ir_mul,
            method="uniform_1d",
            auto_uniform_segments=False,
        )


def test_uniform_1d_jit_options_auto_uniform_segments(monkeypatch: pytest.MonkeyPatch):
    """Verify SegmentedPolynomialFromUniform1dJit stays strict by default and uniformizes when opted in."""
    if u1d_mod.uniform_1d is None:
        monkeypatch.setattr(u1d_mod, "uniform_1d", lambda *args: [])
        monkeypatch.setattr(u1d_mod, "BATCH_DIM_AUTO", -1, raising=False)
        monkeypatch.setattr(u1d_mod, "BATCH_DIM_SHARED", 0, raising=False)
        monkeypatch.setattr(u1d_mod, "BATCH_DIM_BATCHED", 1, raising=False)
        monkeypatch.setattr(u1d_mod, "BATCH_DIM_INDEXED", 2, raising=False)

    poly = cue.descriptors.channelwise_tensor_product(
        cue.Irreps("O3", "128x0e + 64x1o + 32x2e"),
        cue.Irreps("O3", "0e + 1o + 2e"),
    ).polynomial

    # Without options={"auto_uniform_segments": True}, raw SegmentedPolynomialFromUniform1dJit rejects non-uniform poly
    with pytest.raises((RuntimeError, AssertionError)):
        u1d_mod.SegmentedPolynomialFromUniform1dJit(poly)

    # With options={"auto_uniform_segments": True}, it uniformizes to operand_extent=32
    jit_mod = u1d_mod.SegmentedPolynomialFromUniform1dJit(
        poly, options={"auto_uniform_segments": True}
    )
    assert jit_mod.operand_extent == 32

    # Directly passing poly.uniformize_segments() also succeeds with operand_extent=32
    jit_mod2 = u1d_mod.SegmentedPolynomialFromUniform1dJit(poly.uniformize_segments())
    assert jit_mod2.operand_extent == 32
