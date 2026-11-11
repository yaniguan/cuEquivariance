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

import pytest
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
TOL_FP64 = {"atol": 1e-12, "rtol": 1e-12}


def _make_cwtp_poly():
    return (
        cue.descriptors.channelwise_tensor_product(
            cue.Irreps("O3", "4x0e + 4x1o"),
            cue.Irreps("O3", "0e + 1o"),
            cue.Irreps("O3", "0e + 1o"),
        )
        .flatten_coefficient_modes()
        .squeeze_modes()
        .polynomial
    )


def _available_methods_for_cwtp():
    methods = ["naive"]
    if torch.cuda.is_available() and cuet.primitives.segmented_polynomial.HAS_CUE_OPS:
        methods.extend(["uniform_1d", "fused_tp"])
    return methods


# ==============================================================================
# 1. Arbitrary Leading Batch Shapes & Broadcasting Tests
# ==============================================================================


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_1d_unbatched_execution_and_higher_order_grads(layout: cue.IrrepsLayout):
    torch.manual_seed(101)
    irreps_in1 = cue.Irreps("O3", "4x0e + 4x1o")
    irreps_in2 = cue.Irreps("O3", "0e + 1o")
    irreps_out = cue.Irreps("O3", "4x0e + 4x1o")

    # 1a. SegmentedPolynomial with 1D inputs
    poly = _make_cwtp_poly()
    sp = cuet.SegmentedPolynomial(poly, method="naive", math_dtype=torch.float64).to(
        device
    )
    w_1d = torch.randn(
        poly.inputs[0].size, device=device, dtype=torch.float64, requires_grad=True
    )
    x1_1d = torch.randn(
        poly.inputs[1].size, device=device, dtype=torch.float64, requires_grad=True
    )
    x2_1d = torch.randn(
        poly.inputs[2].size, device=device, dtype=torch.float64, requires_grad=True
    )

    out_1d = sp([w_1d, x1_1d, x2_1d])[0]
    assert out_1d.shape == (poly.outputs[0].size,)

    w_2d = w_1d.detach().unsqueeze(0).requires_grad_(True)
    x1_2d = x1_1d.detach().unsqueeze(0).requires_grad_(True)
    x2_2d = x2_1d.detach().unsqueeze(0).requires_grad_(True)
    out_ref = sp([w_2d, x1_2d, x2_2d])[0].squeeze(0)

    torch.testing.assert_close(out_1d, out_ref, **TOL_FP64)

    g1 = torch.autograd.grad(
        out_1d.pow(2).sum(), (w_1d, x1_1d, x2_1d), create_graph=True
    )
    g_ref = torch.autograd.grad(
        out_ref.pow(2).sum(), (w_2d, x1_2d, x2_2d), create_graph=True
    )
    for g, gr in zip(g1, g_ref):
        torch.testing.assert_close(g, gr.squeeze(0), **TOL_FP64)

    gg1 = torch.autograd.grad(sum(t.pow(2).sum() for t in g1), (w_1d, x1_1d, x2_1d))
    gg_ref = torch.autograd.grad(
        sum(t.pow(2).sum() for t in g_ref), (w_2d, x1_2d, x2_2d)
    )
    for gg, ggr in zip(gg1, gg_ref):
        torch.testing.assert_close(gg, ggr.squeeze(0), **TOL_FP64)

    # 1b. High-level Linear, ChannelWiseTensorProduct, FullyConnectedTensorProduct on 1D
    lin = cuet.Linear(
        irreps_in1,
        irreps_out,
        layout=layout,
        device=device,
        dtype=torch.float64,
        method="naive",
    )
    cwtp = cuet.ChannelWiseTensorProduct(
        irreps_in1,
        irreps_in2,
        cue.Irreps("O3", "0e + 1o"),
        layout=layout,
        device=device,
        dtype=torch.float64,
        method="naive",
    )
    fctp = cuet.FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        layout=layout,
        device=device,
        dtype=torch.float64,
        method="naive",
    )

    a_1d = torch.randn(
        irreps_in1.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    b_1d = torch.randn(
        irreps_in2.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    a_2d = a_1d.detach().unsqueeze(0).requires_grad_(True)
    b_2d = b_1d.detach().unsqueeze(0).requires_grad_(True)

    y_lin_1d = lin(a_1d)
    y_lin_ref = lin(a_2d).squeeze(0)
    assert y_lin_1d.shape == (irreps_out.dim,)
    torch.testing.assert_close(y_lin_1d, y_lin_ref, **TOL_FP64)

    y_cwtp_1d = cwtp(a_1d, b_1d)
    y_cwtp_ref = cwtp(a_2d, b_2d).squeeze(0)
    assert y_cwtp_1d.shape == (cwtp.irreps_out.dim,)
    torch.testing.assert_close(y_cwtp_1d, y_cwtp_ref, **TOL_FP64)

    y_fctp_1d = fctp(a_1d, b_1d)
    y_fctp_ref = fctp(a_2d, b_2d).squeeze(0)
    assert y_fctp_1d.shape == (irreps_out.dim,)
    torch.testing.assert_close(y_fctp_1d, y_fctp_ref, **TOL_FP64)


@pytest.mark.parametrize("method", _available_methods_for_cwtp())
def test_shared_operand_bandwidth_preservation(method: str):
    """Verify shared operands ((W,), (1, W), (1, 1, W)) are passed to the 2D backend
    as (1, W) without materializing (R*E, W)."""
    torch.manual_seed(102)
    poly = _make_cwtp_poly()
    sp = cuet.SegmentedPolynomial(poly, method=method, math_dtype=torch.float64).to(
        device
    )

    captured_shapes = []
    orig_m_forward = sp.m.forward

    def spy_forward(inputs, input_indices, output_shapes, output_indices):
        captured_shapes.append([tuple(t.shape) for t in inputs])
        return orig_m_forward(inputs, input_indices, output_shapes, output_indices)

    sp.m.forward = spy_forward

    R, E = 4, 16
    x1 = torch.randn(R, E, poly.inputs[1].size, device=device, dtype=torch.float64)
    x2 = torch.randn(R, E, poly.inputs[2].size, device=device, dtype=torch.float64)

    for w_shape in [
        (poly.inputs[0].size,),
        (1, poly.inputs[0].size),
        (1, 1, poly.inputs[0].size),
    ]:
        captured_shapes.clear()
        w = torch.randn(*w_shape, device=device, dtype=torch.float64)
        out = sp([w, x1, x2])[0]
        assert out.shape == (R, E, poly.outputs[0].size)
        assert len(captured_shapes) == 1
        assert captured_shapes[0][0] == (1, poly.inputs[0].size)
        assert captured_shapes[0][1] == (R * E, poly.inputs[1].size)
        assert captured_shapes[0][2] == (R * E, poly.inputs[2].size)


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("weight_mode", ["shared_2d", "shared_1d", "per_replica"])
def test_3d_replica_batching_and_broadcasting(
    layout: cue.IrrepsLayout, weight_mode: str
):
    """Test 3D Replica-Exchange MD batching (R=4, E=16, D) with shared weights (1, W),
    1D weights (W,), and per-replica weights (R=4, 1, W) across SegmentedPolynomial,
    Linear, ChannelWiseTensorProduct, and FullyConnectedTensorProduct (fwd + 1st & 2nd bwd)."""
    torch.manual_seed(202)
    R, E = 4, 16
    irreps_in1 = cue.Irreps("O3", "4x0e + 4x1o")
    irreps_in2 = cue.Irreps("O3", "0e + 1o")
    irreps_out = cue.Irreps("O3", "4x0e + 4x1o")

    # SegmentedPolynomial
    poly = _make_cwtp_poly()
    sp = cuet.SegmentedPolynomial(poly, method="naive", math_dtype=torch.float64).to(
        device
    )
    if weight_mode == "shared_2d":
        w_sp = torch.randn(
            1, poly.inputs[0].size, device=device, dtype=torch.float64
        ).requires_grad_(True)
    elif weight_mode == "shared_1d":
        w_sp = torch.randn(
            poly.inputs[0].size, device=device, dtype=torch.float64
        ).requires_grad_(True)
    else:
        w_sp = torch.randn(
            R, 1, poly.inputs[0].size, device=device, dtype=torch.float64
        ).requires_grad_(True)

    x1_sp = torch.randn(
        R, E, poly.inputs[1].size, device=device, dtype=torch.float64
    ).requires_grad_(True)
    # Also test operand broadcasting: (1, E, D2) against (R, E, D1)
    x2_sp = torch.randn(
        1, E, poly.inputs[2].size, device=device, dtype=torch.float64
    ).requires_grad_(True)

    out_sp = sp([w_sp, x1_sp, x2_sp])[0]
    assert out_sp.shape == (R, E, poly.outputs[0].size)

    w_sp_ref = w_sp.detach().clone().requires_grad_(True)
    x1_sp_ref = x1_sp.detach().clone().requires_grad_(True)
    x2_sp_ref = x2_sp.detach().clone().requires_grad_(True)
    ref_replicas = []
    for r in range(R):
        if weight_mode == "shared_2d":
            wr = w_sp_ref
        elif weight_mode == "shared_1d":
            wr = w_sp_ref.unsqueeze(0)
        else:
            wr = w_sp_ref[r]  # (1, W)
        ref_replicas.append(sp([wr, x1_sp_ref[r], x2_sp_ref[0]])[0])
    out_sp_ref = torch.stack(ref_replicas, dim=0)

    torch.testing.assert_close(out_sp, out_sp_ref, **TOL_FP64)

    cot_sp = torch.randn_like(out_sp)
    g_sp = torch.autograd.grad(
        (out_sp * cot_sp).sum(), (w_sp, x1_sp, x2_sp), create_graph=True
    )
    g_sp_ref = torch.autograd.grad(
        (out_sp_ref * cot_sp).sum(), (w_sp_ref, x1_sp_ref, x2_sp_ref), create_graph=True
    )
    for g, gr in zip(g_sp, g_sp_ref):
        torch.testing.assert_close(g, gr, **TOL_FP64)

    gg_sp = torch.autograd.grad(sum(t.pow(2).sum() for t in g_sp), (w_sp, x1_sp, x2_sp))
    gg_sp_ref = torch.autograd.grad(
        sum(t.pow(2).sum() for t in g_sp_ref), (w_sp_ref, x1_sp_ref, x2_sp_ref)
    )
    for gg, ggr in zip(gg_sp, gg_sp_ref):
        torch.testing.assert_close(gg, ggr, **TOL_FP64)

    # High-level ops: Linear, ChannelWiseTensorProduct, FullyConnectedTensorProduct
    is_shared = weight_mode != "per_replica"
    lin = cuet.Linear(
        irreps_in1,
        irreps_out,
        layout=layout,
        shared_weights=is_shared,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        method="naive",
    )
    cwtp = cuet.ChannelWiseTensorProduct(
        irreps_in1,
        irreps_in2,
        cue.Irreps("O3", "0e + 1o"),
        layout=layout,
        shared_weights=is_shared,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        method="naive",
    )
    fctp = cuet.FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        layout=layout,
        shared_weights=is_shared,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        method="naive",
    )

    def make_weight(numel: int):
        if weight_mode == "shared_2d":
            return torch.randn(
                1, numel, device=device, dtype=torch.float64, requires_grad=True
            )
        elif weight_mode == "shared_1d":
            return torch.randn(
                numel, device=device, dtype=torch.float64, requires_grad=True
            )
        else:
            return torch.randn(
                R, 1, numel, device=device, dtype=torch.float64, requires_grad=True
            )

    x1 = torch.randn(
        R, E, irreps_in1.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    x2 = torch.randn(
        R, E, irreps_in2.dim, device=device, dtype=torch.float64, requires_grad=True
    )

    for op_name, op, w in [
        ("linear", lin, make_weight(lin.weight_numel)),
        ("cwtp", cwtp, make_weight(cwtp.weight_numel)),
        ("fctp", fctp, make_weight(fctp.weight_numel)),
    ]:
        x1_cur = x1.detach().clone().requires_grad_(True)
        x2_cur = x2.detach().clone().requires_grad_(True)
        w_cur = w.detach().clone().requires_grad_(True)

        x1_r = x1.detach().clone().requires_grad_(True)
        x2_r = x2.detach().clone().requires_grad_(True)
        w_r = w.detach().clone().requires_grad_(True)

        if op_name == "linear":
            out = op(x1_cur, w_cur)
            out_ref = torch.stack(
                [
                    op(
                        x1_r[r],
                        w_r[r].expand(E, -1)
                        if weight_mode == "per_replica"
                        else (w_r.unsqueeze(0) if w_r.ndim == 1 else w_r),
                    )
                    for r in range(R)
                ],
                dim=0,
            )
            vars_mb = (x1_cur, w_cur)
            vars_ref = (x1_r, w_r)
        else:
            out = op(x1_cur, x2_cur, w_cur)
            out_ref = torch.stack(
                [
                    op(
                        x1_r[r],
                        x2_r[r],
                        w_r[r].expand(E, -1)
                        if weight_mode == "per_replica"
                        else (w_r.unsqueeze(0) if w_r.ndim == 1 else w_r),
                    )
                    for r in range(R)
                ],
                dim=0,
            )
            vars_mb = (x1_cur, x2_cur, w_cur)
            vars_ref = (x1_r, x2_r, w_r)

        assert out.shape == out_ref.shape
        torch.testing.assert_close(out, out_ref, **TOL_FP64)

        cot = torch.randn_like(out)
        g_mb = torch.autograd.grad((out * cot).sum(), vars_mb, create_graph=True)
        g_rf = torch.autograd.grad((out_ref * cot).sum(), vars_ref, create_graph=True)
        for g, gr in zip(g_mb, g_rf):
            torch.testing.assert_close(g, gr, **TOL_FP64)

        gg_mb = torch.autograd.grad(sum(t.pow(2).sum() for t in g_mb), vars_mb)
        gg_rf = torch.autograd.grad(sum(t.pow(2).sum() for t in g_rf), vars_ref)
        for gg, ggr in zip(gg_mb, gg_rf):
            torch.testing.assert_close(gg, ggr, **TOL_FP64)


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_4d_md_trajectory_batching(layout: cue.IrrepsLayout):
    """Test 4D MD trajectory batching (B=2, T=3, N=5, D) across SegmentedPolynomial,
    Linear, ChannelWiseTensorProduct, FullyConnectedTensorProduct, and TransposeIrrepsLayout."""
    torch.manual_seed(303)
    B, T, N = 2, 3, 5
    irreps_in1 = cue.Irreps("O3", "4x0e + 4x1o")
    irreps_in2 = cue.Irreps("O3", "0e + 1o")
    irreps_out = cue.Irreps("O3", "4x0e + 4x1o")

    tr = cuet.TransposeIrrepsLayout(
        irreps_in1, source=cue.mul_ir, target=cue.ir_mul, device=device
    )
    tr_inv = cuet.TransposeIrrepsLayout(
        irreps_in1, source=cue.ir_mul, target=cue.mul_ir, device=device
    )
    x_4d = torch.randn(B, T, N, irreps_in1.dim, device=device, dtype=torch.float64)
    torch.testing.assert_close(tr_inv(tr(x_4d)), x_4d, **TOL_FP64)

    lin = cuet.Linear(
        irreps_in1,
        irreps_out,
        layout=layout,
        device=device,
        dtype=torch.float64,
        method="naive",
    )
    cwtp = cuet.ChannelWiseTensorProduct(
        irreps_in1,
        irreps_in2,
        cue.Irreps("O3", "0e + 1o"),
        layout=layout,
        device=device,
        dtype=torch.float64,
        method="naive",
    )
    fctp = cuet.FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        layout=layout,
        device=device,
        dtype=torch.float64,
        method="naive",
    )

    a_4d = torch.randn(
        B, T, N, irreps_in1.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    b_4d = torch.randn(
        B, T, N, irreps_in2.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    a_flat = a_4d.detach().reshape(B * T * N, irreps_in1.dim).requires_grad_(True)
    b_flat = b_4d.detach().reshape(B * T * N, irreps_in2.dim).requires_grad_(True)

    # Linear 4D + 1st & 2nd backward
    out_lin = lin(a_4d)
    out_lin_ref = lin(a_flat).reshape(B, T, N, irreps_out.dim)
    assert out_lin.shape == (B, T, N, irreps_out.dim)
    torch.testing.assert_close(out_lin, out_lin_ref, **TOL_FP64)

    # CWTP 4D + 1st & 2nd backward
    out_cwtp = cwtp(a_4d, b_4d)
    out_cwtp_ref = cwtp(a_flat, b_flat).reshape(B, T, N, cwtp.irreps_out.dim)
    assert out_cwtp.shape == (B, T, N, cwtp.irreps_out.dim)
    torch.testing.assert_close(out_cwtp, out_cwtp_ref, **TOL_FP64)

    # FCTP 4D + 1st & 2nd backward
    out_fctp = fctp(a_4d, b_4d)
    out_fctp_ref = fctp(a_flat, b_flat).reshape(B, T, N, irreps_out.dim)
    assert out_fctp.shape == (B, T, N, irreps_out.dim)
    torch.testing.assert_close(out_fctp, out_fctp_ref, **TOL_FP64)

    loss_4d = out_lin.pow(2).sum() + out_cwtp.pow(2).sum() + out_fctp.pow(2).sum()
    loss_ref = (
        out_lin_ref.pow(2).sum() + out_cwtp_ref.pow(2).sum() + out_fctp_ref.pow(2).sum()
    )
    ga_4d, gb_4d = torch.autograd.grad(loss_4d, (a_4d, b_4d), create_graph=True)
    ga_ref, gb_ref = torch.autograd.grad(loss_ref, (a_flat, b_flat), create_graph=True)
    torch.testing.assert_close(ga_4d, ga_ref.reshape_as(a_4d), **TOL_FP64)
    torch.testing.assert_close(gb_4d, gb_ref.reshape_as(b_4d), **TOL_FP64)

    gga_4d, ggb_4d = torch.autograd.grad(
        ga_4d.pow(2).sum() + gb_4d.pow(2).sum(), (a_4d, b_4d)
    )
    gga_ref, ggb_ref = torch.autograd.grad(
        ga_ref.pow(2).sum() + gb_ref.pow(2).sum(), (a_flat, b_flat)
    )
    torch.testing.assert_close(gga_4d, gga_ref.reshape_as(a_4d), **TOL_FP64)
    torch.testing.assert_close(ggb_4d, ggb_ref.reshape_as(b_4d), **TOL_FP64)


@pytest.mark.parametrize("batch_shape", [(4, 0), (0,), (2, 0, 5)])
def test_empty_batch_dimensions(batch_shape):
    """Verify empty batch shapes like (4, 0, D) never crash and preserve output shapes,
    dtypes, and zero gradients across SegmentedPolynomial, Linear, CWTP, and FCTP."""
    irreps_in1 = cue.Irreps("O3", "4x0e + 4x1o")
    irreps_in2 = cue.Irreps("O3", "0e + 1o")
    irreps_out = cue.Irreps("O3", "4x0e + 4x1o")

    poly = _make_cwtp_poly()
    sp = cuet.SegmentedPolynomial(poly, method="naive", math_dtype=torch.float64).to(
        device
    )
    w = torch.randn(
        1, poly.inputs[0].size, device=device, dtype=torch.float64, requires_grad=True
    )
    x1 = torch.randn(
        *batch_shape,
        poly.inputs[1].size,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    x2 = torch.randn(
        *batch_shape,
        poly.inputs[2].size,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )

    out_sp = sp([w, x1, x2])[0]
    assert out_sp.shape == (*batch_shape, poly.outputs[0].size)
    assert out_sp.dtype == torch.float64
    assert out_sp.requires_grad

    gw, gx1, gx2 = torch.autograd.grad(out_sp.sum(), (w, x1, x2), create_graph=True)
    torch.testing.assert_close(gw, torch.zeros_like(w), **TOL_FP64)
    assert gx1.shape == x1.shape
    assert gx2.shape == x2.shape
    if gw.requires_grad:
        ggw = torch.autograd.grad(gw.pow(2).sum(), w, allow_unused=True)[0]
        if ggw is not None:
            torch.testing.assert_close(ggw, torch.zeros_like(w), **TOL_FP64)

    lin = cuet.Linear(
        irreps_in1,
        irreps_out,
        layout=cue.ir_mul,
        device=device,
        dtype=torch.float64,
        method="naive",
    )
    cwtp = cuet.ChannelWiseTensorProduct(
        irreps_in1,
        irreps_in2,
        cue.Irreps("O3", "0e + 1o"),
        layout=cue.ir_mul,
        device=device,
        dtype=torch.float64,
        method="naive",
    )
    fctp = cuet.FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        layout=cue.ir_mul,
        device=device,
        dtype=torch.float64,
        method="naive",
    )

    a = torch.randn(
        *batch_shape,
        irreps_in1.dim,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    b = torch.randn(
        *batch_shape,
        irreps_in2.dim,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    y_lin = lin(a)
    y_cwtp = cwtp(a, b)
    y_fctp = fctp(a, b)
    assert y_lin.shape == (*batch_shape, irreps_out.dim)
    assert y_cwtp.shape == (*batch_shape, cwtp.irreps_out.dim)
    assert y_fctp.shape == (*batch_shape, irreps_out.dim)
    (y_lin.sum() + y_cwtp.sum() + y_fctp.sum()).backward()
    torch.testing.assert_close(
        lin.weight.grad, torch.zeros_like(lin.weight), **TOL_FP64
    )
    torch.testing.assert_close(
        cwtp.weight.grad, torch.zeros_like(cwtp.weight), **TOL_FP64
    )
    torch.testing.assert_close(
        fctp.weight.grad, torch.zeros_like(fctp.weight), **TOL_FP64
    )


# ==============================================================================
# 2. Fixed-Capacity Verlet Neighbor List Padding (mask + -1 Sentinels)
# ==============================================================================


def _build_verlet_test_graph(idx_dtype: torch.dtype, pad_scheme: str):
    """Construct a real molecular graph (N=7, E_real=15) touching both atom 0 and atom N-1,
    padded to fixed capacity E_max=32."""
    N = 7
    E_real = 15
    E_max = 32

    # Explicitly include edges incident on atom 0 and atom N-1 (6) in both src and dst
    src_real = torch.tensor(
        [0, 6, 1, 2, 0, 3, 4, 5, 6, 2, 1, 4, 3, 5, 0],
        device=device,
        dtype=idx_dtype,
    )
    dst_real = torch.tensor(
        [1, 0, 6, 0, 6, 2, 5, 6, 3, 4, 0, 1, 6, 2, 3],
        device=device,
        dtype=idx_dtype,
    )
    assert src_real.shape == (E_real,)
    assert dst_real.shape == (E_real,)

    n_pad = E_max - E_real
    mask_bool = torch.cat(
        [
            torch.ones(E_real, device=device, dtype=torch.bool),
            torch.zeros(n_pad, device=device, dtype=torch.bool),
        ]
    )

    if pad_scheme == "sentinel_only":
        src_pad = torch.cat(
            [src_real, torch.full((n_pad,), -1, device=device, dtype=idx_dtype)]
        )
        dst_pad = torch.cat(
            [dst_real, torch.full((n_pad,), -1, device=device, dtype=idx_dtype)]
        )
        mask_arg = None
    elif pad_scheme == "sentinel_and_mask":
        src_pad = torch.cat(
            [src_real, torch.full((n_pad,), -1, device=device, dtype=idx_dtype)]
        )
        dst_pad = torch.cat(
            [dst_real, torch.full((n_pad,), -1, device=device, dtype=idx_dtype)]
        )
        mask_arg = mask_bool
    elif pad_scheme == "zero_indices_with_mask":
        # Padded edges point directly to atom 0 (src=0, dst=0) and rely on mask=False!
        src_pad = torch.cat(
            [src_real, torch.zeros((n_pad,), device=device, dtype=idx_dtype)]
        )
        dst_pad = torch.cat(
            [dst_real, torch.zeros((n_pad,), device=device, dtype=idx_dtype)]
        )
        mask_arg = mask_bool
    elif pad_scheme == "mixed_partial_sentinel":
        # Half have src=-1, dst=0; half have src=N-1, dst=-1
        half = n_pad // 2
        rem = n_pad - half
        src_pad = torch.cat(
            [
                src_real,
                torch.full((half,), -1, device=device, dtype=idx_dtype),
                torch.full((rem,), N - 1, device=device, dtype=idx_dtype),
            ]
        )
        dst_pad = torch.cat(
            [
                dst_real,
                torch.zeros((half,), device=device, dtype=idx_dtype),
                torch.full((rem,), -1, device=device, dtype=idx_dtype),
            ]
        )
        mask_arg = None
    else:
        raise ValueError(f"Unknown pad_scheme: {pad_scheme}")

    return N, E_real, E_max, src_real, dst_real, src_pad, dst_pad, mask_arg


@pytest.mark.parametrize("method", _available_methods_for_cwtp())
@pytest.mark.parametrize(
    "pad_scheme",
    [
        "sentinel_only",
        "sentinel_and_mask",
        "zero_indices_with_mask",
        "mixed_partial_sentinel",
    ],
)
@pytest.mark.parametrize("shared_weights", [True, False])
@pytest.mark.parametrize("scatter_to_nodes", [True, False])
@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
def test_segmented_polynomial_verlet_padding_fwd_bwd_force_qat(
    method: str,
    pad_scheme: str,
    shared_weights: bool,
    scatter_to_nodes: bool,
    idx_dtype: torch.dtype,
):
    """Verify forward outputs, atom 0 & atom N-1 forces, shared (1, W) & edge-wise (E_max, W)
    weight gradients, and 2nd-order Force-QAT derivatives match the unpadded E_real=15 graph
    to < 1e-12 in float64."""
    torch.manual_seed(404)
    poly = _make_cwtp_poly()
    sp = cuet.SegmentedPolynomial(poly, method=method, math_dtype=torch.float64).to(
        device
    )

    N, E_real, E_max, src_real, dst_real, src_pad, dst_pad, mask_arg = (
        _build_verlet_test_graph(idx_dtype, pad_scheme)
    )

    # Real unpadded tensors
    x_real = torch.randn(
        N, poly.inputs[1].size, device=device, dtype=torch.float64, requires_grad=True
    )
    edge_sh_real = torch.randn(
        E_real,
        poly.inputs[2].size,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    if shared_weights:
        w_real = torch.randn(
            1,
            poly.inputs[0].size,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
    else:
        w_real = torch.randn(
            E_real,
            poly.inputs[0].size,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )

    # Padded tensors with non-zero junk in invalid slots E_real..E_max-1
    x_pad = x_real.detach().clone().requires_grad_(True)
    edge_sh_pad = (
        torch.cat(
            [
                edge_sh_real.detach(),
                torch.randn(
                    E_max - E_real,
                    poly.inputs[2].size,
                    device=device,
                    dtype=torch.float64,
                ),
            ],
            dim=0,
        )
        .clone()
        .requires_grad_(True)
    )
    if shared_weights:
        w_pad = w_real.detach().clone().requires_grad_(True)
    else:
        w_pad = (
            torch.cat(
                [
                    w_real.detach(),
                    torch.randn(
                        E_max - E_real,
                        poly.inputs[0].size,
                        device=device,
                        dtype=torch.float64,
                    ),
                ],
                dim=0,
            )
            .clone()
            .requires_grad_(True)
        )

    if scatter_to_nodes:
        out_real = sp(
            [w_real, x_real, edge_sh_real],
            input_indices={1: src_real},
            output_shapes={0: torch.empty((N, 1), device=device, dtype=torch.float64)},
            output_indices={0: dst_real},
        )[0]
        out_pad = sp(
            [w_pad, x_pad, edge_sh_pad],
            input_indices={1: src_pad},
            output_shapes={0: torch.empty((N, 1), device=device, dtype=torch.float64)},
            output_indices={0: dst_pad},
            mask=mask_arg,
        )[0]
        assert out_pad.shape == (N, poly.outputs[0].size)
        # Forward exactness (explicitly including atom 0 and atom N-1)
        torch.testing.assert_close(out_pad[0], out_real[0], **TOL_FP64)
        torch.testing.assert_close(out_pad[N - 1], out_real[N - 1], **TOL_FP64)
        torch.testing.assert_close(out_pad, out_real, **TOL_FP64)
    else:
        if pad_scheme == "mixed_partial_sentinel":
            pytest.skip(
                "mixed_partial_sentinel puts -1 in dst_pad, which is not passed when scatter_to_nodes=False"
            )
        out_real = sp(
            [w_real, x_real, edge_sh_real],
            input_indices={1: src_real},
        )[0]
        out_pad = sp(
            [w_pad, x_pad, edge_sh_pad],
            input_indices={1: src_pad},
            mask=mask_arg,
        )[0]
        assert out_pad.shape == (E_max, poly.outputs[0].size)
        torch.testing.assert_close(out_pad[:E_real], out_real, **TOL_FP64)
        torch.testing.assert_close(
            out_pad[E_real:], torch.zeros_like(out_pad[E_real:]), **TOL_FP64
        )

    # Energy -> Forces (1st backward with create_graph=True)
    # Use a non-linear readout of outputs so both output-cotangent and input-hessian paths fire
    energy_real = (out_real.pow(2) * 0.5).sum()
    # Note: for unindexed edge outputs, out_pad[E_real:] is 0.0, so summing over ALL E_max
    # slots directly tests that padded slots contribute 0 to both energy and gradients!
    energy_pad = (out_pad.pow(2) * 0.5).sum()
    torch.testing.assert_close(energy_pad, energy_real, **TOL_FP64)

    gx_real, gsh_real, gw_real = torch.autograd.grad(
        energy_real, (x_real, edge_sh_real, w_real), create_graph=True
    )
    gx_pad, gsh_pad, gw_pad = torch.autograd.grad(
        energy_pad, (x_pad, edge_sh_pad, w_pad), create_graph=True
    )

    # Check 1st-order gradients: atom 0, atom N-1, all atoms, edge_sh, and weights
    torch.testing.assert_close(gx_pad[0], gx_real[0], **TOL_FP64)
    torch.testing.assert_close(gx_pad[N - 1], gx_real[N - 1], **TOL_FP64)
    torch.testing.assert_close(gx_pad, gx_real, **TOL_FP64)
    torch.testing.assert_close(gsh_pad[:E_real], gsh_real, **TOL_FP64)
    torch.testing.assert_close(
        gsh_pad[E_real:], torch.zeros_like(gsh_pad[E_real:]), **TOL_FP64
    )
    if shared_weights:
        torch.testing.assert_close(gw_pad, gw_real, **TOL_FP64)
    else:
        torch.testing.assert_close(gw_pad[:E_real], gw_real, **TOL_FP64)
        torch.testing.assert_close(
            gw_pad[E_real:], torch.zeros_like(gw_pad[E_real:]), **TOL_FP64
        )

    # 2nd-order Force-QAT gradients: differentiate atomic forces (-gx) wrt weights & x!
    force_cotangent = torch.randn_like(gx_real)
    qat_loss_real = (-gx_real * force_cotangent).sum()
    qat_loss_pad = (-gx_pad * force_cotangent).sum()

    ggx_real, ggsh_real, ggw_real = torch.autograd.grad(
        qat_loss_real, (x_real, edge_sh_real, w_real)
    )
    ggx_pad, ggsh_pad, ggw_pad = torch.autograd.grad(
        qat_loss_pad, (x_pad, edge_sh_pad, w_pad)
    )

    torch.testing.assert_close(ggx_pad[0], ggx_real[0], **TOL_FP64)
    torch.testing.assert_close(ggx_pad[N - 1], ggx_real[N - 1], **TOL_FP64)
    torch.testing.assert_close(ggx_pad, ggx_real, **TOL_FP64)
    torch.testing.assert_close(ggsh_pad[:E_real], ggsh_real, **TOL_FP64)
    torch.testing.assert_close(
        ggsh_pad[E_real:], torch.zeros_like(ggsh_pad[E_real:]), **TOL_FP64
    )
    if shared_weights:
        torch.testing.assert_close(ggw_pad, ggw_real, **TOL_FP64)
    else:
        torch.testing.assert_close(ggw_pad[:E_real], ggw_real, **TOL_FP64)
        torch.testing.assert_close(
            ggw_pad[E_real:], torch.zeros_like(ggw_pad[E_real:]), **TOL_FP64
        )


@pytest.mark.parametrize("op_kind", ["channel_wise", "fully_connected"])
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("pad_scheme", ["sentinel_only", "zero_indices_with_mask"])
def test_high_level_indexed_tp_with_mask_and_sentinels(
    op_kind: str, layout: cue.IrrepsLayout, pad_scheme: str
):
    """Verify ChannelWiseTensorProduct and FullyConnectedTensorProduct with
    indices_1, indices_2, indices_out, size_out, and mask (fwd + 1st & 2nd Force-QAT bwd)."""
    torch.manual_seed(505)
    irreps_1 = cue.Irreps("O3", "4x0e + 4x1o")
    irreps_2 = cue.Irreps("O3", "0e + 1o")
    irreps_out = cue.Irreps("O3", "4x0e + 4x1o")

    if op_kind == "channel_wise":
        op = cuet.ChannelWiseTensorProduct(
            irreps_1,
            irreps_2,
            cue.Irreps("O3", "0e + 1o"),
            layout=layout,
            shared_weights=True,
            internal_weights=False,
            device=device,
            dtype=torch.float64,
            method="naive",
        )
    else:
        op = cuet.FullyConnectedTensorProduct(
            irreps_1,
            irreps_2,
            irreps_out,
            layout=layout,
            shared_weights=True,
            internal_weights=False,
            device=device,
            dtype=torch.float64,
            method="naive",
        )

    N, E_real, E_max, src_real, dst_real, src_pad, dst_pad, mask_arg = (
        _build_verlet_test_graph(torch.int64, pad_scheme)
    )

    x1_real = torch.randn(
        N, irreps_1.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    x2_real = torch.randn(
        N, irreps_2.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    w_real = torch.randn(
        1, op.weight_numel, device=device, dtype=torch.float64, requires_grad=True
    )

    x1_pad = x1_real.detach().clone().requires_grad_(True)
    x2_pad = x2_real.detach().clone().requires_grad_(True)
    w_pad = w_real.detach().clone().requires_grad_(True)

    out_real = op(
        x1_real,
        x2_real,
        w_real,
        indices_1=src_real,
        indices_2=dst_real,
        indices_out=dst_real,
        size_out=N,
    )
    out_pad = op(
        x1_pad,
        x2_pad,
        w_pad,
        indices_1=src_pad,
        indices_2=dst_pad,
        indices_out=dst_pad,
        size_out=N,
        mask=mask_arg,
    )

    assert out_pad.shape == (N, op.irreps_out.dim)
    torch.testing.assert_close(out_pad, out_real, **TOL_FP64)

    gx1_r, gx2_r, gw_r = torch.autograd.grad(
        out_real.pow(2).sum(), (x1_real, x2_real, w_real), create_graph=True
    )
    gx1_p, gx2_p, gw_p = torch.autograd.grad(
        out_pad.pow(2).sum(), (x1_pad, x2_pad, w_pad), create_graph=True
    )
    torch.testing.assert_close(gx1_p, gx1_r, **TOL_FP64)
    torch.testing.assert_close(gx2_p, gx2_r, **TOL_FP64)
    torch.testing.assert_close(gw_p, gw_r, **TOL_FP64)

    cot1 = torch.randn_like(gx1_r)
    cot2 = torch.randn_like(gx2_r)
    ggx1_r, ggx2_r, ggw_r = torch.autograd.grad(
        (gx1_r * cot1).sum() + (gx2_r * cot2).sum(), (x1_real, x2_real, w_real)
    )
    ggx1_p, ggx2_p, ggw_p = torch.autograd.grad(
        (gx1_p * cot1).sum() + (gx2_p * cot2).sum(), (x1_pad, x2_pad, w_pad)
    )
    torch.testing.assert_close(ggx1_p, ggx1_r, **TOL_FP64)
    torch.testing.assert_close(ggx2_p, ggx2_r, **TOL_FP64)
    torch.testing.assert_close(ggw_p, ggw_r, **TOL_FP64)


def test_indexed_shared_output_reduction_and_linear_masked():
    """Verify total-energy style shared output reduction (output_shapes=(1, out_dim))
    and element-indexed Linear with -1 sentinels & masks."""
    torch.manual_seed(606)
    poly = _make_cwtp_poly()
    sp = cuet.SegmentedPolynomial(poly, method="naive", math_dtype=torch.float64).to(
        device
    )

    N, E_real, E_max, src_real, _, src_pad, _, _ = _build_verlet_test_graph(
        torch.int64, "sentinel_only"
    )
    w = torch.randn(
        1, poly.inputs[0].size, device=device, dtype=torch.float64, requires_grad=True
    )
    x = torch.randn(
        N, poly.inputs[1].size, device=device, dtype=torch.float64, requires_grad=True
    )
    sh_real = torch.randn(
        E_real,
        poly.inputs[2].size,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    sh_pad = (
        torch.cat(
            [
                sh_real.detach(),
                torch.randn(
                    E_max - E_real,
                    poly.inputs[2].size,
                    device=device,
                    dtype=torch.float64,
                ),
            ],
            dim=0,
        )
        .clone()
        .requires_grad_(True)
    )

    w_p = w.detach().clone().requires_grad_(True)
    x_p = x.detach().clone().requires_grad_(True)

    reduced_real = sp(
        [w, x, sh_real],
        input_indices={1: src_real},
        output_shapes={0: torch.empty((1, 1), device=device, dtype=torch.float64)},
    )[0]
    reduced_pad = sp(
        [w_p, x_p, sh_pad],
        input_indices={1: src_pad},
        output_shapes={0: torch.empty((1, 1), device=device, dtype=torch.float64)},
    )[0]
    assert reduced_pad.shape == (1, poly.outputs[0].size)
    torch.testing.assert_close(reduced_pad, reduced_real, **TOL_FP64)

    gw_r, gx_r = torch.autograd.grad(reduced_real.pow(2).sum(), (w, x))
    gw_p, gx_p = torch.autograd.grad(reduced_pad.pow(2).sum(), (w_p, x_p))
    torch.testing.assert_close(gw_p, gw_r, **TOL_FP64)
    torch.testing.assert_close(gx_p, gx_r, **TOL_FP64)

    # Element-indexed Linear with -1 sentinels
    irreps = cue.Irreps("O3", "4x0e + 4x1o")
    lin = cuet.Linear(
        irreps,
        irreps,
        layout=cue.ir_mul,
        shared_weights=True,
        internal_weights=True,
        weight_classes=3,
        device=device,
        dtype=torch.float64,
        method="naive",
    )
    elem_real = torch.tensor([0, 2, 1, 0, 2], device=device, dtype=torch.int64)
    elem_pad = torch.tensor(
        [0, 2, 1, 0, 2, -1, -1, -1], device=device, dtype=torch.int64
    )
    feat_real = torch.randn(5, irreps.dim, device=device, dtype=torch.float64)
    feat_pad = torch.cat(
        [feat_real, torch.randn(3, irreps.dim, device=device, dtype=torch.float64)],
        dim=0,
    )
    y_real = lin(feat_real, weight_indices=elem_real)
    y_pad = lin(feat_pad, weight_indices=elem_pad)
    torch.testing.assert_close(y_pad[:5], y_real, **TOL_FP64)
    torch.testing.assert_close(y_pad[5:], torch.zeros_like(y_pad[5:]), **TOL_FP64)


# ==============================================================================
# 3. TorchScript, Tracing, torch.compile(fullgraph=True) & Validation Checks
# ==============================================================================


def test_script_trace_and_fullgraph_compile_zero_sync():
    """Verify TorchScript, JIT trace, and torch.compile(fullgraph=True) work with
    zero graph breaks and zero host-device syncs on both 3D multi-batch and -1 sentinels."""
    torch.manual_seed(707)
    irreps_1 = cue.Irreps("O3", "4x0e + 4x1o")
    irreps_2 = cue.Irreps("O3", "0e + 1o")

    cwtp = cuet.ChannelWiseTensorProduct(
        irreps_1,
        irreps_2,
        cue.Irreps("O3", "0e + 1o"),
        layout=cue.ir_mul,
        shared_weights=True,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        method="naive",
    )
    fctp = cuet.FullyConnectedTensorProduct(
        irreps_1,
        irreps_2,
        irreps_1,
        layout=cue.ir_mul,
        shared_weights=True,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        method="naive",
    )

    # 1. TorchScript on 3D multi-batch and masked indexed execution
    cwtp_scripted = torch.jit.script(cwtp)
    fctp_scripted = torch.jit.script(fctp)

    x1_3d = torch.randn(3, 8, irreps_1.dim, device=device, dtype=torch.float64)
    x2_3d = torch.randn(3, 8, irreps_2.dim, device=device, dtype=torch.float64)
    w_cwtp = torch.randn(1, cwtp.weight_numel, device=device, dtype=torch.float64)
    w_fctp = torch.randn(1, fctp.weight_numel, device=device, dtype=torch.float64)

    torch.testing.assert_close(
        cwtp_scripted(x1_3d, x2_3d, w_cwtp), cwtp(x1_3d, x2_3d, w_cwtp), **TOL_FP64
    )
    torch.testing.assert_close(
        fctp_scripted(x1_3d, x2_3d, w_fctp), fctp(x1_3d, x2_3d, w_fctp), **TOL_FP64
    )

    # JIT trace on 3D multi-batch
    cwtp_traced = torch.jit.trace(cwtp, (x1_3d, x2_3d, w_cwtp))
    torch.testing.assert_close(
        cwtp_traced(x1_3d, x2_3d, w_cwtp), cwtp(x1_3d, x2_3d, w_cwtp), **TOL_FP64
    )

    # 2. torch.compile(backend="aot_eager", fullgraph=True) with inline Force-QAT
    N, E_real, E_max, src_real, dst_real, src_pad, dst_pad, mask_arg = (
        _build_verlet_test_graph(torch.int64, "sentinel_and_mask")
    )
    nx_real = torch.randn(
        N, irreps_1.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    ey_real = torch.randn(
        E_real, irreps_2.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    ey_pad = (
        torch.cat(
            [
                ey_real.detach(),
                torch.randn(
                    E_max - E_real, irreps_2.dim, device=device, dtype=torch.float64
                ),
            ],
            dim=0,
        )
        .clone()
        .requires_grad_(True)
    )
    w_r = w_cwtp.detach().clone().requires_grad_(True)
    w_p = w_cwtp.detach().clone().requires_grad_(True)
    nx_pad = nx_real.detach().clone().requires_grad_(True)

    torch.testing.assert_close(
        cwtp_scripted(nx_pad, ey_pad, w_p, src_pad, None, dst_pad, N, mask_arg),
        cwtp(nx_real, ey_real, w_r, src_real, None, dst_real, N),
        **TOL_FP64,
    )

    def energy_and_forces(nx, ey, w, s, d, m):
        node_out = cwtp(
            nx, ey, w, indices_1=s, indices_2=None, indices_out=d, size_out=N, mask=m
        )
        energy = node_out.pow(2).sum()
        forces = -torch.autograd.grad(energy, nx, create_graph=True)[0]
        return energy, forces

    compiled_fn = torch.compile(energy_and_forces, backend="aot_eager", fullgraph=True)
    e_pad, f_pad = compiled_fn(nx_pad, ey_pad, w_p, src_pad, dst_pad, mask_arg)
    e_ref, f_ref = energy_and_forces(nx_real, ey_real, w_r, src_real, dst_real, None)

    torch.testing.assert_close(e_pad, e_ref, **TOL_FP64)
    torch.testing.assert_close(f_pad, f_ref, **TOL_FP64)

    (e_pad + f_pad.pow(2).sum()).backward()
    (e_ref + f_ref.pow(2).sum()).backward()
    torch.testing.assert_close(w_p.grad, w_r.grad, **TOL_FP64)
    torch.testing.assert_close(nx_pad.grad, nx_real.grad, **TOL_FP64)


def test_validation_errors():
    """Ensure invalid shapes/ranks continue to raise clean errors."""
    poly = _make_cwtp_poly()
    sp = cuet.SegmentedPolynomial(poly, method="naive", math_dtype=torch.float64).to(
        device
    )

    # 0D input
    with pytest.raises(ValueError, match="no dimensions"):
        sp(
            [
                torch.tensor(1.0, device=device),
                torch.randn(2, poly.inputs[1].size, device=device),
                torch.randn(2, poly.inputs[2].size, device=device),
            ]
        )

    # Wrong last-dim operand size (including on empty batch!)
    with pytest.raises(ValueError, match="expected shape"):
        sp(
            [
                torch.randn(1, poly.inputs[0].size, device=device),
                torch.randn(4, 0, poly.inputs[1].size + 1, device=device),
                torch.randn(4, 0, poly.inputs[2].size, device=device),
            ]
        )

    # Incompatible broadcast shapes
    with pytest.raises(RuntimeError):
        sp(
            [
                torch.randn(1, poly.inputs[0].size, device=device),
                torch.randn(4, 5, poly.inputs[1].size, device=device),
                torch.randn(3, 5, poly.inputs[2].size, device=device),
            ]
        )

    # Mask rank exceeding batch dims in multi-batch mode (e.g. (4, 5, 1) instead of (4, 5))
    with pytest.raises(ValueError, match="mask"):
        sp(
            [
                torch.randn(1, poly.inputs[0].size, device=device),
                torch.randn(4, 5, poly.inputs[1].size, device=device),
                torch.randn(4, 5, poly.inputs[2].size, device=device),
            ],
            mask=torch.ones(4, 5, 1, dtype=torch.bool, device=device),
        )

    # FullyConnectedTensorProduct missing size_out when indices_out is provided
    fctp = cuet.FullyConnectedTensorProduct(
        cue.Irreps("O3", "2x0e"),
        cue.Irreps("O3", "0e"),
        cue.Irreps("O3", "2x0e"),
        layout=cue.ir_mul,
        device=device,
        method="naive",
    )
    with pytest.raises(ValueError, match="size_out"):
        fctp(
            torch.randn(3, 2, device=device),
            torch.randn(3, 1, device=device),
            indices_out=torch.tensor([0, 1, 0], device=device),
        )


def test_multibatch_output_shapes_2d_mask_and_all_padded_edges():
    """Verify 3D output_shapes (both full and reduced (1, 1, out_size)) with a 2D
    (R, E) boolean mask, plus all-padded (-1) and zero-length (E=0) neighbor lists."""
    torch.manual_seed(909)
    poly = _make_cwtp_poly()
    sp = cuet.SegmentedPolynomial(poly, method="naive", math_dtype=torch.float64).to(
        device
    )

    # 1. 3D Replica batching with 2D mask (R=3, E=10) and reduced output_shapes (1, 1, D_out)
    R, E = 3, 10
    w = torch.randn(
        1, poly.inputs[0].size, device=device, dtype=torch.float64, requires_grad=True
    )
    x1 = torch.randn(
        R,
        E,
        poly.inputs[1].size,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    x2 = torch.randn(
        R,
        E,
        poly.inputs[2].size,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    mask_2d = torch.tensor(
        [
            [True, True, True, False, False, True, False, False, False, False],
            [True, False, True, True, False, False, False, False, False, False],
            [False, False, False, False, True, True, False, False, False, False],
        ],
        device=device,
        dtype=torch.bool,
    )
    # Contaminate masked-out slots with huge garbage values
    x1_pad = (
        torch.where(mask_2d.unsqueeze(-1), x1, torch.full_like(x1, 1e6))
        .detach()
        .requires_grad_(True)
    )
    x2_pad = (
        torch.where(mask_2d.unsqueeze(-1), x2, torch.full_like(x2, -1e6))
        .detach()
        .requires_grad_(True)
    )
    w_pad = w.detach().clone().requires_grad_(True)

    out_full = sp(
        [w_pad, x1_pad, x2_pad],
        output_shapes={0: torch.empty(R, E, 1, device=device, dtype=torch.float64)},
        mask=mask_2d,
    )[0]
    assert out_full.shape == (R, E, poly.outputs[0].size)
    assert torch.all(out_full[~mask_2d] == 0.0)

    out_red = sp(
        [w_pad, x1_pad, x2_pad],
        output_shapes={0: torch.empty(1, 1, 1, device=device, dtype=torch.float64)},
        mask=mask_2d,
    )[0]
    assert out_red.shape == (1, 1, poly.outputs[0].size)

    # Reference over valid slots only
    w_ref = w.detach().clone().requires_grad_(True)
    x1_valid = x1.detach()[mask_2d].requires_grad_(True)
    x2_valid = x2.detach()[mask_2d].requires_grad_(True)
    out_ref_valid = sp([w_ref, x1_valid, x2_valid])[0]
    torch.testing.assert_close(out_full[mask_2d], out_ref_valid, **TOL_FP64)
    torch.testing.assert_close(
        out_red.squeeze(0).squeeze(0), out_ref_valid.sum(dim=0), **TOL_FP64
    )

    gw_pad, gx1_pad, gx2_pad = torch.autograd.grad(
        out_red.pow(2).sum(), (w_pad, x1_pad, x2_pad), create_graph=True
    )
    gw_ref, gx1_ref, gx2_ref = torch.autograd.grad(
        out_ref_valid.sum(dim=0).pow(2).sum(),
        (w_ref, x1_valid, x2_valid),
        create_graph=True,
    )
    torch.testing.assert_close(gw_pad, gw_ref, **TOL_FP64)
    torch.testing.assert_close(gx1_pad[mask_2d], gx1_ref, **TOL_FP64)
    torch.testing.assert_close(gx2_pad[mask_2d], gx2_ref, **TOL_FP64)
    assert torch.all(gx1_pad[~mask_2d] == 0.0)
    assert torch.all(gx2_pad[~mask_2d] == 0.0)

    # 2. All-padded neighbor list (E_real=0, E_max=12 with -1 sentinels) and E=0 empty indices
    N = 5
    nx = torch.randn(
        N, poly.inputs[1].size, device=device, dtype=torch.float64, requires_grad=True
    )
    ey_all_pad = torch.full(
        (12, poly.inputs[2].size),
        999.0,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    src_all_pad = torch.full((12,), -1, device=device, dtype=torch.int64)
    dst_all_pad = torch.full((12,), -1, device=device, dtype=torch.int64)
    out_all_pad = sp(
        [w, nx, ey_all_pad],
        input_indices={1: src_all_pad},
        output_shapes={0: torch.empty(N, 1, device=device, dtype=torch.float64)},
        output_indices={0: dst_all_pad},
    )[0]
    assert out_all_pad.shape == (N, poly.outputs[0].size)
    torch.testing.assert_close(out_all_pad, torch.zeros_like(out_all_pad), **TOL_FP64)
    gnx, gey, gw0 = torch.autograd.grad(out_all_pad.sum(), (nx, ey_all_pad, w))
    torch.testing.assert_close(gnx, torch.zeros_like(nx), **TOL_FP64)
    torch.testing.assert_close(gey, torch.zeros_like(ey_all_pad), **TOL_FP64)
    torch.testing.assert_close(gw0, torch.zeros_like(w), **TOL_FP64)

    # E=0 empty index tensors
    ey_zero = torch.empty(
        (0, poly.inputs[2].size), device=device, dtype=torch.float64, requires_grad=True
    )
    src_zero = torch.empty((0,), device=device, dtype=torch.int64)
    dst_zero = torch.empty((0,), device=device, dtype=torch.int64)
    out_zero = sp(
        [w, nx, ey_zero],
        input_indices={1: src_zero},
        output_shapes={0: torch.empty(N, 1, device=device, dtype=torch.float64)},
        output_indices={0: dst_zero},
    )[0]
    assert out_zero.shape == (N, poly.outputs[0].size)
    torch.testing.assert_close(out_zero, torch.zeros_like(out_zero), **TOL_FP64)
    gnx_z, gey_z, gw_z = torch.autograd.grad(out_zero.sum(), (nx, ey_zero, w))
    torch.testing.assert_close(gnx_z, torch.zeros_like(nx), **TOL_FP64)
    assert gey_z.shape == (0, poly.inputs[2].size)
    torch.testing.assert_close(gw_z, torch.zeros_like(w), **TOL_FP64)
