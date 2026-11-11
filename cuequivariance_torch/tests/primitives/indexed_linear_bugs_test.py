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

import copy
import pickle
import sys
import types
from contextlib import contextmanager
from typing import Tuple
from unittest.mock import patch

import cuequivariance_torch.primitives.segmented_polynomial as sp_mod
import pytest
import torch
import torch._dynamo
from cuequivariance_torch.primitives.segmented_polynomial_indexed_linear import (
    SUBDICT,
    SegmentedPolynomialIndexedLinear,
)
from cuequivariance_torch.primitives.segmented_polynomial_naive import (
    SegmentedPolynomialNaive,
)

import cuequivariance as cue
import cuequivariance_torch as cuet


def _compute_group_ids(counts: torch.Tensor, z_total: int) -> torch.Tensor:
    cum_counts = torch.cumsum(counts.to(dtype=torch.int64), dim=0)
    row_ids = torch.arange(z_total, dtype=torch.int64, device=counts.device)
    return torch.searchsorted(cum_counts, row_ids, right=True)


class _MockIndexedLinearFn(torch.autograd.Function):
    """Faithful grouped-GEMM reference emulation of cuequivariance_ops_torch.indexed_linear."""

    @staticmethod
    def forward(
        ctx,
        input1: torch.Tensor,
        input2: torch.Tensor,
        counts: torch.Tensor,
        u: int,
        v: int,
        C: int,
        Z: int,
        subscripts: Tuple[str, str, str],
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        assert counts.ndim == 1 and counts.shape[0] == C, (
            f"Expected counts shape ({C},), got {counts.shape}"
        )
        assert counts.dtype == torch.int32, f"Expected counts int32, got {counts.dtype}"
        if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
            assert int(counts.sum().item()) == Z, (
                f"counts.sum()={int(counts.sum().item())} != Z={Z}"
            )
            assert bool(torch.all(counts >= 0).item()), "Negative counts not allowed"

        ctx.save_for_backward(input1, input2, counts, coefficients)
        ctx.u = u
        ctx.v = v
        ctx.C = C
        ctx.Z = Z
        ctx.subscripts = subscripts

        group_ids = _compute_group_ids(counts, Z)
        coef = coefficients.to(dtype=input1.dtype, device=input1.device)

        if subscripts == ("uv", "u", "v"):
            assert input1.shape == (C, u, v)
            assert input2.shape == (Z, u)
            w_exp = input1.index_select(0, group_ids)
            return coef * torch.einsum("zuv,zu->zv", w_exp, input2)
        elif subscripts == ("uv", "v", "u"):
            assert input1.shape == (C, u, v)
            assert input2.shape == (Z, v)
            w_exp = input1.index_select(0, group_ids)
            return coef * torch.einsum("zuv,zv->zu", w_exp, input2)
        elif subscripts == ("u", "uv", "v"):
            assert input1.shape == (Z, u)
            assert input2.shape == (C, u, v)
            w_exp = input2.index_select(0, group_ids)
            return coef * torch.einsum("zu,zuv->zv", input1, w_exp)
        elif subscripts == ("u", "vu", "v"):
            assert input1.shape == (Z, u)
            assert input2.shape == (C, v, u)
            w_exp = input2.index_select(0, group_ids)
            return coef * torch.einsum("zu,zvu->zv", input1, w_exp)
        elif subscripts == ("u", "v", "uv"):
            assert input1.shape == (Z, u)
            assert input2.shape == (Z, v)
            outer = coef * torch.einsum("zu,zv->zuv", input1, input2)
            out = torch.zeros((C, u, v), dtype=outer.dtype, device=outer.device)
            return out.index_add(0, group_ids, outer)
        elif subscripts == ("u", "v", "vu"):
            assert input1.shape == (Z, u)
            assert input2.shape == (Z, v)
            outer = coef * torch.einsum("zu,zv->zvu", input1, input2)
            out = torch.zeros((C, v, u), dtype=outer.dtype, device=outer.device)
            return out.index_add(0, group_ids, outer)
        else:
            raise ValueError(f"Unsupported canonical subscripts: {subscripts}")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input1, input2, counts, coefficients = ctx.saved_tensors
        u, v, C, Z = ctx.u, ctx.v, ctx.C, ctx.Z
        subscripts = ctx.subscripts
        grad_output = grad_output.contiguous()

        grad_in1 = None
        grad_in2 = None

        if subscripts == ("uv", "u", "v"):
            if ctx.needs_input_grad[0]:
                grad_in1 = _MockIndexedLinearFn.apply(
                    input2,
                    grad_output,
                    counts,
                    u,
                    v,
                    C,
                    Z,
                    ("u", "v", "uv"),
                    coefficients,
                )
            if ctx.needs_input_grad[1]:
                grad_in2 = _MockIndexedLinearFn.apply(
                    input1,
                    grad_output,
                    counts,
                    u,
                    v,
                    C,
                    Z,
                    ("uv", "v", "u"),
                    coefficients,
                )
        elif subscripts == ("uv", "v", "u"):
            if ctx.needs_input_grad[0]:
                grad_in1 = _MockIndexedLinearFn.apply(
                    grad_output,
                    input2,
                    counts,
                    u,
                    v,
                    C,
                    Z,
                    ("u", "v", "uv"),
                    coefficients,
                )
            if ctx.needs_input_grad[1]:
                grad_in2 = _MockIndexedLinearFn.apply(
                    input1,
                    grad_output,
                    counts,
                    u,
                    v,
                    C,
                    Z,
                    ("uv", "u", "v"),
                    coefficients,
                )
        elif subscripts == ("u", "uv", "v"):
            if ctx.needs_input_grad[0]:
                grad_in1 = _MockIndexedLinearFn.apply(
                    grad_output,
                    input2,
                    counts,
                    v,
                    u,
                    C,
                    Z,
                    ("u", "vu", "v"),
                    coefficients,
                )
            if ctx.needs_input_grad[1]:
                grad_in2 = _MockIndexedLinearFn.apply(
                    input1,
                    grad_output,
                    counts,
                    u,
                    v,
                    C,
                    Z,
                    ("u", "v", "uv"),
                    coefficients,
                )
        elif subscripts == ("u", "vu", "v"):
            if ctx.needs_input_grad[0]:
                grad_in1 = _MockIndexedLinearFn.apply(
                    grad_output,
                    input2,
                    counts,
                    v,
                    u,
                    C,
                    Z,
                    ("u", "uv", "v"),
                    coefficients,
                )
            if ctx.needs_input_grad[1]:
                grad_in2 = _MockIndexedLinearFn.apply(
                    input1,
                    grad_output,
                    counts,
                    u,
                    v,
                    C,
                    Z,
                    ("u", "v", "vu"),
                    coefficients,
                )
        elif subscripts == ("u", "v", "uv"):
            if ctx.needs_input_grad[0]:
                grad_in1 = _MockIndexedLinearFn.apply(
                    grad_output,
                    input2,
                    counts,
                    u,
                    v,
                    C,
                    Z,
                    ("uv", "v", "u"),
                    coefficients,
                )
            if ctx.needs_input_grad[1]:
                grad_in2 = _MockIndexedLinearFn.apply(
                    grad_output,
                    input1,
                    counts,
                    u,
                    v,
                    C,
                    Z,
                    ("uv", "u", "v"),
                    coefficients,
                )
        elif subscripts == ("u", "v", "vu"):
            if ctx.needs_input_grad[0]:
                grad_in1 = _MockIndexedLinearFn.apply(
                    grad_output,
                    input2,
                    counts,
                    v,
                    u,
                    C,
                    Z,
                    ("uv", "u", "v"),
                    coefficients,
                )
            if ctx.needs_input_grad[1]:
                grad_in2 = _MockIndexedLinearFn.apply(
                    grad_output,
                    input1,
                    counts,
                    v,
                    u,
                    C,
                    Z,
                    ("uv", "v", "u"),
                    coefficients,
                )

        return grad_in1, grad_in2, None, None, None, None, None, None, None


def mock_indexed_linear(
    input1: torch.Tensor,
    input2: torch.Tensor,
    counts: torch.Tensor,
    u: int,
    v: int,
    C: int,
    Z: int,
    subscripts: Tuple[str, str, str],
    coefficients: torch.Tensor,
) -> torch.Tensor:
    return _MockIndexedLinearFn.apply(
        input1, input2, counts, u, v, C, Z, subscripts, coefficients
    )


@contextmanager
def patched_cue_ops(fn=mock_indexed_linear):
    fake_mod = types.ModuleType("cuequivariance_ops_torch")
    fake_mod.indexed_linear = fn
    had_prev = "cuequivariance_ops_torch" in sys.modules
    prev_mod = sys.modules.get("cuequivariance_ops_torch")
    sys.modules["cuequivariance_ops_torch"] = fake_mod
    try:
        with patch.object(sp_mod, "HAS_CUE_OPS", True):
            yield fake_mod
    finally:
        if had_prev:
            sys.modules["cuequivariance_ops_torch"] = prev_mod
        else:
            sys.modules.pop("cuequivariance_ops_torch", None)


def _assert_fwd_bwd_bwd_match(
    m_test: torch.nn.Module,
    m_ref: torch.nn.Module,
    inputs,
    input_indices=None,
    output_shapes=None,
    output_indices=None,
    atol: float = 1e-12,
    rtol: float = 1e-12,
):
    input_indices = {} if input_indices is None else input_indices
    output_shapes = {} if output_shapes is None else output_shapes
    output_indices = {} if output_indices is None else output_indices

    in_test = [t.clone().detach().requires_grad_(True) for t in inputs]
    in_ref = [t.clone().detach().requires_grad_(True) for t in inputs]

    out_test = m_test(list(in_test), input_indices, output_shapes, output_indices)
    ref_output_indices = {k: v.to(dtype=torch.int64) for k, v in output_indices.items()}
    out_ref = m_ref(list(in_ref), input_indices, output_shapes, ref_output_indices)

    assert len(out_test) == len(out_ref)
    for y_t, y_r in zip(out_test, out_ref):
        assert y_t.shape == y_r.shape
        torch.testing.assert_close(y_t, y_r, atol=atol, rtol=rtol)

    cotangents = [torch.randn_like(y_r) for y_r in out_ref]
    loss_t = sum((y * g).sum() for y, g in zip(out_test, cotangents))
    loss_r = sum((y * g).sum() for y, g in zip(out_ref, cotangents))

    grads_t = torch.autograd.grad(loss_t, in_test, create_graph=True)
    grads_r = torch.autograd.grad(loss_r, in_ref, create_graph=True)

    for g_t, g_r in zip(grads_t, grads_r):
        torch.testing.assert_close(g_t, g_r, atol=atol, rtol=rtol)

    grad_cotangents = [torch.randn_like(g_r) for g_r in grads_r]
    loss2_t = sum((g * v).sum() for g, v in zip(grads_t, grad_cotangents))
    loss2_r = sum((g * v).sum() for g, v in zip(grads_r, grad_cotangents))

    grads2_t = torch.autograd.grad(loss2_t, in_test)
    grads2_r = torch.autograd.grad(loss2_r, in_ref)

    for g2_t, g2_r in zip(grads2_t, grads2_r):
        torch.testing.assert_close(g2_t, g2_r, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "subscripts,in1_shape,in2_shape",
    [
        (("uv", "u", "v"), (4, 3, 2), (6, 3)),
        (("uv", "v", "u"), (4, 3, 2), (6, 2)),
        (("u", "uv", "v"), (6, 3), (4, 3, 2)),
        (("u", "vu", "v"), (6, 3), (4, 2, 3)),
        (("u", "v", "uv"), (6, 3), (6, 2)),
        (("u", "v", "vu"), (6, 3), (6, 2)),
    ],
)
def test_mock_indexed_linear_gradcheck_all_subscripts(subscripts, in1_shape, in2_shape):
    torch.manual_seed(42)
    counts = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    coef = torch.tensor(0.5773502691896258, dtype=torch.float64)
    in1 = torch.randn(in1_shape, dtype=torch.float64, requires_grad=True)
    in2 = torch.randn(in2_shape, dtype=torch.float64, requires_grad=True)

    def fn(a, b):
        return mock_indexed_linear(a, b, counts, 3, 2, 4, 6, subscripts, coef)

    assert torch.autograd.gradcheck(fn, (in1, in2), eps=1e-6, atol=1e-10, rtol=1e-8)
    assert torch.autograd.gradgradcheck(fn, (in1, in2), eps=1e-6, atol=1e-10, rtol=1e-8)


def test_trailing_shared_operand_batch_size():
    """Bug 1: A trailing shared (1, D) operand must not overwrite batch_size back to 1."""
    torch.manual_seed(0)
    e1 = cue.descriptors.linear(
        cue.Irreps("O3", "3x0e + 2x1o"),
        cue.Irreps("O3", "4x0e + 2x1o"),
    )
    e2 = cue.descriptors.linear(
        cue.Irreps("O3", "2x0e + 1x1o"),
        cue.Irreps("O3", "4x0e + 2x1o"),
    )
    _, d1 = e1.polynomial.operations[0]
    _, d2 = e2.polynomial.operations[0]

    # Polynomial with 4 inputs:
    # input 0: indexed weights for op1 (C=3, W1)
    # input 1: batched features for op1 (B=7, D1)
    # input 2: indexed weights for op2 (C=3, W2)
    # input 3: trailing shared features for op2 (1, D2) -> size == 1!
    poly = cue.SegmentedPolynomial(
        inputs=[
            e1.polynomial.inputs[0],
            e1.polynomial.inputs[1],
            e2.polynomial.inputs[0],
            e2.polynomial.inputs[1],
        ],
        outputs=[e1.polynomial.outputs[0]],
        operations=[
            (cue.Operation([0, 1, 4]), d1),
            (cue.Operation([2, 3, 4]), d2),
        ],
    )

    batch_size = 7
    num_species = 3
    w1 = torch.randn(num_species, poly.inputs[0].size, dtype=torch.float64)
    x1 = torch.randn(batch_size, poly.inputs[1].size, dtype=torch.float64)
    w2 = torch.randn(num_species, poly.inputs[2].size, dtype=torch.float64)
    x2_shared = torch.randn(1, poly.inputs[3].size, dtype=torch.float64)
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.int64)

    with patched_cue_ops():
        m_idx = SegmentedPolynomialIndexedLinear(poly)
        m_ref = SegmentedPolynomialNaive(poly)

        out_idx = m_idx([w1, x1, w2, x2_shared], input_indices={0: idx, 2: idx})[0]
        assert out_idx.shape == (batch_size, poly.outputs[0].size)

        _assert_fwd_bwd_bwd_match(
            m_idx,
            m_ref,
            [w1, x1, w2, x2_shared],
            input_indices={0: idx, 2: idx},
            atol=1e-13,
            rtol=1e-13,
        )

    # Also test single-op where Z operand itself is shared (1, D) while input_indices has B=7
    poly_single = e1.polynomial
    x1_shared = torch.randn(1, poly_single.inputs[1].size, dtype=torch.float64)
    with patched_cue_ops():
        m_idx_s = SegmentedPolynomialIndexedLinear(poly_single)
        m_ref_s = SegmentedPolynomialNaive(poly_single)
        _assert_fwd_bwd_bwd_match(
            m_idx_s,
            m_ref_s,
            [w1, x1_shared],
            input_indices={0: idx},
            atol=1e-13,
            rtol=1e-13,
        )


@pytest.mark.parametrize(
    "group,ir_in,ir_out",
    [
        # Multi-path into single output segment (len(o) > 1)
        ("O3", "2x0e + 3x0e", "4x0e"),
        ("O3", "2x0e + 3x1o + 4x0e", "5x0e + 2x1o"),
        ("O3", "4x0e", "2x0e + 3x0e"),
        # Unused output segments (len(o) == 0) at end, start, and middle + multi-path
        ("O3", "4x0e", "4x0e + 2x1o"),
        ("O3", "4x0e", "2x1o + 4x0e"),
        ("O3", "2x0e + 3x1o + 2x0e", "2x1e + 4x0e + 2x1o"),
        # Zero-multiplicity irreps
        ("SU2", "2x1/2 + 0x1 + 0x1/2 + 1", "3x1/2 + 0x2 + 2x1 + 1x2"),
    ],
)
def test_multipath_and_unused_output_segments(group, ir_in, ir_out):
    """IndexedLinear.forward must sum multi-path segments and zero-fill unused segments."""
    torch.manual_seed(7)
    poly = cue.descriptors.linear(
        cue.Irreps(group, ir_in),
        cue.Irreps(group, ir_out),
    ).polynomial

    w = torch.randn(4, poly.inputs[0].size, dtype=torch.float64)
    x = torch.randn(8, poly.inputs[1].size, dtype=torch.float64)
    idx = torch.tensor([0, 0, 1, 1, 2, 3, 3, 3], dtype=torch.int64)

    with patched_cue_ops():
        m_idx = SegmentedPolynomialIndexedLinear(poly)
        m_ref = SegmentedPolynomialNaive(poly)
        _assert_fwd_bwd_bwd_match(
            m_idx,
            m_ref,
            [w, x],
            input_indices={0: idx},
            atol=1e-13,
            rtol=1e-13,
        )


def test_completely_empty_paths_linear():
    """When no input irrep couples to any output irrep (0 paths total), output & grads are zero."""
    poly = cue.descriptors.linear(
        cue.Irreps("O3", "4x0e"),
        cue.Irreps("O3", "2x1o"),
    ).polynomial
    w = torch.randn(3, poly.inputs[0].size, dtype=torch.float64, requires_grad=True)
    x = torch.randn(5, poly.inputs[1].size, dtype=torch.float64, requires_grad=True)
    idx = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)

    with patched_cue_ops():
        m_idx = SegmentedPolynomialIndexedLinear(poly)
        (out,) = m_idx([w, x], input_indices={0: idx})
        assert out.shape == (5, poly.outputs[0].size)
        assert torch.all(out == 0)
        out.sum().backward()
        assert w.grad is not None and torch.all(w.grad == 0)
        assert x.grad is not None and torch.all(x.grad == 0)


@pytest.mark.parametrize("subscripts", list(SUBDICT.keys()))
def test_all_12_subdict_operations(subscripts: str):
    """Verify every supported subscript in SUBDICT (input0-indexed, input1-indexed, output-indexed)."""
    torch.manual_seed(11)
    dim_map = {"u": 3, "v": 2, "w": 4}
    op_subs = subscripts.split(",")
    d = cue.SegmentedTensorProduct.from_subscripts(f"{subscripts}+")
    for oid, s in enumerate(op_subs):
        seg_shape = tuple(dim_map[ch] for ch in s)
        d.add_segment(oid, seg_shape)
        d.add_segment(oid, seg_shape)
    d.add_path(0, 0, 0, c=0.75)
    d.add_path(1, 1, 0, c=-0.35)  # multi-path into segment 0; segment 1 left unused!

    poly = cue.SegmentedPolynomial(
        inputs=[d.operands[0], d.operands[1]],
        outputs=[d.operands[2]],
        operations=[(cue.Operation([0, 1, 2]), d)],
    )

    sig = SUBDICT[subscripts]
    c_pos = sig[2][0]
    batch_size = 7
    num_classes = 3
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.int64)

    in0_batch = num_classes if c_pos == 0 else batch_size
    in1_batch = num_classes if c_pos == 1 else batch_size
    in0 = torch.randn(in0_batch, poly.inputs[0].size, dtype=torch.float64)
    in1 = torch.randn(in1_batch, poly.inputs[1].size, dtype=torch.float64)

    input_indices = {}
    output_shapes = {}
    output_indices = {}
    if c_pos == 0:
        input_indices = {0: idx}
    elif c_pos == 1:
        input_indices = {1: idx}
    else:
        output_shapes = {0: torch.empty(num_classes, poly.outputs[0].size)}
        output_indices = {0: idx}

    with patched_cue_ops():
        m_idx = SegmentedPolynomialIndexedLinear(poly)
        m_ref = SegmentedPolynomialNaive(poly)
        _assert_fwd_bwd_bwd_match(
            m_idx,
            m_ref,
            [in0, in1],
            input_indices=input_indices,
            output_shapes=output_shapes,
            output_indices=output_indices,
            atol=1e-13,
            rtol=1e-13,
        )


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_unsorted_electrolyte_species_sort_indices(idx_dtype: torch.dtype):
    """Bug 3: Default rejects unsorted indices in eager; options={'sort_indices': True} matches Naive < 1e-12."""
    torch.manual_seed(2025)
    poly = cue.descriptors.linear(
        cue.Irreps("O3", "4x0e + 3x1o + 2x0e + 2x2e"),
        cue.Irreps("O3", "3x0e + 2x1e + 4x1o + 2x2e"),
    ).polynomial

    num_species = 5  # species 4 intentionally absent from batch to test 0-count class
    species = torch.tensor([2, 0, 1, 1, 2, 0, 3], dtype=idx_dtype)
    w = torch.randn(num_species, poly.inputs[0].size, dtype=torch.float64)
    x = torch.randn(species.shape[0], poly.inputs[1].size, dtype=torch.float64)

    with patched_cue_ops():
        # 1. Default sort_indices=False raises AssertionError on unsorted indices
        m_strict = SegmentedPolynomialIndexedLinear(poly)
        with pytest.raises(AssertionError, match="non-sorted indices"):
            m_strict([w, x], input_indices={0: species})

        # 2. options={"sort_indices": True} handles unsorted indices in forward + 1st & 2nd backward
        m_sorted = SegmentedPolynomialIndexedLinear(
            poly, options={"sort_indices": True}
        )
        m_ref = SegmentedPolynomialNaive(poly)
        _assert_fwd_bwd_bwd_match(
            m_sorted,
            m_ref,
            [w, x],
            input_indices={0: species},
            atol=1e-12,
            rtol=1e-12,
        )

        # 3. Verify options forwarding and round-trip pickling/deepcopy via SegmentedPolynomial
        sp_sorted = cuet.SegmentedPolynomial(
            poly, method="indexed_linear", options={"sort_indices": True}
        )
        assert isinstance(sp_sorted.m, SegmentedPolynomialIndexedLinear)
        assert sp_sorted.m.sort_indices is True

        sp_copy = copy.deepcopy(sp_sorted)
        assert sp_copy.m.sort_indices is True
        sp_unpickled = pickle.loads(pickle.dumps(sp_sorted.m))
        assert sp_unpickled.options.get("sort_indices") is True

        # 4. Also test unsorted output-indexed operation ("uv,uw,vw")
        d_out = cue.SegmentedTensorProduct.from_subscripts("uv,uw,vw+")
        d_out.add_segment(0, (3, 4))
        d_out.add_segment(1, (3, 2))
        d_out.add_segment(2, (4, 2))
        d_out.add_path(0, 0, 0, c=0.5773502691896258)
        poly_out = cue.SegmentedPolynomial(
            inputs=[d_out.operands[0], d_out.operands[1]],
            outputs=[d_out.operands[2]],
            operations=[(cue.Operation([0, 1, 2]), d_out)],
        )
        a = torch.randn(species.shape[0], poly_out.inputs[0].size, dtype=torch.float64)
        b = torch.randn(species.shape[0], poly_out.inputs[1].size, dtype=torch.float64)
        out_shape_dict = {0: torch.empty(num_species, poly_out.outputs[0].size)}
        m_out_sorted = SegmentedPolynomialIndexedLinear(
            poly_out, options={"sort_indices": True}
        )
        m_out_ref = SegmentedPolynomialNaive(poly_out)
        _assert_fwd_bwd_bwd_match(
            m_out_sorted,
            m_out_ref,
            [a, b],
            output_shapes=out_shape_dict,
            output_indices={0: species},
            atol=1e-12,
            rtol=1e-12,
        )

        # 5. Also test scatter output_indices and shared (1, D) output reduction on input-indexed op
        scatter_out_idx = torch.tensor([1, 0, 2, 1, 0, 2, 1], dtype=idx_dtype)
        scatter_out_shape = {0: torch.empty(3, poly.outputs[0].size)}
        _assert_fwd_bwd_bwd_match(
            m_sorted,
            m_ref,
            [w, x],
            input_indices={0: species},
            output_shapes=scatter_out_shape,
            output_indices={0: scatter_out_idx},
            atol=1e-12,
            rtol=1e-12,
        )
        shared_out_shape = {0: torch.empty(1, poly.outputs[0].size)}
        _assert_fwd_bwd_bwd_match(
            m_sorted,
            m_ref,
            [w, x],
            input_indices={0: species},
            output_shapes=shared_out_shape,
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.parametrize("batch_size", [0, 1])
@pytest.mark.parametrize("sort_indices", [False, True])
def test_empty_and_singleton_batch_indices(batch_size: int, sort_indices: bool):
    """Edge cases v.numel() == 0 and v.numel() == 1 must work without errors or sync crashes."""
    torch.manual_seed(99)
    poly = cue.descriptors.linear(
        cue.Irreps("O3", "3x0e + 2x1o"),
        cue.Irreps("O3", "2x0e + 3x1o"),
    ).polynomial

    w = torch.randn(3, poly.inputs[0].size, dtype=torch.float64)
    x = torch.randn(batch_size, poly.inputs[1].size, dtype=torch.float64)
    idx = torch.tensor([1][:batch_size], dtype=torch.int64)

    with patched_cue_ops():
        m_idx = SegmentedPolynomialIndexedLinear(
            poly, options={"sort_indices": sort_indices}
        )
        m_ref = SegmentedPolynomialNaive(poly)
        _assert_fwd_bwd_bwd_match(
            m_idx,
            m_ref,
            [w, x],
            input_indices={0: idx},
            atol=1e-13,
            rtol=1e-13,
        )


@pytest.mark.parametrize("sort_indices", [False, True])
def test_torch_compile_fullgraph_zero_sync(sort_indices: bool):
    """Bug 3: torch.compile(..., backend='aot_eager', fullgraph=True) with capture_scalar_outputs=False."""
    torch.manual_seed(123)
    poly = cue.descriptors.linear(
        cue.Irreps("O3", "3x0e + 2x1o + 2x0e"),
        cue.Irreps("O3", "4x0e + 1x1e + 2x1o"),
    ).polynomial

    species = (
        torch.tensor([2, 0, 1, 1, 2, 0, 3], dtype=torch.int64)
        if sort_indices
        else torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int64)
    )
    w = torch.randn(4, poly.inputs[0].size, dtype=torch.float64, requires_grad=True)
    x = torch.randn(
        species.shape[0], poly.inputs[1].size, dtype=torch.float64, requires_grad=True
    )

    prev_scalar = torch._dynamo.config.capture_scalar_outputs
    prev_dynamic = torch._dynamo.config.capture_dynamic_output_shape_ops
    torch._dynamo.reset()
    try:
        torch._dynamo.config.capture_scalar_outputs = False
        torch._dynamo.config.capture_dynamic_output_shape_ops = False

        with patched_cue_ops():
            m_idx = SegmentedPolynomialIndexedLinear(
                poly, options={"sort_indices": sort_indices}
            )
            m_ref = SegmentedPolynomialNaive(poly)
            compiled = torch.compile(m_idx, backend="aot_eager", fullgraph=True)

            (y_comp,) = compiled([w, x], input_indices={0: species})
            w_ref = w.clone().detach().requires_grad_(True)
            x_ref = x.clone().detach().requires_grad_(True)
            (y_ref,) = m_ref(list([w_ref, x_ref]), {0: species}, {}, {})

            torch.testing.assert_close(y_comp, y_ref, atol=1e-12, rtol=1e-12)

            cot = torch.randn_like(y_ref)
            (gw_comp, gx_comp) = torch.autograd.grad((y_comp * cot).sum(), (w, x))
            (gw_ref, gx_ref) = torch.autograd.grad((y_ref * cot).sum(), (w_ref, x_ref))
            torch.testing.assert_close(gw_comp, gw_ref, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(gx_comp, gx_ref, atol=1e-12, rtol=1e-12)
    finally:
        torch._dynamo.config.capture_scalar_outputs = prev_scalar
        torch._dynamo.config.capture_dynamic_output_shape_ops = prev_dynamic
        torch._dynamo.reset()


def test_cpu_runtime_fallback_in_segmented_polynomial():
    """Bug 2: SegmentedPolynomial(method='indexed_linear') must fall back to naive on CPU inputs."""
    torch.manual_seed(321)
    poly = cue.descriptors.linear(
        cue.Irreps("O3", "3x0e + 2x1o"),
        cue.Irreps("O3", "4x0e + 2x1o"),
    ).polynomial

    def cuda_only_indexed_linear(*args, **kwargs):
        raise RuntimeError("cuequivariance_ops_torch.indexed_linear is CUDA-only")

    w_cpu = torch.randn(3, poly.inputs[0].size, dtype=torch.float64, requires_grad=True)
    x_cpu = torch.randn(6, poly.inputs[1].size, dtype=torch.float64, requires_grad=True)
    idx_cpu = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)

    with patched_cue_ops(fn=cuda_only_indexed_linear):
        sp = cuet.SegmentedPolynomial(poly, method="indexed_linear")
        assert sp.method == "indexed_linear"
        assert isinstance(sp.m, SegmentedPolynomialIndexedLinear)

        with pytest.warns(
            UserWarning,
            match="indexed_linear is not supported on CPU. Falling back to naive implementation.",
        ):
            (y_cpu,) = sp([w_cpu, x_cpu], input_indices={0: idx_cpu})

        sp_ref = cuet.SegmentedPolynomial(poly, method="naive")
        w_ref = w_cpu.clone().detach().requires_grad_(True)
        x_ref = x_cpu.clone().detach().requires_grad_(True)
        (y_ref,) = sp_ref([w_ref, x_ref], input_indices={0: idx_cpu})

        torch.testing.assert_close(y_cpu, y_ref, atol=1e-13, rtol=1e-13)

        cot = torch.randn_like(y_ref)
        (gw_cpu, gx_cpu) = torch.autograd.grad((y_cpu * cot).sum(), (w_cpu, x_cpu))
        (gw_ref, gx_ref) = torch.autograd.grad((y_ref * cot).sum(), (w_ref, x_ref))
        torch.testing.assert_close(gw_cpu, gw_ref, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(gx_cpu, gx_ref, atol=1e-13, rtol=1e-13)
