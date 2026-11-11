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
import sys
import types

import pytest
import torch

import cuequivariance_torch as cuet

ALL_DTYPES = [torch.float32, torch.float64, torch.bfloat16, torch.float16]


def _tols(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float64:
        return 1e-11, 1e-11
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 5e-3, 5e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    raise ValueError(f"Unsupported dtype: {dtype}")


# ==============================================================================
# 1. Basic CPU execution, dtypes, shapes, aux outputs, and use_fallback flag
# ==============================================================================


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("four_d", [False, True])
def test_triangle_attention_cpu_dtypes_and_ranks(dtype: torch.dtype, four_d: bool):
    torch.manual_seed(101)
    B, N, H, Q, K, D = 1 if four_d else 2, 5, 2, 5, 6, 8
    bias_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

    q = torch.randn(B, N, H, Q, D, dtype=dtype, requires_grad=True)
    k = torch.randn(B, N, H, K, D, dtype=dtype, requires_grad=True)
    v = torch.randn(B, N, H, K, D, dtype=dtype, requires_grad=True)
    bias = torch.randn(B, 1, H, Q, K, dtype=bias_dtype, requires_grad=True)
    mask = torch.ones(B, N, 1, 1, K, dtype=torch.bool)
    mask[..., -2:] = False

    q_in = q.squeeze(0) if four_d else q
    k_in = k.squeeze(0) if four_d else k
    v_in = v.squeeze(0) if four_d else v
    bias_in = bias.squeeze(0) if four_d else bias
    mask_in = mask.squeeze(0) if four_d else mask

    out, lse, max_val = cuet.triangle_attention(
        q_in, k_in, v_in, bias_in, mask=mask_in, return_aux=True
    )
    expected_shape = (N, H, Q, D) if four_d else (B, N, H, Q, D)
    expected_aux_shape = (N, H, Q, 1) if four_d else (B, N, H, Q, 1)

    assert out.shape == expected_shape
    assert out.dtype == dtype
    assert lse.shape == expected_aux_shape
    assert lse.dtype == torch.float32
    assert max_val.shape == expected_aux_shape
    assert max_val.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    assert torch.isfinite(max_val).all()

    # Prefix mask via kv_lengths must match dense mask
    kv_lengths = cuet.mask_to_kv_lengths(mask)
    kv_in = kv_lengths.squeeze(0) if four_d else kv_lengths
    out_kv = cuet.triangle_attention(
        q_in, k_in, v_in, bias_in, kv_lengths=kv_in, return_aux=False
    )
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(out, out_kv, rtol=rtol, atol=atol)

    # Clamping kv_lengths > K must match unmasked attention
    kv_over = torch.full_like(kv_in, K + 10)
    out_over = cuet.triangle_attention(q_in, k_in, v_in, bias_in, kv_lengths=kv_over)
    out_unmasked = cuet.triangle_attention(q_in, k_in, v_in, bias_in)
    torch.testing.assert_close(out_over, out_unmasked, rtol=rtol, atol=atol)

    loss = out.float().square().sum()
    loss.backward()
    for t in (q, k, v, bias):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("with_weights", [False, True])
def test_triangle_multiplicative_update_cpu_dtypes(
    dtype: torch.dtype, direction: str, with_weights: bool
):
    torch.manual_seed(202)
    B, N, C, D = 2, 5, 8, (6 if with_weights else 8)
    x = torch.randn(B, N, N, C, dtype=dtype, requires_grad=True) * 0.5
    mask = (torch.rand(B, N, N) > 0.25).to(dtype)

    kwargs = {}
    params = [x]
    if with_weights:
        kwargs = dict(
            norm_in_weight=torch.randn(C, dtype=dtype, requires_grad=True),
            norm_in_bias=torch.randn(C, dtype=dtype, requires_grad=True),
            p_in_weight=torch.randn(2 * D, C, dtype=dtype, requires_grad=True) * 0.3,
            p_in_bias=torch.randn(2 * D, dtype=dtype, requires_grad=True) * 0.1,
            g_in_weight=torch.randn(2 * D, C, dtype=dtype, requires_grad=True) * 0.3,
            g_in_bias=torch.randn(2 * D, dtype=dtype, requires_grad=True) * 0.1,
            norm_out_weight=torch.randn(D, dtype=dtype, requires_grad=True),
            norm_out_bias=torch.randn(D, dtype=dtype, requires_grad=True),
            p_out_weight=torch.randn(C, D, dtype=dtype, requires_grad=True) * 0.3,
            p_out_bias=torch.randn(C, dtype=dtype, requires_grad=True) * 0.1,
            g_out_weight=torch.randn(C, C, dtype=dtype, requires_grad=True) * 0.3,
            g_out_bias=torch.randn(C, dtype=dtype, requires_grad=True) * 0.1,
        )
        params.extend(kwargs.values())

    out = cuet.triangle_multiplicative_update(
        x, direction=direction, mask=mask, **kwargs
    )
    assert out.shape == (B, N, N, C)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()

    grads = torch.autograd.grad(out.float().square().mean(), params)
    for g in grads:
        assert torch.isfinite(g).all()


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("norm_kind", ["layer_norm", "rms_norm"])
def test_attention_pair_bias_cpu_dtypes(dtype: torch.dtype, norm_kind: str):
    torch.manual_seed(303)
    B, M, N, H, DH, D, z_dim = 2, 2, 5, 2, 4, 8, 6
    inner = H * DH
    is_ln = norm_kind == "layer_norm"

    single_repr = torch.randn(B * M, N, D, dtype=dtype, requires_grad=True)
    pair_repr = torch.randn(B, N, N, z_dim, dtype=dtype, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, -1] = False

    w_ln_a = torch.randn(D, dtype=dtype, requires_grad=True)
    b_ln_a = torch.randn(D, dtype=dtype, requires_grad=True) if is_ln else None
    w_proj_q = torch.randn(inner, D, dtype=dtype, requires_grad=True) * 0.3
    b_proj_q = torch.randn(inner, dtype=dtype, requires_grad=True) * 0.1
    w_proj_k = torch.randn(inner, D, dtype=dtype, requires_grad=True) * 0.3
    w_proj_v = torch.randn(inner, D, dtype=dtype, requires_grad=True) * 0.3
    w_proj_g = torch.randn(inner, D, dtype=dtype, requires_grad=True) * 0.3
    w_proj_o = torch.randn(D, inner, dtype=dtype, requires_grad=True) * 0.3
    w_proj_z = torch.randn(H, z_dim, dtype=dtype, requires_grad=True) * 0.3
    w_ln_z = torch.randn(z_dim, dtype=dtype, requires_grad=True)
    b_ln_z = torch.randn(z_dim, dtype=dtype, requires_grad=True) if is_ln else None
    w_ln_q = torch.randn(inner, dtype=dtype, requires_grad=True)
    b_ln_q = torch.randn(inner, dtype=dtype, requires_grad=True) if is_ln else None
    w_ln_k = torch.randn(inner, dtype=dtype, requires_grad=True)
    b_ln_k = torch.randn(inner, dtype=dtype, requires_grad=True) if is_ln else None

    out, z_proj = cuet.attention_pair_bias(
        single_repr=single_repr,
        pair_repr=pair_repr,
        mask=mask,
        num_heads=H,
        w_ln_a=w_ln_a,
        b_ln_a=b_ln_a,
        w_proj_q=w_proj_q,
        b_proj_q=b_proj_q,
        w_proj_k=w_proj_k,
        w_proj_v=w_proj_v,
        w_proj_g=w_proj_g,
        w_proj_o=w_proj_o,
        w_proj_z=w_proj_z,
        w_ln_z=w_ln_z,
        b_ln_z=b_ln_z,
        return_z_proj=True,
        w_ln_q=w_ln_q,
        b_ln_q=b_ln_q,
        w_ln_k=w_ln_k,
        b_ln_k=b_ln_k,
        norm_kind=norm_kind,
    )
    assert out.shape == (B * M, N, D)
    assert out.dtype == dtype
    assert z_proj is not None
    assert z_proj.shape == (B, H, N, N)
    assert z_proj.dtype == dtype
    assert torch.isfinite(out).all()
    assert torch.isfinite(z_proj).all()

    loss = out.float().square().mean() + z_proj.float().square().mean()
    loss.backward()
    assert single_repr.grad is not None and torch.isfinite(single_repr.grad).all()
    assert pair_repr.grad is not None and torch.isfinite(pair_repr.grad).all()


def test_use_fallback_bypasses_ops_backend(monkeypatch):
    """When use_fallback=True, the pure-PyTorch fallback is used even if ops backend exists."""
    backend_module = types.ModuleType("cuequivariance_ops_torch")

    def fail_spy(*args, **kwargs):
        raise AssertionError("Backend should not be called when use_fallback=True")

    backend_module.triangle_attention = fail_spy
    backend_module.triangle_multiplicative_update = fail_spy
    backend_module.attention_pair_bias = fail_spy
    monkeypatch.setitem(sys.modules, "cuequivariance_ops_torch", backend_module)

    B, N, H, D = 1, 4, 2, 4
    q = torch.randn(B, N, H, N, D)
    bias = torch.randn(B, 1, H, N, N)
    out_attn = cuet.triangle_attention(q, q, q, bias, use_fallback=True)
    assert out_attn.shape == (B, N, H, N, D)

    x = torch.randn(B, N, N, 8)
    out_tm = cuet.triangle_multiplicative_update(x, use_fallback=True)
    assert out_tm.shape == (B, N, N, 8)

    s = torch.randn(B, N, 8)
    z = torch.randn(B, N, N, 4)
    out_apb, _ = cuet.attention_pair_bias(
        s,
        z,
        None,
        H,
        torch.ones(8),
        torch.zeros(8),
        torch.randn(H * D, 8),
        None,
        torch.randn(H * D, 8),
        torch.randn(H * D, 8),
        torch.randn(H * D, 8),
        torch.randn(8, H * D),
        torch.randn(H, 4),
        use_fallback=True,
    )
    assert out_apb.shape == (B, N, 8)


# ==============================================================================
# 2. S_N Residue Permutation Equivariance across FP32, FP64, BF16, FP16
# ==============================================================================


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_sn_equivariance_triangle_attention(dtype: torch.dtype):
    torch.manual_seed(404)
    B, N, H, D = 2, 6, 2, 8
    perm = torch.tensor([3, 0, 5, 1, 4, 2], dtype=torch.long)

    q = torch.randn(B, N, H, N, D, dtype=dtype)
    k = torch.randn(B, N, H, N, D, dtype=dtype)
    v = torch.randn(B, N, H, N, D, dtype=dtype)
    bias = torch.randn(B, 1, H, N, N, dtype=dtype)
    mask = torch.tensor(
        [
            [True, True, False, True, False, True],
            [False, True, True, False, True, True],
        ]
    ).view(B, 1, 1, 1, N) & torch.tensor([True, False, True, True, True, False]).view(
        1, N, 1, 1, 1
    )

    out, lse, max_val = cuet.triangle_attention(
        q, k, v, bias, mask=mask, return_aux=True
    )

    q_p = q[:, perm][:, :, :, perm, :]
    k_p = k[:, perm][:, :, :, perm, :]
    v_p = v[:, perm][:, :, :, perm, :]
    bias_p = bias[:, :, :, perm, :][:, :, :, :, perm]
    mask_p = mask[:, perm][:, :, :, :, perm]

    out_p, lse_p, max_p = cuet.triangle_attention(
        q_p, k_p, v_p, bias_p, mask=mask_p, return_aux=True
    )

    rtol, atol = _tols(dtype)
    torch.testing.assert_close(
        out_p, out[:, perm][:, :, :, perm, :], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        lse_p, lse[:, perm][:, :, :, perm, :], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        max_p, max_val[:, perm][:, :, :, perm, :], rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_sn_equivariance_triangle_multiplicative_update(
    dtype: torch.dtype, direction: str
):
    torch.manual_seed(505)
    B, N, C, D = 2, 6, 8, 6
    perm = torch.tensor([4, 1, 5, 0, 2, 3], dtype=torch.long)

    x = torch.randn(B, N, N, C, dtype=dtype) * 0.5
    node_mask = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]],
        dtype=dtype,
    )
    mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)

    weights = dict(
        norm_in_weight=torch.randn(C, dtype=dtype),
        norm_in_bias=torch.randn(C, dtype=dtype),
        p_in_weight=torch.randn(2 * D, C, dtype=dtype) * 0.3,
        p_in_bias=torch.randn(2 * D, dtype=dtype) * 0.1,
        g_in_weight=torch.randn(2 * D, C, dtype=dtype) * 0.3,
        g_in_bias=torch.randn(2 * D, dtype=dtype) * 0.1,
        norm_out_weight=torch.randn(D, dtype=dtype),
        norm_out_bias=torch.randn(D, dtype=dtype),
        p_out_weight=torch.randn(C, D, dtype=dtype) * 0.3,
        p_out_bias=torch.randn(C, dtype=dtype) * 0.1,
        g_out_weight=torch.randn(C, C, dtype=dtype) * 0.3,
        g_out_bias=torch.randn(C, dtype=dtype) * 0.1,
    )

    out = cuet.triangle_multiplicative_update(
        x, direction=direction, mask=mask, **weights
    )
    x_p = x[:, perm, :, :][:, :, perm, :]
    mask_p = mask[:, perm, :][:, :, perm]
    out_p = cuet.triangle_multiplicative_update(
        x_p, direction=direction, mask=mask_p, **weights
    )

    rtol, atol = _tols(dtype)
    torch.testing.assert_close(
        out_p, out[:, perm, :, :][:, :, perm, :], rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("multiplicity", [1, 3])
@pytest.mark.parametrize("per_replica_mask", [False, True])
def test_sn_equivariance_attention_pair_bias(
    dtype: torch.dtype, multiplicity: int, per_replica_mask: bool
):
    torch.manual_seed(606)
    B, M, N, H, DH, D, z_dim = 2, multiplicity, 6, 2, 4, 8, 5
    inner = H * DH
    perm = torch.tensor([2, 5, 0, 4, 1, 3], dtype=torch.long)

    single_repr = torch.randn(B * M, N, D, dtype=dtype)
    pair_repr = torch.randn(B, N, N, z_dim, dtype=dtype)
    mask_rows = B * M if per_replica_mask else B
    mask = torch.ones(mask_rows, N, dtype=torch.bool)
    mask[:, 2] = False
    mask[0, 5] = False

    params = dict(
        num_heads=H,
        w_ln_a=torch.randn(D, dtype=dtype),
        b_ln_a=torch.randn(D, dtype=dtype),
        w_proj_q=torch.randn(inner, D, dtype=dtype) * 0.3,
        b_proj_q=torch.randn(inner, dtype=dtype) * 0.1,
        w_proj_k=torch.randn(inner, D, dtype=dtype) * 0.3,
        w_proj_v=torch.randn(inner, D, dtype=dtype) * 0.3,
        w_proj_g=torch.randn(inner, D, dtype=dtype) * 0.3,
        w_proj_o=torch.randn(D, inner, dtype=dtype) * 0.3,
        w_proj_z=torch.randn(H, z_dim, dtype=dtype) * 0.3,
        b_proj_o=torch.randn(D, dtype=dtype) * 0.1,
        b_proj_z=torch.randn(H, dtype=dtype) * 0.1,
        w_ln_z=torch.randn(z_dim, dtype=dtype),
        b_ln_z=torch.randn(z_dim, dtype=dtype),
        b_proj_k=torch.randn(inner, dtype=dtype) * 0.1,
        b_proj_v=torch.randn(inner, dtype=dtype) * 0.1,
        w_ln_q=torch.randn(inner, dtype=dtype),
        b_ln_q=torch.randn(inner, dtype=dtype),
        w_ln_k=torch.randn(inner, dtype=dtype),
        b_ln_k=torch.randn(inner, dtype=dtype),
        b_proj_g=torch.randn(inner, dtype=dtype) * 0.1,
        return_z_proj=True,
    )

    out, z_proj = cuet.attention_pair_bias(
        single_repr=single_repr, pair_repr=pair_repr, mask=mask, **params
    )

    single_p = single_repr[:, perm, :]
    pair_p = pair_repr[:, perm, :, :][:, :, perm, :]
    mask_p = mask[:, perm]

    out_p, z_proj_p = cuet.attention_pair_bias(
        single_repr=single_p, pair_repr=pair_p, mask=mask_p, **params
    )

    rtol, atol = _tols(dtype)
    assert z_proj is not None and z_proj_p is not None
    torch.testing.assert_close(out_p, out[:, perm, :], rtol=rtol, atol=atol)
    torch.testing.assert_close(
        z_proj_p, z_proj[:, :, perm, :][:, :, :, perm], rtol=rtol, atol=atol
    )

    if M == 3:
        perm_m = torch.tensor([2, 0, 1], dtype=torch.long)
        single_m = single_repr.view(B, M, N, D)[:, perm_m].reshape(B * M, N, D)
        mask_m = (
            mask.view(B, M, N)[:, perm_m].reshape(B * M, N)
            if per_replica_mask
            else mask
        )
        out_m, _ = cuet.attention_pair_bias(
            single_repr=single_m, pair_repr=pair_repr, mask=mask_m, **params
        )
        expected_m = out.view(B, M, N, D)[:, perm_m].reshape(B * M, N, D)
        torch.testing.assert_close(out_m, expected_m, rtol=rtol, atol=atol)


# ==============================================================================
# 3. Zero-length rows (kv_lengths == 0 & all-zero mask): 0 NaNs in fwd, g1, g2
# ==============================================================================


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("mode", ["kv_lengths", "mask", "all_zero_mask"])
def test_triangle_attention_zero_length_rows_no_nan_1st_2nd_grad(
    dtype: torch.dtype, mode: str
):
    torch.manual_seed(707)
    B, N, H, Q, K, D = 2, 4, 2, 4, 5, 8
    q = (torch.randn(B, N, H, Q, D, dtype=dtype) * 0.4).requires_grad_(True)
    k = (torch.randn(B, N, H, K, D, dtype=dtype) * 0.4).requires_grad_(True)
    v = (torch.randn(B, N, H, K, D, dtype=dtype) * 0.4).requires_grad_(True)
    bias = (torch.randn(B, 1, H, Q, K, dtype=dtype) * 0.4).requires_grad_(True)

    if mode == "kv_lengths":
        kv_lengths = torch.tensor([[0, 3, 0, 5], [0, 0, 0, 0]], dtype=torch.int32).view(
            B, N, 1, 1, 1
        )
        out, lse, max_val = cuet.triangle_attention(
            q, k, v, bias, kv_lengths=kv_lengths, return_aux=True
        )
        zero_rows = kv_lengths.view(B, N) == 0
    elif mode == "mask":
        mask = torch.zeros(B, N, 1, 1, K, dtype=torch.bool)
        mask[0, 1, :, :, :3] = True
        mask[0, 3, :, :, 1:4] = True
        out, lse, max_val = cuet.triangle_attention(
            q, k, v, bias, mask=mask, return_aux=True
        )
        zero_rows = ~mask.any(dim=-1).view(B, N)
    else:
        mask = torch.zeros(B, N, 1, 1, K, dtype=torch.bool)
        out, lse, max_val = cuet.triangle_attention(
            q, k, v, bias, mask=mask, return_aux=True
        )
        zero_rows = torch.ones(B, N, dtype=torch.bool)

    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    assert torch.isfinite(max_val).all()
    torch.testing.assert_close(out[zero_rows], torch.zeros_like(out[zero_rows]))

    cotangent = torch.randn_like(out, dtype=torch.float32)
    loss1 = (out.float() * cotangent).sum()
    g1 = torch.autograd.grad(loss1, (q, k, v, bias), create_graph=True)
    for g in g1:
        assert torch.isfinite(g).all()
    torch.testing.assert_close(g1[0][zero_rows], torch.zeros_like(g1[0][zero_rows]))
    torch.testing.assert_close(g1[1][zero_rows], torch.zeros_like(g1[1][zero_rows]))
    torch.testing.assert_close(g1[2][zero_rows], torch.zeros_like(g1[2][zero_rows]))

    loss2 = sum(g.float().square().mean() for g in g1)
    g2 = torch.autograd.grad(loss2, (q, k, v, bias))
    for g in g2:
        assert torch.isfinite(g).all()


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("inf_val", [1e6, float("inf")])
def test_attention_pair_bias_all_zero_mask_no_nan_1st_2nd_grad(
    dtype: torch.dtype, inf_val: float
):
    torch.manual_seed(808)
    B, M, N, H, DH, D, z_dim = 2, 3, 4, 2, 4, 8, 4
    inner = H * DH

    single_repr = (torch.randn(B * M, N, D, dtype=dtype) * 0.4).requires_grad_(True)
    pair_repr = (torch.randn(B, N, N, z_dim, dtype=dtype) * 0.4).requires_grad_(True)
    # Per-replica mask where several replicas (and all replicas of batch 1) are all-zero
    mask = torch.zeros(B * M, N, dtype=torch.bool)
    mask[0, :3] = True
    mask[2, 1:] = True

    w_ln_a = torch.ones(D, dtype=dtype, requires_grad=True)
    b_ln_a = torch.zeros(D, dtype=dtype, requires_grad=True)
    w_proj_q = (torch.randn(inner, D, dtype=dtype) * 0.3).requires_grad_(True)
    b_proj_q = (torch.randn(inner, dtype=dtype) * 0.1).requires_grad_(True)
    w_proj_k = (torch.randn(inner, D, dtype=dtype) * 0.3).requires_grad_(True)
    w_proj_v = (torch.randn(inner, D, dtype=dtype) * 0.3).requires_grad_(True)
    w_proj_g = (torch.randn(inner, D, dtype=dtype) * 0.3).requires_grad_(True)
    w_proj_o = (torch.randn(D, inner, dtype=dtype) * 0.3).requires_grad_(True)
    w_proj_z = (torch.randn(H, z_dim, dtype=dtype) * 0.3).requires_grad_(True)
    w_ln_z = torch.ones(z_dim, dtype=dtype, requires_grad=True)
    b_ln_z = torch.zeros(z_dim, dtype=dtype, requires_grad=True)

    out, z_proj = cuet.attention_pair_bias(
        single_repr=single_repr,
        pair_repr=pair_repr,
        mask=mask,
        num_heads=H,
        w_ln_a=w_ln_a,
        b_ln_a=b_ln_a,
        w_proj_q=w_proj_q,
        b_proj_q=b_proj_q,
        w_proj_k=w_proj_k,
        w_proj_v=w_proj_v,
        w_proj_g=w_proj_g,
        w_proj_o=w_proj_o,
        w_proj_z=w_proj_z,
        w_ln_z=w_ln_z,
        b_ln_z=b_ln_z,
        inf=inf_val,
        return_z_proj=True,
    )
    assert z_proj is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(z_proj).all()
    zero_replicas = ~mask.any(dim=-1)
    torch.testing.assert_close(out[zero_replicas], torch.zeros_like(out[zero_replicas]))

    inputs = (
        single_repr,
        pair_repr,
        w_ln_a,
        b_ln_a,
        w_proj_q,
        b_proj_q,
        w_proj_k,
        w_proj_v,
        w_proj_g,
        w_proj_o,
        w_proj_z,
        w_ln_z,
        b_ln_z,
    )
    cotangent = torch.randn_like(out, dtype=torch.float32)
    loss1 = (out.float() * cotangent).sum()
    g1 = torch.autograd.grad(loss1, inputs, create_graph=True)
    for g in g1:
        assert torch.isfinite(g).all()

    loss2 = sum(g.float().square().mean() for g in g1)
    g2 = torch.autograd.grad(loss2, inputs)
    for g in g2:
        assert torch.isfinite(g).all()


# ==============================================================================
# 4. 1st and 2nd-order finite-difference checks (gradcheck & gradgradcheck)
# ==============================================================================


@pytest.mark.parametrize("use_kv_lengths", [False, True])
def test_gradcheck_and_gradgradcheck_triangle_attention(use_kv_lengths: bool):
    torch.manual_seed(901)
    B, N, H, Q, K, D = 1, 3, 2, 3, 4, 4
    dt = torch.float64
    q = torch.randn(B, N, H, Q, D, dtype=dt, requires_grad=True)
    k = torch.randn(B, N, H, K, D, dtype=dt, requires_grad=True)
    v = torch.randn(B, N, H, K, D, dtype=dt, requires_grad=True)
    bias = torch.randn(B, 1, H, Q, K, dtype=dt, requires_grad=True)

    if use_kv_lengths:
        kv_lengths = torch.tensor([[[[[0]]], [[[2]]], [[[4]]]]], dtype=torch.int32)

        def fn(q_, k_, v_, b_):
            return cuet.triangle_attention(q_, k_, v_, b_, kv_lengths=kv_lengths)

    else:
        mask = torch.tensor(
            [
                [
                    [[[False, False, False, False]]],
                    [[[True, False, True, False]]],
                    [[[True, True, True, True]]],
                ]
            ]
        )

        def fn(q_, k_, v_, b_):
            return cuet.triangle_attention(q_, k_, v_, b_, mask=mask)

    assert torch.autograd.gradcheck(fn, (q, k, v, bias), eps=1e-6, atol=1e-4)
    assert torch.autograd.gradgradcheck(fn, (q, k, v, bias), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_gradcheck_and_gradgradcheck_triangle_multiplicative_update(direction: str):
    torch.manual_seed(902)
    B, N, C, D = 1, 3, 4, 3
    dt = torch.float64
    x = torch.randn(B, N, N, C, dtype=dt, requires_grad=True)
    mask = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]], dtype=dt)
    norm_in_w = torch.randn(C, dtype=dt, requires_grad=True)
    norm_in_b = torch.randn(C, dtype=dt, requires_grad=True)
    p_in_w = torch.randn(2 * D, C, dtype=dt, requires_grad=True) * 0.4
    p_in_b = torch.randn(2 * D, dtype=dt, requires_grad=True) * 0.2
    g_in_w = torch.randn(2 * D, C, dtype=dt, requires_grad=True) * 0.4
    g_in_b = torch.randn(2 * D, dtype=dt, requires_grad=True) * 0.2
    norm_out_w = torch.randn(D, dtype=dt, requires_grad=True)
    norm_out_b = torch.randn(D, dtype=dt, requires_grad=True)
    p_out_w = torch.randn(C, D, dtype=dt, requires_grad=True) * 0.4
    p_out_b = torch.randn(C, dtype=dt, requires_grad=True) * 0.2
    g_out_w = torch.randn(C, C, dtype=dt, requires_grad=True) * 0.4
    g_out_b = torch.randn(C, dtype=dt, requires_grad=True) * 0.2

    inputs = (
        x,
        norm_in_w,
        norm_in_b,
        p_in_w,
        p_in_b,
        g_in_w,
        g_in_b,
        norm_out_w,
        norm_out_b,
        p_out_w,
        p_out_b,
        g_out_w,
        g_out_b,
    )

    def fn(
        x_,
        ni_w,
        ni_b,
        pi_w,
        pi_b,
        gi_w,
        gi_b,
        no_w,
        no_b,
        po_w,
        po_b,
        go_w,
        go_b,
    ):
        return cuet.triangle_multiplicative_update(
            x_,
            direction=direction,
            mask=mask,
            norm_in_weight=ni_w,
            norm_in_bias=ni_b,
            p_in_weight=pi_w,
            p_in_bias=pi_b,
            g_in_weight=gi_w,
            g_in_bias=gi_b,
            norm_out_weight=no_w,
            norm_out_bias=no_b,
            p_out_weight=po_w,
            p_out_bias=po_b,
            g_out_weight=go_w,
            g_out_bias=go_b,
        )

    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-4)
    assert torch.autograd.gradgradcheck(fn, inputs, eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("norm_kind", ["layer_norm", "rms_norm"])
def test_gradcheck_and_gradgradcheck_attention_pair_bias(norm_kind: str):
    torch.manual_seed(903)
    B, M, N, H, DH, D, z_dim = 1, 2, 3, 2, 2, 4, 3
    inner = H * DH
    dt = torch.float64
    is_ln = norm_kind == "layer_norm"

    single_repr = torch.randn(B * M, N, D, dtype=dt, requires_grad=True)
    pair_repr = torch.randn(B, N, N, z_dim, dtype=dt, requires_grad=True)
    # Replica 0 has partial mask, replica 1 has all-zero mask row!
    mask = torch.tensor([[True, False, True], [False, False, False]])

    w_ln_a = torch.randn(D, dtype=dt, requires_grad=True)
    w_proj_q = torch.randn(inner, D, dtype=dt, requires_grad=True) * 0.4
    b_proj_q = torch.randn(inner, dtype=dt, requires_grad=True) * 0.2
    w_proj_k = torch.randn(inner, D, dtype=dt, requires_grad=True) * 0.4
    b_proj_k = torch.randn(inner, dtype=dt, requires_grad=True) * 0.2
    w_proj_v = torch.randn(inner, D, dtype=dt, requires_grad=True) * 0.4
    b_proj_v = torch.randn(inner, dtype=dt, requires_grad=True) * 0.2
    w_proj_g = torch.randn(inner, D, dtype=dt, requires_grad=True) * 0.4
    b_proj_g = torch.randn(inner, dtype=dt, requires_grad=True) * 0.2
    w_proj_o = torch.randn(D, inner, dtype=dt, requires_grad=True) * 0.4
    b_proj_o = torch.randn(D, dtype=dt, requires_grad=True) * 0.2
    w_proj_z = torch.randn(H, z_dim, dtype=dt, requires_grad=True) * 0.4
    b_proj_z = torch.randn(H, dtype=dt, requires_grad=True) * 0.2
    w_ln_z = torch.randn(z_dim, dtype=dt, requires_grad=True)
    w_ln_q = torch.randn(inner, dtype=dt, requires_grad=True)
    w_ln_k = torch.randn(inner, dtype=dt, requires_grad=True)

    if is_ln:
        b_ln_a = torch.randn(D, dtype=dt, requires_grad=True)
        b_ln_z = torch.randn(z_dim, dtype=dt, requires_grad=True)
        b_ln_q = torch.randn(inner, dtype=dt, requires_grad=True)
        b_ln_k = torch.randn(inner, dtype=dt, requires_grad=True)
        inputs = (
            single_repr,
            pair_repr,
            w_ln_a,
            b_ln_a,
            w_proj_q,
            b_proj_q,
            w_proj_k,
            b_proj_k,
            w_proj_v,
            b_proj_v,
            w_proj_g,
            b_proj_g,
            w_proj_o,
            b_proj_o,
            w_proj_z,
            b_proj_z,
            w_ln_z,
            b_ln_z,
            w_ln_q,
            b_ln_q,
            w_ln_k,
            b_ln_k,
        )

        def fn(
            s,
            z,
            wla,
            bla,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wg,
            bg,
            wo,
            bo,
            wz,
            bz,
            wlz,
            blz,
            wlq,
            blq,
            wlk,
            blk,
        ):
            out, z_p = cuet.attention_pair_bias(
                s,
                z,
                mask,
                H,
                wla,
                bla,
                wq,
                bq,
                wk,
                wv,
                wg,
                wo,
                wz,
                b_proj_o=bo,
                b_proj_z=bz,
                w_ln_z=wlz,
                b_ln_z=blz,
                return_z_proj=True,
                b_proj_k=bk,
                b_proj_v=bv,
                w_ln_q=wlq,
                b_ln_q=blq,
                w_ln_k=wlk,
                b_ln_k=blk,
                b_proj_g=bg,
                norm_kind=norm_kind,
            )
            return out, z_p

    else:
        inputs = (
            single_repr,
            pair_repr,
            w_ln_a,
            w_proj_q,
            b_proj_q,
            w_proj_k,
            b_proj_k,
            w_proj_v,
            b_proj_v,
            w_proj_g,
            b_proj_g,
            w_proj_o,
            b_proj_o,
            w_proj_z,
            b_proj_z,
            w_ln_z,
            w_ln_q,
            w_ln_k,
        )

        def fn(
            s,
            z,
            wla,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wg,
            bg,
            wo,
            bo,
            wz,
            bz,
            wlz,
            wlq,
            wlk,
        ):
            out, z_p = cuet.attention_pair_bias(
                s,
                z,
                mask,
                H,
                wla,
                None,
                wq,
                bq,
                wk,
                wv,
                wg,
                wo,
                wz,
                b_proj_o=bo,
                b_proj_z=bz,
                w_ln_z=wlz,
                b_ln_z=None,
                return_z_proj=True,
                b_proj_k=bk,
                b_proj_v=bv,
                w_ln_q=wlq,
                b_ln_q=None,
                w_ln_k=wlk,
                b_ln_k=None,
                b_proj_g=bg,
                norm_kind=norm_kind,
            )
            return out, z_p

    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-4)
    assert torch.autograd.gradgradcheck(fn, inputs, eps=1e-6, atol=1e-4)


# ==============================================================================
# 5. TriMul precision="bfloat16" & arbitrary leading batch dimensions
# ==============================================================================


def test_triangle_multiplicative_update_precision_bfloat16_fp32_grads():
    torch.manual_seed(1001)
    B, N, C, D = 2, 5, 8, 6
    x = torch.randn(B, N, N, C, dtype=torch.float32, requires_grad=True)
    mask = (torch.rand(B, N, N, dtype=torch.float32) > 0.2).to(torch.float32)

    params = dict(
        norm_in_weight=torch.randn(C, dtype=torch.float32, requires_grad=True),
        norm_in_bias=torch.randn(C, dtype=torch.float32, requires_grad=True),
        p_in_weight=torch.randn(2 * D, C, dtype=torch.float32, requires_grad=True),
        p_in_bias=torch.randn(2 * D, dtype=torch.float32, requires_grad=True),
        g_in_weight=torch.randn(2 * D, C, dtype=torch.float32, requires_grad=True),
        g_in_bias=torch.randn(2 * D, dtype=torch.float32, requires_grad=True),
        norm_out_weight=torch.randn(D, dtype=torch.float32, requires_grad=True),
        norm_out_bias=torch.randn(D, dtype=torch.float32, requires_grad=True),
        p_out_weight=torch.randn(C, D, dtype=torch.float32, requires_grad=True),
        p_out_bias=torch.randn(C, dtype=torch.float32, requires_grad=True),
        g_out_weight=torch.randn(C, C, dtype=torch.float32, requires_grad=True),
        g_out_bias=torch.randn(C, dtype=torch.float32, requires_grad=True),
    )

    out_bf16_mode = cuet.triangle_multiplicative_update(
        x, direction="outgoing", mask=mask, precision="bfloat16", **params
    )
    assert out_bf16_mode.dtype == torch.float32

    # Must match running directly on explicitly BF16-rounded tensors cast back to FP32
    out_explicit_bf16 = cuet.triangle_multiplicative_update(
        x.detach().to(torch.bfloat16),
        direction="outgoing",
        mask=mask.to(torch.bfloat16),
        **{k: v.detach().to(torch.bfloat16) for k, v in params.items()},
    ).to(torch.float32)
    torch.testing.assert_close(out_bf16_mode, out_explicit_bf16, rtol=0.0, atol=0.0)

    all_tensors = (x, *params.values())
    g1 = torch.autograd.grad(
        out_bf16_mode.square().mean(), all_tensors, create_graph=True
    )
    for name, grad in zip(("x", *params.keys()), g1):
        assert grad.dtype == torch.float32, f"{name} grad dtype mismatch"
        assert torch.isfinite(grad).all(), f"{name} 1st-order grad non-finite"
        assert grad.abs().sum() > 0, f"{name} 1st-order grad unexpectedly zero"

    g2 = torch.autograd.grad(sum(g.square().mean() for g in g1), all_tensors)
    for name, grad in zip(("x", *params.keys()), g2):
        assert grad.dtype == torch.float32
        assert torch.isfinite(grad).all(), f"{name} 2nd-order grad non-finite"


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_triangle_multiplicative_update_arbitrary_batch_shapes(direction: str):
    torch.manual_seed(1002)
    N, C, D = 5, 8, 6
    x = torch.randn(2, 3, N, N, C, dtype=torch.float32, requires_grad=True)
    mask = (torch.rand(2, 3, N, N) > 0.25).to(torch.float32)
    params = dict(
        norm_in_weight=torch.randn(C),
        norm_in_bias=torch.randn(C),
        p_in_weight=torch.randn(2 * D, C) * 0.3,
        p_in_bias=torch.randn(2 * D) * 0.1,
        g_in_weight=torch.randn(2 * D, C) * 0.3,
        g_in_bias=torch.randn(2 * D) * 0.1,
        norm_out_weight=torch.randn(D),
        norm_out_bias=torch.randn(D),
        p_out_weight=torch.randn(C, D) * 0.3,
        p_out_bias=torch.randn(C) * 0.1,
        g_out_weight=torch.randn(C, C) * 0.3,
        g_out_bias=torch.randn(C) * 0.1,
    )

    out_5d = cuet.triangle_multiplicative_update(
        x, direction=direction, mask=mask, **params
    )
    assert out_5d.shape == (2, 3, N, N, C)

    x_flat = x.detach().reshape(6, N, N, C).requires_grad_(True)
    out_flat = cuet.triangle_multiplicative_update(
        x_flat, direction=direction, mask=mask.reshape(6, N, N), **params
    )
    torch.testing.assert_close(
        out_5d, out_flat.view(2, 3, N, N, C), rtol=1e-5, atol=1e-5
    )

    grad_cot = torch.randn_like(out_5d)
    (gx_5d,) = torch.autograd.grad((out_5d * grad_cot).sum(), x)
    (gx_flat,) = torch.autograd.grad(
        (out_flat * grad_cot.view(6, N, N, C)).sum(), x_flat
    )
    torch.testing.assert_close(gx_5d, gx_flat.view(2, 3, N, N, C), rtol=1e-5, atol=1e-5)

    # 3D unbatched (N, N, C) and broadcasted batch dims (1, 3, N, N, C) with (2, 1, N, N)
    out_3d = cuet.triangle_multiplicative_update(
        x[0, 0].detach(), direction=direction, mask=mask[0, 0], **params
    )
    assert out_3d.shape == (N, N, C)
    torch.testing.assert_close(out_3d, out_5d[0, 0], rtol=1e-5, atol=1e-5)

    x_bcast = torch.randn(1, 3, N, N, C)
    mask_bcast = (torch.rand(2, 1, N, N) > 0.25).to(torch.float32)
    out_bcast = cuet.triangle_multiplicative_update(
        x_bcast, direction=direction, mask=mask_bcast, **params
    )
    assert out_bcast.shape == (2, 3, N, N, C)


# ==============================================================================
# 6. APB is_cached_z_proj equivalence and norm_kind="rms_norm" validation
# ==============================================================================


@pytest.mark.parametrize("norm_kind", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("multiplicity", [1, 3])
def test_attention_pair_bias_cached_z_proj_equivalence(
    norm_kind: str, multiplicity: int
):
    torch.manual_seed(1101)
    B, M, N, H, DH, D, z_dim = 2, multiplicity, 6, 3, 4, 12, 5
    inner = H * DH
    is_ln = norm_kind == "layer_norm"

    s1 = torch.randn(B * M, N, D, requires_grad=True)
    z1 = torch.randn(B, N, N, z_dim, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, -2:] = False

    common = dict(
        mask=mask,
        num_heads=H,
        w_ln_a=torch.randn(D),
        b_ln_a=torch.randn(D) if is_ln else None,
        w_proj_q=torch.randn(inner, D) * 0.3,
        b_proj_q=torch.randn(inner) * 0.1,
        w_proj_k=torch.randn(inner, D) * 0.3,
        w_proj_v=torch.randn(inner, D) * 0.3,
        w_proj_g=torch.randn(inner, D) * 0.3,
        w_proj_o=torch.randn(D, inner) * 0.3,
        b_proj_o=torch.randn(D) * 0.1,
        b_proj_k=torch.randn(inner) * 0.1,
        b_proj_v=torch.randn(inner) * 0.1,
        w_ln_q=torch.randn(inner),
        b_ln_q=torch.randn(inner) if is_ln else None,
        w_ln_k=torch.randn(inner),
        b_ln_k=torch.randn(inner) if is_ln else None,
        b_proj_g=torch.randn(inner) * 0.1,
        norm_kind=norm_kind,
    )
    w_proj_z = torch.randn(H, z_dim) * 0.3
    b_proj_z = torch.randn(H) * 0.1
    w_ln_z = torch.randn(z_dim)
    b_ln_z = torch.randn(z_dim) if is_ln else None

    out_no_ret, z_none = cuet.attention_pair_bias(
        single_repr=s1,
        pair_repr=z1,
        w_proj_z=w_proj_z,
        b_proj_z=b_proj_z,
        w_ln_z=w_ln_z,
        b_ln_z=b_ln_z,
        return_z_proj=False,
        **common,
    )
    assert z_none is None

    out_uncached, z_proj = cuet.attention_pair_bias(
        single_repr=s1,
        pair_repr=z1,
        w_proj_z=w_proj_z,
        b_proj_z=b_proj_z,
        w_ln_z=w_ln_z,
        b_ln_z=b_ln_z,
        return_z_proj=True,
        **common,
    )
    assert z_proj is not None
    assert z_proj.shape == (B, H, N, N)
    torch.testing.assert_close(out_no_ret, out_uncached, rtol=0.0, atol=0.0)

    s2 = s1.detach().clone().requires_grad_(True)
    z_cached = z_proj.detach().clone().requires_grad_(True)
    out_cached, z_proj_cached = cuet.attention_pair_bias(
        single_repr=s2,
        pair_repr=z_cached,
        w_proj_z=None,
        is_cached_z_proj=True,
        return_z_proj=True,
        **common,
    )
    torch.testing.assert_close(out_cached, out_uncached, rtol=0.0, atol=0.0)
    assert z_proj_cached is not None
    torch.testing.assert_close(z_proj_cached, z_proj, rtol=0.0, atol=0.0)

    cot = torch.randn_like(out_uncached)
    (gs1,) = torch.autograd.grad((out_uncached * cot).sum(), s1)
    (gs2,) = torch.autograd.grad((out_cached * cot).sum(), s2)
    torch.testing.assert_close(gs1, gs2, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bad_bias_name", ["b_ln_a", "b_ln_z", "b_ln_q", "b_ln_k"])
def test_attention_pair_bias_rms_norm_bias_validation(bad_bias_name: str):
    B, N, H, DH, D, z_dim = 1, 4, 2, 4, 8, 4
    inner = H * DH
    kwargs = dict(
        single_repr=torch.randn(B, N, D),
        pair_repr=torch.randn(B, N, N, z_dim),
        mask=None,
        num_heads=H,
        w_ln_a=torch.ones(D),
        b_ln_a=None,
        w_proj_q=torch.randn(inner, D),
        b_proj_q=None,
        w_proj_k=torch.randn(inner, D),
        w_proj_v=torch.randn(inner, D),
        w_proj_g=torch.randn(inner, D),
        w_proj_o=torch.randn(D, inner),
        w_proj_z=torch.randn(H, z_dim),
        w_ln_z=torch.ones(z_dim),
        b_ln_z=None,
        w_ln_q=torch.ones(inner),
        b_ln_q=None,
        w_ln_k=torch.ones(inner),
        b_ln_k=None,
        norm_kind="rms_norm",
    )
    bias_dims = {"b_ln_a": D, "b_ln_z": z_dim, "b_ln_q": inner, "b_ln_k": inner}
    kwargs[bad_bias_name] = torch.zeros(bias_dims[bad_bias_name])

    with pytest.raises(ValueError, match="rms_norm"):
        cuet.attention_pair_bias(**kwargs)

    kwargs[bad_bias_name] = None
    kwargs["norm_kind"] = "batch_norm"
    with pytest.raises(ValueError, match="norm_kind"):
        cuet.attention_pair_bias(**kwargs)
