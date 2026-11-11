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
"""Architecture Note 06: Triangle Multiplicative Update, Triangle Attention & Pair Bias.

Implements clean, pedagogical pure-PyTorch reference operators matching the exact
mathematical contracts in `cuequivariance_torch/primitives/triangle.py`:
1. `triangle_multiplicative_update_ref` (Outgoing & Incoming):
   - Batched-GEMM `(B, D, N, N)` reformulation of 2-hop triangular contractions.
   - 3D Euclidean geometry propagation along `i -> k -> j` paths on incomplete graphs.
   - Exact S_N node-permutation equivariance: P * TriMul(Z) * P^T == TriMul(P Z P^T).
2. `triangle_attention_ref` + `cuet.mask_to_kv_lengths`:
   - Prefix `kv_lengths` fast-path contract, `lse` / `max` auxiliary Flash statistics,
     and zero-length masked row safety (`kv_lengths == 0` -> exact zero vs `-1e9` leak).
3. `attention_pair_bias_ref`:
   - `return_z_proj` + `is_cached_z_proj` reuse across `M` single-representation replicas
     `(B * M, N, D)` sharing one `(B, N, N, z_dim)` pair tensor, quantifying HBM savings
     across 48 Evoformer / Pairformer / Diffusion blocks.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

import cuequivariance_torch as cuet


def section(title: str) -> None:
    print(f"\n{'=' * 82}\n{title}\n{'=' * 82}")


def _apply_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float = 1e-5,
    norm_kind: str = "layer_norm",
) -> torch.Tensor:
    dim = x.shape[-1]
    if norm_kind == "layer_norm":
        return F.layer_norm(x, (dim,), weight=weight, bias=bias, eps=eps)
    if norm_kind == "rms_norm":
        if bias is not None:
            raise ValueError("rms_norm requires normalization bias to be None")
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        out = x * rms
        return out * weight if weight is not None else out
    raise ValueError(f"Unsupported norm_kind: {norm_kind}")


def triangle_multiplicative_update_ref(
    x: torch.Tensor,
    direction: str = "outgoing",
    mask: Optional[torch.Tensor] = None,
    norm_in_weight: Optional[torch.Tensor] = None,
    norm_in_bias: Optional[torch.Tensor] = None,
    p_in_weight: Optional[torch.Tensor] = None,
    p_in_bias: Optional[torch.Tensor] = None,
    g_in_weight: Optional[torch.Tensor] = None,
    g_in_bias: Optional[torch.Tensor] = None,
    norm_out_weight: Optional[torch.Tensor] = None,
    norm_out_bias: Optional[torch.Tensor] = None,
    p_out_weight: Optional[torch.Tensor] = None,
    p_out_bias: Optional[torch.Tensor] = None,
    g_out_weight: Optional[torch.Tensor] = None,
    g_out_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Pure-PyTorch reference matching `cuequivariance_torch.triangle_multiplicative_update`."""
    if direction not in ("outgoing", "incoming"):
        raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction}")
    n_row, n_col, c_in = x.shape[-3:]
    assert n_row == n_col
    d_hid = c_in if p_in_weight is None else p_in_weight.shape[0] // 2
    c_out = c_in if p_out_weight is None else p_out_weight.shape[0]

    # 1. Input LayerNorm
    x_in = _apply_norm(x, norm_in_weight, norm_in_bias, eps=eps)

    # 2. Fused Sigmoid-Gated Dual GEMM + optional pair mask
    proj_in = (
        F.linear(x_in, p_in_weight, p_in_bias)
        if p_in_weight is not None
        else torch.cat([x_in, x_in], dim=-1)
    )
    gate_in = (
        torch.sigmoid(F.linear(x_in, g_in_weight, g_in_bias))
        if g_in_weight is not None
        else torch.ones_like(proj_in)
    )
    ab = gate_in * proj_in
    if mask is not None:
        ab = ab * mask.to(dtype=ab.dtype).unsqueeze(-1)

    # Transpose to (..., 2*D, N, N) so triangular contraction maps to batched GEMM!
    a_t, b_t = torch.chunk(ab.movedim(-1, -3), 2, dim=-3)
    if direction == "outgoing":
        # sum_k a_{i,k} * b_{j,k} == A @ B^T along last two (N, N) axes
        tri = torch.matmul(a_t, b_t.transpose(-1, -2))
    else:
        # sum_k a_{k,i} * b_{k,j} == A^T @ B along last two (N, N) axes
        tri = torch.matmul(a_t.transpose(-1, -2), b_t)
    tri = tri.movedim(-3, -1)  # (..., N, N, D)
    assert tri.shape[-1] == d_hid

    # 3. Output LayerNorm + Dual-Input Sigmoid-Gated GEMM (gate reads x_in!)
    x_out = _apply_norm(tri, norm_out_weight, norm_out_bias, eps=eps)
    proj_out = (
        F.linear(x_out, p_out_weight, p_out_bias)
        if p_out_weight is not None
        else x_out[..., :c_out]
    )
    gate_out = (
        torch.sigmoid(F.linear(x_in, g_out_weight, g_out_bias))
        if g_out_weight is not None
        else torch.ones_like(proj_out)
    )
    return gate_out * proj_out


def triangle_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    return_aux: bool = False,
    *,
    kv_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference matching `cuequivariance_torch.triangle_attention`."""
    if mask is not None and kv_lengths is not None:
        raise ValueError(
            "triangle_attention: pass either `mask` or `kv_lengths`, not both."
        )

    squeeze_b = q.ndim == 4
    if squeeze_b:
        q, k, v, bias = (
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            bias.unsqueeze(0),
        )
        if mask is not None and mask.ndim == 4:
            mask = mask.unsqueeze(0)
        if kv_lengths is not None and kv_lengths.ndim == 4:
            kv_lengths = kv_lengths.unsqueeze(0)

    orig_dtype = q.dtype
    head_dim = q.shape[-1]
    key_len = k.shape[-2]
    s = (1.0 / math.sqrt(head_dim)) if scale is None else scale

    logits = torch.einsum("bnhqd,bnhkd->bnhqk", q.float() * s, k.float()) + bias.float()

    zero_length_rows: Optional[torch.Tensor] = None
    if mask is not None:
        logits = torch.where(mask.to(torch.bool), logits, torch.full_like(logits, -1e9))
    elif kv_lengths is not None:
        clamped = kv_lengths.clamp(min=0, max=key_len)
        pos = torch.arange(key_len, device=q.device).view(1, 1, 1, 1, key_len)
        prefix_mask = pos < clamped
        zero_length_rows = clamped == 0  # (B, N, 1, 1, 1)
        logits = torch.where(prefix_mask, logits, torch.full_like(logits, -1e9))

    amax = logits.max(dim=-1, keepdim=True).values  # float32 (B, N, H, Q, 1)
    lse = torch.logsumexp(
        logits - amax, dim=-1, keepdim=True
    )  # float32 (B, N, H, Q, 1)
    attn = torch.exp(logits - amax - lse).to(orig_dtype)
    if zero_length_rows is not None:
        attn = torch.where(zero_length_rows, torch.zeros_like(attn), attn)

    out = torch.einsum("bnhqk,bnhkd->bnhqd", attn, v)
    if squeeze_b:
        out, lse, amax = out.squeeze(0), lse.squeeze(0), amax.squeeze(0)
    if return_aux:
        return out, lse, amax
    return out


def attention_pair_bias_ref(
    single_repr: torch.Tensor,
    pair_repr: torch.Tensor,
    mask: Optional[torch.Tensor],
    num_heads: int,
    w_ln_a: torch.Tensor,
    b_ln_a: Optional[torch.Tensor],
    w_proj_q: torch.Tensor,
    b_proj_q: Optional[torch.Tensor],
    w_proj_k: torch.Tensor,
    w_proj_v: torch.Tensor,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    w_proj_z: Optional[torch.Tensor] = None,
    b_proj_o: Optional[torch.Tensor] = None,
    b_proj_z: Optional[torch.Tensor] = None,
    w_ln_z: Optional[torch.Tensor] = None,
    b_ln_z: Optional[torch.Tensor] = None,
    inf: float = 1e6,
    eps: float = 1e-5,
    attn_scale: Optional[float] = None,
    return_z_proj: bool = False,
    is_cached_z_proj: bool = False,
    *,
    b_proj_k: Optional[torch.Tensor] = None,
    b_proj_v: Optional[torch.Tensor] = None,
    w_ln_q: Optional[torch.Tensor] = None,
    b_ln_q: Optional[torch.Tensor] = None,
    w_ln_k: Optional[torch.Tensor] = None,
    b_ln_k: Optional[torch.Tensor] = None,
    b_proj_g: Optional[torch.Tensor] = None,
    norm_kind: str = "layer_norm",
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure-PyTorch reference matching `cuequivariance_torch.attention_pair_bias`."""
    if w_proj_z is None and not is_cached_z_proj:
        raise ValueError("w_proj_z is required unless is_cached_z_proj=True")

    bm, seq_len, d_single = single_repr.shape
    batch_size = pair_repr.shape[0]
    assert bm % batch_size == 0
    multiplicity = bm // batch_size
    inner_dim = w_proj_q.shape[0]
    head_dim = inner_dim // num_heads
    scale = (1.0 / math.sqrt(head_dim)) if attn_scale is None else attn_scale

    # 1. Single representation normalization + Q/K/V/Gate projections
    a_norm = _apply_norm(single_repr, w_ln_a, b_ln_a, eps=eps, norm_kind=norm_kind)
    q = F.linear(a_norm, w_proj_q, b_proj_q)
    k = F.linear(a_norm, w_proj_k, b_proj_k)
    v = F.linear(a_norm, w_proj_v, b_proj_v)
    g = torch.sigmoid(F.linear(a_norm, w_proj_g, b_proj_g))

    if w_ln_q is not None or b_ln_q is not None:
        q = _apply_norm(q, w_ln_q, b_ln_q, eps=eps, norm_kind=norm_kind)
    if w_ln_k is not None or b_ln_k is not None:
        k = _apply_norm(k, w_ln_k, b_ln_k, eps=eps, norm_kind=norm_kind)

    # 2. Pair representation projection or direct cache reuse -> (B, H, N, N)
    if is_cached_z_proj:
        assert pair_repr.shape == (batch_size, num_heads, seq_len, seq_len)
        z_proj = pair_repr
    else:
        z_norm = pair_repr
        if w_ln_z is not None or b_ln_z is not None:
            z_norm = _apply_norm(
                pair_repr, w_ln_z, b_ln_z, eps=eps, norm_kind=norm_kind
            )
        assert w_proj_z is not None
        z_proj = F.linear(z_norm, w_proj_z, b_proj_z).permute(0, 3, 1, 2).contiguous()

    # 3. Multi-replica attention sharing z_proj across multiplicity M
    q_h = q.view(batch_size, multiplicity, seq_len, num_heads, head_dim).permute(
        0, 1, 3, 2, 4
    )
    k_h = k.view(batch_size, multiplicity, seq_len, num_heads, head_dim).permute(
        0, 1, 3, 2, 4
    )
    v_h = v.view(batch_size, multiplicity, seq_len, num_heads, head_dim).permute(
        0, 1, 3, 2, 4
    )

    scores = torch.matmul(q_h * scale, k_h.transpose(-1, -2)) + z_proj.unsqueeze(1)
    if mask is not None:
        if mask.shape == (batch_size, seq_len):
            key_mask = mask[:, None, None, None, :].to(torch.bool)
        elif mask.shape == (bm, seq_len):
            key_mask = mask.view(batch_size, multiplicity, 1, 1, seq_len).to(torch.bool)
        else:
            raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
        scores = scores.masked_fill(~key_mask, -inf)

    attn = torch.softmax(scores.float(), dim=-1).to(single_repr.dtype)
    ctx = torch.matmul(attn, v_h).permute(0, 1, 3, 2, 4).reshape(bm, seq_len, inner_dim)
    out = F.linear(ctx * g, w_proj_o, b_proj_o)
    assert out.shape == (bm, seq_len, d_single)
    return out, (z_proj if return_z_proj else None)


def audit_triangle_multiplicative_update() -> None:
    section(
        "1. Triangle Multiplicative Update: 2-Hop Geometry & S_N Permutation Equivariance"
    )
    torch.manual_seed(101)
    b_sz, n_res, c_pair, d_hid = 2, 24, 32, 32
    z = torch.randn(b_sz, n_res, n_res, c_pair, dtype=torch.float64, requires_grad=True)
    pair_mask = (torch.rand(b_sz, n_res, n_res) > 0.15).to(torch.float64)

    params = {
        "norm_in_weight": torch.randn(c_pair, dtype=torch.float64),
        "norm_in_bias": torch.randn(c_pair, dtype=torch.float64),
        "p_in_weight": torch.randn(2 * d_hid, c_pair, dtype=torch.float64) * 0.2,
        "p_in_bias": torch.randn(2 * d_hid, dtype=torch.float64) * 0.1,
        "g_in_weight": torch.randn(2 * d_hid, c_pair, dtype=torch.float64) * 0.2,
        "g_in_bias": torch.randn(2 * d_hid, dtype=torch.float64) * 0.1,
        "norm_out_weight": torch.randn(d_hid, dtype=torch.float64),
        "norm_out_bias": torch.randn(d_hid, dtype=torch.float64),
        "p_out_weight": torch.randn(c_pair, d_hid, dtype=torch.float64) * 0.2,
        "p_out_bias": torch.randn(c_pair, dtype=torch.float64) * 0.1,
        "g_out_weight": torch.randn(c_pair, c_pair, dtype=torch.float64) * 0.2,
        "g_out_bias": torch.randn(c_pair, dtype=torch.float64) * 0.1,
    }

    perm = torch.randperm(n_res)
    for direction in ("outgoing", "incoming"):
        out = triangle_multiplicative_update_ref(
            z, direction=direction, mask=pair_mask, **params
        )
        z_perm = z[:, perm][:, :, perm].detach().clone().requires_grad_(True)
        mask_perm = pair_mask[:, perm][:, :, perm]
        out_perm = triangle_multiplicative_update_ref(
            z_perm, direction=direction, mask=mask_perm, **params
        )

        # Verify P * TriMul(Z, M) * P^T == TriMul(P Z P^T, P M P^T)
        lhs = out[:, perm][:, :, perm]
        fwd_err = float((lhs - out_perm).detach().abs().max())

        cot = torch.randn_like(out)
        (gz,) = torch.autograd.grad(out, z, grad_outputs=cot, retain_graph=True)
        (gz_perm,) = torch.autograd.grad(
            out_perm, z_perm, grad_outputs=cot[:, perm][:, :, perm]
        )
        bwd_err = float((gz[:, perm][:, :, perm] - gz_perm).detach().abs().max())
        print(
            f"  [{direction:<8s}] S_N equivariance: max |P*TriMul(Z)*P^T - TriMul(P*Z*P^T)| = "
            f"{fwd_err:.3e} (bwd={bwd_err:.3e})"
        )
        assert fwd_err < 1e-13 and bwd_err < 1e-13

    # Demonstrate 2-hop 3D Euclidean geometry recovery (i -> k -> j) on incomplete edges
    coords = torch.randn(n_res, 3, dtype=torch.float64)
    dists = torch.cdist(coords, coords)
    local_adj = (dists < 2.2).to(torch.float64)  # Sparse local neighborhood
    # Channel d in {0, 1, 2} carries coordinate displacement r_k - r_i on observed local edges
    disp_obs = (coords.unsqueeze(0) - coords.unsqueeze(1)) * local_adj.unsqueeze(-1)
    # Outgoing contraction over shared bridge atoms k:
    # sum_{k in N(i) cap N(j)} [(r_k - r_i) - (r_k - r_j)] = |N(i) cap N(j)| * (r_j - r_i)!
    common_k = local_adj @ local_adj.T  # (N, N) count of 2-hop bridges k
    two_hop_disp = torch.einsum("ik,jkd->ijd", local_adj, disp_obs) - torch.einsum(
        "ikd,jk->ijd", disp_obs, local_adj
    )
    unobserved_with_bridge = (local_adj == 0.0) & (common_k > 0.0)
    assert bool(unobserved_with_bridge.any().item())
    recovered_disp = two_hop_disp[unobserved_with_bridge] / common_k[
        unobserved_with_bridge
    ].unsqueeze(-1)
    true_disp = (coords.unsqueeze(1) - coords.unsqueeze(0))[unobserved_with_bridge]
    geom_err = float((recovered_disp - true_disp).abs().max())
    print(
        f"  [2-Hop Geometry Closure] Reconstructed {int(unobserved_with_bridge.sum())} "
        f"unobserved pairs (i,j) via bridges k: max |r_ij_rec - r_ij_true| = {geom_err:.3e}"
    )
    assert geom_err < 1e-14


def audit_triangle_attention_and_kv_lengths() -> None:
    section(
        "2. Triangle Attention: Prefix kv_lengths Contract, Flash Aux & Zero-Row Safety"
    )
    torch.manual_seed(202)
    b_sz, seq_len, n_heads, d_head = 1, 16, 4, 32
    q = torch.randn(b_sz, seq_len, n_heads, seq_len, d_head, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.randn(b_sz, 1, n_heads, seq_len, seq_len, dtype=torch.float32)

    # Row-wise right-padded sequence mask where rows 12..15 have ZERO valid keys!
    valid_len = 12
    pos = torch.arange(seq_len)
    row_valid = pos < valid_len
    col_valid = pos < valid_len
    mask = row_valid.view(1, seq_len, 1, 1, 1) & col_valid.view(1, 1, 1, 1, seq_len)

    kv_lengths = cuet.mask_to_kv_lengths(mask)
    assert (
        kv_lengths.shape == (b_sz, seq_len, 1, 1, 1) and kv_lengths.dtype == torch.int32
    )
    assert int(kv_lengths[0, 0].item()) == valid_len
    assert int(kv_lengths[0, 14].item()) == 0

    # Non-prefix ("holey") mask must be rejected by cuet.mask_to_kv_lengths
    holey_mask = mask.clone()
    holey_mask[0, 0, 0, 0, 3] = False
    rejected_holey = False
    try:
        cuet.mask_to_kv_lengths(holey_mask)
    except ValueError as exc:
        rejected_holey = "prefix-shaped" in str(exc)
    assert rejected_holey

    # Passing both mask and kv_lengths to cuet.triangle_attention must raise ValueError
    rejected_both = False
    try:
        cuet.triangle_attention(q, k, v, bias, mask=mask, kv_lengths=kv_lengths)
    except ValueError as exc:
        rejected_both = "not both" in str(exc)
    assert rejected_both

    out_kvl, lse, amax = triangle_attention_ref(
        q, k, v, bias, return_aux=True, kv_lengths=kv_lengths
    )
    out_dense = triangle_attention_ref(q, k, v, bias, mask=mask)

    # Valid rows (0..11) match dense -1e9 mask; padded rows (12..15) are exact 0 in kv_lengths!
    valid_diff = float(
        (out_kvl[:, :valid_len] - out_dense[:, :valid_len]).float().abs().max()
    )
    zero_row_max = float(out_kvl[:, valid_len:].float().abs().max())
    dense_leak_norm = float(out_dense[:, valid_len:].float().abs().max())

    print(
        f"  cuet.mask_to_kv_lengths valid={valid_len}, zero_rows=4 : rejected_holey={rejected_holey}, rejected_both={rejected_both}"
    )
    print(
        f"  Aux shapes & dtypes (Flash bwd) : lse={tuple(lse.shape)} ({lse.dtype}), max={tuple(amax.shape)} ({amax.dtype})"
    )
    print(f"  Valid rows max |kv_len - dense| : {valid_diff:.3e}")
    print(
        f"  Zero-length row |output|        : kv_lengths={zero_row_max:.1f} vs naive -1e9 softmax leak={dense_leak_norm:.3f}!"
    )
    assert valid_diff == 0.0
    assert zero_row_max == 0.0
    assert dense_leak_norm > 0.05
    assert lse.dtype == torch.float32 and amax.dtype == torch.float32


def audit_attention_pair_bias_cache_and_hbm() -> None:
    section("3. Attention Pair Bias: Multi-Replica z_proj Caching & 48-Block HBM Audit")
    torch.manual_seed(303)
    b_sz, mult_m, seq_len, d_single, z_dim = 2, 4, 20, 64, 32
    n_heads, d_head = 4, 16
    inner = n_heads * d_head

    # Guard check on public wrapper
    caught_missing_wz = False
    try:
        cuet.attention_pair_bias(
            torch.zeros(1, 4, d_single),
            torch.zeros(1, 4, 4, z_dim),
            None,
            n_heads,
            torch.ones(d_single),
            None,
            torch.zeros(inner, d_single),
            None,
            torch.zeros(inner, d_single),
            torch.zeros(inner, d_single),
            torch.zeros(inner, d_single),
            torch.zeros(d_single, inner),
            w_proj_z=None,
            is_cached_z_proj=False,
        )
    except ValueError as exc:
        caught_missing_wz = "w_proj_z is required" in str(exc)
    assert caught_missing_wz

    single_1 = torch.randn(
        b_sz * mult_m, seq_len, d_single, dtype=torch.float64, requires_grad=True
    )
    single_2 = torch.randn(
        b_sz * mult_m, seq_len, d_single, dtype=torch.float64, requires_grad=True
    )
    pair = torch.randn(
        b_sz, seq_len, seq_len, z_dim, dtype=torch.float64, requires_grad=True
    )
    mask = (torch.rand(b_sz, seq_len) > 0.1).to(torch.float64)

    weights = {
        "num_heads": n_heads,
        "w_ln_a": torch.randn(d_single, dtype=torch.float64),
        "b_ln_a": None,
        "w_proj_q": torch.randn(inner, d_single, dtype=torch.float64) * 0.15,
        "b_proj_q": None,
        "w_proj_k": torch.randn(inner, d_single, dtype=torch.float64) * 0.15,
        "w_proj_v": torch.randn(inner, d_single, dtype=torch.float64) * 0.15,
        "w_proj_g": torch.randn(inner, d_single, dtype=torch.float64) * 0.15,
        "w_proj_o": torch.randn(d_single, inner, dtype=torch.float64) * 0.15,
        "w_ln_q": torch.randn(inner, dtype=torch.float64),
        "w_ln_k": torch.randn(inner, dtype=torch.float64),
        "norm_kind": "rms_norm",
    }
    w_proj_z = torch.randn(n_heads, z_dim, dtype=torch.float64) * 0.2
    b_proj_z = torch.randn(n_heads, dtype=torch.float64) * 0.1
    w_ln_z = torch.randn(z_dim, dtype=torch.float64)

    # Uncached 2-step rollout recomputing pair LayerNorm + projection each step
    out1_u, _ = attention_pair_bias_ref(
        single_1,
        pair,
        mask,
        w_proj_z=w_proj_z,
        b_proj_z=b_proj_z,
        w_ln_z=w_ln_z,
        **weights,
    )
    out2_u, _ = attention_pair_bias_ref(
        single_2,
        pair,
        mask,
        w_proj_z=w_proj_z,
        b_proj_z=b_proj_z,
        w_ln_z=w_ln_z,
        **weights,
    )
    (g_pair_u,) = torch.autograd.grad((out1_u + out2_u).sum(), pair, retain_graph=True)

    # Cached rollout: compute z_proj ONCE (`return_z_proj=True`), reuse via `is_cached_z_proj=True`
    out1_c, z_cached = attention_pair_bias_ref(
        single_1,
        pair,
        mask,
        w_proj_z=w_proj_z,
        b_proj_z=b_proj_z,
        w_ln_z=w_ln_z,
        return_z_proj=True,
        **weights,
    )
    assert z_cached is not None and z_cached.shape == (b_sz, n_heads, seq_len, seq_len)
    out2_c, z_none = attention_pair_bias_ref(
        single_2,
        z_cached,
        mask,
        w_proj_z=None,
        is_cached_z_proj=True,
        **weights,
    )
    assert z_none is None
    (g_pair_c,) = torch.autograd.grad((out1_c + out2_c).sum(), pair)

    fwd_diff = float((out2_u - out2_c).detach().abs().max())
    bwd_diff = float((g_pair_u - g_pair_c).detach().abs().max())
    print(
        f"  Multiplicity M={mult_m} sharing z_proj {tuple(z_cached.shape)}: max |uncached - cached| = {fwd_diff:.3e}"
    )
    print(f"  Accumulated dL/d(pair_repr) max |uncached - cached|    = {bwd_diff:.3e}")
    assert fwd_diff == 0.0 and bwd_diff < 1e-13

    # Quantify HBM pair-read traffic across 48 blocks (B=1, N=768, z_dim=128, H=16, M=16, BF16)
    n_tok, z_d, h_cnt, m_rep, n_blocks, bf16_b = 768, 128, 16, 16, 48, 2
    pair_bytes = n_tok * n_tok * z_d * bf16_b
    zproj_bytes = n_tok * n_tok * h_cnt * bf16_b

    hbm_naive_dup = n_blocks * m_rep * pair_bytes  # Replicating pair_repr per sample M
    hbm_uncached_bcast = n_blocks * (
        pair_bytes + zproj_bytes
    )  # Re-projecting each block
    hbm_cached_zproj = (
        pair_bytes + zproj_bytes
    ) + n_blocks * zproj_bytes  # Pre-projected once

    print(
        f"\n  [48-Block Diffusion / Pairformer Pair HBM Traffic (N={n_tok}, z_dim={z_d}, H={h_cnt}, M={m_rep}, BF16)]"
    )
    print(f"    Naive M-replicated pair reads   : {hbm_naive_dup / 1e9:7.2f} GB")
    print(f"    M-broadcast uncached pair reads : {hbm_uncached_bcast / 1e9:7.2f} GB")
    print(
        f"    Cached z_proj + M-broadcast     : {hbm_cached_zproj / 1e9:7.2f} GB "
        f"({hbm_naive_dup / hbm_cached_zproj:.1f}x vs naive, "
        f"{hbm_uncached_bcast / hbm_cached_zproj:.1f}x vs uncached!)"
    )
    assert hbm_cached_zproj < hbm_uncached_bcast < hbm_naive_dup


if __name__ == "__main__":
    audit_triangle_multiplicative_update()
    audit_triangle_attention_and_kv_lengths()
    audit_attention_pair_bias_cache_and_hbm()
    print(
        "\n[PASS] 06_triangle_attention_and_pair_bias.py completed all checks successfully."
    )
