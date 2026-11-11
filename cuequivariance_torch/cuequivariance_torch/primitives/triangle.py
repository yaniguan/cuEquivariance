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
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from cuequivariance_ops_torch import TriMulPrecision
except ImportError:
    import enum

    class TriMulPrecision(enum.IntEnum):  # type: ignore
        """Fallback precision enum when cuequivariance_ops_torch is not available."""

        NONE = -1
        DEFAULT = 0
        TF32 = 1
        TF32x3 = 2
        IEEE = 3


def mask_to_kv_lengths(mask: torch.Tensor) -> torch.Tensor:
    r"""
    Convert a right-padded triangle-attention mask to per-row key/value lengths.

    Args:
        mask (torch.Tensor): Boolean-like mask of shape (B, N, 1, 1, K). For B=1,
            can also be (N, 1, 1, K). The last dimension must be prefix-shaped:
            all True entries first, followed by all False entries.

    Returns:
        torch.Tensor: int32 tensor of shape (B, N, 1, 1, 1) containing each row's
        effective key/value length. Zero-length rows are allowed.

    Raises:
        ValueError: If the mask shape is not supported or the mask is not prefix-shaped.

    Note:
        This helper validates the prefix contract. For torch.compile-heavy code, compute
        lengths before the compiled region and pass them to :func:`triangle_attention`
        with ``kv_lengths=...``.
    """
    mask_bool = mask.to(dtype=torch.bool)
    while len(mask_bool.shape) < 5:
        mask_bool = mask_bool.unsqueeze(0)
    if mask_bool.ndim != 5 or mask_bool.shape[2:4] != (1, 1):
        raise ValueError(
            "mask_to_kv_lengths: mask must have shape (B, N, 1, 1, K) "
            f"after adding leading singleton dimensions, got {tuple(mask_bool.shape)}"
        )
    lengths = mask_bool.to(dtype=torch.int32).sum(dim=-1, keepdim=True).to(torch.int32)
    prefix = (
        torch.arange(mask_bool.shape[-1], device=mask_bool.device).view(
            1, 1, 1, 1, mask_bool.shape[-1]
        )
        < lengths
    )
    if not bool(torch.all(mask_bool == prefix).item()):
        raise ValueError("mask_to_kv_lengths: mask must be right-padded/prefix-shaped")
    return lengths.detach().contiguous()


def _get_ops_backend(op_name: str, is_cuda: bool, use_fallback: bool):
    """Return the CUDA backend callable when available and requested, else None."""
    if use_fallback:
        return None
    try:
        import cuequivariance_ops_torch as ops

        f = getattr(ops, op_name)
    except Exception:
        return None
    if not is_cuda and getattr(ops, "__spec__", None) is not None:
        return None
    return f


def _apply_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    norm_kind: str,
) -> torch.Tensor:
    dim = x.shape[-1]
    if norm_kind == "layer_norm":
        return F.layer_norm(x, (dim,), weight=weight, bias=bias, eps=eps)
    if norm_kind == "rms_norm":
        return F.rms_norm(x, (dim,), weight=weight, eps=eps)
    raise ValueError(
        f"Unsupported norm_kind: {norm_kind!r}. Expected 'layer_norm' or 'rms_norm'."
    )


def _triangle_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    return_aux: bool = False,
    *,
    kv_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.ndim not in (4, 5):
        raise ValueError(
            f"triangle_attention: expected 4D or 5D query tensor, got shape {tuple(q.shape)}"
        )
    squeeze_batch = q.ndim == 4
    if squeeze_batch:
        q = q.unsqueeze(0)
    if k.ndim == 4:
        k = k.unsqueeze(0)
    if v.ndim == 4:
        v = v.unsqueeze(0)
    if bias.ndim == 4:
        bias = bias.unsqueeze(0)
    if mask is not None and mask.ndim == 4:
        mask = mask.unsqueeze(0)
    if kv_lengths is not None and kv_lengths.ndim == 4:
        kv_lengths = kv_lengths.unsqueeze(0)

    B, N, H, Q, D = q.shape
    K = k.shape[-2]
    scale_val = 1.0 / math.sqrt(D) if scale is None else float(scale)

    if K == 0:
        output = torch.zeros_like(q)
        lse = torch.zeros((B, N, H, Q, 1), device=q.device, dtype=torch.float32)
        max_val = torch.zeros((B, N, H, Q, 1), device=q.device, dtype=torch.float32)
        if squeeze_batch:
            output = output.squeeze(0)
            lse = lse.squeeze(0)
            max_val = max_val.squeeze(0)
        return (output, lse, max_val) if return_aux else output

    acc_dtype = (
        torch.float64
        if (q.dtype == torch.float64 or bias.dtype == torch.float64)
        else torch.float32
    )
    q_acc = q.to(acc_dtype)
    k_acc = k.to(acc_dtype)
    v_acc = v.to(acc_dtype)
    bias_acc = bias.to(acc_dtype)

    scores = torch.matmul(q_acc * scale_val, k_acc.transpose(-2, -1)) + bias_acc

    valid_mask: Optional[torch.Tensor] = None
    if kv_lengths is not None:
        lengths = kv_lengths.clamp(min=0, max=K)
        k_idx = torch.arange(K, device=q.device).view(1, 1, 1, 1, K)
        valid_mask = k_idx < lengths
    elif mask is not None:
        valid_mask = mask.to(dtype=torch.bool)

    if valid_mask is not None:
        masked_scores = torch.where(valid_mask, scores, torch.full_like(scores, -1e9))
        any_valid = valid_mask.any(dim=-1, keepdim=True)
        attn = torch.where(
            any_valid,
            torch.softmax(masked_scores, dim=-1),
            torch.zeros_like(masked_scores),
        )
    else:
        masked_scores = scores
        attn = torch.softmax(masked_scores, dim=-1)

    output = torch.matmul(attn, v_acc).to(q.dtype)

    if return_aux:
        max_val = masked_scores.amax(dim=-1, keepdim=True).to(torch.float32)
        lse = torch.logsumexp(masked_scores, dim=-1, keepdim=True).to(torch.float32)
        if squeeze_batch:
            output = output.squeeze(0)
            lse = lse.squeeze(0)
            max_val = max_val.squeeze(0)
        return output, lse, max_val

    if squeeze_batch:
        output = output.squeeze(0)
    return output


def triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    return_aux: bool = False,
    *,
    kv_lengths: Optional[torch.Tensor] = None,
    use_fallback: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Triangle Attention

    .. math::

        \text{Attention}_q(Q, K, V, B, M) = \sum_k\left[\text{softmax}_k\left(\begin{cases}
        s\, Q_q \cdot K_k + B_{qk} & \text{if } M_k = 1 \\
        -10^9 & \text{otherwise}
        \end{cases}\right) V_k \right]


    Args:
        q (torch.Tensor): Query tensor of shape (B, N, H, Q, D). For B=1, can also be (N, H, Q, D).
        k (torch.Tensor): Key tensor of shape (B, N, H, K, D). For B=1, can also be (N, H, K, D).
        v (torch.Tensor): Value tensor of shape (B, N, H, K, D). For B=1, can also be (N, H, K, D).
        bias (torch.Tensor): Bias tensor of shape (B, 1, H, Q, K), For B=1, can also be (1, H, Q, K).
            Will be cast to float32 for standard kernels. On Blackwell GPUs (sm100f, compute
            capability 10.0 or 10.3), will be cast to match q/k/v dtype (bf16/fp16) for best
            performance.
        mask (torch.Tensor, optional): Mask tensor of shape (B, N, 1, 1, K). For B=1, can also be (N, 1, 1, K).
            Will be cast to bool internally. Dense masks accept any pattern and route to the
            correct fallback path. For right-padded masks, pass ``kv_lengths`` instead to use
            the Blackwell sm100f length fast path.
        scale (float, optional): Float scale for q (s in the equation). If None, value 1/sqrt(d) is used.
        return_aux (bool): If True, two auxiliary tensors are returned along with the result.
            Defaults to False.
        kv_lengths (torch.Tensor, optional): int32 tensor of shape (B, N, 1, 1, 1)
            containing each row's effective K length. This represents a right-padded /
            prefix mask: positions ``j < kv_lengths[b, n]`` are valid and later positions
            are masked. Pass either ``mask`` or ``kv_lengths``, not both. On supported
            Blackwell cu13 builds, ``kv_lengths`` selects the sm100f length fast path.
            Each length must be in ``[0, K]``; values greater than ``K`` are clamped
            to ``K`` (the row then attends the full key sequence).
        use_fallback (bool, optional): If True, force the pure-PyTorch reference
            fallback even when CUDA kernels are available. Defaults to False.

    Note:
        - B: batch size
        - N: number of tokens
        - H: number of heads
        - Q: number of query tokens
        - K: number of key tokens
        - D: attention dimension

    Returns:
        - output(torch.Tensor): Output tensor of shape (B, N, H, Q, D). dtype=q.dtype
        - lse(torch.Tensor): Auxiliary result (for special use only). dtype=float32
        - max(torch.Tensor): Auxiliary result (for special use only). dtype=float32

    Notes:
        (1) Context is saved for backward pass. You don't need to save it manually.
        (2) Kernel precision (fp32, bf16, fp16) is based on input dtypes. For tf32, set it from torch global scope
        (3) Triangle attention kernel supports: all hidden_dim<=32 and divisible by 4 for tf32/fp32, and for all hidden_dim<=128 and divisible by 8 for bf16/fp16 (standard kernels). On Blackwell GPUs (compute capability 10.0 or 10.3), the sm100f kernel supports hidden_dim<=256 for forward passes and hidden_dim<=128 for backward passes. In the rare instance that the kernel does not support an input config, fallback to torch is enabled instead of erroring out.
        (4) Blackwell-optimized kernels (for compute capabilities 10.0 and 10.3) provide superior performance especially for long sequences and higher head dimensions. These kernels require the key/value sequence length K to be a multiple of 8 for the forward pass; pad the sequence if necessary. Use ``kv_lengths`` for right-padded sequence masks to select the sm100f length fast path. A dense ``mask`` without ``kv_lengths`` remains correct for arbitrary or holey patterns, but it routes to the fallback path.

    Example:
        >>> import torch
        >>> import math
        >>> from cuequivariance_torch import mask_to_kv_lengths, triangle_attention
        >>> if torch.cuda.is_available():  # doctest: +SKIP
        ...     device = torch.device("cuda")
        ...     # Set up dimensions
        ...     batch_size, seq_len, num_heads, hidden_dim = 1, 128, 2, 32
        ...     # Create input tensors on GPU with float16 precision
        ...     q = torch.randn(batch_size, seq_len, num_heads, seq_len, hidden_dim,
        ...                     device=device, dtype=torch.float16, requires_grad=True)
        ...     k = torch.randn(batch_size, seq_len, num_heads, seq_len, hidden_dim,
        ...                     device=device, dtype=torch.float16, requires_grad=True)
        ...     v = torch.randn(batch_size, seq_len, num_heads, seq_len, hidden_dim,
        ...                     device=device, dtype=torch.float16, requires_grad=True)
        ...     bias = torch.randn(batch_size, 1, num_heads, seq_len, seq_len,
        ...                        device=device, dtype=torch.float32, requires_grad=True)
        ...     # Right-padded sequence mask: valid tokens first, padding last.
        ...     seq_lengths = torch.tensor([96], device=device, dtype=torch.int32)
        ...     positions = torch.arange(seq_len, device=device)
        ...     row_valid = positions.view(1, seq_len) < seq_lengths.view(batch_size, 1)
        ...     col_valid = positions.view(1, seq_len) < seq_lengths.view(batch_size, 1)
        ...     mask = row_valid.view(batch_size, seq_len, 1, 1, 1) & col_valid.view(
        ...         batch_size, 1, 1, 1, seq_len)
        ...     kv_lengths = mask_to_kv_lengths(mask)
        ...     # Calculate scale
        ...     scale = 1 / math.sqrt(hidden_dim)
        ...     # Forward pass using the Blackwell sm100f length fast path when available.
        ...     output, lse, max_val = triangle_attention(
        ...         q=q, k=k, v=v, bias=bias, scale=scale, return_aux=True,
        ...         kv_lengths=kv_lengths)
        ...     # Arbitrary dense masks are still correct; they use the fallback path.
        ...     arbitrary_mask = torch.rand(batch_size, seq_len, 1, 1, seq_len,
        ...                                 device=device) < 0.5
        ...     fallback_output = triangle_attention(
        ...         q=q, k=k, v=v, bias=bias, mask=arbitrary_mask, scale=scale)
        ...     print(output.shape)  # torch.Size([1, 128, 2, 128, 32])
        ...     # Create gradient tensor and perform backward pass
        ...     grad_out = torch.randn_like(output)
        ...     output.backward(grad_out)
        ...     # Access gradients
        ...     print(q.grad.shape)  # torch.Size([1, 128, 2, 128, 32])
        ...     print(k.grad.shape)  # torch.Size([1, 128, 2, 128, 32])
        ...     print(v.grad.shape)  # torch.Size([1, 128, 2, 128, 32])
        ...     print(bias.grad.shape)  # torch.Size([1, 1, 2, 128, 128])
        torch.Size([1, 128, 2, 128, 32])
        torch.Size([1, 128, 2, 128, 32])
        torch.Size([1, 128, 2, 128, 32])
        torch.Size([1, 128, 2, 128, 32])
        torch.Size([1, 1, 2, 128, 128])
    """

    if mask is not None and kv_lengths is not None:
        raise ValueError(
            "triangle_attention: pass either `mask` or `kv_lengths`, not both. "
            "`kv_lengths` selects the SM100f length fast path; a dense `mask` uses "
            "the fallback path."
        )

    f = _get_ops_backend("triangle_attention", q.is_cuda, use_fallback)
    if f is not None:
        return f(q, k, v, bias, mask, scale, return_aux, kv_lengths=kv_lengths)
    return _triangle_attention_fallback(
        q,
        k,
        v,
        bias,
        mask=mask,
        scale=scale,
        return_aux=return_aux,
        kv_lengths=kv_lengths,
    )


def _triangle_multiplicative_update_fallback(
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
    precision: Optional[TriMulPrecision | str] = None,
) -> torch.Tensor:
    if x.ndim < 3 or x.shape[-3] != x.shape[-2]:
        raise ValueError(
            f"triangle_multiplicative_update: expected x of shape (..., N, N, C), got {tuple(x.shape)}"
        )

    orig_dtype = x.dtype
    is_bf16_mode = (
        isinstance(precision, str) and precision.lower() == "bfloat16"
    ) or getattr(precision, "name", None) == "BFLOAT16"

    if is_bf16_mode:
        x = x.to(torch.bfloat16)
        if mask is not None:
            mask = mask.to(torch.bfloat16)
        if norm_in_weight is not None:
            norm_in_weight = norm_in_weight.to(torch.bfloat16)
        if norm_in_bias is not None:
            norm_in_bias = norm_in_bias.to(torch.bfloat16)
        if p_in_weight is not None:
            p_in_weight = p_in_weight.to(torch.bfloat16)
        if p_in_bias is not None:
            p_in_bias = p_in_bias.to(torch.bfloat16)
        if g_in_weight is not None:
            g_in_weight = g_in_weight.to(torch.bfloat16)
        if g_in_bias is not None:
            g_in_bias = g_in_bias.to(torch.bfloat16)
        if norm_out_weight is not None:
            norm_out_weight = norm_out_weight.to(torch.bfloat16)
        if norm_out_bias is not None:
            norm_out_bias = norm_out_bias.to(torch.bfloat16)
        if p_out_weight is not None:
            p_out_weight = p_out_weight.to(torch.bfloat16)
        if p_out_bias is not None:
            p_out_bias = p_out_bias.to(torch.bfloat16)
        if g_out_weight is not None:
            g_out_weight = g_out_weight.to(torch.bfloat16)
        if g_out_bias is not None:
            g_out_bias = g_out_bias.to(torch.bfloat16)

    N = x.shape[-2]
    C = x.shape[-1]
    batch_dims = x.shape[:-3]

    if mask is not None:
        if mask.ndim < 2 or mask.shape[-2:] != (N, N):
            raise ValueError(
                f"triangle_multiplicative_update: expected mask of shape (..., {N}, {N}), "
                f"got {tuple(mask.shape)}"
            )
        batch_dims = torch.broadcast_shapes(batch_dims, mask.shape[:-2])
        x = x.expand(batch_dims + (N, N, C))
        mask = mask.expand(batch_dims + (N, N))

    if p_in_weight is not None:
        if (
            p_in_weight.ndim != 2
            or p_in_weight.shape[0] % 2 != 0
            or p_in_weight.shape[1] != C
        ):
            raise ValueError(
                f"p_in_weight must have shape (2*D, {C}), got {tuple(p_in_weight.shape)}"
            )
        D = p_in_weight.shape[0] // 2
    elif g_in_weight is not None:
        if (
            g_in_weight.ndim != 2
            or g_in_weight.shape[0] % 2 != 0
            or g_in_weight.shape[1] != C
        ):
            raise ValueError(
                f"g_in_weight must have shape (2*D, {C}), got {tuple(g_in_weight.shape)}"
            )
        D = g_in_weight.shape[0] // 2
    elif p_in_bias is not None:
        D = p_in_bias.shape[0] // 2
    elif g_in_bias is not None:
        D = g_in_bias.shape[0] // 2
    elif norm_out_weight is not None:
        D = norm_out_weight.shape[0]
    elif norm_out_bias is not None:
        D = norm_out_bias.shape[0]
    elif p_out_weight is not None:
        D = p_out_weight.shape[1]
    else:
        D = C

    if p_out_weight is not None:
        C_out = p_out_weight.shape[0]
    elif g_out_weight is not None:
        C_out = g_out_weight.shape[0]
    elif p_out_bias is not None:
        C_out = p_out_bias.shape[0]
    elif g_out_bias is not None:
        C_out = g_out_bias.shape[0]
    else:
        C_out = C

    x_in = F.layer_norm(x, (C,), weight=norm_in_weight, bias=norm_in_bias, eps=eps)

    if p_in_weight is not None:
        p_proj = F.linear(x_in, p_in_weight, p_in_bias)
    else:
        if D != C:
            raise ValueError(f"p_in_weight is required when D ({D}) != C ({C})")
        p_proj = torch.cat([x_in, x_in], dim=-1)
        if p_in_bias is not None:
            p_proj = p_proj + p_in_bias

    if g_in_weight is not None:
        g_proj = F.linear(x_in, g_in_weight, g_in_bias)
    else:
        if D != C:
            raise ValueError(f"g_in_weight is required when D ({D}) != C ({C})")
        g_proj = torch.cat([x_in, x_in], dim=-1)
        if g_in_bias is not None:
            g_proj = g_proj + g_in_bias

    ab = torch.sigmoid(g_proj) * p_proj
    if mask is not None:
        ab = ab * mask.unsqueeze(-1).to(dtype=ab.dtype)

    a, b = ab.chunk(2, dim=-1)

    a_perm = a.movedim(-1, -3)
    b_perm = b.movedim(-1, -3)
    if direction == "outgoing":
        tri = torch.matmul(a_perm, b_perm.transpose(-2, -1)).movedim(-3, -1)
    else:
        tri = torch.matmul(a_perm.transpose(-2, -1), b_perm).movedim(-3, -1)

    x_out = F.layer_norm(tri, (D,), weight=norm_out_weight, bias=norm_out_bias, eps=eps)

    if p_out_weight is not None:
        out_proj = F.linear(x_out, p_out_weight, p_out_bias)
    else:
        if C_out != D:
            raise ValueError(
                f"p_out_weight is required when C_out ({C_out}) != D ({D})"
            )
        out_proj = x_out + p_out_bias if p_out_bias is not None else x_out

    if g_out_weight is not None:
        out_gate_proj = F.linear(x_in, g_out_weight, g_out_bias)
    else:
        if C_out != C:
            raise ValueError(
                f"g_out_weight is required when C_out ({C_out}) != C ({C})"
            )
        out_gate_proj = x_in + g_out_bias if g_out_bias is not None else x_in

    out = torch.sigmoid(out_gate_proj) * out_proj
    return out.to(orig_dtype)


def triangle_multiplicative_update(
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
    precision: Optional[TriMulPrecision | str] = None,
    *,
    use_fallback: bool = False,
) -> torch.Tensor:
    """Apply triangle multiplicative update operation.

    This function performs a triangle multiplicative update operation, which is a key component
    in the AlphaFold2 architecture. The operation consists of:

    1. Input normalization and gating
    2. Triangular projection (either outgoing or incoming)
    3. Output normalization and gating

    The function supports both ahead-of-time (AOT) tuning and just-in-time (JIT) tuning.
    Auto-tuning behavior can be controlled through environment variables:

    - Quick testing: Default configuration where tuning configs, if existent, are looked-up. If not, then falls back to default kernel parameters. No tuning is performed.
    - On-Demand tuning: Set `CUEQ_TRITON_TUNING= "ONDEMAND"` to auto-tune for new shapes encountered on first run (may take several minutes)
    - AOT tuning: Set `CUEQ_TRITON_TUNING= "AOT"` to perform full ahead-of-time tuning for optimal performance **(may take several hours)**
    - Ignore user cache: Set CUEQ_TRITON_IGNORE_EXISTING_CACHE to ignore both the default settings that come with the package and any user-local settings previously saved with AOT/ONDEMAND tuning. May be used to regenerate optimal settings for a particular setup.
    - Cache directory: Set `CUEQ_TRITON_CACHE_DIR` to specify where tuning configurations are stored
    - Note: When using Docker with default or on-demand tuning enabled, commit the container to persist tuning changes

    Args:
        x (torch.Tensor): Input pair tensor of shape (B, N, N, C) where:
            B is the batch size
            N is the sequence length
            C is the pair width (the channel size of the pair representation)
            The hidden width D (the channel size inside the update) is set by the
            projection weights: D = p_in_weight.shape[0] // 2. Often C = D.
        direction (str): Direction of the triangular projection. Must be either "outgoing" or "incoming".
        mask (torch.Tensor): Optional Mask tensor of shape (B, N, N) for masking the output.
        norm_in_weight (torch.Tensor): Optional weight tensor for input normalization of shape (C,).
        norm_in_bias (torch.Tensor): Optional bias tensor for input normalization of shape (C,).
        p_in_weight (torch.Tensor): Optional weight tensor for input projection of shape (2D, C).
        p_in_bias (torch.Tensor): Optional bias tensor for input projection of shape (2D,).
        g_in_weight (torch.Tensor): Optional weight tensor for input gating of shape (2D, C).
        g_in_bias (torch.Tensor): Optional bias tensor for input gating of shape (2D,).
        norm_out_weight (torch.Tensor): Optional weight tensor for output normalization of shape (D,).
        norm_out_bias (torch.Tensor): Optional bias tensor for output normalization of shape (D,).
        p_out_weight (torch.Tensor): Optional weight tensor for output projection of shape (C, D).
        p_out_bias (torch.Tensor): Optional bias tensor for output projection of shape (C,).
        g_out_weight (torch.Tensor): Optional weight tensor for output gating of shape (C, C).
        g_out_bias (torch.Tensor): Optional bias tensor for output gating of shape (C,).
        eps (float, optional): Small constant for numerical stability in normalization. Defaults to 1e-5.
        precision (TriMulPrecision or str, optional): Precision mode for matrix multiplications.
            Available options:
            - None: Defaults to triton language dot's default for non-32b input and for 32b input, tf32/tf32x3 based on 1/0 value set in torch.backends.cuda.matmul.allow_tf32
            - IEEE: Use IEEE 754 precision
            - "bfloat16": Round the input pair, the mask and all weights and biases to BF16 before
              the operation (the LayerNorms then read BF16 values). The rounding is differentiable,
              so gradients reach FP32 parameters. Under BF16 autocast with precision None, the
              LayerNorms read the FP32 pair and only the matrix multiplications use BF16.
        use_fallback (bool, optional): If True, force the pure-PyTorch reference
            fallback even when CUDA kernels are available. Defaults to False.

    Returns:
        Output tensor of shape (B, N, N, C)

    Notes:
        (1) Context is saved for backward pass. You don't need to save it manually.
        (2) Kernel precision (fp32, bf16, fp16) is based on input dtypes. For tf32, set it from torch global scope using torch.backends.cuda.matmul.allow_tf32
        (3) **Limitation**: Currently only supports hidden_dim values that are multiples of 32.
        (4) We have moved away from the default round-towards-zero (RZ) implementation to round-nearest (RN) for better tf32 accuracy in cuex.triangle_multiplicative_update. In rare circumstances, this may cause minor differences in results observed.
        (5) When using torch compile, use `cueuivariance_ops_torch.init_triton_cache()` to initialize triton cache before calling torch compiled triangular multiplicative update.
        (6) Although the example demonstrates the most common case of one batch dimension, the API supports variable number of leading batch dimensions.

    Example:
        >>> import torch
        >>> from cuequivariance_torch import triangle_multiplicative_update
        >>> if torch.cuda.is_available():  # doctest: +SKIP
        ...     device = torch.device("cuda")
        ...     batch_size, seq_len, hidden_dim = 1, 128, 128
        ...     # Create input tensor
        ...     x = torch.randn(batch_size, seq_len, seq_len, hidden_dim, requires_grad=True, device=device)
        ...     # Create mask (1 for valid positions, 0 for masked)
        ...     mask = torch.ones(batch_size, seq_len, seq_len, device=device)
        ...     # Perform triangular multiplication
        ...     output = triangle_multiplicative_update(
        ...         x=x,
        ...         direction="outgoing",  # or "incoming"
        ...         mask=mask,
        ...     )
        ...     print(output.shape)  # torch.Size([1, 128, 128, 128])
        ...     # Create gradient tensor and perform backward pass
        ...     grad_out = torch.randn_like(output)
        ...     output.backward(grad_out)
        ...     # Access gradients
        ...     print(x.grad.shape)  # torch.Size([1, 128, 128, 128])
        torch.Size([1, 128, 128, 128])
        torch.Size([1, 128, 128, 128])
    """
    if direction not in ("outgoing", "incoming"):
        raise ValueError(
            f"direction must be either 'outgoing' or 'incoming', got {direction!r}"
        )

    f = _get_ops_backend("triangle_multiplicative_update", x.is_cuda, use_fallback)
    if f is not None:
        return f(
            x,
            direction,
            mask=mask,
            norm_in_weight=norm_in_weight,
            norm_in_bias=norm_in_bias,
            p_in_weight=p_in_weight,
            p_in_bias=p_in_bias,
            g_in_weight=g_in_weight,
            g_in_bias=g_in_bias,
            norm_out_weight=norm_out_weight,
            norm_out_bias=norm_out_bias,
            p_out_weight=p_out_weight,
            p_out_bias=p_out_bias,
            g_out_weight=g_out_weight,
            g_out_bias=g_out_bias,
            eps=eps,
            precision=precision,
        )
    return _triangle_multiplicative_update_fallback(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        p_in_bias=p_in_bias,
        g_in_weight=g_in_weight,
        g_in_bias=g_in_bias,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        p_out_bias=p_out_bias,
        g_out_weight=g_out_weight,
        g_out_bias=g_out_bias,
        eps=eps,
        precision=precision,
    )


def _attention_pair_bias_fallback(
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
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if single_repr.ndim != 3:
        raise ValueError(
            f"attention_pair_bias: expected single_repr of shape (B * M, N, D), "
            f"got {tuple(single_repr.shape)}"
        )
    BM, N, _ = single_repr.shape
    H = num_heads
    if H <= 0:
        raise ValueError(f"attention_pair_bias: num_heads must be positive, got {H}")

    if is_cached_z_proj:
        if pair_repr.ndim != 4 or pair_repr.shape[1:] != (H, N, N):
            raise ValueError(
                f"attention_pair_bias: with is_cached_z_proj=True, expected pair_repr "
                f"of shape (B, {H}, {N}, {N}), got {tuple(pair_repr.shape)}"
            )
    else:
        if pair_repr.ndim != 4 or pair_repr.shape[1:3] != (N, N):
            raise ValueError(
                f"attention_pair_bias: expected pair_repr of shape (B, {N}, {N}, z_dim), "
                f"got {tuple(pair_repr.shape)}"
            )

    B = pair_repr.shape[0]
    if B <= 0 or BM % B != 0:
        raise ValueError(
            f"attention_pair_bias: single_repr first dim ({BM}) must be divisible by "
            f"pair_repr batch dim ({B})"
        )
    M = BM // B

    a_norm = _apply_norm(single_repr, w_ln_a, b_ln_a, eps, norm_kind)
    q = F.linear(a_norm, w_proj_q, b_proj_q)
    k = F.linear(a_norm, w_proj_k, b_proj_k)
    v = F.linear(a_norm, w_proj_v, b_proj_v)
    g = torch.sigmoid(F.linear(a_norm, w_proj_g, b_proj_g))

    if w_ln_q is not None or b_ln_q is not None:
        q = _apply_norm(q, w_ln_q, b_ln_q, eps, norm_kind)
    if w_ln_k is not None or b_ln_k is not None:
        k = _apply_norm(k, w_ln_k, b_ln_k, eps, norm_kind)

    inner = q.shape[-1]
    if inner % H != 0:
        raise ValueError(
            f"attention_pair_bias: projected inner dim ({inner}) must be divisible by "
            f"num_heads ({H})"
        )
    DH = inner // H

    q_heads = q.view(B, M, N, H, DH).permute(0, 1, 3, 2, 4)
    k_heads = k.view(B, M, N, H, DH).permute(0, 1, 3, 2, 4)
    v_heads = v.view(B, M, N, H, DH).permute(0, 1, 3, 2, 4)

    if is_cached_z_proj:
        z_proj = pair_repr
    else:
        assert w_proj_z is not None
        z_in = (
            _apply_norm(pair_repr, w_ln_z, b_ln_z, eps, norm_kind)
            if (w_ln_z is not None or b_ln_z is not None)
            else pair_repr
        )
        z_proj = F.linear(z_in, w_proj_z, b_proj_z).permute(0, 3, 1, 2).contiguous()

    scale = 1.0 / math.sqrt(DH) if attn_scale is None else float(attn_scale)
    acc_dtype = (
        torch.float64
        if (single_repr.dtype == torch.float64 or pair_repr.dtype == torch.float64)
        else torch.float32
    )
    q_acc = q_heads.to(acc_dtype)
    k_acc = k_heads.to(acc_dtype)
    v_acc = v_heads.to(acc_dtype)

    scores = torch.matmul(q_acc * scale, k_acc.transpose(-2, -1)) + z_proj.to(
        acc_dtype
    ).unsqueeze(1)

    if mask is not None:
        if mask.shape == (B, N):
            valid_mask = (mask != 0).view(B, 1, 1, 1, N)
        elif mask.shape == (BM, N):
            valid_mask = (mask != 0).view(B, M, 1, 1, N)
        else:
            raise ValueError(
                f"attention_pair_bias: expected mask of shape ({B}, {N}) or "
                f"({BM}, {N}), got {tuple(mask.shape)}"
            )
        if N == 0:
            attn = torch.zeros_like(scores)
        else:
            any_valid = valid_mask.any(dim=-1, keepdim=True)
            safe_scores = torch.where(
                valid_mask,
                scores,
                torch.where(
                    any_valid,
                    torch.full_like(scores, -float(inf)),
                    torch.zeros_like(scores),
                ),
            )
            attn = torch.where(
                any_valid,
                torch.softmax(safe_scores, dim=-1),
                torch.zeros_like(scores),
            )
    else:
        attn = torch.softmax(scores, dim=-1) if N > 0 else torch.zeros_like(scores)

    attn_out = torch.matmul(attn, v_acc).to(single_repr.dtype)
    attn_flat = attn_out.permute(0, 1, 3, 2, 4).reshape(BM, N, inner)
    output = F.linear(g * attn_flat, w_proj_o, b_proj_o)

    return output, (z_proj if return_z_proj else None)


def attention_pair_bias(
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
    use_fallback: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute attention with a pairwise bias.

    Normalizes the single and pair representations according to ``norm_kind``,
    then computes the Q/K/V and gating projections, pair-bias term, biased and
    masked softmax attention, output gate, and output projection.
    The backend selects a supported, profitable optimized route for the current
    inputs and otherwise uses the native PyTorch implementation (see Notes). Q, K,
    V and the gate are all projected from the same normalized ``single_repr``
    (there is no separate ``s`` input).

    Dimensions: ``B`` batch, ``M`` multiplicity (single-rep replicas), ``N`` sequence
    length, ``D`` single-rep feature dim, ``H`` heads, ``DH`` head dim (so the
    attention inner dim is ``H * DH``), ``z_dim`` pair feature dim.

    Args:
        single_repr: Single/token representation of shape (B * M, N, D).
            Normalization and the Q/K/V/gate projections are applied to this tensor
            inside the op (the gate is computed from the normalized representation).
        pair_repr: Pairwise tensor of shape (B, N, N, z_dim). When
            ``is_cached_z_proj`` is True, ``pair_repr`` is instead the
            already-projected bias of shape (B, H, N, N).
        mask: Attention mask of shape (B, N) or (B * M, N) (0 = masked, 1 = valid).
            If None, all positions are treated as valid.
        num_heads: Number of attention heads.
        w_ln_a: Normalization weight for ``single_repr`` of shape (D,).
        b_ln_a: Normalization bias for ``single_repr`` of shape (D,). May be None.
        w_proj_q: Weight for the query projection of shape (H * DH, D).
        b_proj_q: Bias for the query projection of shape (H * DH,). May be None.
        w_proj_k: Weight for the key projection of shape (H * DH, D).
        w_proj_v: Weight for the value projection of shape (H * DH, D).
        w_proj_g: Weight for the gating projection of shape (H * DH, D).
        w_proj_o: Weight for the output projection of shape (D, H * DH).
        w_proj_z: Weight for the pair projection of shape (H, z_dim). May be None
            when ``is_cached_z_proj`` is True because ``pair_repr`` is already
            projected. Defaults to None.
        b_proj_o: Bias for the output projection of shape (D,). Defaults to None.
        b_proj_z: Bias for the pair projection of shape (H,). Defaults to None.
        w_ln_z: Normalization weight for ``pair_repr`` of shape (z_dim,). May be None.
        b_ln_z: Normalization bias for ``pair_repr`` of shape (z_dim,). May be None.
        inf: Large value used for masking invalid attention positions. Defaults to 1e6.
        eps: Epsilon value for normalization. Defaults to 1e-5.
        attn_scale: Scaling factor for attention scores. If None, uses
            1/sqrt(head_dim). Defaults to None.
        return_z_proj: Whether to return the projected pair tensor as the second
            output. Defaults to False.
        is_cached_z_proj: Whether ``pair_repr`` is already projected and cached. If
            True, ``pair_repr`` should be of shape (B, H, N, N). Defaults to False.
        b_proj_k: Optional key-projection bias of shape (H * DH,). Defaults to None.
        b_proj_v: Optional value-projection bias of shape (H * DH,). Defaults to None.
        w_ln_q: Optional weight for projected-query LayerNorm of shape (H * DH,).
            The norm is applied over H * DH before splitting heads. Defaults to None.
        b_ln_q: Optional bias for projected-query LayerNorm of shape (H * DH,).
            Weight and bias are independently optional. Defaults to None.
        w_ln_k: Optional weight for projected-key LayerNorm of shape (H * DH,).
            The norm is applied over H * DH before splitting heads. Defaults to None.
        b_ln_k: Optional bias for projected-key LayerNorm of shape (H * DH,).
            Weight and bias are independently optional. Defaults to None.
        b_proj_g: Optional gating-projection bias of shape (H * DH,). Defaults to None.
        norm_kind: Normalization applied to the single, optional Q/K, and pair
            representations. ``"layer_norm"`` preserves the existing behavior;
            ``"rms_norm"`` requires every normalization bias to be ``None``.
        use_fallback: If True, force the pure-PyTorch reference fallback even when
            CUDA kernels are available. Defaults to False.

    Returns:
        - output (torch.Tensor): Attention output of shape (B * M, N, D) with the
          pairwise bias applied.
        - z_proj (torch.Tensor | None): Projected pair tensor of shape (B, H, N, N)
          reusable via ``is_cached_z_proj``, or ``None`` when ``return_z_proj`` is
          False.

    Notes:
        - The backend chooses among its supported optimized routes according to the
          device, dtype, shape, gradient mode, and projection options. It uses the
          native PyTorch implementation whenever no optimized route is supported
          and profitable. The exact routing policy is an implementation detail.
        - Multiplicity (M) is inferred from the shapes so multiple single-rep
          replicas can share one pair representation in a single forward pass.
    """

    if w_proj_z is None and not is_cached_z_proj:
        raise ValueError("w_proj_z is required unless is_cached_z_proj=True")
    if norm_kind not in ("layer_norm", "rms_norm"):
        raise ValueError(
            f"Unsupported norm_kind: {norm_kind!r}. Expected 'layer_norm' or 'rms_norm'."
        )
    if norm_kind == "rms_norm" and any(
        b is not None for b in (b_ln_a, b_ln_z, b_ln_q, b_ln_k)
    ):
        raise ValueError(
            "norm_kind='rms_norm' requires all normalization biases "
            "(b_ln_a, b_ln_z, b_ln_q, b_ln_k) to be None"
        )

    f = _get_ops_backend("attention_pair_bias", single_repr.is_cuda, use_fallback)
    if f is not None:
        norm_kind_kwargs = {} if norm_kind == "layer_norm" else {"norm_kind": norm_kind}
        return f(
            single_repr,
            pair_repr,
            mask,
            num_heads,
            w_ln_a,
            b_ln_a,
            w_proj_q,
            b_proj_q,
            w_proj_k,
            w_proj_v,
            w_proj_g,
            w_proj_o,
            w_proj_z,
            b_proj_o=b_proj_o,
            b_proj_z=b_proj_z,
            w_ln_z=w_ln_z,
            b_ln_z=b_ln_z,
            inf=inf,
            eps=eps,
            attn_scale=attn_scale,
            return_z_proj=return_z_proj,
            is_cached_z_proj=is_cached_z_proj,
            b_proj_k=b_proj_k,
            b_proj_v=b_proj_v,
            w_ln_q=w_ln_q,
            b_ln_q=b_ln_q,
            w_ln_k=w_ln_k,
            b_ln_k=b_ln_k,
            b_proj_g=b_proj_g,
            **norm_kind_kwargs,
        )

    return _attention_pair_bias_fallback(
        single_repr,
        pair_repr,
        mask,
        num_heads,
        w_ln_a,
        b_ln_a,
        w_proj_q,
        b_proj_q,
        w_proj_k,
        w_proj_v,
        w_proj_g,
        w_proj_o,
        w_proj_z=w_proj_z,
        b_proj_o=b_proj_o,
        b_proj_z=b_proj_z,
        w_ln_z=w_ln_z,
        b_ln_z=b_ln_z,
        inf=inf,
        eps=eps,
        attn_scale=attn_scale,
        return_z_proj=return_z_proj,
        is_cached_z_proj=is_cached_z_proj,
        b_proj_k=b_proj_k,
        b_proj_v=b_proj_v,
        w_ln_q=w_ln_q,
        b_ln_q=b_ln_q,
        w_ln_k=w_ln_k,
        b_ln_k=b_ln_k,
        b_proj_g=b_proj_g,
        norm_kind=norm_kind,
    )
