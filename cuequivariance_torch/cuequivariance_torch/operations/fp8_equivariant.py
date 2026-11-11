# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""H100 O(3)-equivariant FP8 block quantization and linear layers."""

from __future__ import annotations

import math
from typing import Optional

import torch
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
)

import cuequivariance as cue
from cuequivariance import descriptors

_SUPPORTED_FP8_DTYPES: tuple[torch.dtype, ...] = tuple(
    getattr(torch, name)
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if hasattr(torch, name)
)


def _coerce_irreps(irreps: cue.Irreps | str) -> cue.Irreps:
    if isinstance(irreps, cue.Irreps):
        return irreps
    if isinstance(irreps, str):
        return cue.Irreps(cue.O3, irreps)
    (parsed,) = default_irreps(irreps)
    return parsed


def _coerce_layout(
    layout: Optional[cue.IrrepsLayout | str],
    default: cue.IrrepsLayout = cue.ir_mul,
) -> cue.IrrepsLayout:
    if layout is None:
        return default
    if isinstance(layout, str):
        return cue.IrrepsLayout[layout]
    return layout


def _validate_fp8_dtype(fp8_dtype: torch.dtype) -> float:
    if fp8_dtype not in _SUPPORTED_FP8_DTYPES:
        raise ValueError(
            f"Unsupported FP8 dtype {fp8_dtype}. Expected one of {_SUPPORTED_FP8_DTYPES}."
        )
    return float(torch.finfo(fp8_dtype).max)


def irrep_fp8_quantize(
    x: torch.Tensor,
    irreps: cue.Irreps | str,
    *,
    layout: cue.IrrepsLayout = cue.ir_mul,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    granularity: str = "per_irrep",
    eps: float = 1e-12,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize an equivariant tensor into FP8 using :math:`O(3)`-invariant :math:`L_2` block scales.

    Standard :math:`L_\infty` (`amax`) scaling across magnetic components :math:`m \in [-\ell, \ell]`
    is not rotationally invariant because :math:`\|D^{(\ell)}(R)\mathbf{x}\|_\infty \neq \|\mathbf{x}\|_\infty`.
    Instead, this function derives the FP8 scale from the Euclidean :math:`L_2` norm
    :math:`s = \max(\|\mathbf{x}\|_2, \varepsilon) / \text{fp8\_max}`, which is strictly invariant under
    orthogonal Wigner-:math:`D` transformations (:math:`\|D^{(\ell,p)}(g)\mathbf{x}\|_2 = \|\mathbf{x}\|_2`)
    and guarantees zero saturation (:math:`|x_m| \le \|\mathbf{x}\|_2 \le s \cdot \text{fp8\_max}`) under
    arbitrary :math:`O(3)` rotations and reflections.

    Args:
        x: Input tensor of shape ``(*batch, irreps.dim)``.
        irreps: Irreducible representations describing the last dimension of ``x``.
        layout: Memory layout of ``x`` (``cue.ir_mul`` or ``cue.mul_ir``). Handled natively
            without transposing memory.
        fp8_dtype: Target FP8 dtype (``torch.float8_e4m3fn`` or ``torch.float8_e5m2``).
        granularity: Quantization block granularity:
            - ``"per_irrep"``: One :math:`O(3)`-invariant :math:`L_2` scale per ``(*batch, irrep_segment)``
              shared across ``(mul, ir.dim)``. Scale shape: ``(*batch, len(irreps))``.
            - ``"per_channel"``: One :math:`O(3)`-invariant :math:`L_2` scale per ``(*batch, channel)``
              shared across ``ir.dim``. Scale shape: ``(*batch, irreps.num_irreps)``.
            - ``"per_atom"`` (or ``"per_row"``): One :math:`L_2` scale across the full ``irreps.dim``
              feature vector per batch element. Scale shape: ``(*batch, 1)``.
            - ``"per_tensor"``: One global :math:`O(3)`-invariant scale ``(1,)`` from the maximum
              per-element :math:`L_2` norm across the batch.
        eps: Numerical floor on the :math:`L_2` norm to prevent division by zero.
        scale_dtype: Floating-point dtype for the returned scale tensor (default: ``torch.float32``).

    Returns:
        Tuple ``(q_x, scale)`` where ``q_x`` has shape ``(*batch, irreps.dim)`` and dtype ``fp8_dtype``,
        and ``scale`` has dtype ``scale_dtype``.
    """
    irreps_obj = _coerce_irreps(irreps)
    layout_obj = _coerce_layout(layout, default=cue.ir_mul)
    fp8_max = _validate_fp8_dtype(fp8_dtype)

    if x.ndim < 1 or x.shape[-1] != irreps_obj.dim:
        raise ValueError(
            f"Expected input tensor last dimension {irreps_obj.dim} for irreps '{irreps_obj}', "
            f"got shape {tuple(x.shape)}."
        )

    batch_shape = x.shape[:-1]
    calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

    if granularity == "per_irrep":
        if len(irreps_obj) == 0:
            return (
                x.to(fp8_dtype),
                torch.empty((*batch_shape, 0), dtype=scale_dtype, device=x.device),
            )
        q_parts: list[torch.Tensor] = []
        s_parts: list[torch.Tensor] = []
        for sl, (mul, _ir) in zip(irreps_obj.slices(), irreps_obj):
            seg = x[..., sl]
            if mul == 0:
                s_i = torch.full(
                    (*batch_shape, 1),
                    eps / fp8_max,
                    dtype=scale_dtype,
                    device=x.device,
                )
                q_parts.append(seg.to(fp8_dtype))
                s_parts.append(s_i)
                continue
            seg_c = seg.to(calc_dtype)
            norm = torch.linalg.vector_norm(seg_c, ord=2, dim=-1, keepdim=True).clamp(
                min=eps
            )
            s_i = (norm / fp8_max).to(scale_dtype)
            q_i = (seg_c / s_i.to(calc_dtype)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
            q_parts.append(q_i)
            s_parts.append(s_i)
        return torch.cat(q_parts, dim=-1), torch.cat(s_parts, dim=-1)

    if granularity == "per_channel":
        if irreps_obj.num_irreps == 0:
            return (
                x.to(fp8_dtype),
                torch.empty((*batch_shape, 0), dtype=scale_dtype, device=x.device),
            )
        q_parts = []
        s_parts = []
        for sl, (mul, ir) in zip(irreps_obj.slices(), irreps_obj):
            if mul == 0:
                continue
            d = ir.dim
            seg = x[..., sl]
            if layout_obj == cue.mul_ir:
                seg_v = seg.reshape(*batch_shape, mul, d).to(calc_dtype)
                norm = torch.linalg.vector_norm(seg_v, ord=2, dim=-1).clamp(min=eps)
                s_i = (norm / fp8_max).to(scale_dtype)
                q_i = (
                    (seg_v / s_i.to(calc_dtype).unsqueeze(-1))
                    .clamp(-fp8_max, fp8_max)
                    .to(fp8_dtype)
                    .reshape(*batch_shape, mul * d)
                )
            elif layout_obj == cue.ir_mul:
                seg_v = seg.reshape(*batch_shape, d, mul).to(calc_dtype)
                norm = torch.linalg.vector_norm(seg_v, ord=2, dim=-2).clamp(min=eps)
                s_i = (norm / fp8_max).to(scale_dtype)
                q_i = (
                    (seg_v / s_i.to(calc_dtype).unsqueeze(-2))
                    .clamp(-fp8_max, fp8_max)
                    .to(fp8_dtype)
                    .reshape(*batch_shape, d * mul)
                )
            else:
                raise ValueError(f"Unsupported layout {layout_obj}.")
            q_parts.append(q_i)
            s_parts.append(s_i)
        return torch.cat(q_parts, dim=-1), torch.cat(s_parts, dim=-1)

    if granularity in ("per_atom", "per_row"):
        x_c = x.to(calc_dtype)
        norm = torch.linalg.vector_norm(x_c, ord=2, dim=-1, keepdim=True).clamp(min=eps)
        scale = (norm / fp8_max).to(scale_dtype)
        q_x = (x_c / scale.to(calc_dtype)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        return q_x, scale

    if granularity == "per_tensor":
        x_c = x.to(calc_dtype)
        norms = torch.linalg.vector_norm(x_c, ord=2, dim=-1)
        max_norm = (
            norms.max()
            if norms.numel() > 0
            else torch.tensor(eps, dtype=calc_dtype, device=x.device)
        ).clamp(min=eps)
        scale = (max_norm.reshape(1) / fp8_max).to(scale_dtype)
        q_x = (x_c / scale.to(calc_dtype)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        return q_x, scale

    raise ValueError(
        f"Unsupported granularity '{granularity}'. Expected 'per_irrep', 'per_channel', "
        "'per_atom', or 'per_tensor'."
    )


def irrep_fp8_dequantize(
    q_x: torch.Tensor,
    scale: torch.Tensor,
    irreps: cue.Irreps | str,
    *,
    layout: cue.IrrepsLayout = cue.ir_mul,
    granularity: str = "per_irrep",
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize an FP8 equivariant tensor back to floating-point precision.

    Args:
        q_x: Quantized FP8 tensor of shape ``(*batch, irreps.dim)``.
        scale: Scale tensor produced by :func:`irrep_fp8_quantize`.
        irreps: Irreducible representations describing the last dimension of ``q_x``.
        layout: Memory layout of ``q_x`` (``cue.ir_mul`` or ``cue.mul_ir``).
        granularity: Quantization block granularity matching ``scale``.
        out_dtype: Target floating-point dtype for the dequantized output.

    Returns:
        Dequantized tensor of shape ``(*batch, irreps.dim)`` and dtype ``out_dtype``.
    """
    irreps_obj = _coerce_irreps(irreps)
    layout_obj = _coerce_layout(layout, default=cue.ir_mul)

    if q_x.ndim < 1 or q_x.shape[-1] != irreps_obj.dim:
        raise ValueError(
            f"Expected quantized tensor last dimension {irreps_obj.dim} for irreps '{irreps_obj}', "
            f"got shape {tuple(q_x.shape)}."
        )

    batch_shape = q_x.shape[:-1]

    if granularity == "per_irrep":
        if len(irreps_obj) == 0:
            return q_x.to(out_dtype)
        parts: list[torch.Tensor] = []
        for idx, (sl, (mul, _ir)) in enumerate(zip(irreps_obj.slices(), irreps_obj)):
            seg = q_x[..., sl].to(out_dtype)
            if mul == 0:
                parts.append(seg)
                continue
            s_i = scale[..., idx : idx + 1].to(out_dtype)
            parts.append(seg * s_i)
        return torch.cat(parts, dim=-1)

    if granularity == "per_channel":
        if irreps_obj.num_irreps == 0:
            return q_x.to(out_dtype)
        parts = []
        chan_offset = 0
        for sl, (mul, ir) in zip(irreps_obj.slices(), irreps_obj):
            if mul == 0:
                continue
            d = ir.dim
            s_i = scale[..., chan_offset : chan_offset + mul].to(out_dtype)
            chan_offset += mul
            seg = q_x[..., sl].to(out_dtype)
            if layout_obj == cue.mul_ir:
                deq = (seg.reshape(*batch_shape, mul, d) * s_i.unsqueeze(-1)).reshape(
                    *batch_shape, mul * d
                )
            elif layout_obj == cue.ir_mul:
                deq = (seg.reshape(*batch_shape, d, mul) * s_i.unsqueeze(-2)).reshape(
                    *batch_shape, d * mul
                )
            else:
                raise ValueError(f"Unsupported layout {layout_obj}.")
            parts.append(deq)
        return torch.cat(parts, dim=-1)

    if granularity in ("per_atom", "per_row", "per_tensor"):
        return q_x.to(out_dtype) * scale.to(out_dtype)

    raise ValueError(
        f"Unsupported granularity '{granularity}'. Expected 'per_irrep', 'per_channel', "
        "'per_atom', or 'per_tensor'."
    )


def _smooth_block_ste(
    seg: torch.Tensor,
    norm_dim: int | tuple[int, ...],
    fp8_dtype: torch.dtype,
    fp8_max: float,
    eps: float,
) -> torch.Tensor:
    calc_dtype = torch.float64 if seg.dtype == torch.float64 else torch.float32
    seg_c = seg.to(calc_dtype)
    sum_sq = (seg_c * seg_c).sum(dim=norm_dim, keepdim=True)
    scale = torch.sqrt(sum_sq + eps * eps) / fp8_max
    u = seg_c / scale
    u_q = u.detach().clamp(-fp8_max, fp8_max).to(fp8_dtype).to(calc_dtype)
    u_ste = u + (u_q - u).detach()
    return (u_ste * scale).to(seg.dtype)


def irrep_fp8_ste(
    x: torch.Tensor,
    irreps: cue.Irreps | str,
    *,
    layout: cue.IrrepsLayout = cue.ir_mul,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    granularity: str = "per_irrep",
    eps: float = 1e-12,
    scale_grad: bool = False,
) -> torch.Tensor:
    """Differentiable Straight-Through Estimator (STE) fake-quantization for Force-QAT.

    In forward evaluation, returns the exact FP8 quantized-and-dequantized tensor.
    In backward evaluation (including 2nd-order ``create_graph=True`` Force+Stress QAT),
    detaches the discrete FP8 rounding residual ``(x_deq - x).detach()`` (or optionally
    propagates smooth :math:`C^\\infty` gradients through the :math:`O(3)`-invariant :math:`L_2`
    scale when ``scale_grad=True``).

    Args:
        x: Input tensor of shape ``(*batch, irreps.dim)``.
        irreps: Irreducible representations describing the last dimension of ``x``.
        layout: Memory layout of ``x`` (``cue.ir_mul`` or ``cue.mul_ir``).
        fp8_dtype: Target FP8 dtype (``torch.float8_e4m3fn`` or ``torch.float8_e5m2``).
        granularity: Quantization block granularity (``"per_irrep"``, ``"per_channel"``,
            ``"per_atom"``, or ``"per_tensor"``).
        eps: Numerical floor on the :math:`L_2` norm.
        scale_grad: If ``False`` (default), uses classic identity-gradient STE
            ``x + (x_deq - x).detach()``. If ``True``, propagates smooth 1st and 2nd
            derivatives through the :math:`O(3)`-invariant :math:`L_2` block scale.

    Returns:
        Fake-quantized tensor of shape ``(*batch, irreps.dim)`` and dtype ``x.dtype``.
    """
    if not scale_grad:
        with torch.no_grad():
            q_x, scale = irrep_fp8_quantize(
                x,
                irreps,
                layout=layout,
                fp8_dtype=fp8_dtype,
                granularity=granularity,
                eps=eps,
            )
            x_deq = irrep_fp8_dequantize(
                q_x,
                scale,
                irreps,
                layout=layout,
                granularity=granularity,
                out_dtype=x.dtype,
            )
        return x + (x_deq - x).detach()

    irreps_obj = _coerce_irreps(irreps)
    layout_obj = _coerce_layout(layout, default=cue.ir_mul)
    fp8_max = _validate_fp8_dtype(fp8_dtype)

    if x.ndim < 1 or x.shape[-1] != irreps_obj.dim:
        raise ValueError(
            f"Expected input tensor last dimension {irreps_obj.dim} for irreps '{irreps_obj}', "
            f"got shape {tuple(x.shape)}."
        )

    batch_shape = x.shape[:-1]

    if granularity == "per_irrep":
        if len(irreps_obj) == 0:
            return x
        parts: list[torch.Tensor] = []
        for sl, (mul, _ir) in zip(irreps_obj.slices(), irreps_obj):
            seg = x[..., sl]
            if mul == 0:
                parts.append(seg)
                continue
            parts.append(_smooth_block_ste(seg, -1, fp8_dtype, fp8_max, eps))
        return torch.cat(parts, dim=-1)

    if granularity == "per_channel":
        if irreps_obj.num_irreps == 0:
            return x
        parts = []
        for sl, (mul, ir) in zip(irreps_obj.slices(), irreps_obj):
            if mul == 0:
                continue
            d = ir.dim
            seg = x[..., sl]
            if layout_obj == cue.mul_ir:
                seg_v = seg.reshape(*batch_shape, mul, d)
                parts.append(
                    _smooth_block_ste(seg_v, -1, fp8_dtype, fp8_max, eps).reshape(
                        *batch_shape, mul * d
                    )
                )
            elif layout_obj == cue.ir_mul:
                seg_v = seg.reshape(*batch_shape, d, mul)
                parts.append(
                    _smooth_block_ste(seg_v, -2, fp8_dtype, fp8_max, eps).reshape(
                        *batch_shape, d * mul
                    )
                )
            else:
                raise ValueError(f"Unsupported layout {layout_obj}.")
        return torch.cat(parts, dim=-1)

    if granularity in ("per_atom", "per_row"):
        return _smooth_block_ste(x, -1, fp8_dtype, fp8_max, eps)

    if granularity == "per_tensor":
        all_dims = tuple(range(x.ndim))
        return _smooth_block_ste(x, all_dims, fp8_dtype, fp8_max, eps)

    raise ValueError(
        f"Unsupported granularity '{granularity}'. Expected 'per_irrep', 'per_channel', "
        "'per_atom', or 'per_tensor'."
    )


class EquivariantFP8Linear(torch.nn.Module):
    r"""H100 :math:`O(3)`-equivariant FP8 linear layer with Force-QAT STE support.

    Applies Schur's Lemma block-diagonal channel mixing ``irreps_in -> irreps_out`` where:
    - Activations of each irrep block ``(batch, ir.dim, mul_in)`` are quantized into FP8 using
      strictly :math:`O(3)`-invariant :math:`L_2` block norms so rotation never shifts the FP8
      exponent scale or causes saturation clipping.
    - Scalar (:math:`0e`) multiplicity weights ``(mul_in, mul_out)`` are quantized into FP8 per
      irrep block.
    - On Ada/Hopper+ GPUs (CUDA compute capability ``>= (8, 9)``) when GEMM dimensions
      ``(M, K, N) = (batch * ir.dim, mul_in, mul_out)`` satisfy 16-byte Tensor Core alignment,
      dispatches directly to ``torch._scaled_mm``.
    - On CPU or unaligned shapes (and for 1st/2nd-order ``create_graph=True`` Force-QAT backward),
      executes exact FP8-quantized matrix multiplication with Straight-Through Estimator (STE)
      matching H100 FP8 semantics.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps | str,
        irreps_out: cue.Irreps | str,
        *,
        layout: Optional[cue.IrrepsLayout] = cue.ir_mul,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        shared_weights: bool = True,
        internal_weights: Optional[bool] = None,
        fp8_dtype_act: torch.dtype = torch.float8_e4m3fn,
        fp8_dtype_weight: torch.dtype = torch.float8_e4m3fn,
        act_granularity: str = "per_irrep",
        scale_grad: bool = False,
        eps: float = 1e-12,
        use_scaled_mm: Optional[bool] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        irreps_in_obj = _coerce_irreps(irreps_in)
        irreps_out_obj = _coerce_irreps(irreps_out)
        assert_same_group(irreps_in_obj, irreps_out_obj)

        self.fp8_max_act = _validate_fp8_dtype(fp8_dtype_act)
        self.fp8_max_weight = _validate_fp8_dtype(fp8_dtype_weight)

        if act_granularity not in (
            "per_irrep",
            "per_channel",
            "per_atom",
            "per_row",
            "per_tensor",
        ):
            raise ValueError(f"Unsupported act_granularity '{act_granularity}'.")

        self.irreps_in = irreps_in_obj
        self.irreps_out = irreps_out_obj
        self.layout_in = _coerce_layout(
            layout_in if layout_in is not None else layout, default=cue.ir_mul
        )
        self.layout_out = _coerce_layout(
            layout_out if layout_out is not None else layout, default=cue.ir_mul
        )
        self.fp8_dtype_act = fp8_dtype_act
        self.fp8_dtype_weight = fp8_dtype_weight
        self.act_granularity = act_granularity
        self.scale_grad = scale_grad
        self.eps = eps
        self.use_scaled_mm = use_scaled_mm

        e = descriptors.linear(irreps_in_obj, irreps_out_obj)
        d = e.polynomial.operations[0][1]
        self.weight_numel = e.inputs[0].dim

        self.shared_weights = shared_weights
        self.internal_weights = (
            internal_weights if internal_weights is not None else shared_weights
        )
        param_dtype = dtype if dtype is not None else torch.get_default_dtype()

        if self.internal_weights:
            if not self.shared_weights:
                raise ValueError("Internal weights require shared_weights=True.")
            self.weight = torch.nn.Parameter(
                torch.randn(1, self.weight_numel, device=device, dtype=param_dtype)
            )
        else:
            self.register_parameter("weight", None)

        # Precompute block structure per unique output irrep matching descriptors.linear
        w_slices = d.operands[0].segment_slices()
        in_slices = irreps_in_obj.slices()
        out_slices = irreps_out_obj.slices()

        path_lookup: dict[tuple[int, int], tuple[slice, float]] = {}
        for path in d.paths:
            w_idx, x_idx, y_idx = path.indices
            path_lookup[(x_idx, y_idx)] = (w_slices[w_idx], float(path.coefficients))

        seen_irs: list[cue.Irrep] = []
        for mul_out, ir_out in irreps_out_obj:
            if mul_out > 0 and ir_out not in seen_irs:
                seen_irs.append(ir_out)

        self._blocks: list[dict] = []
        for ir in seen_irs:
            in_entries = [
                (i, in_slices[i], mul_i)
                for i, (mul_i, ir_i) in enumerate(irreps_in_obj)
                if ir_i == ir and mul_i > 0
            ]
            out_entries = [
                (j, out_slices[j], mul_j)
                for j, (mul_j, ir_j) in enumerate(irreps_out_obj)
                if ir_j == ir and mul_j > 0
            ]
            k_ir = sum(m for _, _, m in in_entries)
            n_ir = sum(m for _, _, m in out_entries)
            if k_ir == 0 or n_ir == 0:
                continue

            w_specs: list[tuple[int, int, int, int, slice, float]] = []
            r_off = 0
            for i_idx, _sl_i, mul_i in in_entries:
                c_off = 0
                for j_idx, _sl_j, mul_j in out_entries:
                    if (i_idx, j_idx) in path_lookup:
                        w_sl, coeff = path_lookup[(i_idx, j_idx)]
                        w_specs.append(
                            (r_off, r_off + mul_i, c_off, c_off + mul_j, w_sl, coeff)
                        )
                    c_off += mul_j
                r_off += mul_i

            block_irreps = cue.Irreps(irreps_in_obj.irrep_class, [(k_ir, ir)])
            self._blocks.append(
                {
                    "ir": ir,
                    "d_ir": ir.dim,
                    "k_ir": k_ir,
                    "n_ir": n_ir,
                    "in_entries": in_entries,
                    "out_entries": out_entries,
                    "w_specs": w_specs,
                    "block_irreps": block_irreps,
                }
            )

    def extra_repr(self) -> str:
        return (
            f"irreps_in={self.irreps_in}, irreps_out={self.irreps_out}, "
            f"fp8_act={self.fp8_dtype_act}, fp8_weight={self.fp8_dtype_weight}, "
            f"granularity={self.act_granularity}"
        )

    def _can_dispatch_scaled_mm(self, x: torch.Tensor, m: int, k: int, n: int) -> bool:
        if self.use_scaled_mm is False:
            return False
        if not self.shared_weights:
            return False
        if self.act_granularity == "per_channel":
            return False
        if (
            self.fp8_dtype_act == torch.float8_e5m2
            and self.fp8_dtype_weight == torch.float8_e5m2
        ):
            return False
        if (
            m <= 0
            or k <= 0
            or n <= 0
            or (m % 16 != 0)
            or (k % 16 != 0)
            or (n % 16 != 0)
        ):
            return False
        if not x.is_cuda:
            return False
        if self.use_scaled_mm is None:
            major, minor = torch.cuda.get_device_capability(x.device)
            if (major, minor) < (8, 9):
                return False
        return True

    def _assemble_weight_block(
        self, weight: torch.Tensor, b_flat: int, block: dict
    ) -> torch.Tensor:
        k_ir: int = block["k_ir"]
        n_ir: int = block["n_ir"]
        w_specs: list[tuple[int, int, int, int, slice, float]] = block["w_specs"]

        if self.shared_weights:
            w_vec = weight.squeeze(0) if weight.ndim == 2 else weight
            if len(w_specs) == 1:
                r0, r1, c0, c1, w_sl, coeff = w_specs[0]
                if (r1 - r0) == k_ir and (c1 - c0) == n_ir:
                    return w_vec[w_sl].reshape(k_ir, n_ir) * coeff
            w_mat = weight.new_zeros((k_ir, n_ir))
            for r0, r1, c0, c1, w_sl, coeff in w_specs:
                w_mat[r0:r1, c0:c1] = w_vec[w_sl].reshape(r1 - r0, c1 - c0) * coeff
            return w_mat

        if len(w_specs) == 1:
            r0, r1, c0, c1, w_sl, coeff = w_specs[0]
            if (r1 - r0) == k_ir and (c1 - c0) == n_ir:
                return weight[:, w_sl].reshape(b_flat, k_ir, n_ir) * coeff
        w_mat = weight.new_zeros((b_flat, k_ir, n_ir))
        for r0, r1, c0, c1, w_sl, coeff in w_specs:
            w_mat[:, r0:r1, c0:c1] = (
                weight[:, w_sl].reshape(b_flat, r1 - r0, c1 - c0) * coeff
            )
        return w_mat

    def _quantize_weight_block(
        self, w_ir: torch.Tensor, calc_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_c = w_ir.to(calc_dtype)
        amax = w_c.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
        s_w = (amax / self.fp8_max_weight).to(torch.float32)
        q_w = (
            (w_c / s_w.to(calc_dtype))
            .clamp(-self.fp8_max_weight, self.fp8_max_weight)
            .to(self.fp8_dtype_weight)
        )
        return q_w, s_w

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the equivariant FP8 linear transformation.

        Args:
            x: Input tensor of shape ``(*batch, irreps_in.dim)``.
            weight: Optional external weight tensor of shape ``(1, weight_numel)`` or
                ``(weight_numel,)`` when ``shared_weights=True``, or ``(batch, weight_numel)``
                when ``shared_weights=False``.

        Returns:
            Output tensor of shape ``(*batch, irreps_out.dim)``.
        """
        if self.internal_weights:
            if weight is not None:
                raise ValueError("Internal weights are enabled; pass weight=None.")
            weight = self.weight
        elif weight is None:
            raise ValueError(
                "External weights must be provided when internal_weights=False."
            )

        if x.ndim < 1 or x.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Expected x last dimension {self.irreps_in.dim}, got shape {tuple(x.shape)}."
            )

        batch_shape = x.shape[:-1]
        b_flat = math.prod(batch_shape) if len(batch_shape) > 0 else 1
        x_flat = x.reshape(b_flat, self.irreps_in.dim)

        if len(self.irreps_out) == 0 or b_flat == 0:
            return x.new_zeros((*batch_shape, self.irreps_out.dim))

        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        need_grad = torch.is_grad_enabled() and (
            x.requires_grad or weight.requires_grad
        )

        out_segments: list[Optional[torch.Tensor]] = [None] * len(self.irreps_out)

        for block in self._blocks:
            d_ir: int = block["d_ir"]
            k_ir: int = block["k_ir"]
            n_ir: int = block["n_ir"]
            in_entries = block["in_entries"]
            out_entries = block["out_entries"]
            block_irreps: cue.Irreps = block["block_irreps"]

            # Gather input block in (b_flat, d_ir, k_ir) ir_mul order
            in_parts: list[torch.Tensor] = []
            for _i_idx, sl_in, mul_i in in_entries:
                seg_in = x_flat[:, sl_in]
                if self.layout_in == cue.ir_mul:
                    in_parts.append(seg_in.reshape(b_flat, d_ir, mul_i))
                else:
                    in_parts.append(seg_in.reshape(b_flat, mul_i, d_ir).transpose(1, 2))
            x_ir = in_parts[0] if len(in_parts) == 1 else torch.cat(in_parts, dim=-1)
            x_ir_flat = x_ir.reshape(b_flat, d_ir * k_ir)

            w_ir = self._assemble_weight_block(weight, b_flat, block)

            with torch.no_grad():
                q_x_flat, s_x = irrep_fp8_quantize(
                    x_ir_flat,
                    block_irreps,
                    layout=cue.ir_mul,
                    fp8_dtype=self.fp8_dtype_act,
                    granularity=self.act_granularity,
                    eps=self.eps,
                )
                q_w, s_w = self._quantize_weight_block(w_ir, calc_dtype)

            m_dim = b_flat * d_ir
            use_hw = self._can_dispatch_scaled_mm(x, m_dim, k_ir, n_ir)

            if use_hw:
                q_x_2d = q_x_flat.reshape(m_dim, k_ir).contiguous()
                q_w_col = q_w.t().contiguous().t()
                ones_a = torch.ones((1,), dtype=torch.float32, device=x.device)
                scale_b = s_w.reshape(1)
                raw_mm = torch._scaled_mm(
                    q_x_2d,
                    q_w_col,
                    ones_a,
                    scale_b,
                    out_dtype=torch.float32,
                )
                s_x_bcast = s_x.to(torch.float32).reshape(-1, 1, 1)
                y_hw = (raw_mm.reshape(b_flat, d_ir, n_ir) * s_x_bcast).to(x.dtype)
            else:
                y_hw = None

            if need_grad:
                if self.scale_grad:
                    x_ste_flat = irrep_fp8_ste(
                        x_ir_flat.to(calc_dtype),
                        block_irreps,
                        layout=cue.ir_mul,
                        fp8_dtype=self.fp8_dtype_act,
                        granularity=self.act_granularity,
                        eps=self.eps,
                        scale_grad=True,
                    )
                else:
                    x_deq_flat = irrep_fp8_dequantize(
                        q_x_flat,
                        s_x,
                        block_irreps,
                        layout=cue.ir_mul,
                        granularity=self.act_granularity,
                        out_dtype=calc_dtype,
                    )
                    x_ir_c = x_ir_flat.to(calc_dtype)
                    x_ste_flat = x_ir_c + (x_deq_flat - x_ir_c).detach()

                x_ste = x_ste_flat.reshape(b_flat, d_ir, k_ir)
                w_c = w_ir.to(calc_dtype)
                w_deq = q_w.to(calc_dtype) * s_w.to(calc_dtype)
                w_ste = w_c + (w_deq - w_c).detach()
                y_ste = torch.matmul(x_ste, w_ste).to(x.dtype)
                y_ir = y_ste + (y_hw - y_ste).detach() if y_hw is not None else y_ste
            elif y_hw is not None:
                y_ir = y_hw
            else:
                x_deq = irrep_fp8_dequantize(
                    q_x_flat,
                    s_x,
                    block_irreps,
                    layout=cue.ir_mul,
                    granularity=self.act_granularity,
                    out_dtype=calc_dtype,
                ).reshape(b_flat, d_ir, k_ir)
                w_deq = q_w.to(calc_dtype) * s_w.to(calc_dtype)
                y_ir = torch.matmul(x_deq, w_deq).to(x.dtype)

            c_off = 0
            for j_idx, _sl_out, mul_j in out_entries:
                part_out = y_ir[:, :, c_off : c_off + mul_j]
                c_off += mul_j
                if self.layout_out == cue.ir_mul:
                    out_segments[j_idx] = part_out.reshape(b_flat, d_ir * mul_j)
                else:
                    out_segments[j_idx] = (
                        part_out.transpose(1, 2)
                        .contiguous()
                        .reshape(b_flat, mul_j * d_ir)
                    )

        for j_idx, (mul_j, ir_j) in enumerate(self.irreps_out):
            if out_segments[j_idx] is None:
                out_segments[j_idx] = x.new_zeros((b_flat, mul_j * ir_j.dim))

        y_flat = torch.cat([seg for seg in out_segments if seg is not None], dim=-1)
        return y_flat.reshape(*batch_shape, self.irreps_out.dim)


__all__ = [
    "irrep_fp8_quantize",
    "irrep_fp8_dequantize",
    "irrep_fp8_ste",
    "EquivariantFP8Linear",
]
