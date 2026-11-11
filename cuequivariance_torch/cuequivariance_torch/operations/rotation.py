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
import math
import warnings
from typing import Dict, List, Optional

import cuequivariance as cue
import torch
from cuequivariance import descriptors
from cuequivariance.group_theory.irreps_array.misc_ui import (
    default_irreps,
    default_layout,
)

import cuequivariance_torch as cuet


def _select_rotation_method(
    same_shape: bool,
    method: Optional[str],
    use_fallback: Optional[bool],
) -> str:
    if method is None:
        if use_fallback is None:
            if same_shape:
                # No warning here as it's the default behavior
                return "uniform_1d"
            warnings.warn(
                "Segments are not the same shape, falling back to naive method\n"
                "You can consider making the segments uniform in the descriptor."
            )
            return "naive"
        warnings.warn(
            "`use_fallback` is deprecated, please use `method` instead",
            DeprecationWarning,
        )
        if not same_shape and not use_fallback:
            raise ValueError(
                "Uniform 1D method requires segments to be the same shape\n"
                "You can consider making the segments uniform in the descriptor."
            )
        return "naive" if use_fallback else "uniform_1d"

    if method == "uniform_1d" and not same_shape:
        raise ValueError(
            "Uniform 1D method requires segments to be the same shape\n"
            "You can consider making the segments uniform in the descriptor."
        )
    return method


def _flatten_for_batch(
    t: torch.Tensor,
    out_batch: torch.Size,
    n_flat: int,
) -> torch.Tensor:
    op_batch = t.shape[:-1]
    op_dim = t.shape[-1]
    if math.prod(op_batch) == 1:
        return t.reshape(1, op_dim)
    if op_batch == out_batch:
        return t.reshape(n_flat, op_dim)
    return t.expand(*out_batch, op_dim).reshape(n_flat, op_dim)


def _execute_rotation_polynomial(
    f: "cuet.SegmentedPolynomial",
    encodings: List[torch.Tensor],
    x: torch.Tensor,
    transpose_in: "cuet.TransposeIrrepsLayout",
    transpose_out: "cuet.TransposeIrrepsLayout",
    out_dim: int,
    indices_in: Optional[torch.Tensor],
    indices_out: Optional[torch.Tensor],
    size_out: Optional[int],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.ndim == 0 or x.shape[-1] != out_dim:
        raise ValueError(
            f"Expected input x with last dimension {out_dim}, got shape {tuple(x.shape)}"
        )

    x_arg_idx = len(encodings)

    if indices_in is not None or indices_out is not None:
        if indices_out is not None and size_out is None:
            raise ValueError("size_out should be provided if indices_out is provided")
        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D input x of shape (N_in, {out_dim}) when using indices, "
                f"got shape {tuple(x.shape)}"
            )

        encodings_2d: List[torch.Tensor] = []
        for enc in encodings:
            if enc.ndim == 1:
                enc = enc.unsqueeze(0)
            elif enc.ndim != 2:
                raise ValueError(
                    f"Expected 1D angle batch when using indices, got encoding shape {tuple(enc.shape)}"
                )
            encodings_2d.append(enc)

        if indices_in is not None and indices_out is not None:
            if indices_in.shape[0] != indices_out.shape[0]:
                raise ValueError(
                    f"indices_in and indices_out must have the same length, "
                    f"got {indices_in.shape[0]} and {indices_out.shape[0]}"
                )
            num_edges = indices_in.shape[0]
        elif indices_in is not None:
            num_edges = indices_in.shape[0]
        else:
            assert indices_out is not None
            num_edges = indices_out.shape[0]

        if (
            not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
            and num_edges == 0
        ):
            out_rows = size_out if indices_out is not None else 0
            assert out_rows is not None
            out = torch.zeros((out_rows, out_dim), dtype=x.dtype, device=x.device)
            if x.requires_grad or any(e.requires_grad for e in encodings_2d):
                zero_touch = x.sum() * 0.0
                for e in encodings_2d:
                    zero_touch = zero_touch + e.sum() * 0.0
                out = out + zero_touch
            return out

        idx_in = (
            indices_in.to(torch.int64)
            if (
                indices_in is not None
                and f.method == "naive"
                and indices_in.dtype == torch.int32
            )
            else indices_in
        )
        idx_out = (
            indices_out.to(torch.int64)
            if (
                indices_out is not None
                and f.method == "naive"
                and indices_out.dtype == torch.int32
            )
            else indices_out
        )

        # Transpose layout on node tensor BEFORE indexed gather inside f
        x_t = transpose_in(x)

        input_indices: Dict[int, torch.Tensor] = {}
        if idx_in is not None:
            input_indices[x_arg_idx] = idx_in

        output_indices: Dict[int, torch.Tensor] = {}
        output_shapes: Dict[int, torch.Tensor] = {}
        if idx_out is not None:
            assert size_out is not None
            output_indices[0] = idx_out
            output_shapes[0] = torch.empty((size_out, 1), device=x.device)
        elif size_out is not None:
            output_shapes[0] = torch.empty((size_out, 1), device=x.device)

        output = f(
            [*encodings_2d, x_t],
            input_indices=input_indices,
            output_shapes=output_shapes,
            output_indices=output_indices,
            mask=mask,
        )
        # Transpose layout on node tensor AFTER indexed scatter inside f
        return transpose_out(output[0])

    # Unindexed execution: fast path for standard non-empty 2D tensors
    if (
        size_out is None
        and mask is None
        and x.ndim == 2
        and all(e.ndim == 2 for e in encodings)
        and (
            torch.jit.is_tracing()
            or torch.compiler.is_compiling()
            or (x.shape[0] > 0 and all(e.shape[0] > 0 for e in encodings))
        )
    ):
        output = f([*encodings, transpose_in(x)])
        return transpose_out(output[0])

    # General arbitrary-leading-batch / masked path: (*batch, dim)
    out_batch = torch.broadcast_shapes(*(e.shape[:-1] for e in encodings), x.shape[:-1])
    n_flat = math.prod(out_batch)
    if n_flat == 0:
        out_shape = (
            (size_out, out_dim) if size_out is not None else (*out_batch, out_dim)
        )
        out = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        if x.requires_grad or any(e.requires_grad for e in encodings):
            zero_touch = x.sum() * 0.0
            for e in encodings:
                zero_touch = zero_touch + e.sum() * 0.0
            out = out + zero_touch
        return out

    encodings_flat = [_flatten_for_batch(e, out_batch, n_flat) for e in encodings]
    x_flat = _flatten_for_batch(x, out_batch, n_flat)
    mask_flat = (
        mask.to(torch.bool).expand(out_batch).reshape(n_flat)
        if mask is not None
        else None
    )
    output_shapes_unindexed: Optional[Dict[int, torch.Tensor]] = (
        {0: torch.empty((size_out, 1), device=x.device)}
        if size_out is not None
        else None
    )
    output = f(
        [*encodings_flat, transpose_in(x_flat)],
        output_shapes=output_shapes_unindexed,
        mask=mask_flat,
    )
    y = transpose_out(output[0])
    if size_out is not None:
        return y
    return y.reshape(*out_batch, out_dim)


class Rotation(torch.nn.Module):
    """
    A class that represents a rotation layer for SO3 or O3 representations.

    Args:
        irreps (Irreps): The irreducible representations of the tensor to rotate.
        layout (IrrepsLayout, optional): The memory layout of the tensor, ``cue.ir_mul`` is preferred.
        layout_in (IrrepsLayout, optional): The layout of the input irreducible representations, by default ``layout``.
        layout_out (IrrepsLayout, optional): The layout of the output irreducible representations, by default ``layout``.
        device (torch.device, optional): The device to use for the operation.
        math_dtype (torch.dtype or string, optional): The dtype to use for the math operations, by default it follows the dtype of the input tensors,
            if possible, or the torch default dtype (see SegmentedPolynomial for more details).
        method (str, optional): The method to use for the operation, by default "uniform_1d" (using a CUDA kernel)
            if all segments have the same shape, otherwise "naive" (using a PyTorch implementation).
        use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
            If `True` the "naive" method is used.
            If `False` the "uniform_1d" method is used (make sure all segments have the same shape).
        auto_uniform_segments (bool, optional): Whether to automatically split non-uniform 1D segments by their
            greatest common divisor when cost-effective so the ``"uniform_1d"`` method can be used. Default is True.
    """

    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        device: Optional[torch.device] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
        auto_uniform_segments: bool = True,
    ):
        super().__init__()
        (irreps,) = default_irreps(irreps)

        if irreps.irrep_class not in [cue.SO3, cue.O3]:
            raise ValueError(
                f"Unsupported irrep class {irreps.irrep_class}. Must be SO3 or O3."
            )

        e = descriptors.yxy_rotation(irreps)
        if auto_uniform_segments:
            e = e.uniformize_segments()
        same_shape = (
            e.all_same_segment_shape()
            and all(o.ndim in (0, 1) for o in e.polynomial.operands)
            and len(
                {
                    o.segment_shape
                    for o in e.polynomial.operands
                    if o.ndim == 1 and len(o.segments) > 0
                }
            )
            <= 1
        )

        self.irreps = irreps
        self.lmax = max((ir.l for _, ir in irreps), default=0)

        layout_in = default_layout(layout_in or layout)
        self.transpose_in = cuet.TransposeIrrepsLayout(
            e.inputs[3].irreps,
            source=layout_in,
            target=e.inputs[3].layout,
            device=device,
            use_fallback=use_fallback,
        )

        layout_out = default_layout(layout_out or layout)
        self.transpose_out = cuet.TransposeIrrepsLayout(
            e.outputs[0].irreps,
            source=e.outputs[0].layout,
            target=layout_out,
            device=device,
            use_fallback=use_fallback,
        )

        self.method = _select_rotation_method(same_shape, method, use_fallback)

        self.f = cuet.SegmentedPolynomial(
            e.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

    def forward(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        x: torch.Tensor,
        *,
        indices_in: Optional[torch.Tensor] = None,
        indices_out: Optional[torch.Tensor] = None,
        size_out: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the rotation layer.

        Args:
            gamma (torch.Tensor): The gamma angles. First rotation around the y-axis.
            beta (torch.Tensor): The beta angles. Second rotation around the x-axis.
            alpha (torch.Tensor): The alpha angles. Third rotation around the y-axis.
            x (torch.Tensor): The input tensor of shape ``(*batch, irreps.dim)`` (or ``(N_in, irreps.dim)`` when ``indices_in`` is provided).
            indices_in (torch.Tensor, optional): 1D integer index tensor to gather rows from ``x`` after input layout transposition.
            indices_out (torch.Tensor, optional): 1D integer index tensor to scatter-add rotated rows before output layout transposition.
            size_out (int, optional): Output batch size, required when ``indices_out`` is provided.
            mask (torch.Tensor, optional): Optional boolean/float mask for fixed-capacity execution.

        Returns:
            torch.Tensor: The rotated tensor.
        """
        gamma = torch.as_tensor(gamma, dtype=x.dtype, device=x.device)
        beta = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
        alpha = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)

        encodings_gamma = encode_rotation_angle(gamma, self.lmax)
        encodings_beta = encode_rotation_angle(beta, self.lmax)
        encodings_alpha = encode_rotation_angle(alpha, self.lmax)

        return _execute_rotation_polynomial(
            self.f,
            [encodings_gamma, encodings_beta, encodings_alpha],
            x,
            self.transpose_in,
            self.transpose_out,
            self.irreps.dim,
            indices_in,
            indices_out,
            size_out,
            mask=mask,
        )


class EdgeRotation(torch.nn.Module):
    r"""2-angle Wigner rotation layer for eSCN / UMA edge frame alignment on :math:`SO(3)` or :math:`O(3)` irreps.

    Given a 3D edge vector :math:`\vec{v}` with Euler angles :math:`(\beta, \alpha)` from
    :func:`vector_to_euler_angles` satisfying

    .. math::

        R_y(\alpha) R_x(\beta) \vec{e}_y = \hat{v},

    this module applies the 2-angle rotation without the redundant third :math:`y`-axis stage of
    :class:`Rotation` (~41.5% fewer paths and FLOPs than ``yxy_rotation``), using the algebraic
    Chebyshev trigonometric encoding (:func:`encode_edge_rotations`) directly on :math:`\vec{v}` by default:

    - **Forward edge alignment** (``inverse=False``, default): applies
      :math:`D(R_x(-\beta) R_y(-\alpha))` via :func:`cuequivariance.descriptors.xy_rotation`
      (equivalent to ``Rotation(irreps)(-alpha, -beta, 0, x)``). On a ``1o`` / ``SO3(1)`` vector
      :math:`\vec{v}`, this maps :math:`\vec{v} \mapsto [0, \|\vec{v}\|, 0]` along :math:`+\vec{e}_y`,
      diagonalizing eSCN tensor products across :math:`|m|`.
    - **Inverse edge rotation** (``inverse=True``): applies
      :math:`D(R_y(\alpha) R_x(\beta))` via :func:`cuequivariance.descriptors.yx_rotation`
      (equivalent to ``Rotation(irreps)(0, beta, alpha, x)``), mapping features from the edge-aligned
      frame back to the global frame so that
      ``EdgeRotation(irreps, inverse=True)(v, EdgeRotation(irreps, inverse=False)(v, x)) == x``.

    Supports fused ``node -> edge`` gather + rotate via ``indices_in`` and ``edge -> node`` rotate +
    scatter-add via ``indices_out`` / ``size_out``, performing layout transposes at the node level.

    Args:
        irreps (Irreps): The irreducible representations of the tensor to rotate.
        inverse (bool, optional): If ``False`` (default), rotate from the global frame into the
            edge-aligned frame (:math:`\hat{v} \to +\vec{e}_y`). If ``True``, rotate from the
            edge-aligned frame back to the global frame (:math:`+\vec{e}_y \to \hat{v}`).
        layout (IrrepsLayout, optional): The memory layout of the tensor, ``cue.ir_mul`` is preferred.
        layout_in (IrrepsLayout, optional): The layout of the input irreducible representations, by default ``layout``.
        layout_out (IrrepsLayout, optional): The layout of the output irreducible representations, by default ``layout``.
        device (torch.device, optional): The device to use for the operation.
        math_dtype (torch.dtype or string, optional): The dtype to use for the math operations.
        method (str, optional): ``"uniform_1d"`` (default when segments have uniform shape) or ``"naive"``.
        use_fallback (bool, optional, deprecated): Deprecated alias for ``method``.
    """

    def __init__(
        self,
        irreps: cue.Irreps,
        inverse: bool = False,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        device: Optional[torch.device] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
        auto_uniform_segments: bool = True,
    ):
        super().__init__()
        (irreps,) = default_irreps(irreps)

        if irreps.irrep_class not in [cue.SO3, cue.O3]:
            raise ValueError(
                f"Unsupported irrep class {irreps.irrep_class}. Must be SO3 or O3."
            )

        self.irreps = irreps
        self.inverse = bool(inverse)
        self.lmax = max((ir.l for _, ir in irreps), default=0)

        e = (
            descriptors.yx_rotation(irreps)
            if self.inverse
            else descriptors.xy_rotation(irreps)
        )
        if auto_uniform_segments:
            e = e.uniformize_segments()
        same_shape = (
            e.all_same_segment_shape()
            and all(o.ndim in (0, 1) for o in e.polynomial.operands)
            and len(
                {
                    o.segment_shape
                    for o in e.polynomial.operands
                    if o.ndim == 1 and len(o.segments) > 0
                }
            )
            <= 1
        )

        layout_in = default_layout(layout_in or layout)
        self.transpose_in = cuet.TransposeIrrepsLayout(
            e.inputs[2].irreps,
            source=layout_in,
            target=e.inputs[2].layout,
            device=device,
            use_fallback=use_fallback,
        )

        layout_out = default_layout(layout_out or layout)
        self.transpose_out = cuet.TransposeIrrepsLayout(
            e.outputs[0].irreps,
            source=e.outputs[0].layout,
            target=layout_out,
            device=device,
            use_fallback=use_fallback,
        )

        self.method = _select_rotation_method(same_shape, method, use_fallback)

        self.f = cuet.SegmentedPolynomial(
            e.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return f"irreps={self.irreps}, inverse={self.inverse}"

    def forward(
        self,
        vectors: Optional[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = None,
        x: Optional[torch.Tensor] = None,
        *,
        beta: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        encodings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        indices_in: Optional[torch.Tensor] = None,
        indices_out: Optional[torch.Tensor] = None,
        size_out: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the 2-angle edge rotation.

        Args:
            vectors (torch.Tensor or tuple[torch.Tensor, torch.Tensor], optional):
                3D edge vectors of shape ``(*batch, 3)`` (default algebraic path), or a
                ``(beta, alpha)`` angle tuple. May be omitted when ``beta``/``alpha`` or
                ``encodings`` are passed by keyword.
            x (torch.Tensor, optional): Input irreps tensor of shape ``(*batch, irreps.dim)``
                (or ``(N_in, irreps.dim)`` when ``indices_in`` is provided).
            beta (torch.Tensor, optional): Explicit polar Euler angle :math:`\\beta`.
            alpha (torch.Tensor, optional): Explicit azimuthal Euler angle :math:`\\alpha`.
            encodings (tuple[torch.Tensor, torch.Tensor], optional): Precomputed
                ``(enc_beta, enc_alpha)`` from :func:`encode_edge_rotations` with ``ell >= self.lmax``.
            indices_in (torch.Tensor, optional): 1D integer indices for fused ``node -> edge`` gather.
            indices_out (torch.Tensor, optional): 1D integer indices for fused ``edge -> node`` scatter-add.
            size_out (int, optional): Output batch dimension, required when ``indices_out`` is provided.

        Returns:
            torch.Tensor: The rotated irreps tensor.
        """
        if (
            x is None
            and isinstance(vectors, torch.Tensor)
            and (beta is not None or alpha is not None or encodings is not None)
        ):
            x, vectors = vectors, None

        if x is None:
            raise ValueError("Input feature tensor `x` must be provided.")

        if isinstance(vectors, tuple):
            if beta is not None or alpha is not None or encodings is not None:
                raise ValueError(
                    "Specify only one of `vectors`, `(beta, alpha)`, or `encodings`."
                )
            if len(vectors) != 2:
                raise ValueError(
                    f"Expected (beta, alpha) tuple of length 2, got length {len(vectors)}"
                )
            beta, alpha = vectors
            vectors = None

        num_sources = (
            (vectors is not None)
            + (beta is not None or alpha is not None)
            + (encodings is not None)
        )
        if num_sources != 1:
            raise ValueError(
                "Exactly one of `vectors`, `(beta, alpha)`, or `encodings` must be provided."
            )

        if vectors is not None:
            cos_beta, sin_beta, cos_alpha, sin_alpha = _vector_to_edge_trig(vectors)
            if self.inverse:
                # yx_rotation(beta, alpha)
                enc_0 = encode_rotation_trig(cos_beta, sin_beta, self.lmax)
                enc_1 = encode_rotation_trig(cos_alpha, sin_alpha, self.lmax)
            else:
                # xy_rotation(-alpha, -beta)
                enc_0 = encode_rotation_trig(cos_alpha, -sin_alpha, self.lmax)
                enc_1 = encode_rotation_trig(cos_beta, -sin_beta, self.lmax)
            enc_0 = enc_0.to(dtype=x.dtype, device=x.device)
            enc_1 = enc_1.to(dtype=x.dtype, device=x.device)
        elif encodings is not None:
            enc_beta, enc_alpha = encodings
            enc_beta = _slice_encoding(enc_beta, self.lmax).to(
                dtype=x.dtype, device=x.device
            )
            enc_alpha = _slice_encoding(enc_alpha, self.lmax).to(
                dtype=x.dtype, device=x.device
            )
            if self.inverse:
                enc_0, enc_1 = enc_beta, enc_alpha
            else:
                enc_0 = _negate_encoding(enc_alpha, self.lmax)
                enc_1 = _negate_encoding(enc_beta, self.lmax)
        else:
            if beta is None or alpha is None:
                raise ValueError("Both `beta` and `alpha` must be provided together.")
            beta_t = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
            alpha_t = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
            if self.inverse:
                enc_0 = encode_rotation_angle(beta_t, self.lmax)
                enc_1 = encode_rotation_angle(alpha_t, self.lmax)
            else:
                enc_0 = encode_rotation_angle(-alpha_t, self.lmax)
                enc_1 = encode_rotation_angle(-beta_t, self.lmax)

        return _execute_rotation_polynomial(
            self.f,
            [enc_0, enc_1],
            x,
            self.transpose_in,
            self.transpose_out,
            self.irreps.dim,
            indices_in,
            indices_out,
            size_out,
            mask=mask,
        )


def _slice_encoding(enc: torch.Tensor, ell: int) -> torch.Tensor:
    target = 2 * ell + 1
    if enc.shape[-1] == target:
        return enc
    if enc.shape[-1] < target or enc.shape[-1] % 2 != 1:
        raise ValueError(
            f"Encoding last dimension {enc.shape[-1]} is incompatible with lmax={ell} "
            f"(expected odd size >= {target})"
        )
    src_ell = (enc.shape[-1] - 1) // 2
    return enc[..., src_ell - ell : src_ell + ell + 1]


def _negate_encoding(enc: torch.Tensor, ell: int) -> torch.Tensor:
    if ell == 0:
        return enc
    return torch.cat([enc[..., : ell + 1], -enc[..., ell + 1 :]], dim=-1)


def encode_rotation_angle(angle: torch.Tensor, ell: int) -> torch.Tensor:
    """Encode a angle into a tensor of cosines and sines.

    The encoding is::

        [cos(l * angle), cos((l - 1) * angle), ..., cos(angle), 1, sin(angle), sin(2 * angle), ..., sin(l * angle)].

    This encoding is used to feed the segmented tensor products that perform rotations.
    """
    angle = torch.as_tensor(angle)
    angle = angle.unsqueeze(-1)

    m = torch.arange(1, ell + 1, device=angle.device, dtype=angle.dtype)
    c = torch.cos(m * angle)
    s = torch.sin(m * angle)
    one = torch.ones_like(angle)
    return torch.cat([c.flip(-1), one, s], dim=-1)


def encode_rotation_trig(
    cos_angle: torch.Tensor,
    sin_angle: torch.Tensor,
    ell: int,
) -> torch.Tensor:
    r"""Encode :math:`(\cos\theta, \sin\theta)` into multiple-angle cosines and sines via Chebyshev recurrence.

    Uses the angle-addition recurrence

    .. math::

        \cos(m\theta) = \cos((m-1)\theta)\cos\theta - \sin((m-1)\theta)\sin\theta, \quad
        \sin(m\theta) = \sin((m-1)\theta)\cos\theta + \cos((m-1)\theta)\sin\theta

    with zero transcendental calls, producing the exact layout of :func:`encode_rotation_angle`::

        [cos(ell * a), ..., cos(a), 1, sin(a), ..., sin(ell * a)].

    Args:
        cos_angle (torch.Tensor): Cosine of the base angle, shape ``(*shape)``.
        sin_angle (torch.Tensor): Sine of the base angle, shape ``(*shape)``.
        ell (int): Maximum harmonic order ``ell >= 0``.

    Returns:
        torch.Tensor: Encoded tensor of shape ``(*shape, 2 * ell + 1)``.
    """
    if ell < 0:
        raise ValueError(f"ell must be non-negative, got {ell}")

    cos_angle = torch.as_tensor(cos_angle)
    sin_angle = torch.as_tensor(
        sin_angle, dtype=cos_angle.dtype, device=cos_angle.device
    )
    if cos_angle.shape != sin_angle.shape:
        cos_angle, sin_angle = torch.broadcast_tensors(cos_angle, sin_angle)

    orig_dtype = cos_angle.dtype
    work_dtype = (
        torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    )
    c1 = cos_angle.to(work_dtype)
    s1 = sin_angle.to(work_dtype)
    one = torch.ones_like(c1)

    if ell == 0:
        return one.unsqueeze(-1).to(orig_dtype)

    cos_terms = [c1]
    sin_terms = [s1]
    cm, sm = c1, s1
    for _ in range(2, ell + 1):
        cm, sm = cm * c1 - sm * s1, sm * c1 + cm * s1
        cos_terms.append(cm)
        sin_terms.append(sm)

    return torch.stack(cos_terms[::-1] + [one] + sin_terms, dim=-1).to(orig_dtype)


def _vector_to_edge_trig(
    vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if vector.shape[-1] != 3:
        raise ValueError(
            f"Expected last dimension of size 3, got {tuple(vector.shape)}"
        )

    orig_dtype = vector.dtype
    work_dtype = (
        torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    )
    v = vector.to(work_dtype)
    x, y, z = v.unbind(-1)

    r_xz_sq = x * x + z * z
    on_y_axis = r_xz_sq == 0.0
    r_xz_safe = torch.sqrt(torch.where(on_y_axis, torch.ones_like(r_xz_sq), r_xz_sq))
    r_xz = torch.where(on_y_axis, torch.zeros_like(r_xz_sq), r_xz_safe)

    r_sq = r_xz_sq + y * y
    is_zero = r_sq == 0.0
    inv_r = torch.where(
        is_zero,
        torch.zeros_like(r_sq),
        torch.rsqrt(torch.where(is_zero, torch.ones_like(r_sq), r_sq)),
    )

    cos_beta = torch.where(is_zero, torch.ones_like(y), y * inv_r)
    sin_beta = r_xz * inv_r
    cos_alpha = torch.where(on_y_axis, torch.ones_like(z), z / r_xz_safe)
    sin_alpha = torch.where(on_y_axis, torch.zeros_like(x), x / r_xz_safe)
    return cos_beta, sin_beta, cos_alpha, sin_alpha


def encode_edge_rotations(
    vector: torch.Tensor,
    ell: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Algebraically encode 3D edge vectors into ``(enc_beta, enc_alpha)`` rotation encodings.

    Computes the multiple-angle trigonometric encodings matching
    ``encode_rotation_angle(beta, ell)`` and ``encode_rotation_angle(alpha, ell)`` for
    ``beta, alpha = vector_to_euler_angles(vector)`` directly from ``(x, y, z)`` via safe
    double-``where`` norms and :func:`encode_rotation_trig`, avoiding ``acos``/``atan2``/``cos``/``sin``
    and maintaining finite first and second derivatives at the poles ``[0, ±1, 0]``, the zero vector
    ``[0, 0, 0]``, and in ``bfloat16``/``float16`` near the poles.

    Args:
        vector (torch.Tensor): 3D vectors of shape ``(*shape, 3)``.
        ell (int): Maximum spherical harmonic degree ``ell >= 0``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(enc_beta, enc_alpha)``, each of shape ``(*shape, 2 * ell + 1)``.
    """
    cos_beta, sin_beta, cos_alpha, sin_alpha = _vector_to_edge_trig(vector)
    enc_beta = encode_rotation_trig(cos_beta, sin_beta, ell).to(vector.dtype)
    enc_alpha = encode_rotation_trig(cos_alpha, sin_alpha, ell).to(vector.dtype)
    return enc_beta, enc_alpha


def vector_to_euler_angles(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert a 3D vector to Euler angles.

    .. math::

        R_y(\alpha) R_x(\beta) \vec{e}_y = \vec{v}

    Uses a pole- and zero-safe ``atan2`` formulation with double-``where`` guards so that
    forward values and first/second derivatives remain finite at ``[0, ±1, 0]``, ``[0, 0, 0]``,
    and near the poles in low-precision floating-point types (``bfloat16`` / ``float16``).

    Args:
        vector (torch.Tensor): The 3D vector.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The beta and alpha angles.
    """
    assert vector.shape[-1] == 3
    orig_dtype = vector.dtype
    work_dtype = (
        torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    )
    v = vector.to(work_dtype)
    x, y, z = v.unbind(-1)

    r_xz_sq = x * x + z * z
    on_y_axis = r_xz_sq == 0.0
    r_xz = torch.where(
        on_y_axis,
        torch.zeros_like(r_xz_sq),
        torch.sqrt(torch.where(on_y_axis, torch.ones_like(r_xz_sq), r_xz_sq)),
    )

    is_zero = on_y_axis & (y == 0.0)
    y_safe = torch.where(is_zero, torch.ones_like(y), y)
    x_safe = torch.where(on_y_axis, torch.zeros_like(x), x)
    z_safe = torch.where(on_y_axis, torch.ones_like(z), z)

    beta = torch.where(is_zero, torch.zeros_like(y), torch.atan2(r_xz, y_safe))
    alpha = torch.where(on_y_axis, torch.zeros_like(x), torch.atan2(x_safe, z_safe))

    return beta.to(orig_dtype), alpha.to(orig_dtype)


class Inversion(torch.nn.Module):
    """
    Inversion layer for :math:`O(3)` representations.

    Args:
        irreps (Irreps): The irreducible representations of the tensor to invert.
        layout (IrrepsLayout, optional): The memory layout of the tensor, ``cue.ir_mul`` is preferred.
        layout_in (IrrepsLayout, optional): The layout of the input irreducible representations, by default ``layout``.
        layout_out (IrrepsLayout, optional): The layout of the output irreducible representations, by default ``layout``.
        device (torch.device, optional): The device to use for the linear layer.
        math_dtype (torch.dtype, optional): The dtype to use for the math operations, by default it follows the dtype of the input tensors.
        method (str, optional): The method to use for the linear layer, by default "uniform_1d" (using a CUDA kernel)
            if all segments have the same shape, otherwise "naive" (using a PyTorch implementation).
        use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
            If `True` the "naive" method is used.
            If `False` the "uniform_1d" method is used (make sure all segments have the same shape).
        auto_uniform_segments (bool, optional): Whether to automatically split non-uniform 1D segments by their
            greatest common divisor when cost-effective so the ``"uniform_1d"`` method can be used. Default is True.
    """

    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
        auto_uniform_segments: bool = True,
    ):
        super().__init__()
        (irreps,) = default_irreps(irreps)

        if irreps.irrep_class not in [cue.O3]:
            raise ValueError(
                f"Unsupported irrep class {irreps.irrep_class}. Must be O3."
            )

        e = descriptors.inversion(irreps)
        if auto_uniform_segments:
            e = e.uniformize_segments()
        same_shape = (
            e.all_same_segment_shape()
            and all(o.ndim in (0, 1) for o in e.polynomial.operands)
            and len(
                {
                    o.segment_shape
                    for o in e.polynomial.operands
                    if o.ndim == 1 and len(o.segments) > 0
                }
            )
            <= 1
        )

        layout_in = default_layout(layout_in or layout)
        self.transpose_in = cuet.TransposeIrrepsLayout(
            e.inputs[0].irreps,
            source=layout_in,
            target=e.inputs[0].layout,
            device=device,
            use_fallback=use_fallback,
        )

        layout_out = default_layout(layout_out or layout)
        self.transpose_out = cuet.TransposeIrrepsLayout(
            e.outputs[0].irreps,
            source=e.outputs[0].layout,
            target=layout_out,
            device=device,
            use_fallback=use_fallback,
        )

        if method is None:
            if use_fallback is None:
                if same_shape:
                    # No warning here as it's the default behavior
                    self.method = "uniform_1d"
                else:
                    warnings.warn(
                        "Segments are not the same shape, falling back to naive method\n"
                        "You can consider making the segments uniform in the descriptor."
                    )
                    self.method = "naive"
            else:
                warnings.warn(
                    "`use_fallback` is deprecated, please use `method` instead",
                    DeprecationWarning,
                )
                if not same_shape and not use_fallback:
                    raise ValueError(
                        "Uniform 1D method requires segments to be the same shape\n"
                        "You can consider making the segments uniform in the descriptor."
                    )
                self.method = "naive" if use_fallback else "uniform_1d"
        else:
            if method == "uniform_1d" and not same_shape:
                raise ValueError(
                    "Uniform 1D method requires segments to be the same shape\n"
                    "You can consider making the segments uniform in the descriptor."
                )
            self.method = method

        self.f = cuet.SegmentedPolynomial(
            e.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inversion layer."""
        output = self.f([self.transpose_in(x)])
        return self.transpose_out(output[0])
