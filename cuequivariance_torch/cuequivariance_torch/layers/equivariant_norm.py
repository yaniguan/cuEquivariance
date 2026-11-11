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
from typing import Optional

import torch
import torch.nn as nn

import cuequivariance as cue
from cuequivariance_torch.operations.gate import _resolve_irreps_list, _resolve_layout


def _safe_inv_rms(norm_sq: torch.Tensor, eps: float) -> torch.Tensor:
    """Double-where zero-safe inverse RMS for 1st and 2nd derivatives even at eps=0."""
    is_pos = norm_sq > 0.0
    safe_sq = torch.where(is_pos, norm_sq, torch.ones_like(norm_sq))
    return torch.where(is_pos, torch.rsqrt(safe_sq + eps), torch.zeros_like(norm_sq))


class _EquivariantNormBase(nn.Module):
    __constants__ = [
        "_blocks",
        "_is_mul_ir",
        "_dim",
        "_num_features",
        "_num_scalars",
        "_num_scalar_blocks",
        "_has_bias",
        "eps",
        "affine",
        "centering",
        "norm_mode",
        "normalization",
    ]

    def __init__(
        self,
        irreps: cue.Irreps | str,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        include_bias: bool = False,
        centering: bool = False,
        norm_mode: str = "per_irrep",
        normalization: str = "component",
        group: type[cue.Irrep] | str | None = None,
        layout: cue.IrrepsLayout | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        (resolved_irreps,) = _resolve_irreps_list(irreps, group=group)
        if any(mul < 0 for mul, _ in resolved_irreps):
            raise ValueError(
                f"irreps cannot contain negative multiplicities, got {resolved_irreps}."
            )

        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}.")
        if norm_mode not in ("per_irrep", "per_channel", "global"):
            raise ValueError(
                f"norm_mode must be one of ('per_irrep', 'per_channel', 'global'), "
                f"got {norm_mode!r}."
            )
        if normalization not in ("component", "norm"):
            raise ValueError(
                f"normalization must be 'component' or 'norm', got {normalization!r}."
            )

        resolved_layout = _resolve_layout(layout)

        self.irreps = resolved_irreps
        self.irreps_in = resolved_irreps
        self.irreps_out = resolved_irreps
        self.layout = resolved_layout
        self._is_mul_ir: bool = resolved_layout == cue.mul_ir

        self.eps: float = float(eps)
        self.affine: bool = bool(affine)
        self.centering: bool = bool(centering)
        self.norm_mode: str = norm_mode
        self.normalization: str = normalization
        self._has_bias: bool = bool(affine and include_bias)

        blocks: list[tuple[int, int, bool, int, int, int, int]] = []
        offset = 0
        w_offset = 0
        b_offset = 0
        num_scalar_blocks = 0
        for mul, ir in resolved_irreps:
            d = ir.dim
            block_dim = mul * d
            is_scalar = ir.is_scalar()
            if is_scalar and d != 1:
                raise ValueError(f"Scalar irrep {ir} must have dim == 1, got {d}.")
            if mul > 0:
                blocks.append(
                    (
                        offset,
                        offset + block_dim,
                        is_scalar,
                        mul,
                        d,
                        w_offset,
                        b_offset,
                    )
                )
                if is_scalar:
                    num_scalar_blocks += 1
            offset += block_dim
            w_offset += mul
            if is_scalar:
                b_offset += mul

        self._blocks = blocks
        self._dim: int = offset
        self._num_features: int = w_offset
        self._num_scalars: int = b_offset
        self._num_scalar_blocks: int = num_scalar_blocks

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones(self._num_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)

        if self._has_bias:
            self.bias = nn.Parameter(
                torch.zeros(self._num_scalars, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        """Reset learnable affine parameters to identity defaults."""
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize an equivariant feature tensor.

        Args:
            x (torch.Tensor): Input tensor of shape ``(*batch, irreps.dim)``.
            mask (torch.Tensor, optional): Optional boolean mask of shape ``(*batch,)``
                where ``True`` marks valid atoms/entries and ``False`` zeroes invalid
                padding entries without ``NaN`` propagation.

        Returns:
            torch.Tensor: Normalized tensor of shape ``(*batch, irreps.dim)`` and dtype ``x.dtype``.
        """
        if x.shape[-1] != self._dim:
            raise ValueError(f"Expected input last dim {self._dim}, got {x.shape[-1]}.")

        batch_shape = x.shape[:-1]
        if mask is not None:
            if mask.shape != batch_shape:
                raise ValueError(
                    f"Expected mask shape {batch_shape}, got {mask.shape}."
                )
            mask_exp = mask.to(dtype=torch.bool).unsqueeze(-1)
            x = torch.where(mask_exp, x, torch.zeros_like(x))

        if self._dim == 0:
            return x.clone()

        orig_dtype = x.dtype
        acc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        x_acc = x.to(dtype=acc_dtype)

        if self._is_mul_ir:
            mul_axis = -2
            m_axis = -1
        else:
            m_axis = -2
            mul_axis = -1

        global_scalar_mean: Optional[torch.Tensor] = None
        if (
            self.centering
            and self.norm_mode == "global"
            and self._num_scalar_blocks > 1
        ):
            scalar_slices: list[torch.Tensor] = []
            for start, end, is_scalar, _, _, _, _ in self._blocks:
                if is_scalar:
                    scalar_slices.append(x_acc[..., start:end])
            global_scalar_mean = (
                torch.cat(scalar_slices, dim=-1)
                .mean(dim=-1, keepdim=True)
                .unsqueeze(-1)
            )

        fields: list[torch.Tensor] = []
        sq_chans: list[torch.Tensor] = []
        for start, end, is_scalar, mul, d, _, _ in self._blocks:
            block = x_acc[..., start:end]
            if self._is_mul_ir:
                field = block.reshape(batch_shape + (mul, d))
            else:
                field = block.reshape(batch_shape + (d, mul))

            if self.centering and is_scalar:
                if global_scalar_mean is not None:
                    field = field - global_scalar_mean
                else:
                    field = field - field.mean(dim=mul_axis, keepdim=True)

            field_sq = field * field
            if self.normalization == "component":
                sq_chan = field_sq.mean(dim=m_axis, keepdim=True)
            else:
                sq_chan = field_sq.sum(dim=m_axis, keepdim=True)

            fields.append(field)
            sq_chans.append(sq_chan)

        w_acc: Optional[torch.Tensor] = None
        if self.affine and self.weight is not None:
            w_acc = self.weight.to(dtype=acc_dtype)

        b_acc: Optional[torch.Tensor] = None
        if self._has_bias and self.bias is not None:
            b_acc = self.bias.to(dtype=acc_dtype)

        global_inv_rms: Optional[torch.Tensor] = None
        if self.norm_mode == "global":
            total_sq = sq_chans[0].sum(dim=mul_axis, keepdim=True)
            for i in range(1, len(sq_chans)):
                total_sq = total_sq + sq_chans[i].sum(dim=mul_axis, keepdim=True)
            global_norm_sq = total_sq / float(self._num_features)
            global_inv_rms = _safe_inv_rms(global_norm_sq, self.eps)

        out_blocks: list[torch.Tensor] = []
        for idx, (_, _, is_scalar, mul, d, w_start, b_start) in enumerate(self._blocks):
            field = fields[idx]
            sq_chan = sq_chans[idx]

            if global_inv_rms is not None:
                inv_rms = global_inv_rms
            elif self.norm_mode == "per_irrep":
                norm_sq = sq_chan.mean(dim=mul_axis, keepdim=True)
                inv_rms = _safe_inv_rms(norm_sq, self.eps)
            else:
                inv_rms = _safe_inv_rms(sq_chan, self.eps)

            if w_acc is not None:
                w_block = w_acc[w_start : w_start + mul]
                if self._is_mul_ir:
                    scale = inv_rms * w_block.unsqueeze(-1)
                else:
                    scale = inv_rms * w_block.unsqueeze(-2)
            else:
                scale = inv_rms

            field_out = field * scale

            if is_scalar and b_acc is not None:
                b_block = b_acc[b_start : b_start + mul]
                if self._is_mul_ir:
                    field_out = field_out + b_block.unsqueeze(-1)
                else:
                    field_out = field_out + b_block.unsqueeze(-2)

            out_blocks.append(field_out.reshape(batch_shape + (mul * d,)))

        if len(out_blocks) == 1:
            out_acc = out_blocks[0]
        else:
            out_acc = torch.cat(out_blocks, dim=-1)

        out = out_acc.to(dtype=orig_dtype)
        if mask is not None:
            mask_exp = mask.to(dtype=torch.bool).unsqueeze(-1)
            out = torch.where(mask_exp, out, torch.zeros_like(out))

        return out


class EquivariantRMSNorm(_EquivariantNormBase):
    r"""Equivariant Root-Mean-Square Normalization for :math:`SO(3)` / :math:`O(3)` representations.

    Accumulates squared norms in ``float32`` (or ``float64`` for ``float64`` inputs) to prevent
    ``bfloat16`` / ``float16`` overflow, uses a double-``where`` inverse RMS so exact zero inputs
    produce exact ``0.0`` in forward, first, and second derivatives even at ``eps=0.0``, and
    executes natively in both ``cue.mul_ir`` and ``cue.ir_mul`` layouts with zero transposes.

    Args:
        irreps (Irreps or str): Input and output irreducible representations.
        eps (float, optional): Small constant added inside ``rsqrt`` for numerical stability.
            Defaults to ``1e-5``. Supports ``0.0`` safely via double-``where``.
        affine (bool, optional): If ``True``, learns one :math:`O(3)`-invariant scalar weight
            per channel ``(mul,)`` across all irrep blocks. Defaults to ``True``.
        centering (bool, optional): If ``True``, subtracts the channel mean **only** on invariant
            scalar irreps (``ir.is_scalar()``, e.g. ``0e``), never on :math:`\ell > 0` or
            pseudoscalars ``0o``. Defaults to ``False``.
        norm_mode (str, optional): Variance aggregation mode:
            ``"per_irrep"`` (across ``mul`` and ``m`` per ``(mul, ir)`` block, default),
            ``"per_channel"`` (across ``m`` for each individual channel), or
            ``"global"`` (across all channels of the entire representation, as in EquiformerV2).
        normalization (str, optional): ``"component"`` (mean over ``ir.dim`` so unit-variance
            components stay unit-variance, default) or ``"norm"`` (sum over ``ir.dim``).
        group (type[Irrep] or str, optional): Symmetry group used when parsing string irreps.
        layout (IrrepsLayout, optional): Memory layout (``cue.mul_ir`` or ``cue.ir_mul``).
        device (torch.device, optional): Parameter device.
        dtype (torch.dtype, optional): Parameter dtype.
    """

    __constants__ = _EquivariantNormBase.__constants__

    def __init__(
        self,
        irreps: cue.Irreps | str,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        centering: bool = False,
        norm_mode: str = "per_irrep",
        normalization: str = "component",
        group: type[cue.Irrep] | str | None = None,
        layout: cue.IrrepsLayout | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            irreps,
            eps=eps,
            affine=affine,
            include_bias=False,
            centering=centering,
            norm_mode=norm_mode,
            normalization=normalization,
            group=group,
            layout=layout,
            device=device,
            dtype=dtype,
        )

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return (
            f"{self.irreps}, layout={self.layout}, eps={self.eps}, "
            f"affine={self.affine}, centering={self.centering}, "
            f"norm_mode={self.norm_mode!r}, normalization={self.normalization!r}"
        )


class EquivariantLayerNorm(_EquivariantNormBase):
    r"""Equivariant Layer Normalization for :math:`SO(3)` / :math:`O(3)` representations.

    Centers channel means and applies learnable ``bias`` **only** on invariant scalar irreps
    (``ir.is_scalar()``, e.g. ``0e`` in :math:`O(3)` or ``0`` in :math:`SO(3)`), while non-scalar
    irreps (including pseudoscalars ``0o`` and all :math:`\ell > 0` irreps) receive **no** mean
    subtraction and **no** bias so exact :math:`O(3)` rotation and parity equivariance is
    preserved. All irrep blocks are normalized by zero-safe FP32/FP64 RMS norm with optional
    per-channel affine ``weight``.

    Args:
        irreps (Irreps or str): Input and output irreducible representations.
        eps (float, optional): Small constant added inside ``rsqrt`` for numerical stability.
            Defaults to ``1e-5``. Supports ``0.0`` safely via double-``where``.
        affine (bool, optional): If ``True``, learns per-channel ``weight`` across all irrep
            blocks and (when ``bias=True``) per-channel ``bias`` on invariant scalars.
            Defaults to ``True``.
        bias (bool, optional): If ``True`` (and ``affine=True``), includes learnable scalar
            bias on invariant scalar irreps (``ir.is_scalar()``). Defaults to ``True``.
        norm_mode (str, optional): Variance aggregation mode:
            ``"per_irrep"`` (default), ``"per_channel"``, or ``"global"``.
        normalization (str, optional): ``"component"`` (default) or ``"norm"``.
        group (type[Irrep] or str, optional): Symmetry group used when parsing string irreps.
        layout (IrrepsLayout, optional): Memory layout (``cue.mul_ir`` or ``cue.ir_mul``).
        device (torch.device, optional): Parameter device.
        dtype (torch.dtype, optional): Parameter dtype.
    """

    __constants__ = _EquivariantNormBase.__constants__

    def __init__(
        self,
        irreps: cue.Irreps | str,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = True,
        norm_mode: str = "per_irrep",
        normalization: str = "component",
        group: type[cue.Irrep] | str | None = None,
        layout: cue.IrrepsLayout | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            irreps,
            eps=eps,
            affine=affine,
            include_bias=bias,
            centering=True,
            norm_mode=norm_mode,
            normalization=normalization,
            group=group,
            layout=layout,
            device=device,
            dtype=dtype,
        )

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return (
            f"{self.irreps}, layout={self.layout}, eps={self.eps}, "
            f"affine={self.affine}, bias={self._has_bias}, "
            f"norm_mode={self.norm_mode!r}, normalization={self.normalization!r}"
        )
