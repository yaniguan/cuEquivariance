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
from collections.abc import Callable, Sequence
from typing import Optional

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
    default_layout,
)

import cuequivariance as cue


class _CallableActivation(nn.Module):
    """Fallback wrapper for non-FX-traceable Python callables."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self._fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x)

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return getattr(self._fn, "__name__", repr(self._fn))


_BUILTIN_ACTIVATIONS: dict[
    Callable[[torch.Tensor], torch.Tensor], Callable[[], nn.Module]
] = {
    torch.sigmoid: nn.Sigmoid,
    F.sigmoid: nn.Sigmoid,
    torch.tanh: nn.Tanh,
    F.tanh: nn.Tanh,
    F.silu: nn.SiLU,
    torch.relu: nn.ReLU,
    F.relu: nn.ReLU,
    F.gelu: nn.GELU,
    F.elu: nn.ELU,
    F.celu: nn.CELU,
    F.selu: nn.SELU,
    F.mish: nn.Mish,
    F.softplus: nn.Softplus,
    F.softsign: nn.Softsign,
    F.leaky_relu: nn.LeakyReLU,
    F.hardswish: nn.Hardswish,
    F.hardsigmoid: nn.Hardsigmoid,
}


def _to_activation_module(
    act: Callable[[torch.Tensor], torch.Tensor] | None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        if device is not None or dtype is not None:
            return act.to(device=device, dtype=dtype)
        return act
    if not callable(act):
        raise TypeError(
            f"Activation must be a callable or None, got {type(act).__name__}."
        )
    factory = _BUILTIN_ACTIVATIONS.get(act)
    if factory is not None:
        return factory()
    try:
        return torch.fx.symbolic_trace(lambda x: act(x))
    except Exception:
        return _CallableActivation(act)


def _resolve_group(group: type[cue.Irrep] | cue.Irrep | str) -> type[cue.Irrep]:
    if isinstance(group, str):
        resolved = getattr(cue, group, None)
        if not (isinstance(resolved, type) and issubclass(resolved, cue.Irrep)):
            raise ValueError(f"Unknown symmetry group {group!r}.")
        return resolved
    if isinstance(group, cue.Irrep):
        return type(group)
    if isinstance(group, type) and issubclass(group, cue.Irrep):
        return group
    raise ValueError(f"Invalid symmetry group {group!r}.")


def _resolve_irreps_list(
    *irreps_in: cue.Irreps | str,
    group: type[cue.Irrep] | str | None = None,
) -> list[cue.Irreps]:
    if group is not None:
        irrep_class = _resolve_group(group)
        resolved: list[cue.Irreps] = []
        for irr in irreps_in:
            if isinstance(irr, cue.Irreps):
                if irr.irrep_class != irrep_class:
                    raise ValueError(
                        f"Irreps {irr} has group {irr.irrep_class}, "
                        f"expected {irrep_class}."
                    )
                resolved.append(irr)
            else:
                resolved.append(cue.Irreps(irrep_class, irr))
        return resolved

    scope = cue.get_irrep_scope(raising=False)
    if scope is not None:
        resolved = [
            irr if isinstance(irr, cue.Irreps) else cue.Irreps(scope, irr)
            for irr in irreps_in
        ]
        assert_same_group(*resolved)
        return resolved

    known_groups = {irr.irrep_class for irr in irreps_in if isinstance(irr, cue.Irreps)}
    if len(known_groups) > 1:
        raise ValueError("The provided irreps are not of the same group.")
    if len(known_groups) == 1:
        inferred = next(iter(known_groups))
        resolved = []
        for irr in irreps_in:
            if isinstance(irr, cue.Irreps):
                resolved.append(irr)
            elif isinstance(irr, str):
                resolved.append(cue.Irreps(inferred, irr))
            else:
                (conv,) = default_irreps(irr)
                resolved.append(conv)
        assert_same_group(*resolved)
        return resolved

    resolved = list(default_irreps(*irreps_in))
    assert_same_group(*resolved)
    return resolved


def _resolve_layout(layout: cue.IrrepsLayout | str | None) -> cue.IrrepsLayout:
    if layout is None:
        scoped = cue.get_layout_scope(raising=False)
        if scoped is not None:
            return scoped
    resolved = default_layout(layout)
    if resolved not in (cue.mul_ir, cue.ir_mul):
        raise ValueError(f"Invalid layout {resolved!r}.")
    return resolved


class Gate(nn.Module):
    r"""Equivariant gated nonlinearity for :math:`SO(3)` / :math:`O(3)` representations.

    Applies elementwise activations ``act_scalars`` to invariant scalar channels
    ``irreps_scalars``, applies ``act_gates`` to invariant scalar gate channels
    ``irreps_gates``, and multiplies each channel in ``irreps_gated`` by its
    corresponding activated scalar gate. Executes natively in both ``cue.mul_ir``
    and ``cue.ir_mul`` layouts with zero layout transpositions.

    The concatenated input representation is
    ``irreps_in = irreps_scalars + irreps_gates + irreps_gated`` and the output
    representation is ``irreps_out = irreps_scalars + irreps_gated``.

    Args:
        irreps_scalars (Irreps or str): Invariant scalar irreps passed to the output
            after applying ``act_scalars``. Every irrep must satisfy ``ir.is_scalar()``.
        act_scalars (Sequence[Callable or None]): Activation function (or ``None`` for
            identity) for each ``(mul, ir)`` block in ``irreps_scalars``.
        irreps_gates (Irreps or str): Invariant scalar irreps used as multiplicative
            gates after applying ``act_gates``. Every irrep must satisfy ``ir.is_scalar()``.
        act_gates (Sequence[Callable or None]): Activation function (or ``None`` for
            identity) for each ``(mul, ir)`` block in ``irreps_gates``.
        irreps_gated (Irreps or str): Irreps whose channels are multiplicatively gated
            by ``irreps_gates``. Total multiplicity must equal that of ``irreps_gates``.
        group (type[Irrep] or str, optional): Symmetry group (e.g. ``cue.O3`` or ``"O3"``)
            used when parsing string irreps.
        layout (IrrepsLayout, optional): Memory layout (``cue.mul_ir`` or ``cue.ir_mul``).
        device (torch.device, optional): Target device for any stateful activation modules.
        dtype (torch.dtype, optional): Target dtype for any stateful activation modules.
    """

    __constants__ = [
        "_scalar_slices",
        "_gate_slices",
        "_gated_blocks",
        "_is_mul_ir",
        "_scalars_dim",
        "_gates_dim",
        "_gated_dim",
        "_in_dim",
        "_out_dim",
    ]

    def __init__(
        self,
        irreps_scalars: cue.Irreps | str,
        act_scalars: Sequence[Callable[[torch.Tensor], torch.Tensor] | None],
        irreps_gates: cue.Irreps | str,
        act_gates: Sequence[Callable[[torch.Tensor], torch.Tensor] | None],
        irreps_gated: cue.Irreps | str,
        *,
        group: type[cue.Irrep] | str | None = None,
        layout: cue.IrrepsLayout | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        irreps_scalars_r, irreps_gates_r, irreps_gated_r = _resolve_irreps_list(
            irreps_scalars, irreps_gates, irreps_gated, group=group
        )

        for name, irr in (
            ("irreps_scalars", irreps_scalars_r),
            ("irreps_gates", irreps_gates_r),
            ("irreps_gated", irreps_gated_r),
        ):
            if any(mul < 0 for mul, _ in irr):
                raise ValueError(
                    f"{name} cannot contain negative multiplicities, got {irr}."
                )

        if any(not ir.is_scalar() or ir.dim != 1 for _, ir in irreps_scalars_r):
            raise ValueError(
                f"All irreps in irreps_scalars must be scalar (is_scalar()), "
                f"got {irreps_scalars_r}."
            )
        if any(not ir.is_scalar() or ir.dim != 1 for _, ir in irreps_gates_r):
            raise ValueError(
                f"All irreps in irreps_gates must be scalar (is_scalar()), "
                f"got {irreps_gates_r}."
            )

        act_scalars_list = list(act_scalars)
        act_gates_list = list(act_gates)
        if len(act_scalars_list) != len(irreps_scalars_r):
            raise ValueError(
                f"Length of act_scalars ({len(act_scalars_list)}) must match "
                f"number of blocks in irreps_scalars ({len(irreps_scalars_r)})."
            )
        if len(act_gates_list) != len(irreps_gates_r):
            raise ValueError(
                f"Length of act_gates ({len(act_gates_list)}) must match "
                f"number of blocks in irreps_gates ({len(act_gates_list)} vs {len(irreps_gates_r)})."
            )

        num_gates = irreps_gates_r.num_irreps
        num_gated = irreps_gated_r.num_irreps
        if num_gates != num_gated:
            raise ValueError(
                f"Total multiplicity of irreps_gates ({num_gates}) must equal "
                f"total multiplicity of irreps_gated ({num_gated})."
            )

        resolved_layout = _resolve_layout(layout)

        self.irreps_scalars = irreps_scalars_r
        self.irreps_gates = irreps_gates_r
        self.irreps_gated = irreps_gated_r
        self.irreps_in = irreps_scalars_r + irreps_gates_r + irreps_gated_r
        self.irreps_out = irreps_scalars_r + irreps_gated_r
        self.layout = resolved_layout
        self._is_mul_ir: bool = resolved_layout == cue.mul_ir

        self.act_scalars = nn.ModuleList(
            [
                _to_activation_module(a, device=device, dtype=dtype)
                for a in act_scalars_list
            ]
        )
        self.act_gates = nn.ModuleList(
            [
                _to_activation_module(a, device=device, dtype=dtype)
                for a in act_gates_list
            ]
        )

        scalar_slices: list[tuple[int, int]] = []
        offset = 0
        for mul, ir in irreps_scalars_r:
            d = mul * ir.dim
            scalar_slices.append((offset, offset + d))
            offset += d
        self._scalar_slices = scalar_slices
        self._scalars_dim: int = offset

        gate_slices: list[tuple[int, int]] = []
        offset = 0
        for mul, ir in irreps_gates_r:
            d = mul * ir.dim
            gate_slices.append((offset, offset + d))
            offset += d
        self._gate_slices = gate_slices
        self._gates_dim: int = offset

        gated_blocks: list[tuple[int, int, int, int, int, int]] = []
        v_offset = 0
        g_offset = 0
        for mul, ir in irreps_gated_r:
            block_dim = mul * ir.dim
            gated_blocks.append(
                (v_offset, v_offset + block_dim, g_offset, g_offset + mul, mul, ir.dim)
            )
            v_offset += block_dim
            g_offset += mul
        self._gated_blocks = gated_blocks
        self._gated_dim: int = v_offset

        self._in_dim: int = self._scalars_dim + self._gates_dim + self._gated_dim
        self._out_dim: int = self._scalars_dim + self._gated_dim

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return (
            f"{self.irreps_scalars} + {self.irreps_gates} + {self.irreps_gated} "
            f"-> {self.irreps_out}, layout={self.layout}"
        )

    def forward(
        self,
        x: torch.Tensor,
        gated: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the equivariant gate nonlinearity.

        Supports either a single concatenated tensor ``x`` of shape
        ``(*batch, irreps_in.dim)`` or two separate tensors ``(scalars_and_gates, gated)``
        of shapes ``(*batch, irreps_scalars.dim + irreps_gates.dim)`` and
        ``(*batch, irreps_gated.dim)``.

        Args:
            x (torch.Tensor): Concatenated input tensor of shape ``(*batch, irreps_in.dim)``,
                or scalar+gate tensor of shape ``(*batch, irreps_scalars.dim + irreps_gates.dim)``
                when ``gated`` is provided as a separate feature tensor.
            gated (torch.Tensor, optional): Optional separate gated feature tensor of shape
                ``(*batch, irreps_gated.dim)`` (or a boolean ``mask`` if passed positionally
                with a single concatenated ``x``).
            mask (torch.Tensor, optional): Optional boolean mask of shape ``(*batch,)`` where
                ``True`` marks valid entries and ``False`` zeroes masked entries cleanly
                without ``NaN`` propagation.

        Returns:
            torch.Tensor: Output tensor of shape ``(*batch, irreps_out.dim)``.
        """
        if gated is not None and gated.dtype == torch.bool and mask is None:
            mask = gated
            gated = None

        batch_shape = x.shape[:-1]

        if gated is None:
            if x.shape[-1] != self._in_dim:
                raise ValueError(
                    f"Expected input last dim {self._in_dim}, got {x.shape[-1]}."
                )
            if mask is not None:
                if mask.shape != batch_shape:
                    raise ValueError(
                        f"Expected mask shape {batch_shape}, got {mask.shape}."
                    )
                mask_exp = mask.to(dtype=torch.bool).unsqueeze(-1)
                x = torch.where(mask_exp, x, torch.zeros_like(x))
            scalars_in = x[..., : self._scalars_dim]
            gates_in = x[..., self._scalars_dim : self._scalars_dim + self._gates_dim]
            gated_in = x[..., self._scalars_dim + self._gates_dim :]
        else:
            if gated.shape[:-1] != batch_shape:
                raise ValueError(
                    f"Leading batch shapes of x ({batch_shape}) and gated "
                    f"({gated.shape[:-1]}) must match."
                )
            expected_sg = self._scalars_dim + self._gates_dim
            if x.shape[-1] != expected_sg:
                raise ValueError(
                    f"Expected first input last dim {expected_sg}, got {x.shape[-1]}."
                )
            if gated.shape[-1] != self._gated_dim:
                raise ValueError(
                    f"Expected gated input last dim {self._gated_dim}, got {gated.shape[-1]}."
                )
            if mask is not None:
                if mask.shape != batch_shape:
                    raise ValueError(
                        f"Expected mask shape {batch_shape}, got {mask.shape}."
                    )
                mask_exp = mask.to(dtype=torch.bool).unsqueeze(-1)
                x = torch.where(mask_exp, x, torch.zeros_like(x))
                gated = torch.where(mask_exp, gated, torch.zeros_like(gated))
            scalars_in = x[..., : self._scalars_dim]
            gates_in = x[..., self._scalars_dim :]
            gated_in = gated

        out_parts: list[torch.Tensor] = []
        for i, act_s in enumerate(self.act_scalars):
            s_start, s_end = self._scalar_slices[i]
            out_parts.append(act_s(scalars_in[..., s_start:s_end]))

        if len(self._gated_blocks) > 0:
            gate_parts: list[torch.Tensor] = []
            for i, act_g in enumerate(self.act_gates):
                g_start, g_end = self._gate_slices[i]
                gate_parts.append(act_g(gates_in[..., g_start:g_end]))

            if len(gate_parts) == 0:
                gates_act = gates_in
            elif len(gate_parts) == 1:
                gates_act = gate_parts[0]
            else:
                gates_act = torch.cat(gate_parts, dim=-1)

            for v_start, v_end, g_start, g_end, mul, ir_dim in self._gated_blocks:
                v_slice = gated_in[..., v_start:v_end]
                g_block = gates_act[..., g_start:g_end]
                if ir_dim == 1:
                    out_parts.append(v_slice * g_block)
                elif self._is_mul_ir:
                    v_reshaped = v_slice.reshape(batch_shape + (mul, ir_dim))
                    v_out = (v_reshaped * g_block.unsqueeze(-1)).reshape(
                        batch_shape + (mul * ir_dim,)
                    )
                    out_parts.append(v_out)
                else:
                    v_reshaped = v_slice.reshape(batch_shape + (ir_dim, mul))
                    v_out = (v_reshaped * g_block.unsqueeze(-2)).reshape(
                        batch_shape + (mul * ir_dim,)
                    )
                    out_parts.append(v_out)

        if len(out_parts) == 0:
            out = x.new_zeros(batch_shape + (0,))
        elif len(out_parts) == 1:
            out = out_parts[0]
        else:
            out = torch.cat(out_parts, dim=-1)

        if mask is not None:
            mask_exp = mask.to(dtype=torch.bool).unsqueeze(-1)
            out = torch.where(mask_exp, out, torch.zeros_like(out))

        return out
