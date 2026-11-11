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
from __future__ import annotations

import math
from typing import Optional, Sequence

import cuequivariance as cue
import torch
import torch.nn as nn
from cuequivariance import descriptors
from cuequivariance.group_theory.irreps_array.misc_ui import default_layout

import cuequivariance_torch as cuet


def _resolve_torch_layout(
    layout: Optional[cue.IrrepsLayout | str],
) -> cue.IrrepsLayout:
    if layout is None:
        scope_layout = cue.get_layout_scope(raising=False)
        if scope_layout is not None:
            return scope_layout
    return default_layout(layout)


class CartesianToIrreps(nn.Module):
    r"""PyTorch module mapping Cartesian tensors to spherical harmonic irreducible representations.

    Computes the orthogonal projection :math:`c = Q^T T_{\text{cart}}` where :math:`Q^T Q = I`,
    supporting arbitrary permutation symmetry formulas (monopole ``""``, dipole ``"i"``,
    antisymmetric rank-2 ``"ij=-ji"``, polarizability/stress ``"ij=ji"``, octupole
    ``"ijk=jik=ikj"``, etc.) and optional traceless harmonic restriction (:math:`\ell = k`).

    Args:
        formula (str): Index-symmetry formula (e.g. ``""``, ``"i"``, ``"ij=ji"``, ``"ijk=jik=ikj"``).
        group (type[Irrep] or str, optional): Symmetry group, ``cue.O3`` (default) or ``cue.SO3``.
        vector_parity (int, optional): Parity of a 3D Cartesian vector under inversion for ``O3``
            (``-1`` for polar ``1o``, ``+1`` for axial ``1e``). Defaults to ``-1``.
        traceless (bool, optional): If ``True``, restricts output to the symmetric traceless
            harmonic subspace :math:`\ell = \text{degree}`. Defaults to ``False``.
        keep_ir (sequence of Irrep or Irreps, optional): Optional filter of output irreps to keep.
        multiplicity (int, optional): Number of channels. Defaults to ``1``.
        layout (IrrepsLayout or str, optional): Memory layout (``cue.ir_mul`` or ``cue.mul_ir``).
        device (torch.device, optional): Target device for module buffers.
        math_dtype (torch.dtype or str, optional): Computation dtype for ``SegmentedPolynomial``.
        method (str, optional): Execution backend for ``SegmentedPolynomial``
            (defaults to ``"uniform_1d"``, falling back cleanly to ``"naive"`` when unavailable).
    """

    def __init__(
        self,
        formula: str,
        *,
        group: type[cue.Irrep] | str = cue.O3,
        vector_parity: int = -1,
        traceless: bool = False,
        keep_ir: Optional[Sequence[cue.Irrep] | cue.Irreps] = None,
        multiplicity: int = 1,
        layout: Optional[cue.IrrepsLayout | str] = None,
        device: Optional[torch.device] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        method: Optional[str] = None,
    ):
        super().__init__()
        resolved_layout = _resolve_torch_layout(layout)

        e = descriptors.cartesian_to_irreps(
            formula,
            group=group,
            vector_parity=vector_parity,
            traceless=traceless,
            keep_ir=keep_ir,
            multiplicity=multiplicity,
            layout=cue.ir_mul,
        )
        cart_rep = e.inputs[0]
        assert isinstance(cart_rep, descriptors.CartesianTensorRep)

        self.formula: str = formula
        self.degree: int = cart_rep.degree
        self.multiplicity: int = cart_rep.multiplicity
        self.traceless: bool = traceless
        self.layout: cue.IrrepsLayout = resolved_layout
        self.irreps_out: cue.Irreps = e.outputs[0].irreps
        self.cart_dim: int = 3**self.degree
        self.total_cart_dim: int = self.multiplicity * self.cart_dim
        self._polynomial_orig = e.polynomial
        self._math_dtype_orig = math_dtype
        self._explicit_method = method

        self.transpose_out = cuet.TransposeIrrepsLayout(
            self.irreps_out,
            source=cue.ir_mul,
            target=self.layout,
            device=device,
        )

        if self.irreps_out.dim == 0:
            self.method = "uniform_1d" if method is None else method
            self.f = None
        else:
            if (
                method is None
                and device is not None
                and torch.device(device).type == "cpu"
            ):
                req_method = "naive"
            else:
                req_method = "uniform_1d" if method is None else method
            self.f = cuet.SegmentedPolynomial(
                e.polynomial,
                method=req_method,
                math_dtype=math_dtype,
            ).to(device)
            self.method = self.f.method

    def extra_repr(self) -> str:
        return (
            f"formula={self.formula!r}, irreps_out={self.irreps_out}, "
            f"degree={self.degree}, multiplicity={self.multiplicity}, "
            f"traceless={self.traceless}, layout={self.layout}"
        )

    def _apply_sp(self, x_2d: torch.Tensor) -> torch.Tensor:
        assert self.f is not None
        if (
            x_2d.device.type == "cpu"
            and self.f.method == "uniform_1d"
            and self._explicit_method is None
        ):
            if not hasattr(self, "_cpu_fallback"):
                object.__setattr__(
                    self,
                    "_cpu_fallback",
                    cuet.SegmentedPolynomial(
                        self._polynomial_orig,
                        method="naive",
                        math_dtype=self._math_dtype_orig,
                    ),
                )
            [out] = self._cpu_fallback([x_2d])
            return out
        [out] = self.f([x_2d])
        return out

    def _prepare_input(self, x: torch.Tensor) -> tuple[tuple[int, ...], torch.Tensor]:
        k = self.degree
        M = self.multiplicity
        cart_dim = self.cart_dim
        total_cart_dim = self.total_cart_dim

        if M == 1:
            if k == 0:
                if x.ndim >= 2 and x.shape[-1] == 1:
                    return tuple(x.shape[:-1]), x
                return tuple(x.shape), x.unsqueeze(-1)
            if k == 1:
                if x.ndim < 1 or x.shape[-1] != 3:
                    raise ValueError(
                        f"Expected input shape (*batch, 3), got {tuple(x.shape)}."
                    )
                return tuple(x.shape[:-1]), x
            if x.ndim >= k and x.shape[-k:] == (3,) * k:
                batch_shape = tuple(x.shape[:-k])
                return batch_shape, x.reshape(batch_shape + (cart_dim,))
            if x.ndim >= 1 and x.shape[-1] == cart_dim:
                return tuple(x.shape[:-1]), x
            raise ValueError(
                f"Expected input shape (*batch, {', '.join(['3'] * k)}) or "
                f"(*batch, {cart_dim}), got {tuple(x.shape)}."
            )

        # M > 1
        if k == 0:
            if x.ndim < 1 or x.shape[-1] != M:
                raise ValueError(
                    f"Expected input shape (*batch, {M}), got {tuple(x.shape)}."
                )
            return tuple(x.shape[:-1]), x

        if self.layout == cue.mul_ir:
            shaped_suffix = (M,) + (3,) * k
            if x.ndim >= k + 1 and x.shape[-(k + 1) :] == shaped_suffix:
                batch_shape = tuple(x.shape[: -(k + 1)])
                x_ir_mul = (
                    x.reshape(batch_shape + (M, cart_dim))
                    .transpose(-2, -1)
                    .reshape(batch_shape + (total_cart_dim,))
                )
                return batch_shape, x_ir_mul
            if x.ndim >= 1 and x.shape[-1] == total_cart_dim:
                batch_shape = tuple(x.shape[:-1])
                x_ir_mul = (
                    x.reshape(batch_shape + (M, cart_dim))
                    .transpose(-2, -1)
                    .reshape(batch_shape + (total_cart_dim,))
                )
                return batch_shape, x_ir_mul
            raise ValueError(
                f"Expected input shape (*batch, {M}, {', '.join(['3'] * k)}) or "
                f"(*batch, {total_cart_dim}) for mul_ir layout, got {tuple(x.shape)}."
            )

        # self.layout == cue.ir_mul
        shaped_suffix = (3,) * k + (M,)
        if x.ndim >= k + 1 and x.shape[-(k + 1) :] == shaped_suffix:
            batch_shape = tuple(x.shape[: -(k + 1)])
            return batch_shape, x.reshape(batch_shape + (total_cart_dim,))
        if x.ndim >= 1 and x.shape[-1] == total_cart_dim:
            return tuple(x.shape[:-1]), x
        raise ValueError(
            f"Expected input shape (*batch, {', '.join(['3'] * k)}, {M}) or "
            f"(*batch, {total_cart_dim}) for ir_mul layout, got {tuple(x.shape)}."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project Cartesian tensor ``x`` onto spherical harmonic irreps."""
        batch_shape, x_ir_mul = self._prepare_input(x)
        batch_numel = math.prod(batch_shape)
        x_2d = x_ir_mul.reshape(batch_numel, self.total_cart_dim).contiguous()

        if self.f is None or batch_numel == 0:
            if self.irreps_out.dim == 0:
                return x_2d[:, :0].reshape(batch_shape + (0,))
            return x_2d.new_zeros(batch_shape + (self.irreps_out.dim,))

        out_ir_mul_2d = self._apply_sp(x_2d)
        out_2d = self.transpose_out(out_ir_mul_2d)
        return out_2d.reshape(batch_shape + (self.irreps_out.dim,))


class IrrepsToCartesian(nn.Module):
    r"""PyTorch module synthesizing Cartesian tensors from spherical harmonic irreps.

    Computes :math:`T_{\text{cart}} = Q c`, the exact transpose of :class:`CartesianToIrreps`.
    Because :math:`Q^T Q = I`, spherical coefficients satisfy
    ``cart2ir(ir2cart(c)) == c`` to machine precision, and reconstructed multipole tensors
    automatically satisfy exact index permutation symmetry and trace-freeness.

    Args:
        formula (str): Index-symmetry formula (e.g. ``""``, ``"i"``, ``"ij=ji"``, ``"ijk=jik=ikj"``).
        group (type[Irrep] or str, optional): Symmetry group, ``cue.O3`` (default) or ``cue.SO3``.
        vector_parity (int, optional): Parity of a 3D Cartesian vector under inversion for ``O3``
            (``-1`` for polar ``1o``, ``+1`` for axial ``1e``). Defaults to ``-1``.
        traceless (bool, optional): If ``True``, restricts input to the symmetric traceless
            harmonic subspace :math:`\ell = \text{degree}`. Defaults to ``False``.
        keep_ir (sequence of Irrep or Irreps, optional): Optional filter of input irreps to include.
        multiplicity (int, optional): Number of channels. Defaults to ``1``.
        layout (IrrepsLayout or str, optional): Memory layout (``cue.ir_mul`` or ``cue.mul_ir``).
        reshape_output (bool, optional): If ``True`` (default), reshapes Cartesian axes to
            ``(3, ..., 3)``; if ``False``, returns flattened ``3**degree`` (scaled by ``multiplicity``).
        device (torch.device, optional): Target device for module buffers.
        math_dtype (torch.dtype or str, optional): Computation dtype for ``SegmentedPolynomial``.
        method (str, optional): Execution backend for ``SegmentedPolynomial``
            (defaults to ``"uniform_1d"``, falling back cleanly to ``"naive"`` when unavailable).
    """

    def __init__(
        self,
        formula: str,
        *,
        group: type[cue.Irrep] | str = cue.O3,
        vector_parity: int = -1,
        traceless: bool = False,
        keep_ir: Optional[Sequence[cue.Irrep] | cue.Irreps] = None,
        multiplicity: int = 1,
        layout: Optional[cue.IrrepsLayout | str] = None,
        reshape_output: bool = True,
        device: Optional[torch.device] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        method: Optional[str] = None,
    ):
        super().__init__()
        resolved_layout = _resolve_torch_layout(layout)

        e = descriptors.irreps_to_cartesian(
            formula,
            group=group,
            vector_parity=vector_parity,
            traceless=traceless,
            keep_ir=keep_ir,
            multiplicity=multiplicity,
            layout=cue.ir_mul,
        )
        cart_rep = e.outputs[0]
        assert isinstance(cart_rep, descriptors.CartesianTensorRep)

        self.formula: str = formula
        self.irreps_in: cue.Irreps = e.inputs[0].irreps
        self.degree: int = cart_rep.degree
        self.multiplicity: int = cart_rep.multiplicity
        self.traceless: bool = traceless
        self.layout: cue.IrrepsLayout = resolved_layout
        self.reshape_output: bool = reshape_output
        self.cart_dim: int = 3**self.degree
        self.total_cart_dim: int = self.multiplicity * self.cart_dim
        self._polynomial_orig = e.polynomial
        self._math_dtype_orig = math_dtype
        self._explicit_method = method

        self.transpose_in = cuet.TransposeIrrepsLayout(
            self.irreps_in,
            source=self.layout,
            target=cue.ir_mul,
            device=device,
        )

        if self.irreps_in.dim == 0:
            self.method = "uniform_1d" if method is None else method
            self.f = None
        else:
            if (
                method is None
                and device is not None
                and torch.device(device).type == "cpu"
            ):
                req_method = "naive"
            else:
                req_method = "uniform_1d" if method is None else method
            self.f = cuet.SegmentedPolynomial(
                e.polynomial,
                method=req_method,
                math_dtype=math_dtype,
            ).to(device)
            self.method = self.f.method

    def extra_repr(self) -> str:
        return (
            f"formula={self.formula!r}, irreps_in={self.irreps_in}, "
            f"degree={self.degree}, multiplicity={self.multiplicity}, "
            f"traceless={self.traceless}, layout={self.layout}, "
            f"reshape_output={self.reshape_output}"
        )

    def _apply_sp(self, x_2d: torch.Tensor) -> torch.Tensor:
        assert self.f is not None
        if (
            x_2d.device.type == "cpu"
            and self.f.method == "uniform_1d"
            and self._explicit_method is None
        ):
            if not hasattr(self, "_cpu_fallback"):
                object.__setattr__(
                    self,
                    "_cpu_fallback",
                    cuet.SegmentedPolynomial(
                        self._polynomial_orig,
                        method="naive",
                        math_dtype=self._math_dtype_orig,
                    ),
                )
            [out] = self._cpu_fallback([x_2d])
            return out
        [out] = self.f([x_2d])
        return out

    def _format_output(
        self, cart_ir_mul_2d: torch.Tensor, batch_shape: tuple[int, ...]
    ) -> torch.Tensor:
        M = self.multiplicity
        k = self.degree
        if M == 1:
            if self.reshape_output:
                return cart_ir_mul_2d.reshape(batch_shape + (3,) * k)
            return cart_ir_mul_2d.reshape(batch_shape + (self.cart_dim,))

        if self.layout == cue.ir_mul:
            if self.reshape_output:
                return cart_ir_mul_2d.reshape(batch_shape + (3,) * k + (M,))
            return cart_ir_mul_2d.reshape(batch_shape + (self.total_cart_dim,))

        # self.layout == cue.mul_ir
        batch_numel = cart_ir_mul_2d.shape[0]
        cart_mul_ir = cart_ir_mul_2d.reshape(batch_numel, self.cart_dim, M).transpose(
            -2, -1
        )
        if self.reshape_output:
            return cart_mul_ir.reshape(batch_shape + (M,) + (3,) * k)
        return cart_mul_ir.reshape(batch_shape + (self.total_cart_dim,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Synthesize Cartesian tensor from spherical harmonic irreps ``x``."""
        if x.ndim < 1 or x.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Expected input shape (*batch, {self.irreps_in.dim}), got {tuple(x.shape)}."
            )
        batch_shape = tuple(x.shape[:-1])
        batch_numel = math.prod(batch_shape)

        if self.f is None or batch_numel == 0:
            cart_2d = x.new_zeros((batch_numel, self.total_cart_dim))
            return self._format_output(cart_2d, batch_shape)

        x_2d = x.reshape(batch_numel, self.irreps_in.dim)
        x_ir_mul = self.transpose_in(x_2d).contiguous()
        cart_ir_mul_2d = self._apply_sp(x_ir_mul)
        return self._format_output(cart_ir_mul_2d, batch_shape)
