# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import numpy as np

import cuequivariance as cue


def _irrep_reflection_sigma(ir: cue.Irrep) -> int:
    """Return reflection symmetry sigma = (-1)**l * p in {+1, -1} (always +1 for SO3)."""
    if isinstance(ir, cue.O3):
        return ((-1) ** ir.l) * ir.p
    return 1


# The function escn_iu_ju_ku below is a 1:1 adaptation of https://github.com/e3nn/e3nn-jax/blob/a2a81ab451b9cd597d7be27b3e1faba79457475d/e3nn_jax/experimental/linear_shtp.py#L38-L165
def escn_tp(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    m_max: Optional[int] = None,
    l_max: Optional[int] = None,
) -> cue.EquivariantPolynomial:
    """
    subsrcipts: ``weights[uv],input[u],output[v]``

    Tensor Product part of the eSCN convolution introduced in https://arxiv.org/pdf/2302.03655.pdf

    Args:
        irreps_in (Irreps): Irreps of the input.
        irreps_out (Irreps): Irreps of the output.
        m_max (int, optional): Maximum angular resolution around the principal axis.
        l_max (int, optional): Maximum angular resolution along the principal axis.

    Returns:
        EquivariantPolynomial:
            Descriptor of the tensor product part of the eSCN convolution.

            - Operand 0: weights
            - Operand 1: input
            - Operand 2: output
    """
    assert irreps_in.irrep_class == irreps_out.irrep_class
    G = irreps_in.irrep_class
    if G not in [cue.SO3, cue.O3]:
        # TODO: we could support SU2 since it shares the same Clebsch-Gordan coefficients as SO3 and O3
        raise NotImplementedError("Only SO3 and O3 are supported")

    if l_max is not None:

        def pr(mul_ir: cue.MulIrrep) -> bool:
            ir = mul_ir.ir

            return any(abs(other.l - ir.l) <= l_max for _, other in irreps_in)

        irreps_out = irreps_out.filter(keep=pr)

    d = cue.SegmentedTensorProduct.from_subscripts("iuv,ju,kv+ijk")

    for mul, ir in irreps_in:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps_out:
        d.add_segment(2, (ir.dim, mul))

    for i1, (mul1, ir1) in enumerate(irreps_in):
        for i2, (mul2, ir2) in enumerate(irreps_out):
            ell = min(ir1.l, ir2.l)

            if l_max is not None:
                if abs(ir1.l - ir2.l) > l_max:
                    continue
            if m_max is not None:
                ell = min(ell, m_max)

            # Scaled rotation (2 degrees of freedom per |m|)
            c = np.zeros((2 * ell + 1, ir1.dim, ir2.dim))
            for m in range(-ell, ell + 1):
                # "cosine" part
                c[ell - abs(m), ir1.l + m, ir2.l + m] = 1.0

                # "sine" part
                if m != 0:
                    c[ell + abs(m), ir1.l - m, ir2.l + m] = 1.0 if m > 0 else -1.0

            if G == cue.SO3:
                pass  # keep all the degrees of freedom
            elif G == cue.O3:
                if (-1) ** (ir1.l + ir2.l) == ir1.p * ir2.p:
                    # Symmetric case: keep only the "cosine" part
                    c = c[: ell + 1]
                elif ell > 0:
                    # Antisymmetric case: keep only the "sine" part
                    c = c[ell + 1 :]
                else:
                    c = None

            if c is not None:
                d.add_path(None, i1, i2, c=c)

    d = d.normalize_paths_for_operand(2)
    d = d.flatten_coefficient_modes()
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_in.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps_in, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def escn_tp_compact(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    m_max: Optional[int] = None,
) -> cue.SegmentedPolynomial:
    """
    subsrcipts: ``weights[uv],input[u],output[v]``

    Tensor Product part of the eSCN convolution introduced in https://arxiv.org/pdf/2302.03655.pdf

    This "compact" implementation puts the L index contiguous in memory (M-major order).
    For ``O3``, each ``m``-block is split into the two reflection-symmetry subspaces
    ``(-1)**l * p == +1`` and ``-1`` so symmetric cosine couplings and antisymmetric sine
    couplings remain single dense matrix-vector blocks per ``(m, sigma)``.
    This allows to create bigger segments and fewer paths than :func:`escn_tp`.

    Use :func:`irreps_to_spherical_signal` and :func:`spherical_signal_to_irreps` to permute
    inputs and outputs between standard L-major :class:`cue.Irreps` and compact M-major order,
    and :func:`escn_weights_to_compact` to map :func:`escn_tp` weights to compact layout.

    Args:
        irreps_in (Irreps): Irreps of the input.
        irreps_out (Irreps): Irreps of the output.
        m_max (int, optional): Maximum angular resolution around the principal axis.

    Returns:
        SegmentedPolynomial:
            Descriptor of the tensor product part of the eSCN convolution.

            - Operand 0: weights
            - Operand 1: input (in M-major compact order)
            - Operand 2: output (in M-major compact order)
    """
    assert irreps_in.irrep_class == irreps_out.irrep_class
    G = irreps_in.irrep_class
    if G not in [cue.SO3, cue.O3]:
        raise NotImplementedError("Only SO3 and O3 are supported")

    d = cue.SegmentedTensorProduct.from_subscripts("uv,u,v")
    sigmas = (1,) if G == cue.SO3 else (1, -1)

    l_max_in = max((ir.l for _, ir in irreps_in), default=-1)
    seg_in: dict[tuple[int, int], int] = {}
    for m in range(-l_max_in, l_max_in + 1):
        for sigma in sigmas:
            mul = sum(
                mul
                for mul, ir in irreps_in
                if abs(m) <= ir.l and _irrep_reflection_sigma(ir) == sigma
            )
            if G == cue.SO3 or mul > 0:
                seg_in[(m, sigma)] = d.add_segment(1, (mul,))

    l_max_out = max((ir.l for _, ir in irreps_out), default=-1)
    seg_out: dict[tuple[int, int], int] = {}
    for m in range(-l_max_out, l_max_out + 1):
        for sigma in sigmas:
            mul = sum(
                mul
                for mul, ir in irreps_out
                if abs(m) <= ir.l and _irrep_reflection_sigma(ir) == sigma
            )
            if G == cue.SO3 or mul > 0:
                seg_out[(m, sigma)] = d.add_segment(2, (mul,))

    if m_max is None:
        m_max_eff = min(l_max_in, l_max_out)
    else:
        m_max_eff = min(m_max, l_max_in, l_max_out)

    # m = 0: symmetric cosine coupling only (sigma_in == sigma_out)
    if m_max_eff >= 0:
        for sigma in sigmas:
            if (0, sigma) in seg_in and (0, sigma) in seg_out:
                d.add_path(None, seg_in[(0, sigma)], seg_out[(0, sigma)], c=1.0)

    for m in range(1, m_max_eff + 1):
        # "cosine" part: sigma_in == sigma_out
        for sigma in sigmas:
            if (-m, sigma) in seg_in and (-m, sigma) in seg_out:
                d.add_path(None, seg_in[(-m, sigma)], seg_out[(-m, sigma)], c=1.0)
                i = d.operands[0].num_segments - 1
                d.add_path(i, seg_in[(m, sigma)], seg_out[(m, sigma)], c=1.0)

        # "sine" part: sigma_out == +1 for SO3, sigma_out == -sigma_in for O3
        for sigma_in in sigmas:
            sigma_out = 1 if G == cue.SO3 else -sigma_in
            if (m, sigma_in) in seg_in and (-m, sigma_out) in seg_out:
                d.add_path(None, seg_in[(m, sigma_in)], seg_out[(-m, sigma_out)], c=1.0)
                i = d.operands[0].num_segments - 1
                d.add_path(i, seg_in[(-m, sigma_in)], seg_out[(m, sigma_out)], c=-1.0)

    d = d.normalize_paths_for_operand(2)
    return cue.SegmentedPolynomial.eval_last_operand(d)


class SphericalSignal(cue.Rep):
    r"""Group representation of a band-limited spherical signal in M-major compact order.

    Represents :math:`\bigoplus_{\ell=0}^{l_{\max}} \mathrm{mul} \times D^{(\ell)}` restricted to
    magnetic quantum numbers :math:`|m| \le m_{\max}`, ordered outer-by-:math:`m` from
    :math:`-m_{\max}` to :math:`+m_{\max}`, then :math:`\ell \in [|m|, l_{\max}]`, then channel
    :math:`u \in [0, \mathrm{mul})`.

    - When ``m_max == l_max`` (and ``axial=False``), the full 3-generator :math:`\mathfrak{so}(3)`
      Lie algebra acts on the representation (with spatial inversion ``H`` of parity
      ``parity * (-1)**ell`` on each :math:`\ell` when ``group == cue.O3``).
    - When ``m_max < l_max`` (or ``axial=True``), only rotations around the principal :math:`y`-axis
      leave the :math:`|m| \le m_{\max}` subspace invariant, so the continuous symmetry is the 1D
      axial :math:`\mathfrak{so}(2)_y` Lie algebra (``A = zeros((1, 1, 1))``) generated by :math:`J_y`.

    Args:
        mul (int): Channel multiplicity per :math:`(\ell, m)` component.
        l_max (int): Maximum spherical harmonic degree :math:`l_{\max} \ge 0`.
        m_max (int, optional): Maximum order :math:`m_{\max}` around the :math:`y`-axis.
            Defaults to ``l_max`` (clamped to ``min(m_max, l_max)``).
        group (type[cue.Irrep], optional): ``cue.SO3`` (default) or ``cue.O3``.
        parity (int, optional): Reflection symmetry ``(-1)**l * p`` in ``{+1, -1}`` when
            ``group == cue.O3`` (``+1`` for true scalar signals ``0e + 1o + 2e + ...``,
            ``-1`` for pseudoscalar signals ``0o + 1e + 2o + ...``). Ignored for ``cue.SO3``.
        axial (bool, optional): Force the 1D axial :math:`SO(2)_y` Lie algebra even when
            ``m_max == l_max`` (used e.g. by :meth:`trivial` of an axial representation).
    """

    def __init__(
        self,
        mul: int,
        l_max: int,
        m_max: Optional[int] = None,
        *,
        group: type[cue.Irrep] = cue.SO3,
        parity: int = 1,
        axial: Optional[bool] = None,
    ):
        if mul < 0:
            raise ValueError(f"mul must be non-negative, got {mul}")
        if l_max < 0:
            raise ValueError(f"l_max must be non-negative, got {l_max}")
        if m_max is None:
            m_max_eff = l_max
        else:
            if m_max < 0:
                raise ValueError(f"m_max must be non-negative, got {m_max}")
            m_max_eff = min(m_max, l_max)

        if group not in (cue.SO3, cue.O3):
            raise ValueError(f"Unsupported group {group}. Must be cue.SO3 or cue.O3.")
        if parity not in (1, -1):
            raise ValueError(f"parity must be 1 or -1, got {parity}")

        if axial is None:
            axial_eff = m_max_eff < l_max
        else:
            axial_eff = bool(axial)
            if not axial_eff and m_max_eff < l_max:
                raise ValueError(
                    "Full 3D Lie algebra (axial=False) requires m_max == l_max."
                )

        self.mul = int(mul)
        self.l_max = int(l_max)
        self.m_max = int(m_max_eff)
        self.group = group
        self.parity = 1 if group == cue.SO3 else int(parity)
        self.axial = axial_eff

    def _dim(self) -> int:
        num_lm = (2 * self.m_max + 1) * (self.l_max + 1) - self.m_max * (self.m_max + 1)
        return self.mul * num_lm

    def algebra(self) -> np.ndarray:
        if self.axial:
            return np.zeros((1, 1, 1), dtype=np.float64)
        return self.group.algebra()

    def continuous_generators(self) -> np.ndarray:
        lie_dim = 1 if self.axial else 3
        dim = self._dim()
        X = np.zeros((lie_dim, dim, dim), dtype=np.float64)
        if dim == 0:
            return X

        offsets: dict[tuple[int, int], int] = {}
        pos = 0
        for m in range(-self.m_max, self.m_max + 1):
            for ell in range(abs(m), self.l_max + 1):
                offsets[(m, ell)] = pos
                pos += self.mul

        eye = np.eye(self.mul, dtype=np.float64)
        for ell in range(self.l_max + 1):
            X_ell = cue.SO3(ell).continuous_generators()
            if self.axial:
                X_ell = X_ell[1:2]
            m_lim = min(ell, self.m_max)
            for m1 in range(-m_lim, m_lim + 1):
                for m2 in range(-m_lim, m_lim + 1):
                    vals = X_ell[:, ell + m1, ell + m2]
                    if np.any(vals != 0.0):
                        i1 = offsets[(m1, ell)]
                        i2 = offsets[(m2, ell)]
                        X[:, i1 : i1 + self.mul, i2 : i2 + self.mul] = (
                            vals[:, None, None] * eye[None, :, :]
                        )
        return X

    def discrete_generators(self) -> np.ndarray:
        dim = self._dim()
        if self.group == cue.SO3:
            return np.zeros((0, dim, dim), dtype=np.float64)

        if dim == 0:
            return np.zeros((1, 0, 0), dtype=np.float64)

        diag = np.empty(dim, dtype=np.float64)
        pos = 0
        for m in range(-self.m_max, self.m_max + 1):
            for ell in range(abs(m), self.l_max + 1):
                diag[pos : pos + self.mul] = self.parity * ((-1) ** ell)
                pos += self.mul
        return np.diag(diag)[None, :, :]

    def trivial(self) -> SphericalSignal:
        return SphericalSignal(1, 0, 0, group=self.group, parity=1, axial=self.axial)

    def __repr__(self) -> str:
        parts = [f"mul={self.mul}", f"l_max={self.l_max}", f"m_max={self.m_max}"]
        if self.group == cue.O3:
            parts.append("group=O3")
            parts.append(f"parity={self.parity}")
        if self.axial and self.m_max == self.l_max:
            parts.append("axial=True")
        return f"SphericalSignal({', '.join(parts)})"

    def __hash__(self) -> int:
        return hash(
            (self.mul, self.l_max, self.m_max, self.group, self.parity, self.axial)
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SphericalSignal):
            if self.dim == 0 and other.dim == 0:
                return self.lie_dim == other.lie_dim and len(self.H) == len(other.H)
            return (
                self.mul == other.mul
                and self.l_max == other.l_max
                and self.m_max == other.m_max
                and self.group == other.group
                and self.parity == other.parity
                and self.axial == other.axial
            )
        if isinstance(other, cue.Rep):
            return (
                self.dim == other.dim
                and self.lie_dim == other.lie_dim
                and len(self.H) == len(other.H)
                and cue.Rep.__eq__(self, other)
            )
        return False


class _CompactIrrepsRep(cue.Rep):
    """M-major compact representation for arbitrary SO(3)/O(3) Irreps."""

    def __init__(self, irreps: cue.Irreps, m_max: Optional[int] = None):
        G = irreps.irrep_class
        if G not in (cue.SO3, cue.O3):
            raise ValueError(f"Unsupported irrep class {G}. Must be SO3 or O3.")
        l_max = max((ir.l for _, ir in irreps), default=0)
        m_lim = l_max if m_max is None else min(m_max, l_max)
        if m_lim < 0:
            raise ValueError(f"m_max must be non-negative, got {m_max}")

        self.irreps = irreps
        self.l_max = l_max
        self.m_max = m_lim
        self.axial = m_lim < l_max

        # Map each compact entry index -> L-major (ir_mul) index
        sigmas = (1,) if G == cue.SO3 else (1, -1)
        ir_base_offsets: list[int] = []
        cursor = 0
        for mul, ir in irreps:
            ir_base_offsets.append(cursor)
            cursor += ir.dim * mul

        perm: list[int] = []
        for m in range(-m_lim, m_lim + 1):
            for sigma in sigmas:
                for idx, (mul, ir) in enumerate(irreps):
                    if (
                        mul > 0
                        and abs(m) <= ir.l
                        and _irrep_reflection_sigma(ir) == sigma
                    ):
                        start = ir_base_offsets[idx] + (ir.l + m) * mul
                        perm.extend(range(start, start + mul))
        self._perm = np.array(perm, dtype=np.int64)

    def _dim(self) -> int:
        return int(self._perm.shape[0])

    def algebra(self) -> np.ndarray:
        if self.axial:
            return np.zeros((1, 1, 1), dtype=np.float64)
        return self.irreps.irrep_class.algebra()

    def continuous_generators(self) -> np.ndarray:
        rep_ir = cue.IrrepsAndLayout(self.irreps, cue.ir_mul)
        X_ir = rep_ir.continuous_generators()
        if self.axial:
            X_ir = X_ir[1:2]
        if self._dim() == 0:
            return np.zeros((X_ir.shape[0], 0, 0), dtype=np.float64)
        return X_ir[:, self._perm, :][:, :, self._perm]

    def discrete_generators(self) -> np.ndarray:
        rep_ir = cue.IrrepsAndLayout(self.irreps, cue.ir_mul)
        H_ir = rep_ir.discrete_generators()
        if self._dim() == 0:
            return np.zeros((H_ir.shape[0], 0, 0), dtype=np.float64)
        return H_ir[:, self._perm, :][:, :, self._perm]

    def trivial(self) -> cue.Rep:
        return SphericalSignal(
            1, 0, 0, group=self.irreps.irrep_class, parity=1, axial=self.axial
        )

    def __repr__(self) -> str:
        return f"CompactSphericalRep({self.irreps}, m_max={self.m_max})"

    def __hash__(self) -> int:
        return hash((self.irreps, self.m_max, self.axial))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CompactIrrepsRep):
            return (
                self.irreps == other.irreps
                and self.m_max == other.m_max
                and self.axial == other.axial
            )
        if isinstance(other, cue.Rep):
            return (
                self.dim == other.dim
                and self.lie_dim == other.lie_dim
                and len(self.H) == len(other.H)
                and cue.Rep.__eq__(self, other)
            )
        return False


def _compact_rep_for_irreps(irreps: cue.Irreps, m_max: Optional[int] = None) -> cue.Rep:
    G = irreps.irrep_class
    if len(irreps) > 0:
        muls = {mul for mul, _ in irreps}
        sigmas = {_irrep_reflection_sigma(ir) for _, ir in irreps}
        l_max = max(ir.l for _, ir in irreps)
        expected_ls = list(range(l_max + 1))
        actual_ls = [ir.l for _, ir in irreps]
        if len(muls) == 1 and len(sigmas) == 1 and actual_ls == expected_ls:
            (mul,) = muls
            (sigma,) = sigmas
            return SphericalSignal(
                mul=mul,
                l_max=l_max,
                m_max=m_max,
                group=G,
                parity=sigma,
            )
    return _CompactIrrepsRep(irreps, m_max=m_max)


def irreps_to_spherical_signal(
    irreps: cue.Irreps,
    layout: Optional[cue.IrrepsLayout] = None,
    m_max: Optional[int] = None,
) -> cue.EquivariantPolynomial:
    """Create the permutation descriptor from L-major :class:`cue.Irreps` to M-major compact order.

    Args:
        irreps (Irreps): Input ``SO3`` or ``O3`` irreps.
        layout (IrrepsLayout, optional): Input memory layout (defaults to active layout scope or ``cue.ir_mul``).
        m_max (int, optional): If provided, restrict the output to ``|m| <= m_max``.
            When ``None`` (default), keeps all ``m in [-l_max, l_max]`` matching :func:`escn_tp_compact`.

    Returns:
        EquivariantPolynomial: Permutation polynomial mapping ``IrrepsAndLayout(irreps, layout)``
        to :class:`SphericalSignal` (or general M-major compact representation).
    """
    G = irreps.irrep_class
    if G not in (cue.SO3, cue.O3):
        raise NotImplementedError("Only SO3 and O3 are supported")

    if layout is None:
        layout = cue.get_layout_scope(raising=False) or cue.ir_mul

    sigmas = (1,) if G == cue.SO3 else (1, -1)
    l_max = max((ir.l for _, ir in irreps), default=-1)
    m_lim = l_max if m_max is None else min(m_max, l_max)

    d = cue.SegmentedTensorProduct.from_subscripts(
        "ui,u+i" if layout == cue.mul_ir else "iu,u+i"
    )
    ir_segs: list[int] = []
    for mul, ir in irreps:
        seg_shape = (mul, ir.dim) if layout == cue.mul_ir else (ir.dim, mul)
        ir_segs.append(d.add_segment(0, seg_shape))

    if m_lim >= 0:
        for m in range(-m_lim, m_lim + 1):
            for sigma in sigmas:
                for idx, (mul, ir) in enumerate(irreps):
                    if (
                        mul > 0
                        and abs(m) <= ir.l
                        and _irrep_reflection_sigma(ir) == sigma
                    ):
                        s_out = d.add_segment(1, (mul,))
                        c = np.zeros((ir.dim,), dtype=np.float64)
                        c[ir.l + m] = 1.0
                        d.add_path(ir_segs[idx], s_out, c=c)

    if layout == cue.ir_mul:
        d = d.flatten_coefficient_modes()
    rep_out = _compact_rep_for_irreps(irreps, m_max=m_max)
    return cue.EquivariantPolynomial(
        [cue.IrrepsAndLayout(irreps, layout)],
        [rep_out],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def spherical_signal_to_irreps(
    irreps: cue.Irreps,
    layout: Optional[cue.IrrepsLayout] = None,
    m_max: Optional[int] = None,
) -> cue.EquivariantPolynomial:
    """Create the permutation descriptor from M-major compact order back to L-major :class:`cue.Irreps`.

    Args:
        irreps (Irreps): Target ``SO3`` or ``O3`` irreps.
        layout (IrrepsLayout, optional): Output memory layout (defaults to active layout scope or ``cue.ir_mul``).
        m_max (int, optional): Maximum ``|m|`` present in the compact input (defaults to ``l_max``).

    Returns:
        EquivariantPolynomial: Permutation polynomial mapping M-major compact order back to
        ``IrrepsAndLayout(irreps, layout)``.
    """
    fwd = irreps_to_spherical_signal(irreps, layout=layout, m_max=m_max)
    bwd, _ = fwd.transpose(is_undefined_primal=[True], has_cotangent=[True])
    return bwd


def escn_weights_to_compact(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    m_max: Optional[int] = None,
) -> cue.EquivariantPolynomial:
    """Map weights of :func:`escn_tp` to the compact weight layout of :func:`escn_tp_compact`.

    Accounts for the M-major block packing, the antisymmetric sine sign convention, and the
    per-output-irrep vs per-``(m, sigma)``-segment variance normalization so that::

        x_cmp = irreps_to_spherical_signal(irreps_in)(x)[0]
        w_cmp = escn_weights_to_compact(irreps_in, irreps_out, m_max)(w)[0]
        y_cmp = escn_tp_compact(irreps_in, irreps_out, m_max)(w_cmp, x_cmp)[0]
        y = spherical_signal_to_irreps(irreps_out)(y_cmp)[0]

    matches ``escn_tp(irreps_in, irreps_out, m_max)(w, x)[0]`` to machine precision for both
    ``SO3`` and ``O3``.
    """
    assert irreps_in.irrep_class == irreps_out.irrep_class
    G = irreps_in.irrep_class
    if G not in (cue.SO3, cue.O3):
        raise NotImplementedError("Only SO3 and O3 are supported")

    sigmas = (1,) if G == cue.SO3 else (1, -1)
    p_std = escn_tp(irreps_in, irreps_out, m_max=m_max).polynomial
    p_cmp = escn_tp_compact(irreps_in, irreps_out, m_max=m_max)
    stp_std = p_std.operations[0][1]
    stp_cmp = p_cmp.operations[0][1]

    W = stp_std.operands[0].size
    assert W == stp_cmp.operands[0].size

    ir2_offset = 0
    ir2_seg_ranges: list[tuple[int, int]] = []
    for _, ir2 in irreps_out:
        ir2_seg_ranges.append((ir2_offset, ir2_offset + ir2.dim))
        ir2_offset += ir2.dim

    c_std_by_i2: dict[int, float] = {}
    for path in stp_std.paths:
        out_seg = path.indices[2]
        for i2, (s0, s1) in enumerate(ir2_seg_ranges):
            if s0 <= out_seg < s1:
                c_std_by_i2[i2] = abs(float(path.coefficients))
                break

    c_cmp_by_w: dict[int, float] = {}
    for path in stp_cmp.paths:
        w_idx = path.indices[0]
        if w_idx not in c_cmp_by_w:
            c_cmp_by_w[w_idx] = abs(float(path.coefficients))

    l_max_in = max((ir.l for _, ir in irreps_in), default=-1)
    l_max_out = max((ir.l for _, ir in irreps_out), default=-1)
    m_max_eff = (
        min(l_max_in, l_max_out) if m_max is None else min(m_max, l_max_in, l_max_out)
    )

    in_offsets: dict[tuple[int, int, int], int] = {}
    in_sizes: dict[tuple[int, int], int] = {}
    for m in range(-l_max_in, l_max_in + 1):
        for sigma in sigmas:
            off = 0
            for i1, (mul1, ir1) in enumerate(irreps_in):
                if (
                    mul1 > 0
                    and abs(m) <= ir1.l
                    and _irrep_reflection_sigma(ir1) == sigma
                ):
                    in_offsets[(m, sigma, i1)] = off
                    off += mul1
            if G == cue.SO3 or off > 0:
                in_sizes[(m, sigma)] = off

    out_offsets: dict[tuple[int, int, int], int] = {}
    out_sizes: dict[tuple[int, int], int] = {}
    for m in range(-l_max_out, l_max_out + 1):
        for sigma in sigmas:
            off = 0
            for i2, (mul2, ir2) in enumerate(irreps_out):
                if (
                    mul2 > 0
                    and abs(m) <= ir2.l
                    and _irrep_reflection_sigma(ir2) == sigma
                ):
                    out_offsets[(m, sigma, i2)] = off
                    off += mul2
            if G == cue.SO3 or off > 0:
                out_sizes[(m, sigma)] = off

    cmp_blocks: dict[tuple[str, int, int], tuple[int, int, int, int]] = {}
    w_slices = stp_cmp.operands[0].segment_slices()
    w_cursor = 0
    if m_max_eff >= 0:
        for sigma in sigmas:
            if (0, sigma) in in_sizes and (0, sigma) in out_sizes:
                cmp_blocks[("cos", 0, sigma)] = (
                    w_cursor,
                    w_slices[w_cursor].start,
                    in_sizes[(0, sigma)],
                    out_sizes[(0, sigma)],
                )
                w_cursor += 1
    for m in range(1, m_max_eff + 1):
        for sigma in sigmas:
            if (-m, sigma) in in_sizes and (-m, sigma) in out_sizes:
                cmp_blocks[("cos", m, sigma)] = (
                    w_cursor,
                    w_slices[w_cursor].start,
                    in_sizes[(-m, sigma)],
                    out_sizes[(-m, sigma)],
                )
                w_cursor += 1
        for sigma_in in sigmas:
            sigma_out = 1 if G == cue.SO3 else -sigma_in
            if (m, sigma_in) in in_sizes and (-m, sigma_out) in out_sizes:
                cmp_blocks[("sin", m, sigma_in)] = (
                    w_cursor,
                    w_slices[w_cursor].start,
                    in_sizes[(m, sigma_in)],
                    out_sizes[(-m, sigma_out)],
                )
                w_cursor += 1
    assert w_cursor == stp_cmp.operands[0].num_segments

    # Each row u in (mul1, mul2) is a contiguous slice of length mul2 in both w_std and w_cmp
    # Collect (dst_start, mul2, src_seg_idx, ratio) and sort by dst_start to add operand 1 segments in order
    d = cue.SegmentedTensorProduct.from_subscripts("v,v")
    dst_entries: list[tuple[int, int, int, float]] = []

    for i1, (mul1, ir1) in enumerate(irreps_in):
        for i2, (mul2, ir2) in enumerate(irreps_out):
            if mul1 == 0 or mul2 == 0:
                continue
            ell = min(ir1.l, ir2.l)
            if m_max is not None:
                ell = min(ell, m_max)
            sigma1 = _irrep_reflection_sigma(ir1)
            sigma2 = _irrep_reflection_sigma(ir2)

            modes: list[tuple[str, int, float]] = []
            if G == cue.SO3 or sigma1 == sigma2:
                for k in range(ell + 1):
                    modes.append(("cos", ell - k, 1.0))
            if G == cue.SO3 or (sigma1 != sigma2 and ell > 0):
                for m in range(1, ell + 1):
                    modes.append(("sin", m, -1.0))

            for kind, m, sgn in modes:
                w_cmp_idx, base, _u_tot, v_tot = cmp_blocks[(kind, m, sigma1)]
                m_in = m if kind == "sin" else (-m if m > 0 else 0)
                m_out = -m if kind == "sin" else (-m if m > 0 else 0)
                u0 = in_offsets[(m_in, sigma1, i1)]
                v0 = out_offsets[(m_out, sigma2, i2)]
                c_cmp = c_cmp_by_w.get(w_cmp_idx, 0.0)
                ratio = sgn * (c_std_by_i2.get(i2, 0.0) / c_cmp) if c_cmp > 0.0 else 0.0
                for u in range(mul1):
                    src_seg = d.add_segment(0, (mul2,))
                    dst_start = base + (u0 + u) * v_tot + v0
                    dst_entries.append((dst_start, mul2, src_seg, ratio))

    dst_entries.sort(key=lambda item: item[0])
    expected_pos = 0
    for dst_start, mul2, src_seg, ratio in dst_entries:
        assert dst_start == expected_pos
        expected_pos += mul2
        dst_seg = d.add_segment(1, (mul2,))
        d.add_path(src_seg, dst_seg, c=ratio)
    assert expected_pos == W

    scalars_w = cue.IrrepsAndLayout(irreps_in.new_scalars(W), cue.ir_mul)
    return cue.EquivariantPolynomial(
        [scalars_w],
        [scalars_w],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )
