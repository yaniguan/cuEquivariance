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

import dataclasses
import functools
from typing import Optional, Sequence

import numpy as np

import cuequivariance as cue
from cuequivariance.group_theory.irreps_array.irrep_utils import into_list_of_irrep
from cuequivariance.group_theory.irreps_array.reduced_tensor_product import (
    germinate_perm_repr,
)

_SYMMETRIC_INDEX_LETTERS = "ijklmnopqrstuvwxyzabcdefgh"


def _resolve_group_and_vector_irrep(
    group: Optional[type[cue.Irrep] | str],
    vector_parity: int,
    vector_irrep: Optional[cue.Irrep] = None,
) -> tuple[type[cue.Irrep], cue.Irrep]:
    if vector_parity not in (-1, 1):
        raise ValueError(f"vector_parity must be -1 or 1, got {vector_parity!r}.")

    if isinstance(group, str):
        if group == "O3":
            group = cue.O3
        elif group == "SO3":
            group = cue.SO3
        else:
            raise ValueError(
                f"Unsupported group {group!r}; expected cue.O3 or cue.SO3."
            )

    if vector_irrep is not None:
        if not isinstance(vector_irrep, (cue.O3, cue.SO3)) or vector_irrep.dim != 3:
            raise ValueError(
                f"vector_irrep must be a 3D O3 or SO3 vector irrep, got {vector_irrep!r}."
            )
        if group is not None and group is not type(vector_irrep):
            raise ValueError(
                f"group {group} does not match vector_irrep {vector_irrep!r} "
                f"of group {type(vector_irrep)}."
            )
        group = type(vector_irrep)
        return group, vector_irrep

    if group is None:
        group = cue.O3

    if group is cue.O3:
        return cue.O3, cue.O3(1, vector_parity)
    if group is cue.SO3:
        return cue.SO3, cue.SO3(1)

    raise ValueError(f"Unsupported group {group!r}; expected cue.O3 or cue.SO3.")


def _resolve_layout(layout: Optional[cue.IrrepsLayout | str]) -> cue.IrrepsLayout:
    if layout is None:
        layout = cue.get_layout_scope()
    elif isinstance(layout, str):
        layout = cue.IrrepsLayout[layout]
    if not isinstance(layout, cue.IrrepsLayout):
        raise TypeError(f"Expected cue.IrrepsLayout, got {type(layout)}.")
    return layout


def _validate_multiplicity(multiplicity: int) -> int:
    if (
        not isinstance(multiplicity, int)
        or isinstance(multiplicity, bool)
        or multiplicity < 1
    ):
        raise ValueError(f"multiplicity must be an integer >= 1, got {multiplicity!r}.")
    return multiplicity


def _validate_degree(degree: int) -> int:
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
        raise ValueError(f"degree must be an integer >= 0, got {degree!r}.")
    return degree


def symmetric_cartesian_formula(degree: int) -> str:
    """Return the index-symmetry formula string for a totally symmetric rank-``degree`` tensor."""
    degree = _validate_degree(degree)
    if degree > len(_SYMMETRIC_INDEX_LETTERS):
        raise ValueError(
            f"Symmetric formula string supports degree <= {len(_SYMMETRIC_INDEX_LETTERS)}, "
            f"got {degree}."
        )
    if degree == 0:
        return ""
    if degree == 1:
        return _SYMMETRIC_INDEX_LETTERS[0]
    base = _SYMMETRIC_INDEX_LETTERS[:degree]
    transpositions = []
    for r in range(degree - 1):
        chars = list(base)
        chars[r], chars[r + 1] = chars[r + 1], chars[r]
        transpositions.append("".join(chars))
    return "=".join([base] + transpositions)


@dataclasses.dataclass(init=False, frozen=True)
class CartesianTensorRep(cue.Rep):
    r"""Group representation of a rank-:math:`k` 3D Cartesian tensor space :math:`(\mathbb{R}^3)^{\otimes k}`.

    Supports an optional channel ``multiplicity`` and memory ``layout``:

    - For ``cue.ir_mul``, elements are ordered as ``(3**degree, multiplicity)``.
    - For ``cue.mul_ir``, elements are ordered as ``(multiplicity, 3**degree)``.

    Args:
        degree (int): Tensor rank :math:`k \ge 0`.
        multiplicity (int, optional): Number of channels. Defaults to ``1``.
        layout (IrrepsLayout, optional): Memory layout (``cue.ir_mul`` or ``cue.mul_ir``).
            Defaults to ``cue.ir_mul``.
        group (type[Irrep], optional): Symmetry group, ``cue.O3`` or ``cue.SO3``.
            Defaults to ``cue.O3``.
        vector_parity (int, optional): Parity of the base 3D vector under inversion for ``O3``
            (``-1`` for polar vectors ``1o``, ``+1`` for axial vectors ``1e``). Defaults to ``-1``.
        vector_irrep (Irrep, optional): Explicit 3D vector irrep (overrides ``group``/``vector_parity``).
    """

    degree: int
    group: type[cue.Irrep]
    vector_irrep: cue.Irrep
    multiplicity: int
    layout: cue.IrrepsLayout

    def __init__(
        self,
        degree: int,
        multiplicity: int = 1,
        layout: Optional[cue.IrrepsLayout | str] = cue.ir_mul,
        *,
        group: Optional[type[cue.Irrep] | str] = None,
        vector_parity: int = -1,
        vector_irrep: Optional[cue.Irrep] = None,
    ):
        degree = _validate_degree(degree)
        multiplicity = _validate_multiplicity(multiplicity)
        resolved_group, resolved_vec = _resolve_group_and_vector_irrep(
            group, vector_parity, vector_irrep
        )
        resolved_layout = _resolve_layout(layout)

        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "group", resolved_group)
        object.__setattr__(self, "vector_irrep", resolved_vec)
        object.__setattr__(self, "multiplicity", multiplicity)
        object.__setattr__(self, "layout", resolved_layout)

    @property
    def vector_parity(self) -> int:
        return self.vector_irrep.p if isinstance(self.vector_irrep, cue.O3) else 1

    def _dim(self) -> int:
        return self.multiplicity * (3**self.degree)

    def algebra(self) -> np.ndarray:
        return self.group.algebra()

    def continuous_generators(self) -> np.ndarray:
        X_vec = self.vector_irrep.continuous_generators()
        lie_dim = X_vec.shape[0]
        k = self.degree

        if k == 0:
            X_cart = np.zeros((lie_dim, 1, 1), dtype=np.float64)
        else:
            cart_dim = 3**k
            X_cart = np.zeros((lie_dim, cart_dim, cart_dim), dtype=np.float64)
            for r in range(k):
                left = np.eye(3**r, dtype=np.float64)[None, :, :]
                right = np.eye(3 ** (k - 1 - r), dtype=np.float64)[None, :, :]
                X_cart = X_cart + np.kron(np.kron(left, X_vec), right)

        if self.multiplicity == 1:
            return X_cart

        I_mul = np.eye(self.multiplicity, dtype=np.float64)[None, :, :]
        if self.layout == cue.ir_mul:
            return np.kron(X_cart, I_mul)
        return np.kron(I_mul, X_cart)

    def discrete_generators(self) -> np.ndarray:
        H_vec = self.vector_irrep.discrete_generators()
        num_H = H_vec.shape[0]
        if num_H == 0:
            return np.zeros((0, self.dim, self.dim), dtype=np.float64)

        k = self.degree
        if k == 0:
            H_cart = np.ones((num_H, 1, 1), dtype=np.float64)
        else:
            H_list = []
            for n in range(num_H):
                h = H_vec[n]
                h_k = functools.reduce(np.kron, [h] * k)
                H_list.append(h_k)
            H_cart = np.stack(H_list, axis=0)

        if self.multiplicity == 1:
            return H_cart

        I_mul = np.eye(self.multiplicity, dtype=np.float64)[None, :, :]
        if self.layout == cue.ir_mul:
            return np.kron(H_cart, I_mul)
        return np.kron(I_mul, H_cart)

    def trivial(self) -> cue.Rep:
        return CartesianTensorRep(
            0,
            multiplicity=1,
            layout=self.layout,
            group=self.group,
            vector_irrep=self.vector_irrep,
        )

    def is_scalar(self) -> bool:
        return self.degree == 0

    def __repr__(self) -> str:
        return (
            f"CartesianTensorRep(degree={self.degree}, vector_irrep={self.vector_irrep}, "
            f"multiplicity={self.multiplicity}, layout={self.layout})"
        )

    def _canonical_key(self) -> tuple:
        effective_vec = self.group.trivial() if self.degree == 0 else self.vector_irrep
        effective_layout = (
            cue.ir_mul if (self.degree == 0 or self.multiplicity == 1) else self.layout
        )
        return (
            CartesianTensorRep,
            self.degree,
            self.group,
            effective_vec,
            self.multiplicity,
            effective_layout,
        )

    def __hash__(self) -> int:
        return hash(self._canonical_key())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CartesianTensorRep):
            return self._canonical_key() == other._canonical_key()
        if isinstance(other, cue.Rep):
            return cue.Rep.__eq__(self, other)
        return False


def _parse_cartesian_formula(formula: str) -> tuple[int, str, bool]:
    if not isinstance(formula, str):
        raise TypeError(f"formula must be a string, got {type(formula)}.")
    formula = formula.strip()
    if formula == "":
        return 0, "", False
    if formula == "-":
        raise ValueError("Invalid formula '-'.")
    f0, perm_repr = germinate_perm_repr(formula)
    degree = len(f0)
    is_zero_antisym = degree == 0 and (-1, ()) in perm_repr
    return degree, f0, is_zero_antisym


def _cartesian_basis(
    formula: str,
    group: type[cue.Irrep],
    vec_ir: cue.Irrep,
    traceless: bool,
    keep_ir: Optional[Sequence[cue.Irrep] | cue.Irreps],
    epsilon: float,
) -> tuple[int, cue.NumpyIrrepsArray]:
    degree, f0, is_zero_antisym = _parse_cartesian_formula(formula)

    user_keep: Optional[frozenset[cue.Irrep]] = None
    if keep_ir is not None:
        user_keep = frozenset(into_list_of_irrep(group, keep_ir))

    if traceless:
        if group is cue.O3:
            harmonic_ir: cue.Irrep = cue.O3(degree, (vec_ir.p) ** degree)
        else:
            harmonic_ir = cue.SO3(degree)
        traceless_keep = frozenset([harmonic_ir])
        effective_keep = (
            traceless_keep if user_keep is None else (user_keep & traceless_keep)
        )
    else:
        effective_keep = user_keep

    if degree == 0:
        trivial_ir = group.trivial()
        if (not is_zero_antisym) and (
            effective_keep is None or trivial_ir in effective_keep
        ):
            single_irreps = cue.Irreps(group, [(1, trivial_ir)])
            basis = cue.NumpyIrrepsArray(
                single_irreps, np.ones((1,), dtype=np.float64), cue.ir_mul
            )
        else:
            single_irreps = cue.Irreps(group, [])
            basis = cue.NumpyIrrepsArray(
                single_irreps, np.zeros((0,), dtype=np.float64), cue.ir_mul
            )
        return 0, basis

    if effective_keep is not None and len(effective_keep) == 0:
        single_irreps = cue.Irreps(group, [])
        basis = cue.NumpyIrrepsArray(
            single_irreps,
            np.zeros((3,) * degree + (0,), dtype=np.float64),
            cue.ir_mul,
        )
        return degree, basis

    vec_irreps = cue.Irreps(group, [vec_ir])
    basis = cue.reduced_tensor_product_basis(
        formula,
        layout=cue.ir_mul,
        epsilon=epsilon,
        keep_ir=list(effective_keep) if effective_keep is not None else None,
        **{idx: vec_irreps for idx in f0},
    )
    basis = _orthonormalize_basis_segments(basis, degree)
    return degree, basis


def _orthonormalize_basis_segments(
    basis: cue.NumpyIrrepsArray, degree: int
) -> cue.NumpyIrrepsArray:
    """Ensure multiplicity channels within each irrep segment are strictly orthonormal.

    ``cue.reduced_tensor_product_basis`` applies ``_sparsify`` across multiplicity channels
    when ``m_s > 1``, which preserves equivariance and linear independence but need not
    preserve unit norm / orthogonality between multiplicity copies of the same irrep.
    Transforming only the multiplicity axis via reduced QR restores ``Q^T Q = I`` while
    commuting strictly with the group action ``D_ir(g) (x) I_{m_s}``.
    """
    cart_dim = 3**degree
    new_segments: list[np.ndarray] = []
    modified = False

    for (m_s, ir), seg in zip(basis.irreps, basis.segments):
        seg_3d = seg.reshape(cart_dim, ir.dim, m_s)
        mat = seg_3d.reshape(cart_dim * ir.dim, m_s) / np.sqrt(ir.dim)
        gram = mat.T @ mat
        if np.allclose(gram, np.eye(m_s), atol=1e-14, rtol=1e-14):
            new_segments.append(seg)
            continue

        _, r_factor = np.linalg.qr(mat, mode="reduced")
        signs = np.where(np.diag(r_factor) < 0.0, -1.0, 1.0)
        r_factor = signs[:, None] * r_factor
        seg_orth = np.linalg.solve(
            r_factor.T, seg_3d.reshape(cart_dim * ir.dim, m_s).T
        ).T.reshape(seg.shape)
        new_segments.append(seg_orth)
        modified = True

    if not modified:
        return basis
    return cue.from_segments(basis.irreps, new_segments, cue.ir_mul, (3,) * degree)


def _build_cartesian_irreps_stp(
    degree: int,
    basis: cue.NumpyIrrepsArray,
    multiplicity: int,
    layout: cue.IrrepsLayout,
    direction: str,
) -> tuple[cue.SegmentedTensorProduct, cue.Irreps]:
    cart_dim = 3**degree
    single_irreps = basis.irreps
    irreps_full = multiplicity * single_irreps

    if layout == cue.ir_mul:
        Q = basis.array.reshape(cart_dim, single_irreps.dim)
        d = cue.SegmentedTensorProduct.from_subscripts("iu,ju+ij")
        if direction == "cart_to_ir":
            d.add_segment(0, (cart_dim, multiplicity))
            if single_irreps.dim > 0:
                d.add_segment(1, (single_irreps.dim, multiplicity))
                d.add_path(0, 0, c=Q, dims={"u": multiplicity})
        else:
            if single_irreps.dim > 0:
                d.add_segment(0, (single_irreps.dim, multiplicity))
            d.add_segment(1, (cart_dim, multiplicity))
            if single_irreps.dim > 0:
                d.add_path(0, 0, c=Q.T, dims={"u": multiplicity})
        d = d.flatten_coefficient_modes()
        return d, irreps_full

    # layout == cue.mul_ir
    if multiplicity == 1:
        if single_irreps.dim > 0:
            segs_mul_ir = [
                np.swapaxes(seg.reshape(cart_dim, ir.dim, m_s), -2, -1).reshape(
                    cart_dim, m_s * ir.dim
                )
                for (m_s, ir), seg in zip(single_irreps, basis.segments)
            ]
            Q_mul_ir = np.concatenate(segs_mul_ir, axis=1)
        else:
            Q_mul_ir = np.zeros((cart_dim, 0), dtype=np.float64)

        d = cue.SegmentedTensorProduct.from_subscripts("iu,ju+ij")
        if direction == "cart_to_ir":
            d.add_segment(0, (cart_dim, 1))
            if single_irreps.dim > 0:
                d.add_segment(1, (single_irreps.dim, 1))
                d.add_path(0, 0, c=Q_mul_ir, dims={"u": 1})
        else:
            if single_irreps.dim > 0:
                d.add_segment(0, (single_irreps.dim, 1))
            d.add_segment(1, (cart_dim, 1))
            if single_irreps.dim > 0:
                d.add_path(0, 0, c=Q_mul_ir.T, dims={"u": 1})
        d = d.flatten_coefficient_modes()
        return d, irreps_full

    # layout == cue.mul_ir and multiplicity > 1
    if direction == "cart_to_ir":
        d = cue.SegmentedTensorProduct.from_subscripts("ui,auj+iaj")
        d.add_segment(0, (multiplicity, cart_dim))
        for (m_s, ir), seg in zip(single_irreps, basis.segments):
            q_s = np.swapaxes(seg.reshape(cart_dim, ir.dim, m_s), -2, -1)
            sid = d.add_segment(1, (m_s, multiplicity, ir.dim))
            d.add_path(0, sid, c=q_s)
        d = d.flatten_coefficient_modes(force=True)
        return d, irreps_full

    d = cue.SegmentedTensorProduct.from_subscripts("auj,ui+iaj")
    d.add_segment(1, (multiplicity, cart_dim))
    for (m_s, ir), seg in zip(single_irreps, basis.segments):
        q_s = np.swapaxes(seg.reshape(cart_dim, ir.dim, m_s), -2, -1)
        sid = d.add_segment(0, (m_s, multiplicity, ir.dim))
        d.add_path(sid, 0, c=q_s)
    d = d.flatten_coefficient_modes(force=True)
    return d, irreps_full


def cartesian_to_irreps(
    formula: str,
    *,
    group: type[cue.Irrep] | str = cue.O3,
    vector_parity: int = -1,
    traceless: bool = False,
    keep_ir: Optional[Sequence[cue.Irrep] | cue.Irreps] = None,
    multiplicity: int = 1,
    layout: cue.IrrepsLayout | str = cue.ir_mul,
    epsilon: float = 1e-5,
) -> cue.EquivariantPolynomial:
    r"""Construct an orthogonal change-of-basis descriptor from Cartesian tensors to spherical irreps.

    Maps a rank-:math:`k` Cartesian tensor :math:`T \in (\mathbb{R}^3)^{\otimes k}` to its
    spherical harmonic irreducible components :math:`c = Q^T T`, where :math:`Q` has
    orthonormal columns (:math:`Q^T Q = I`).

    Args:
        formula (str): Index-symmetry formula (e.g. ``""`` for monopole, ``"i"`` for dipole,
            ``"ij=ji"`` for symmetric rank-2 polarizability/stress, ``"ij=-ji"`` for antisymmetric
            rank-2 tensor, ``"ijk=jik=ikj"`` for symmetric rank-3 octupole).
        group (type[Irrep], optional): ``cue.O3`` (default) or ``cue.SO3``.
        vector_parity (int, optional): Parity of a 3D Cartesian vector under inversion
            (``-1`` for polar ``1o``, ``+1`` for axial ``1e``). Defaults to ``-1``.
        traceless (bool, optional): If ``True``, restricts to the highest-weight harmonic
            subspace :math:`\ell = k` (symmetric traceless multipoles). Defaults to ``False``.
        keep_ir (sequence of Irrep or Irreps, optional): Subset of output irreps to keep.
        multiplicity (int, optional): Channel multiplicity. Defaults to ``1``.
        layout (IrrepsLayout, optional): Memory layout (``cue.ir_mul`` or ``cue.mul_ir``).
            Defaults to ``cue.ir_mul``.
        epsilon (float, optional): Gram-Schmidt tolerance for basis construction. Defaults to ``1e-5``.

    Returns:
        :class:`~cuequivariance.EquivariantPolynomial`: Orthogonal Cartesian-to-spherical descriptor.
    """
    multiplicity = _validate_multiplicity(multiplicity)
    resolved_group, vec_ir = _resolve_group_and_vector_irrep(group, vector_parity)
    resolved_layout = _resolve_layout(layout)
    degree, basis = _cartesian_basis(
        formula, resolved_group, vec_ir, traceless, keep_ir, epsilon
    )
    d, irreps_out = _build_cartesian_irreps_stp(
        degree, basis, multiplicity, resolved_layout, direction="cart_to_ir"
    )
    cart_rep = CartesianTensorRep(
        degree,
        multiplicity=multiplicity,
        layout=resolved_layout,
        group=resolved_group,
        vector_irrep=vec_ir,
    )
    return cue.EquivariantPolynomial(
        [cart_rep],
        [cue.IrrepsAndLayout(irreps_out, resolved_layout)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def irreps_to_cartesian(
    formula: str,
    *,
    group: type[cue.Irrep] | str = cue.O3,
    vector_parity: int = -1,
    traceless: bool = False,
    keep_ir: Optional[Sequence[cue.Irrep] | cue.Irreps] = None,
    multiplicity: int = 1,
    layout: cue.IrrepsLayout | str = cue.ir_mul,
    epsilon: float = 1e-5,
) -> cue.EquivariantPolynomial:
    r"""Construct an orthogonal synthesis descriptor from spherical irreps to Cartesian tensors.

    Maps spherical harmonic irreducible coefficients :math:`c` to a rank-:math:`k` Cartesian
    tensor :math:`T = Q c`, the exact transpose of :func:`cartesian_to_irreps`. Because
    :math:`Q^T Q = I`, ``cartesian_to_irreps(irreps_to_cartesian(c)) == c``, and
    ``irreps_to_cartesian(cartesian_to_irreps(T))`` orthogonally projects :math:`T` onto
    the target permutation-symmetry (and optional traceless) subspace.

    Args:
        formula (str): Index-symmetry formula (e.g. ``""``, ``"i"``, ``"ij=ji"``, ``"ijk=jik=ikj"``).
        group (type[Irrep], optional): ``cue.O3`` (default) or ``cue.SO3``.
        vector_parity (int, optional): Parity of a 3D Cartesian vector under inversion
            (``-1`` for polar ``1o``, ``+1`` for axial ``1e``). Defaults to ``-1``.
        traceless (bool, optional): If ``True``, restricts to the highest-weight harmonic
            subspace :math:`\ell = k` (symmetric traceless multipoles). Defaults to ``False``.
        keep_ir (sequence of Irrep or Irreps, optional): Subset of input irreps to include.
        multiplicity (int, optional): Channel multiplicity. Defaults to ``1``.
        layout (IrrepsLayout, optional): Memory layout (``cue.ir_mul`` or ``cue.mul_ir``).
            Defaults to ``cue.ir_mul``.
        epsilon (float, optional): Gram-Schmidt tolerance for basis construction. Defaults to ``1e-5``.

    Returns:
        :class:`~cuequivariance.EquivariantPolynomial`: Orthogonal spherical-to-Cartesian descriptor.
    """
    multiplicity = _validate_multiplicity(multiplicity)
    resolved_group, vec_ir = _resolve_group_and_vector_irrep(group, vector_parity)
    resolved_layout = _resolve_layout(layout)
    degree, basis = _cartesian_basis(
        formula, resolved_group, vec_ir, traceless, keep_ir, epsilon
    )
    d, irreps_in = _build_cartesian_irreps_stp(
        degree, basis, multiplicity, resolved_layout, direction="ir_to_cart"
    )
    cart_rep = CartesianTensorRep(
        degree,
        multiplicity=multiplicity,
        layout=resolved_layout,
        group=resolved_group,
        vector_irrep=vec_ir,
    )
    return cue.EquivariantPolynomial(
        [cue.IrrepsAndLayout(irreps_in, resolved_layout)],
        [cart_rep],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def symmetric_traceless_multipole_to_irreps(
    degree: int,
    *,
    group: type[cue.Irrep] | str = cue.O3,
    vector_parity: int = -1,
    multiplicity: int = 1,
    layout: cue.IrrepsLayout | str = cue.ir_mul,
    epsilon: float = 1e-5,
) -> cue.EquivariantPolynomial:
    """Construct the descriptor mapping a rank-``degree`` Cartesian multipole tensor to its spherical harmonic irrep."""
    formula = symmetric_cartesian_formula(degree)
    return cartesian_to_irreps(
        formula,
        group=group,
        vector_parity=vector_parity,
        traceless=True,
        multiplicity=multiplicity,
        layout=layout,
        epsilon=epsilon,
    )


def irreps_to_symmetric_traceless_multipole(
    degree: int,
    *,
    group: type[cue.Irrep] | str = cue.O3,
    vector_parity: int = -1,
    multiplicity: int = 1,
    layout: cue.IrrepsLayout | str = cue.ir_mul,
    epsilon: float = 1e-5,
) -> cue.EquivariantPolynomial:
    """Construct the descriptor mapping a spherical harmonic irrep to a symmetric traceless rank-``degree`` Cartesian multipole tensor."""
    formula = symmetric_cartesian_formula(degree)
    return irreps_to_cartesian(
        formula,
        group=group,
        vector_parity=vector_parity,
        traceless=True,
        multiplicity=multiplicity,
        layout=layout,
        epsilon=epsilon,
    )


def _cartesian_operand_irreps_metadata(
    degree: int,
    multiplicity: int,
    group: type[cue.Irrep],
    vec_ir: cue.Irrep,
) -> cue.Irreps:
    if degree == 1:
        return cue.Irreps(group, [(multiplicity, vec_ir)])
    cart_size = multiplicity * (3**degree)
    return cue.Irreps(group, [(cart_size, group.trivial())])


def cartesian_to_irreps_ir_dict(
    formula: str,
    *,
    group: type[cue.Irrep] | str = cue.O3,
    vector_parity: int = -1,
    traceless: bool = False,
    keep_ir: Optional[Sequence[cue.Irrep] | cue.Irreps] = None,
    multiplicity: int = 1,
    layout: cue.IrrepsLayout | str = cue.ir_mul,
    epsilon: float = 1e-5,
) -> cue.IrDictPolynomial:
    """``ir_dict`` variant of :func:`cartesian_to_irreps` returning an :class:`~cuequivariance.IrDictPolynomial`."""
    resolved_group, vec_ir = _resolve_group_and_vector_irrep(group, vector_parity)
    e = cartesian_to_irreps(
        formula,
        group=resolved_group,
        vector_parity=vector_parity,
        traceless=traceless,
        keep_ir=keep_ir,
        multiplicity=multiplicity,
        layout=layout,
        epsilon=epsilon,
    )
    cart_rep: CartesianTensorRep = e.inputs[0]  # type: ignore[assignment]
    irreps_out: cue.Irreps = e.outputs[0].irreps  # type: ignore[attr-defined]
    cart_irreps = _cartesian_operand_irreps_metadata(
        cart_rep.degree, cart_rep.multiplicity, resolved_group, vec_ir
    )
    poly = cue.split_polynomial_by_irreps(e.polynomial, -1, irreps_out)
    return cue.IrDictPolynomial(
        polynomial=poly,
        input_irreps=(cart_irreps,),
        output_irreps=(irreps_out,),
    )


def irreps_to_cartesian_ir_dict(
    formula: str,
    *,
    group: type[cue.Irrep] | str = cue.O3,
    vector_parity: int = -1,
    traceless: bool = False,
    keep_ir: Optional[Sequence[cue.Irrep] | cue.Irreps] = None,
    multiplicity: int = 1,
    layout: cue.IrrepsLayout | str = cue.ir_mul,
    epsilon: float = 1e-5,
) -> cue.IrDictPolynomial:
    """``ir_dict`` variant of :func:`irreps_to_cartesian` returning an :class:`~cuequivariance.IrDictPolynomial`."""
    resolved_group, vec_ir = _resolve_group_and_vector_irrep(group, vector_parity)
    e = irreps_to_cartesian(
        formula,
        group=resolved_group,
        vector_parity=vector_parity,
        traceless=traceless,
        keep_ir=keep_ir,
        multiplicity=multiplicity,
        layout=layout,
        epsilon=epsilon,
    )
    irreps_in: cue.Irreps = e.inputs[0].irreps  # type: ignore[attr-defined]
    cart_rep: CartesianTensorRep = e.outputs[0]  # type: ignore[assignment]
    cart_irreps = _cartesian_operand_irreps_metadata(
        cart_rep.degree, cart_rep.multiplicity, resolved_group, vec_ir
    )
    poly = cue.split_polynomial_by_irreps(e.polynomial, 0, irreps_in)
    return cue.IrDictPolynomial(
        polynomial=poly,
        input_irreps=(irreps_in,),
        output_irreps=(cart_irreps,),
    )
