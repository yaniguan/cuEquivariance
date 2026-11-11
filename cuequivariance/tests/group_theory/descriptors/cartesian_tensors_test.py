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
import functools
import itertools

import cuequivariance as cue
import numpy as np
import pytest
from cuequivariance.group_theory.descriptors import (
    CartesianTensorRep,
    cartesian_to_irreps,
    cartesian_to_irreps_ir_dict,
    irreps_to_cartesian,
    irreps_to_cartesian_ir_dict,
    irreps_to_symmetric_traceless_multipole,
    symmetric_cartesian_formula,
    symmetric_traceless_multipole_to_irreps,
)


def _symmetrize_tensor(T: np.ndarray) -> np.ndarray:
    k = T.ndim
    if k <= 1:
        return T.copy()
    perms = list(itertools.permutations(range(k)))
    return sum(np.transpose(T, p) for p in perms) / len(perms)


def _analytic_traceless_quadrupole(T: np.ndarray) -> np.ndarray:
    S = 0.5 * (T + T.T)
    return S - (np.trace(S) / 3.0) * np.eye(3)


def _analytic_traceless_octupole(T: np.ndarray) -> np.ndarray:
    S = _symmetrize_tensor(T)
    tr = np.einsum("aak->k", S)
    eye = np.eye(3)
    detraced = (
        S
        - (
            np.einsum("ij,k->ijk", eye, tr)
            + np.einsum("jk,i->ijk", eye, tr)
            + np.einsum("ki,j->ijk", eye, tr)
        )
        / 5.0
    )
    return detraced


def _analytic_traceless_hexadecapole(T: np.ndarray) -> np.ndarray:
    S = _symmetrize_tensor(T)
    tr2 = np.einsum("aakl->kl", S)
    tr4 = np.einsum("aa->", tr2)
    eye = np.eye(3)
    pair_terms = (
        np.einsum("ij,kl->ijkl", eye, tr2)
        + np.einsum("ik,jl->ijkl", eye, tr2)
        + np.einsum("il,jk->ijkl", eye, tr2)
        + np.einsum("jk,il->ijkl", eye, tr2)
        + np.einsum("jl,ik->ijkl", eye, tr2)
        + np.einsum("kl,ij->ijkl", eye, tr2)
    )
    double_trace_terms = (
        np.einsum("ij,kl->ijkl", eye, eye)
        + np.einsum("ik,jl->ijkl", eye, eye)
        + np.einsum("il,jk->ijkl", eye, eye)
    ) * tr4
    return S - pair_terms / 7.0 + double_trace_terms / 35.0


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("multiplicity", [1, 3])
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize(
    "group, vector_parity",
    [(cue.O3, -1), (cue.O3, 1), (cue.SO3, 1)],
)
def test_cartesian_tensor_rep_lie_algebra_and_generators(
    degree: int,
    multiplicity: int,
    layout: cue.IrrepsLayout,
    group: type[cue.Irrep],
    vector_parity: int,
):
    rep = CartesianTensorRep(
        degree,
        multiplicity=multiplicity,
        layout=layout,
        group=group,
        vector_parity=vector_parity,
    )
    assert rep.dim == multiplicity * (3**degree)
    assert rep.dim == cue.Rep.dim.fget(rep)

    A = rep.A
    X = rep.X
    H = rep.H
    assert A.shape == (3, 3, 3)
    assert X.shape == (3, rep.dim, rep.dim)

    # Skew-symmetry of real orthogonal generators
    np.testing.assert_allclose(X, -np.swapaxes(X, -1, -2), atol=1e-14)

    # Lie bracket: [X_i, X_j] = A_{ijk} X_k
    lhs = np.einsum("ils,jsm->ijlm", X, X) - np.einsum("jls,ism->ijlm", X, X)
    rhs = np.einsum("ijk,klm->ijlm", A, X)
    np.testing.assert_allclose(lhs, rhs, atol=1e-13, rtol=1e-13)

    # Discrete generators
    if group is cue.O3:
        assert H.shape == (1, rep.dim, rep.dim)
        np.testing.assert_allclose(H[0] @ H[0], np.eye(rep.dim), atol=1e-14)
        expected_parity = 1 if degree == 0 else (vector_parity**degree)
        np.testing.assert_allclose(H[0], expected_parity * np.eye(rep.dim), atol=1e-14)
        # Inversion commutes with rotations: H X_i = X_i H
        np.testing.assert_allclose(
            np.einsum("ls,ism->ilm", H[0], X),
            np.einsum("ils,sm->ilm", X, H[0]),
            atol=1e-14,
        )
    else:
        assert H.shape == (0, rep.dim, rep.dim)

    # Scalar & trivial checks for degree == 0 vs degree >= 1
    assert rep.is_scalar() == (degree == 0)
    assert rep.is_scalar() == cue.Rep.is_scalar(rep)
    assert rep.is_trivial() == (degree == 0 and multiplicity == 1)
    assert rep.trivial().is_trivial()

    # Exponential map matches explicit Kronecker power of 3x3 O(3)/SO(3) matrix
    rng = np.random.default_rng(42 + degree + 10 * multiplicity)
    rot_params = rng.normal(size=(3,))
    disc_params = np.array([1]) if group is cue.O3 else np.array([], dtype=int)

    D_rep = rep.exp_map(rot_params, disc_params)
    R_vec = rep.vector_irrep.exp_map(rot_params, disc_params)
    if degree == 0:
        R_cart = np.ones((1, 1))
    else:
        R_cart = functools.reduce(np.kron, [R_vec] * degree)
    I_mul = np.eye(multiplicity)
    expected_D = (
        np.kron(R_cart, I_mul) if layout == cue.ir_mul else np.kron(I_mul, R_cart)
    )
    np.testing.assert_allclose(D_rep, expected_D, atol=1e-13, rtol=1e-13)


def test_cartesian_tensor_rep_hash_eq_and_validation():
    r1 = CartesianTensorRep(2, multiplicity=1, layout=cue.ir_mul)
    r2 = CartesianTensorRep(2, multiplicity=1, layout=cue.mul_ir)
    assert r1 == r2
    assert hash(r1) == hash(r2)

    r3 = CartesianTensorRep(2, multiplicity=2, layout=cue.ir_mul)
    r4 = CartesianTensorRep(2, multiplicity=2, layout=cue.mul_ir)
    assert r3 != r4
    assert "CartesianTensorRep" in repr(r3)

    # Degree 0 is independent of vector_parity
    r0_odd = CartesianTensorRep(0, multiplicity=2, group=cue.O3, vector_parity=-1)
    r0_even = CartesianTensorRep(0, multiplicity=2, group=cue.O3, vector_parity=1)
    assert r0_odd == r0_even
    assert hash(r0_odd) == hash(r0_even)

    # Rank-1 polar vector CartesianTensorRep equals IrrepsAndLayout("1o") as a Rep
    assert CartesianTensorRep(1, group=cue.O3, vector_parity=-1) == cue.IrrepsAndLayout(
        cue.Irreps("O3", "1o"), cue.ir_mul
    )

    with pytest.raises(ValueError, match="degree"):
        CartesianTensorRep(-1)
    with pytest.raises(ValueError, match="multiplicity"):
        CartesianTensorRep(1, multiplicity=0)
    with pytest.raises(ValueError, match="vector_parity"):
        CartesianTensorRep(1, vector_parity=0)
    with pytest.raises(ValueError, match="Unsupported group"):
        CartesianTensorRep(1, group=cue.SU2)
    with pytest.raises(ValueError, match="does not match"):
        CartesianTensorRep(1, group=cue.SO3, vector_irrep=cue.O3(1, -1))


PHYSICAL_CASES = [
    ("", False, "0e"),  # monopole
    ("i", False, "1o"),  # polar dipole
    ("ij=-ji", False, "1e"),  # magnetic dipole / angular momentum (axial)
    ("ij=ji", False, "0e + 2e"),  # polarizability / virial stress
    ("ij=ji", True, "2e"),  # traceless quadrupole Theta_ij
    ("ijk=jik=ikj", False, "1o + 3o"),  # symmetric rank-3
    ("ijk=jik=ikj", True, "3o"),  # traceless octupole Omega_ijk
    ("ijkl=jikl=ikjl=ijlk", True, "4e"),  # traceless hexadecapole
    ("ij", False, "0e + 1e + 2e"),  # general rank-2
    ("ijk", False, "0o + 3x1o + 2x2o + 3o"),  # general rank-3 (m_s > 1)
    ("ijkl=klij=jikl", False, "2x0e + 2x2e + 4e"),  # elasticity tensor (21 comp)
]


@pytest.mark.parametrize("formula, traceless, expected_irreps", PHYSICAL_CASES)
@pytest.mark.parametrize("multiplicity", [1, 3])
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_cartesian_irreps_equivariance_o3_and_so3(
    formula: str,
    traceless: bool,
    expected_irreps: str,
    multiplicity: int,
    layout: cue.IrrepsLayout,
):
    c2i = cartesian_to_irreps(
        formula,
        group=cue.O3,
        vector_parity=-1,
        traceless=traceless,
        multiplicity=multiplicity,
        layout=layout,
    )
    i2c = irreps_to_cartesian(
        formula,
        group=cue.O3,
        vector_parity=-1,
        traceless=traceless,
        multiplicity=multiplicity,
        layout=layout,
    )

    assert c2i.all_same_segment_shape()
    assert i2c.all_same_segment_shape()
    if layout == cue.ir_mul:
        assert c2i.polynomial.inputs[0].segment_shape == (multiplicity,)
        assert c2i.polynomial.outputs[0].segment_shape == (multiplicity,)

    expected = multiplicity * cue.Irreps(cue.O3, expected_irreps)
    assert c2i.outputs[0].irreps == expected
    assert i2c.inputs[0].irreps == expected

    cart_rep = c2i.inputs[0]
    ir_rep = c2i.outputs[0]

    rng = np.random.default_rng(1234)
    T = rng.normal(size=(cart_rep.dim,))
    c = rng.normal(size=(ir_rep.dim,))

    # Test both proper rotation (det=+1) and improper rotoinversion / pure inversion (det=-1)
    for rot_params, disc_params in [
        (rng.normal(size=(3,)), np.array([0])),
        (np.zeros(3), np.array([1])),  # pure inversion -I
        (rng.normal(size=(3,)), np.array([1])),  # general improper O(3) element
    ]:
        D_cart = cart_rep.exp_map(rot_params, disc_params)
        D_ir = ir_rep.exp_map(rot_params, disc_params)

        [lhs_c] = c2i(D_cart @ T)
        [c_T] = c2i(T)
        rhs_c = D_ir @ c_T
        np.testing.assert_allclose(lhs_c, rhs_c, atol=1e-13, rtol=1e-13)

        [lhs_T] = i2c(D_ir @ c)
        [T_c] = i2c(c)
        rhs_T = D_cart @ T_c
        np.testing.assert_allclose(lhs_T, rhs_T, atol=1e-13, rtol=1e-13)

    # Also verify SO3 group variant
    c2i_so3 = cartesian_to_irreps(
        formula,
        group=cue.SO3,
        traceless=traceless,
        multiplicity=multiplicity,
        layout=layout,
    )
    rot_params = rng.normal(size=(3,))
    D_cart_so3 = c2i_so3.inputs[0].exp_map(rot_params, np.array([], dtype=int))
    D_ir_so3 = c2i_so3.outputs[0].exp_map(rot_params, np.array([], dtype=int))
    [y1] = c2i_so3(D_cart_so3 @ T)
    [y0] = c2i_so3(T)
    np.testing.assert_allclose(y1, D_ir_so3 @ y0, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("formula, traceless, _", PHYSICAL_CASES)
@pytest.mark.parametrize("multiplicity", [1, 4])
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_isometry_parseval_and_orthogonal_projection(
    formula: str,
    traceless: bool,
    _: str,
    multiplicity: int,
    layout: cue.IrrepsLayout,
):
    c2i = cartesian_to_irreps(
        formula, traceless=traceless, multiplicity=multiplicity, layout=layout
    )
    i2c = irreps_to_cartesian(
        formula, traceless=traceless, multiplicity=multiplicity, layout=layout
    )

    rng = np.random.default_rng(2025)
    c = rng.normal(size=(11, i2c.inputs[0].dim))

    # Synthesis -> Analysis is exact identity: c == cart2ir(ir2cart(c))
    [T_from_c] = i2c(c)
    [c_roundtrip] = c2i(T_from_c)
    np.testing.assert_allclose(c_roundtrip, c, atol=1e-13, rtol=1e-13)

    # Parseval norm preservation on subspace: ||c||_2^2 == ||T_from_c||_F^2
    np.testing.assert_allclose(
        np.sum(c**2, axis=-1),
        np.sum(T_from_c**2, axis=-1),
        atol=1e-13,
        rtol=1e-13,
    )

    # Arbitrary Cartesian tensor projection idempotency & Pythagorean identity
    T_arb = rng.normal(size=(11, c2i.inputs[0].dim))
    [c_proj] = c2i(T_arb)
    [T_proj] = i2c(c_proj)
    [T_proj2] = i2c(c2i(T_proj)[0])
    np.testing.assert_allclose(T_proj2, T_proj, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(
        np.sum(c_proj**2, axis=-1),
        np.sum(T_proj**2, axis=-1),
        atol=1e-13,
        rtol=1e-13,
    )
    # Orthogonal complement: <T_proj, T_arb - T_proj> == 0
    residual = T_arb - T_proj
    inner = np.sum(T_proj * residual, axis=-1)
    np.testing.assert_allclose(inner, 0.0, atol=1e-13)


def test_multipole_wrappers_trace_free_symmetry_and_analytic_projectors():
    rng = np.random.default_rng(77)

    # Degree 0 (monopole), 1 (dipole), 2 (quadrupole), 3 (octupole), 4 (hexadecapole)
    expected_dims = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}
    for deg, dim in expected_dims.items():
        c2m = symmetric_traceless_multipole_to_irreps(deg)
        m2c = irreps_to_symmetric_traceless_multipole(deg)
        assert c2m.outputs[0].dim == dim
        assert m2c.inputs[0].dim == dim

        c = rng.normal(size=(dim,))
        [T_flat] = m2c(c)
        [c_rec] = c2m(T_flat)
        np.testing.assert_allclose(c_rec, c, atol=1e-13, rtol=1e-13)

        if deg >= 2:
            T = T_flat.reshape((3,) * deg)
            # Full permutation symmetry
            for perm in itertools.permutations(range(deg)):
                np.testing.assert_allclose(T, np.transpose(T, perm), atol=1e-14)
            # Zero trace across every pair of axes
            for ax1, ax2 in itertools.combinations(range(deg), 2):
                tr = np.trace(T, axis1=ax1, axis2=ax2)
                np.testing.assert_allclose(tr, 0.0, atol=1e-14)

    # Exact agreement of Q Q^T with analytical Cartesian multipole detracers!
    for deg, analytic_fn in [
        (2, _analytic_traceless_quadrupole),
        (3, _analytic_traceless_octupole),
        (4, _analytic_traceless_hexadecapole),
    ]:
        c2m = symmetric_traceless_multipole_to_irreps(deg)
        m2c = irreps_to_symmetric_traceless_multipole(deg)
        T_raw = rng.normal(size=(3,) * deg)
        [c] = c2m(T_raw.reshape(-1))
        [T_proj_flat] = m2c(c)
        T_proj = T_proj_flat.reshape((3,) * deg)
        T_expected = analytic_fn(T_raw)
        np.testing.assert_allclose(T_proj, T_expected, atol=1e-13, rtol=1e-13)


def test_layout_consistency_between_ir_mul_and_mul_ir():
    # Include "ijk" where internal irrep multiplicities m_s > 1 (3x1o + 2x2o)
    rng = np.random.default_rng(99)
    for formula, traceless in [("ij=ji", False), ("ijk", False), ("ijk=jik=ikj", True)]:
        mul = 4
        c2i_irmul = cartesian_to_irreps(
            formula, traceless=traceless, multiplicity=mul, layout=cue.ir_mul
        )
        c2i_mulir = cartesian_to_irreps(
            formula, traceless=traceless, multiplicity=mul, layout=cue.mul_ir
        )
        irreps_out = c2i_irmul.outputs[0].irreps
        tr_ir_to_mul = cue.descriptors.transpose(irreps_out, cue.ir_mul, cue.mul_ir)

        deg = c2i_irmul.inputs[0].degree
        T_mul_ir = rng.normal(size=(5, mul, 3**deg))
        T_ir_mul = np.swapaxes(T_mul_ir, -2, -1)

        [out_irmul] = c2i_irmul(T_ir_mul.reshape(5, -1))
        [out_irmul_as_mulir] = tr_ir_to_mul(out_irmul)
        [out_mulir] = c2i_mulir(T_mul_ir.reshape(5, -1))
        np.testing.assert_allclose(
            out_mulir, out_irmul_as_mulir, atol=1e-14, rtol=1e-14
        )


def test_ir_dict_variants_and_keep_ir_filtering():
    rng = np.random.default_rng(314)
    for formula, traceless in [
        ("", False),
        ("i", False),
        ("ij=ji", False),
        ("ijk", False),
    ]:
        e_c2i = cartesian_to_irreps(formula, traceless=traceless, multiplicity=2)
        d_c2i = cartesian_to_irreps_ir_dict(
            formula, traceless=traceless, multiplicity=2
        )
        assert d_c2i.polynomial == e_c2i.split_operand_by_irrep(-1).polynomial
        assert d_c2i.output_irreps == (e_c2i.outputs[0].irreps,)

        T = rng.normal(size=(3, e_c2i.inputs[0].dim))
        [flat_out] = e_c2i(T)
        split_out = np.concatenate(d_c2i.polynomial(T), axis=-1)
        np.testing.assert_allclose(flat_out, split_out, atol=1e-14)

        e_i2c = irreps_to_cartesian(formula, traceless=traceless, multiplicity=2)
        d_i2c = irreps_to_cartesian_ir_dict(
            formula, traceless=traceless, multiplicity=2
        )
        assert d_i2c.polynomial == e_i2c.split_operand_by_irrep(0).polynomial
        assert d_i2c.input_irreps == (e_i2c.inputs[0].irreps,)

        c = rng.normal(size=(3, e_i2c.inputs[0].dim))
        c_parts = []
        offset = 0
        for mul, ir in e_i2c.inputs[0].irreps:
            sz = mul * ir.dim
            c_parts.append(c[:, offset : offset + sz])
            offset += sz
        [cart_flat] = e_i2c(c)
        [cart_from_dict] = d_i2c.polynomial(*c_parts)
        np.testing.assert_allclose(cart_flat, cart_from_dict, atol=1e-14)

    # keep_ir filtering: extract isotropic scalar trace 0e from symmetric rank-2
    iso = cartesian_to_irreps("ij=ji", keep_ir=[cue.O3(0, 1)])
    assert iso.outputs[0].irreps == cue.Irreps("O3", "0e")

    # Empty subspace: antisymmetric rank-2 with traceless=True -> empty irreps
    empty_desc = cartesian_to_irreps("ij=-ji", traceless=True)
    assert empty_desc.outputs[0].irreps.dim == 0
    [empty_out] = empty_desc(np.ones((2, 9)))
    assert empty_out.shape == (2, 0)

    assert symmetric_cartesian_formula(0) == ""
    assert symmetric_cartesian_formula(1) == "i"
    assert symmetric_cartesian_formula(2) == "ij=ji"
    assert symmetric_cartesian_formula(3) == "ijk=jik=ikj"
