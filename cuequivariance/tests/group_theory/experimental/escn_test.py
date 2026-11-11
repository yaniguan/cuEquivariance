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
import numpy as np
import pytest
from cuequivariance.group_theory.experimental.escn import (
    SphericalSignal,
    escn_tp,
    escn_tp_compact,
    escn_weights_to_compact,
    irreps_to_spherical_signal,
    spherical_signal_to_irreps,
)

import cuequivariance as cue


def test_escn():
    escn_tp(
        cue.Irreps("O3", "32x0e + 32x1o"),
        cue.Irreps("O3", "32x0e + 32x1o"),
        m_max=2,
    )


def test_escn_compact():
    escn_tp_compact(
        cue.Irreps("SO3", "32x0 + 32x1"),
        cue.Irreps("SO3", "32x0 + 32x1"),
        m_max=2,
    )


@pytest.mark.parametrize("group", [cue.SO3, cue.O3])
@pytest.mark.parametrize("parity", [1, -1])
@pytest.mark.parametrize(
    "mul,l_max,m_max",
    [
        (1, 0, None),
        (2, 2, None),
        (3, 3, 3),
        (2, 3, 1),
        (2, 2, 0),
        (2, 1, 4),  # m_max > l_max clamps cleanly to l_max
    ],
)
def test_spherical_signal_rep_invariants(
    group: type[cue.Irrep],
    parity: int,
    mul: int,
    l_max: int,
    m_max: int | None,
):
    if group == cue.SO3 and parity == -1:
        pytest.skip("parity only distinguishes O3 signals")

    # Check positional (mul, l_max, m_max) compatibility as well as keyword group/parity
    rep = (
        SphericalSignal(mul, l_max, m_max)
        if group == cue.SO3
        else SphericalSignal(mul, l_max, m_max, group=group, parity=parity)
    )

    expected_m = l_max if m_max is None else min(m_max, l_max)
    assert rep.m_max == expected_m
    expected_dim = sum(
        mul * (l_max - abs(m) + 1) for m in range(-expected_m, expected_m + 1)
    )
    assert rep.dim == expected_dim

    A = rep.A
    X = rep.X
    H = rep.H
    expected_lie_dim = 1 if expected_m < l_max else 3
    assert A.shape == (expected_lie_dim, expected_lie_dim, expected_lie_dim)
    assert X.shape == (expected_lie_dim, rep.dim, rep.dim)
    assert H.shape == (0 if group == cue.SO3 else 1, rep.dim, rep.dim)

    # Skew-symmetry of real orthogonal generators
    np.testing.assert_allclose(X + np.swapaxes(X, -1, -2), 0.0, atol=1e-12)

    # Lie algebra commutation: [X_i, X_j] = A_{ijk} X_k
    xx = np.einsum("iuv,jvw->ijuw", X, X)
    comm = xx - np.swapaxes(xx, 0, 1)
    rhs = np.einsum("ijk,kuv->ijuv", A, X)
    np.testing.assert_allclose(comm, rhs, atol=1e-11, rtol=0)

    # Discrete generator involution H @ H = I and commutation with X
    for h in H:
        np.testing.assert_allclose(h @ h, np.eye(rep.dim), atol=1e-12)
        for x_gen in X:
            np.testing.assert_allclose(h @ x_gen, x_gen @ h, atol=1e-12)

    # Trivial representation invariants
    triv = rep.trivial()
    assert triv.is_trivial()
    assert triv.is_scalar()
    assert triv.lie_dim == rep.lie_dim
    assert len(triv.H) == len(rep.H)

    # Hash, equality, repr, and exp_map orthogonality
    rep_copy = SphericalSignal(mul, l_max, expected_m, group=group, parity=parity)
    assert rep == rep_copy
    assert hash(rep) == hash(rep_copy)
    assert "SphericalSignal" in repr(rep)

    rng = np.random.default_rng(42)
    c_params = rng.standard_normal(rep.lie_dim)
    d_params = np.array([1]) if group == cue.O3 else np.array([], dtype=int)
    D = rep.exp_map(c_params, d_params)
    np.testing.assert_allclose(D @ D.T, np.eye(rep.dim), atol=1e-12)


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize(
    "group,irreps_str,m_max,expect_spherical_signal",
    [
        (cue.SO3, "3x0 + 3x1 + 3x2", None, True),
        (cue.O3, "3x0e + 3x1o + 3x2e", None, True),
        (cue.O3, "2x0o + 2x1e + 2x2o", 1, True),
        (cue.O3, "2x0e + 3x0o + 2x1o + 4x1e + 2x2e", None, False),
        (cue.O3, "2x0e + 3x0o + 2x1o + 4x1e + 2x2e", 1, False),
    ],
)
def test_irreps_spherical_signal_permutation_and_equivariance(
    layout: cue.IrrepsLayout,
    group: type[cue.Irrep],
    irreps_str: str,
    m_max: int | None,
    expect_spherical_signal: bool,
):
    rng = np.random.default_rng(7)
    irreps = cue.Irreps(group, irreps_str)
    to_sig = irreps_to_spherical_signal(irreps, layout=layout, m_max=m_max)
    from_sig = spherical_signal_to_irreps(irreps, layout=layout, m_max=m_max)

    if expect_spherical_signal:
        assert isinstance(to_sig.outputs[0], SphericalSignal)
        assert isinstance(from_sig.inputs[0], SphericalSignal)

    x = rng.standard_normal(irreps.dim)
    [sig] = to_sig(x)
    [x_rec] = from_sig(sig)

    if m_max is None:
        np.testing.assert_allclose(x_rec, x, atol=1e-14)
    else:
        # Projecting a second time is idempotent
        [sig2] = to_sig(x_rec)
        np.testing.assert_allclose(sig2, sig, atol=1e-14)

    # Check exact equivariance of the permutation / projection
    rep_in = to_sig.inputs[0]
    rep_out = to_sig.outputs[0]
    if m_max is None:
        c_params = rng.standard_normal(3)
        c_params_out = c_params
    else:
        theta = float(rng.standard_normal())
        c_params = np.array([0.0, theta, 0.0])
        c_params_out = np.array([theta])
    d_params = np.array([1]) if group == cue.O3 else np.array([], dtype=int)

    D_in = rep_in.exp_map(c_params, d_params)
    D_out = rep_out.exp_map(c_params_out, d_params)
    np.testing.assert_allclose(to_sig(D_in @ x)[0], D_out @ sig, atol=1e-12)
    np.testing.assert_allclose(from_sig(D_out @ sig)[0], D_in @ x_rec, atol=1e-12)


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize(
    "group,s_in,s_out,m_max",
    [
        (cue.SO3, "4x0 + 4x1 + 4x2", "4x0 + 4x1 + 4x2", None),
        (cue.SO3, "2x0 + 3x1 + 2x2", "4x0 + 2x1 + 3x3", 1),
        (cue.SO3, "2x0 + 3x1 + 2x2", "2x0 + 3x1 + 2x2", 0),
        (cue.O3, "4x0e + 4x1o + 4x2e", "4x0e + 4x1o + 4x2e", None),
        (cue.O3, "4x0e + 4x1o + 4x2e", "4x0e + 4x1o + 4x2e", 1),
        (
            cue.O3,
            "2x0e + 3x0o + 2x1o + 4x1e + 2x2e + 3x2o",
            "3x0e + 2x0o + 4x1o + 2x1e + 3x2e + 2x2o",
            2,
        ),
        (
            cue.O3,
            "2x0e + 3x0o + 2x1o + 4x1e + 2x2e + 3x2o",
            "3x0e + 2x0o + 4x1o + 2x1e + 3x2e + 2x2o",
            1,
        ),
        (cue.O3, "2x0e + 3x1o + 2x2o", "4x0o + 2x1e + 3x2e", 2),
        (cue.O3, "2x0e + 3x1o + 2x2e", "2x0e + 3x1o + 2x2e", 0),
    ],
)
def test_escn_tp_compact_matches_escn_tp(
    layout: cue.IrrepsLayout,
    group: type[cue.Irrep],
    s_in: str,
    s_out: str,
    m_max: int | None,
):
    rng = np.random.default_rng(123)
    irreps_in = cue.Irreps(group, s_in)
    irreps_out = cue.Irreps(group, s_out)

    poly_std = escn_tp(irreps_in, irreps_out, m_max=m_max)
    poly_cmp = escn_tp_compact(irreps_in, irreps_out, m_max=m_max)
    assert isinstance(poly_cmp, cue.SegmentedPolynomial)
    assert poly_std.inputs[0].dim == poly_cmp.inputs[0].size
    assert (
        poly_cmp.operations[0][1].num_paths
        <= poly_std.polynomial.operations[0][1].num_paths
    )

    to_cmp = irreps_to_spherical_signal(irreps_in, layout=layout)
    from_cmp = spherical_signal_to_irreps(irreps_out, layout=layout)
    w_map = escn_weights_to_compact(irreps_in, irreps_out, m_max=m_max)

    w = rng.standard_normal(poly_std.inputs[0].dim)
    x_ir_mul = rng.standard_normal(irreps_in.dim)
    x_layout = (
        cue.descriptors.transpose(irreps_in, cue.ir_mul, layout)(x_ir_mul)[0]
        if layout != cue.ir_mul
        else x_ir_mul
    )

    [y_std_ir_mul] = poly_std(w, x_ir_mul)
    y_std = (
        cue.descriptors.transpose(irreps_out, cue.ir_mul, layout)(y_std_ir_mul)[0]
        if layout != cue.ir_mul
        else y_std_ir_mul
    )

    [w_cmp] = w_map(w)
    [x_cmp] = to_cmp(x_layout)
    [y_cmp] = poly_cmp(w_cmp, x_cmp)
    [y_got] = from_cmp(y_cmp)

    np.testing.assert_allclose(y_got, y_std, atol=1e-12, rtol=1e-12)
