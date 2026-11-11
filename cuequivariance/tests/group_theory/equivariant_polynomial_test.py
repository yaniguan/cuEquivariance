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
import numpy as np
import pytest

import cuequivariance as cue


def test_transpose():
    cue.descriptors.transpose(cue.Irreps(cue.O3, "3x1e"), cue.mul_ir, cue.ir_mul)


def test_commutativity_squeeze_flatten():
    irreps1 = cue.Irreps("O3", "32x0e + 32x1o")
    irreps2 = cue.Irreps("O3", "1x0e + 1x1o")
    irreps3 = cue.Irreps("O3", "32x0e + 32x1o")

    poly = cue.descriptors.fully_connected_tensor_product(irreps1, irreps2, irreps3)
    assert (
        poly.squeeze_modes().flatten_coefficient_modes()
        == poly.flatten_coefficient_modes().squeeze_modes()
    )

    poly = cue.descriptors.full_tensor_product(irreps1, irreps2, irreps3)
    assert (
        poly.squeeze_modes().flatten_coefficient_modes()
        == poly.flatten_coefficient_modes().squeeze_modes()
    )

    poly = cue.descriptors.channelwise_tensor_product(irreps1, irreps2, irreps3)
    assert (
        poly.squeeze_modes().flatten_coefficient_modes()
        == poly.flatten_coefficient_modes().squeeze_modes()
    )

    poly = cue.descriptors.linear(irreps1, irreps2)
    assert (
        poly.squeeze_modes().flatten_coefficient_modes()
        == poly.flatten_coefficient_modes().squeeze_modes()
    )


@pytest.mark.parametrize("ell", [1, 2, 3, 4])
def test_spherical_harmonics(ell: int):
    poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [ell])

    vec = np.random.randn(3)
    axis = np.random.randn(3)
    angle = np.random.rand()

    [yl] = poly(vec)

    R = cue.SO3(1).rotation(axis, angle)
    Rl = cue.SO3(ell).rotation(axis, angle)

    [yl1] = poly(R @ vec)
    yl2 = Rl @ yl

    np.testing.assert_allclose(yl1, yl2, rtol=1e-06, atol=1e-13)
    np.testing.assert_allclose(np.sum(yl**2), (2 * ell + 1) * np.sum(vec**2) ** ell)


@pytest.mark.parametrize("ir_vec", [cue.SO3(1), cue.O3(1, -1), cue.O3(1, 1)])
@pytest.mark.parametrize(
    "layout_arg, expected_layout",
    [
        (cue.ir_mul, cue.ir_mul),
        (cue.mul_ir, cue.mul_ir),
        ("ir_mul", cue.ir_mul),
        ("mul_ir", cue.mul_ir),
    ],
)
@pytest.mark.parametrize("ls", [[0], [1], [0, 1, 2, 3], [2, 0, 3]])
def test_spherical_harmonics_layout_and_equivariance(
    ir_vec: cue.Irrep,
    layout_arg,
    expected_layout: cue.IrrepsLayout,
    ls: list[int],
):
    poly = cue.descriptors.spherical_harmonics(ir_vec, ls, layout=layout_arg)
    assert poly.inputs[0].layout == expected_layout
    assert poly.outputs[0].layout == expected_layout

    rng = np.random.default_rng(42)
    vec = rng.standard_normal(3)
    angles = rng.standard_normal(poly.inputs[0].lie_dim)

    discrete_cases = (
        [np.array([])]
        if isinstance(ir_vec, cue.SO3)
        else [np.array([0]), np.array([1])]
    )
    for discrete in discrete_cases:
        D_in = poly.inputs[0].exp_map(angles, discrete)
        D_out = poly.outputs[0].exp_map(angles, discrete)

        [yl_rot_in] = poly(D_in @ vec)
        [yl] = poly(vec)
        yl_rot_out = D_out @ yl
        np.testing.assert_allclose(yl_rot_in, yl_rot_out, rtol=1e-12, atol=1e-12)


def test_spherical_harmonics_layout_context_and_invalid():
    with cue.assume(cue.O3, cue.mul_ir):
        poly = cue.descriptors.spherical_harmonics(
            cue.O3(1, -1), [0, 1, 2], layout=None
        )
        assert poly.inputs[0].layout == cue.mul_ir
        assert poly.outputs[0].layout == cue.mul_ir

    with pytest.raises(ValueError, match="Invalid layout"):
        cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1], layout="invalid")


@pytest.mark.parametrize("ell", [0, 1, 2, 3, 4])
def test_yxy_rotation(ell: int):
    alpha = 0.3
    beta = 0.4
    gamma = -0.5

    irrep = cue.SO3(ell)
    poly = cue.descriptors.yxy_rotation(cue.Irreps("SO3", [irrep]))

    def enc(th: float):
        m = np.arange(1, ell + 1)
        c = np.cos(m * th)
        s = np.sin(m * th)
        return np.concatenate([c[::-1], [1.0], s])

    x = np.random.randn(irrep.dim)
    [y1] = poly(enc(gamma), enc(beta), enc(alpha), x)

    A = irrep.rotation(np.array([0.0, 1.0, 0.0]), alpha)
    B = irrep.rotation(np.array([1.0, 0.0, 0.0]), beta)
    C = irrep.rotation(np.array([0.0, 1.0, 0.0]), gamma)
    y2 = A @ B @ C @ x

    np.testing.assert_allclose(y1, y2)


@pytest.mark.parametrize("ell", [0, 1, 2, 3, 4])
def test_yx_rotation(ell: int):
    phi = 0.3
    theta = 0.4

    irrep = cue.SO3(ell)
    poly = cue.descriptors.yx_rotation(cue.Irreps("SO3", [irrep]))

    def enc(th: float):
        m = np.arange(1, ell + 1)
        c = np.cos(m * th)
        s = np.sin(m * th)
        return np.concatenate([c[::-1], [1.0], s])

    x = np.random.randn(irrep.dim)
    [y1] = poly(enc(phi), enc(theta), x)

    # Rotation around x-axis followed by rotation around y-axis
    Rx = irrep.rotation(np.array([1.0, 0.0, 0.0]), phi)
    Ry = irrep.rotation(np.array([0.0, 1.0, 0.0]), theta)
    y2 = Ry @ Rx @ x

    np.testing.assert_allclose(y1, y2)


def test_elementwise_tensor_product():
    irreps1 = cue.Irreps("O3", "8x0e + 7x1o + 4x2e")
    irreps2 = cue.Irreps("O3", "8x0e + 7x1o + 4x0e")
    irreps3 = cue.Irreps("O3", "0e + 1o + 1e")
    poly = cue.descriptors.elementwise_tensor_product(irreps1, irreps2, irreps3)
    assert (
        poly.squeeze_modes().flatten_coefficient_modes()
        == poly.flatten_coefficient_modes().squeeze_modes()
    )
    assert poly.num_inputs == 2
    assert poly.num_outputs == 1


def test_symmetric_contraction():
    irreps_in = 16 * cue.Irreps("SO3", "0 + 1 + 2")
    irreps_out = 16 * cue.Irreps("SO3", "0 + 1")
    poly = cue.descriptors.symmetric_contraction(irreps_in, irreps_out, [0, 1, 2, 3])
    assert poly.num_inputs == 2
    assert poly.num_outputs == 1

    [_, _, _, (_, d)] = poly.polynomial.operations
    assert d.num_paths == 437

    poly = poly.polynomial.unsymmetrize_for_identical_operands()
    [_, _, _, (_, d)] = poly.operations
    assert d.num_paths == 105
