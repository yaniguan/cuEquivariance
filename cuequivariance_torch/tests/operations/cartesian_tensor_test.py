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
import copy
import itertools

import cuequivariance as cue
import cuequivariance_torch as cuet
import numpy as np
import pytest
import torch
from cuequivariance_torch.operations import CartesianToIrreps, IrrepsToCartesian

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _rotate_cartesian_tensor(
    T: torch.Tensor,
    R: torch.Tensor,
    degree: int,
    multiplicity: int,
    layout: cue.IrrepsLayout,
    parity_sign: float = 1.0,
) -> torch.Tensor:
    """Apply O(3) transformation (parity_sign * R)^{tensor degree} to shaped Cartesian tensor T."""
    M_mat = parity_sign * R  # (3, 3)
    if degree == 0:
        return T.clone()

    letters_in = "abcdef"[:degree]
    letters_out = "ijklmn"[:degree]
    r_terms = ",".join(f"{o}{a}" for o, a in zip(letters_out, letters_in))

    if multiplicity == 1:
        eq = f"{r_terms},...{letters_in}->...{letters_out}"
    elif layout == cue.mul_ir:
        eq = f"{r_terms},...u{letters_in}->...u{letters_out}"
    else:
        eq = f"{r_terms},...{letters_in}u->...{letters_out}u"

    return torch.einsum(eq, *([M_mat] * degree), T)


EQUIVARIANCE_CASES = [
    ("", False, "0e"),
    ("i", False, "1o"),
    ("ij=-ji", False, "1e"),
    ("ij=ji", False, "0e + 2e"),
    ("ij=ji", True, "2e"),
    ("ijk=jik=ikj", True, "3o"),
    ("ij", False, "0e + 1e + 2e"),
    ("ijk", False, "0o + 3x1o + 2x2o + 3o"),
]


@pytest.mark.parametrize("formula, traceless, expected_irreps", EQUIVARIANCE_CASES)
@pytest.mark.parametrize("multiplicity", [1, 4])
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
def test_torch_cartesian_equivariance_rotations_and_inversion(
    formula: str,
    traceless: bool,
    expected_irreps: str,
    multiplicity: int,
    layout: cue.IrrepsLayout,
):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    c2i = cuet.CartesianToIrreps(
        formula,
        traceless=traceless,
        multiplicity=multiplicity,
        layout=layout,
        device=device,
    )
    i2c = cuet.IrrepsToCartesian(
        formula,
        traceless=traceless,
        multiplicity=multiplicity,
        layout=layout,
        device=device,
    )

    assert c2i is not None and isinstance(c2i, CartesianToIrreps)
    assert i2c is not None and isinstance(i2c, IrrepsToCartesian)
    assert c2i.irreps_out == multiplicity * cue.Irreps("O3", expected_irreps)
    assert i2c.irreps_in == c2i.irreps_out

    deg = c2i.degree
    batch = 6
    if multiplicity == 1:
        T_shape = (batch,) + (3,) * deg
    elif layout == cue.mul_ir:
        T_shape = (batch, multiplicity) + (3,) * deg
    else:
        T_shape = (batch,) + (3,) * deg + (multiplicity,)

    T = torch.randn(T_shape, dtype=torch.float64, device=device)
    c = torch.randn(batch, i2c.irreps_in.dim, dtype=torch.float64, device=device)

    ir_rep = cue.IrrepsAndLayout(c2i.irreps_out, layout)

    # Test both random SO(3) rotation and improper O(3) transformations (pure inversion & rotoinversion)
    for rot_params, disc in [
        (rng.normal(size=(3,)), 0),
        (np.zeros(3), 1),
        (rng.normal(size=(3,)), 1),
    ]:
        R_np = cue.SO3(1).exp_map(rot_params, np.array([], dtype=int))
        R = torch.from_numpy(R_np).to(dtype=torch.float64, device=device)
        parity_sign = -1.0 if disc == 1 else 1.0

        D_ir_np = ir_rep.exp_map(rot_params, np.array([disc]))
        D_ir = torch.from_numpy(D_ir_np).to(dtype=torch.float64, device=device)

        T_rot = _rotate_cartesian_tensor(
            T, R, deg, multiplicity, layout, parity_sign=parity_sign
        )
        lhs_c = c2i(T_rot)
        rhs_c = c2i(T) @ D_ir.T
        torch.testing.assert_close(lhs_c, rhs_c, atol=1e-13, rtol=1e-13)

        c_rot = c @ D_ir.T
        lhs_T = i2c(c_rot)
        rhs_T = _rotate_cartesian_tensor(
            i2c(c), R, deg, multiplicity, layout, parity_sign=parity_sign
        )
        torch.testing.assert_close(lhs_T, rhs_T, atol=1e-13, rtol=1e-13)


def test_physical_trace_free_symmetry_isometry_and_parseval():
    torch.manual_seed(123)

    # Quadrupole Theta_ij (2e, dim=5)
    q_c2i = cuet.CartesianToIrreps(
        "ij=ji", traceless=True, layout=cue.ir_mul, device=device
    )
    q_i2c = cuet.IrrepsToCartesian(
        "ij=ji", traceless=True, layout=cue.ir_mul, device=device
    )
    assert q_c2i.irreps_out == cue.Irreps("O3", "2e")

    c_quad = torch.randn(16, 5, dtype=torch.float64, device=device)
    Theta = q_i2c(c_quad)  # (16, 3, 3)
    assert Theta.shape == (16, 3, 3)

    # Symmetry Theta_ij == Theta_ji and trace sum_i Theta_ii == 0
    assert (Theta - Theta.transpose(-1, -2)).abs().max().item() < 1e-13
    assert (
        torch.diagonal(Theta, dim1=-2, dim2=-1).sum(dim=-1).abs().max().item() < 1e-13
    )

    # Parseval norm preservation & exact roundtrip
    torch.testing.assert_close(
        (c_quad**2).sum(dim=-1), (Theta**2).sum(dim=(-2, -1)), atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(q_c2i(Theta), c_quad, atol=1e-13, rtol=1e-13)

    # Octupole Omega_ijk (3o, dim=7)
    o_c2i = cuet.CartesianToIrreps(
        "ijk=jik=ikj", traceless=True, layout=cue.ir_mul, device=device
    )
    o_i2c = cuet.IrrepsToCartesian(
        "ijk=jik=ikj", traceless=True, layout=cue.ir_mul, device=device
    )
    assert o_c2i.irreps_out == cue.Irreps("O3", "3o")

    c_oct = torch.randn(12, 7, dtype=torch.float64, device=device)
    Omega = o_i2c(c_oct)  # (12, 3, 3, 3)
    assert Omega.shape == (12, 3, 3, 3)

    # Totally symmetric across all permutations of (i, j, k)
    for p in itertools.permutations((1, 2, 3)):
        assert (Omega - Omega.permute(0, *p)).abs().max().item() < 1e-13

    # Trace-free along all index pairs
    assert torch.einsum("biik->bk", Omega).abs().max().item() < 1e-13
    assert torch.einsum("biki->bk", Omega).abs().max().item() < 1e-13
    assert torch.einsum("bkii->bk", Omega).abs().max().item() < 1e-13

    # Parseval norm preservation & exact roundtrip
    torch.testing.assert_close(
        (c_oct**2).sum(dim=-1),
        (Omega**2).sum(dim=(-3, -2, -1)),
        atol=1e-13,
        rtol=1e-13,
    )
    torch.testing.assert_close(o_c2i(Omega), c_oct, atol=1e-13, rtol=1e-13)

    # Exact transpose / adjoint identity: <cart2ir(T), c> == <T, ir2cart(c)>
    T_rand = torch.randn(12, 3, 3, 3, dtype=torch.float64, device=device)
    lhs_adj = (o_c2i(T_rand) * c_oct).sum()
    rhs_adj = (T_rand * Omega).sum()
    torch.testing.assert_close(lhs_adj, rhs_adj, atol=1e-13, rtol=1e-13)

    # Multi-irrep (m_s > 1) full rank-3 "ijk" (dim=27) exact isometry & roundtrip
    full3_c2i = cuet.CartesianToIrreps(
        "ijk", multiplicity=2, layout=cue.mul_ir, device=device
    )
    full3_i2c = cuet.IrrepsToCartesian(
        "ijk", multiplicity=2, layout=cue.mul_ir, device=device
    )
    T_full3 = torch.randn(8, 2, 3, 3, 3, dtype=torch.float64, device=device)
    c_full3 = full3_c2i(T_full3)
    torch.testing.assert_close(full3_i2c(c_full3), T_full3, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        (c_full3**2).sum(dim=-1),
        (T_full3**2).sum(dim=(-4, -3, -2, -1)),
        atol=1e-13,
        rtol=1e-13,
    )


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("multiplicity", [1, 3])
def test_shapes_batches_flattened_and_reshape_output(
    layout: cue.IrrepsLayout, multiplicity: int
):
    torch.manual_seed(7)
    c2i = cuet.CartesianToIrreps(
        "ij=ji", multiplicity=multiplicity, layout=layout, device=device
    )
    i2c_shaped = cuet.IrrepsToCartesian(
        "ij=ji",
        multiplicity=multiplicity,
        layout=layout,
        reshape_output=True,
        device=device,
    )
    i2c_flat = cuet.IrrepsToCartesian(
        "ij=ji",
        multiplicity=multiplicity,
        layout=layout,
        reshape_output=False,
        device=device,
    )

    for batch_shape in [(), (1,), (5,), (2, 4), (0, 3)]:
        c = torch.randn(
            batch_shape + (i2c_shaped.irreps_in.dim,),
            dtype=torch.float64,
            device=device,
        )
        T_shaped = i2c_shaped(c)
        T_flat = i2c_flat(c)

        if multiplicity == 1:
            assert T_shaped.shape == batch_shape + (3, 3)
            assert T_flat.shape == batch_shape + (9,)
        elif layout == cue.mul_ir:
            assert T_shaped.shape == batch_shape + (multiplicity, 3, 3)
            assert T_flat.shape == batch_shape + (multiplicity * 9,)
        else:
            assert T_shaped.shape == batch_shape + (3, 3, multiplicity)
            assert T_flat.shape == batch_shape + (9 * multiplicity,)

        # Both shaped and flattened Cartesian inputs produce identical irreps
        c_from_shaped = c2i(T_shaped)
        c_from_flat = c2i(T_flat)
        torch.testing.assert_close(c_from_shaped, c, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(c_from_flat, c, atol=1e-13, rtol=1e-13)

    # Monopole "" shape roundtrips for (), (1,), (4,) and flattened (4, 1)
    if multiplicity == 1:
        m_c2i = cuet.CartesianToIrreps("", layout=layout, device=device)
        m_i2c = cuet.IrrepsToCartesian("", layout=layout, device=device)
        for b_shape in [(), (1,), (4,)]:
            c0 = torch.randn(b_shape + (1,), dtype=torch.float64, device=device)
            t0 = m_i2c(c0)
            assert t0.shape == b_shape
            torch.testing.assert_close(m_c2i(t0), c0, atol=1e-13, rtol=1e-13)
        t0_flat = torch.randn(4, 1, dtype=torch.float64, device=device)
        torch.testing.assert_close(m_c2i(t0_flat), t0_flat, atol=1e-13, rtol=1e-13)

    # Deepcopy & cue.assume context & error validation
    c2i_copy = copy.deepcopy(c2i)
    x_test = torch.randn(2, c2i.total_cart_dim, dtype=torch.float64, device=device)
    torch.testing.assert_close(c2i(x_test), c2i_copy(x_test))

    with cue.assume(cue.O3, cue.ir_mul):
        m_ctx = cuet.CartesianToIrreps("ij=ji", traceless=True, device=device)
        assert m_ctx.layout == cue.ir_mul

    with pytest.raises(ValueError, match="Expected input shape"):
        c2i(torch.randn(2, 5, dtype=torch.float64, device=device))
    with pytest.raises(ValueError, match="Expected input shape"):
        i2c_shaped(torch.randn(2, 999, dtype=torch.float64, device=device))


def test_autograd_gradcheck_and_gradgradcheck():
    torch.manual_seed(2026)

    # 1. Direct gradcheck and gradgradcheck on CartesianToIrreps & IrrepsToCartesian
    for layout in [cue.mul_ir, cue.ir_mul]:
        c2i = cuet.CartesianToIrreps(
            "ijk=jik=ikj",
            traceless=True,
            multiplicity=2,
            layout=layout,
            device=device,
        )
        i2c = cuet.IrrepsToCartesian(
            "ij=ji",
            traceless=False,
            multiplicity=2,
            layout=layout,
            device=device,
        )

        T_inp = (
            torch.randn(3, 2, 3, 3, 3, dtype=torch.float64, device=device)
            if layout == cue.mul_ir
            else torch.randn(3, 3, 3, 3, 2, dtype=torch.float64, device=device)
        ).requires_grad_(True)
        assert torch.autograd.gradcheck(c2i, (T_inp,), eps=1e-6, atol=1e-5, rtol=1e-5)
        assert torch.autograd.gradgradcheck(
            c2i, (T_inp,), eps=1e-6, atol=1e-5, rtol=1e-5
        )

        c_inp = torch.randn(
            3, i2c.irreps_in.dim, dtype=torch.float64, device=device, requires_grad=True
        )
        assert torch.autograd.gradcheck(i2c, (c_inp,), eps=1e-6, atol=1e-5, rtol=1e-5)
        assert torch.autograd.gradgradcheck(
            i2c, (c_inp,), eps=1e-6, atol=1e-5, rtol=1e-5
        )

    # 2. Polarizable Force-Field / Force-QAT response pipeline:
    #    R -> spherical harmonics Y_{0,1,2}(r_ij) -> Cartesian polarizability alpha_ij(R)
    #    & traceless quadrupole Theta_ij(R) -> polarization energy E_pol(R, E_field, grad_E)
    #    Testing d(alpha_ij)/dR, forces -dE_pol/dR, and Hessian d^2 E_pol / dR^2!
    sh = cuet.SphericalHarmonics([0, 1, 2], normalize=False, device=device)
    alpha_synth = cuet.IrrepsToCartesian(
        "ij=ji", traceless=False, layout=cue.ir_mul, device=device
    )  # 0e + 2e -> (3, 3)
    quad_synth = cuet.IrrepsToCartesian(
        "ij=ji", traceless=True, layout=cue.ir_mul, device=device
    )  # 2e -> (3, 3)

    def polarizability_from_positions(pos: torch.Tensor) -> torch.Tensor:
        # Pair vectors between 3 atoms: (3, 3) -> pairs (3, 3)
        disp = pos[[1, 2, 0]] - pos[[0, 1, 2]]  # (3, 3)
        r2 = (disp**2).sum(dim=-1, keepdim=True)
        decay = torch.exp(-0.25 * r2)
        yl = sh(disp) * decay  # (3, 1 + 3 + 5 = 9): [0e (1), 1o (3), 2e (5)]
        alpha_irreps = torch.cat([yl[:, :1], yl[:, 4:9]], dim=-1)  # 0e + 2e (dim 6)
        alpha_cart = alpha_synth(alpha_irreps).sum(dim=0)  # molecular alpha_ij (3, 3)
        return alpha_cart

    def polarization_energy(
        pos: torch.Tensor, e_field: torch.Tensor, field_grad: torch.Tensor
    ) -> torch.Tensor:
        disp = pos[[1, 2, 0]] - pos[[0, 1, 2]]
        yl = sh(disp) * torch.exp(-0.25 * (disp**2).sum(dim=-1, keepdim=True))
        dipole = yl[:, 1:4].sum(dim=0)  # (3,)
        alpha_irreps = torch.cat([yl[:, :1], yl[:, 4:9]], dim=-1)
        alpha_cart = alpha_synth(alpha_irreps).sum(dim=0)  # (3, 3)
        theta_cart = quad_synth(yl[:, 4:9]).sum(dim=0)  # (3, 3) traceless quadrupole

        e_dip = -torch.dot(dipole, e_field)
        e_ind = -0.5 * torch.einsum("i,ij,j->", e_field, alpha_cart, e_field)
        e_quad = -(1.0 / 3.0) * torch.einsum("ij,ij->", theta_cart, field_grad)
        return e_dip + e_ind + e_quad

    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.95, 0.15, -0.10], [-0.30, 0.88, 0.25]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    e_field = torch.tensor(
        [0.12, -0.08, 0.20], dtype=torch.float64, device=device, requires_grad=True
    )
    field_grad = torch.tensor(
        [[0.05, -0.02, 0.01], [-0.02, -0.03, 0.04], [0.01, 0.04, -0.02]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    # Raman tensor / polarizability derivative d(alpha_ij)/dR & 2nd derivative
    assert torch.autograd.gradcheck(
        polarizability_from_positions, (pos,), eps=1e-6, atol=1e-5, rtol=1e-5
    )
    assert torch.autograd.gradgradcheck(
        polarizability_from_positions, (pos,), eps=1e-6, atol=1e-5, rtol=1e-5
    )

    # Full polarizable FF energy 1st & 2nd derivatives wrt (R, E, grad_E)
    assert torch.autograd.gradcheck(
        polarization_energy,
        (pos, e_field, field_grad),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.autograd.gradgradcheck(
        polarization_energy,
        (pos, e_field, field_grad),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-5,
    )

    # Verify O(3) rotational invariance of E_pol(R, E, grad_E) and equivariance of forces -dE_pol/dR!
    R_rot = torch.from_numpy(cue.SO3(1).rotation(np.array([0.4, -0.7, 0.5]), 1.1)).to(
        dtype=torch.float64, device=device
    )

    e1 = polarization_energy(pos, e_field, field_grad)
    (f1,) = torch.autograd.grad(-e1, pos, create_graph=True)

    pos_rot = (pos.detach() @ R_rot.T).requires_grad_(True)
    e_field_rot = e_field.detach() @ R_rot.T
    field_grad_rot = R_rot @ field_grad.detach() @ R_rot.T
    e2 = polarization_energy(pos_rot, e_field_rot, field_grad_rot)
    (f2,) = torch.autograd.grad(-e2, pos_rot, create_graph=True)

    torch.testing.assert_close(e1, e2, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(f1 @ R_rot.T, f2, atol=1e-13, rtol=1e-13)
