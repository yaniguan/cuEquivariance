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
import torch
import torch.nn as nn
import torch.nn.functional as F
from cuequivariance_torch.layers import (
    EquivariantLayerNorm as LayersEquivariantLayerNorm,
)
from cuequivariance_torch.layers import (
    EquivariantRMSNorm as LayersEquivariantRMSNorm,
)
from cuequivariance_torch.operations import Gate as OperationsGate

import cuequivariance as cue
import cuequivariance_torch as cuet


def _o3_matrix(
    irreps: cue.Irreps,
    layout: cue.IrrepsLayout,
    angles: tuple[float, float, float],
    parity: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    rep = cue.IrrepsAndLayout(irreps, layout)
    mat = rep.exp_map(np.array(angles, dtype=np.float64), np.array([parity]))
    return torch.from_numpy(mat).to(dtype=dtype)


def test_exports_and_zero_transpose_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cuet.Gate is OperationsGate
    assert cuet.operations.Gate is OperationsGate
    assert cuet.EquivariantRMSNorm is LayersEquivariantRMSNorm
    assert cuet.layers.EquivariantRMSNorm is LayersEquivariantRMSNorm
    assert cuet.EquivariantLayerNorm is LayersEquivariantLayerNorm
    assert cuet.layers.EquivariantLayerNorm is LayersEquivariantLayerNorm

    def _forbid_transpose(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Runtime layout transpose is forbidden!")

    monkeypatch.setattr(cuet.TransposeIrrepsLayout, "forward", _forbid_transpose)
    monkeypatch.setattr(cuet.TransposeSegments, "forward", _forbid_transpose)
    monkeypatch.setattr(torch, "transpose", _forbid_transpose)
    monkeypatch.setattr(torch.Tensor, "transpose", _forbid_transpose)
    monkeypatch.setattr(torch.Tensor, "permute", _forbid_transpose)

    for layout in (cue.mul_ir, cue.ir_mul):
        gate = cuet.Gate(
            "4x0e",
            [F.silu],
            "6x0e",
            [torch.sigmoid],
            "2x0o + 2x1o + 2x2e",
            group=cue.O3,
            layout=layout,
        )
        rms = cuet.EquivariantRMSNorm(
            "4x0e + 2x0o + 2x1o + 2x2e", group="O3", layout=layout
        )
        ln = cuet.EquivariantLayerNorm(
            "4x0e + 2x0o + 2x1o + 2x2e", group="O3", layout=layout
        )
        assert gate.irreps_in == cue.Irreps("O3", "4x0e + 6x0e + 2x0o + 2x1o + 2x2e")
        assert gate.irreps_out == cue.Irreps("O3", "4x0e + 2x0o + 2x1o + 2x2e")

        for mod in (gate, rms, ln):
            assert repr(mod)
            for sub in mod.modules():
                assert not isinstance(
                    sub, (cuet.TransposeIrrepsLayout, cuet.TransposeSegments)
                )

        x = torch.randn(4, gate.irreps_in.dim, requires_grad=True)
        out = ln(rms(gate(x)))
        out.sum().backward()
        assert x.grad is not None


def test_so3_group_and_context_scope() -> None:
    with cue.assume("SO3", cue.ir_mul):
        gate = cuet.Gate("3x0", [torch.tanh], "4x0", [torch.sigmoid], "2x1 + 2x2")
        rms = cuet.EquivariantRMSNorm("3x0 + 2x1 + 2x2")
        ln = cuet.EquivariantLayerNorm("3x0 + 2x1 + 2x2")

    assert gate.irreps_in.irrep_class == cue.SO3
    assert gate.layout == cue.ir_mul
    x = torch.randn(5, gate.irreps_in.dim, dtype=torch.float64)
    y = ln(rms(gate(x)))
    assert y.shape == (5, gate.irreps_out.dim)


def test_validation_errors() -> None:
    # Non-scalar in irreps_scalars (0o is pseudoscalar, not scalar!)
    with pytest.raises(ValueError, match="irreps_scalars must be scalar"):
        cuet.Gate(
            "2x0o",
            [torch.tanh],
            "2x0e",
            [torch.sigmoid],
            "2x1o",
            group="O3",
            layout=cue.mul_ir,
        )

    # Non-scalar in irreps_gates
    with pytest.raises(ValueError, match="irreps_gates must be scalar"):
        cuet.Gate(
            "2x0e",
            [torch.tanh],
            "2x1o",
            [torch.sigmoid],
            "2x1o",
            group="O3",
            layout=cue.mul_ir,
        )

    # Multiplicity mismatch between gates and gated
    with pytest.raises(ValueError, match="Total multiplicity"):
        cuet.Gate(
            "2x0e",
            [torch.tanh],
            "3x0e",
            [torch.sigmoid],
            "2x1o",
            group="O3",
            layout=cue.mul_ir,
        )

    # Activation list length mismatch
    with pytest.raises(ValueError, match="act_scalars"):
        cuet.Gate(
            "2x0e + 2x0e",
            [torch.tanh],
            "2x0e",
            [torch.sigmoid],
            "2x1o",
            group="O3",
            layout=cue.mul_ir,
        )

    with pytest.raises(ValueError, match="act_gates"):
        cuet.Gate(
            "2x0e",
            [torch.tanh],
            "1x0e + 1x0e",
            [torch.sigmoid],
            "2x1o",
            group="O3",
            layout=cue.mul_ir,
        )

    # Norm argument validation
    with pytest.raises(ValueError, match="norm_mode"):
        cuet.EquivariantRMSNorm(
            "4x0e", group="O3", layout=cue.mul_ir, norm_mode="invalid"
        )

    with pytest.raises(ValueError, match="normalization"):
        cuet.EquivariantRMSNorm(
            "4x0e", group="O3", layout=cue.mul_ir, normalization="invalid"
        )

    with pytest.raises(ValueError, match="eps"):
        cuet.EquivariantLayerNorm("4x0e", group="O3", layout=cue.mul_ir, eps=-1e-5)


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("parity", [0, 1])
def test_gate_o3_rotation_and_parity_equivariance(
    layout: cue.IrrepsLayout, parity: int
) -> None:
    torch.manual_seed(42)
    irreps_scalars = cue.Irreps("O3", "3x0e + 2x0e")
    irreps_gates = cue.Irreps("O3", "6x0e + 6x0e")
    irreps_gated = cue.Irreps("O3", "3x0o + 3x1o + 3x1e + 3x2e")

    gate = cuet.Gate(
        irreps_scalars,
        [F.silu, None],
        irreps_gates,
        [torch.sigmoid, torch.tanh],
        irreps_gated,
        layout=layout,
        dtype=torch.float64,
    )

    angles = (0.73, -1.19, 0.41)
    d_in = _o3_matrix(gate.irreps_in, layout, angles, parity)
    d_out = _o3_matrix(gate.irreps_out, layout, angles, parity)

    x = torch.randn(7, gate.irreps_in.dim, dtype=torch.float64)
    out_rot = gate(x @ d_in.T)
    expected = gate(x) @ d_out.T
    torch.testing.assert_close(out_rot, expected, atol=1e-13, rtol=1e-13)

    # Two-input mode equivariance
    sg_dim = irreps_scalars.dim + irreps_gates.dim
    d_sg = _o3_matrix(irreps_scalars + irreps_gates, layout, angles, parity)
    d_gated = _o3_matrix(irreps_gated, layout, angles, parity)
    x_sg = x[:, :sg_dim]
    x_gated = x[:, sg_dim:]
    out_two = gate(x_sg @ d_sg.T, x_gated @ d_gated.T)
    torch.testing.assert_close(out_two, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("norm_mode", ["per_irrep", "per_channel", "global"])
@pytest.mark.parametrize("normalization", ["component", "norm"])
@pytest.mark.parametrize("centering", [False, True])
@pytest.mark.parametrize("parity", [0, 1])
def test_equivariant_rmsnorm_o3_equivariance(
    layout: cue.IrrepsLayout,
    norm_mode: str,
    normalization: str,
    centering: bool,
    parity: int,
) -> None:
    torch.manual_seed(101)
    irreps = cue.Irreps("O3", "4x0e + 3x0o + 3x1o + 2x1e + 3x2e")
    norm = cuet.EquivariantRMSNorm(
        irreps,
        centering=centering,
        norm_mode=norm_mode,
        normalization=normalization,
        affine=True,
        layout=layout,
        dtype=torch.float64,
    )
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))

    angles = (-0.61, 0.94, 1.37)
    d_mat = _o3_matrix(irreps, layout, angles, parity)

    x = torch.randn(6, irreps.dim, dtype=torch.float64) + 1.5
    out_rot = norm(x @ d_mat.T)
    expected = norm(x) @ d_mat.T
    torch.testing.assert_close(out_rot, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("norm_mode", ["per_irrep", "per_channel", "global"])
@pytest.mark.parametrize("normalization", ["component", "norm"])
@pytest.mark.parametrize("parity", [0, 1])
def test_equivariant_layernorm_o3_equivariance(
    layout: cue.IrrepsLayout,
    norm_mode: str,
    normalization: str,
    parity: int,
) -> None:
    torch.manual_seed(202)
    irreps = cue.Irreps("O3", "4x0e + 3x0o + 3x1o + 2x1e + 3x2e")
    ln = cuet.EquivariantLayerNorm(
        irreps,
        norm_mode=norm_mode,
        normalization=normalization,
        affine=True,
        bias=True,
        layout=layout,
        dtype=torch.float64,
    )
    with torch.no_grad():
        ln.weight.copy_(torch.randn_like(ln.weight))
        ln.bias.copy_(torch.randn_like(ln.bias) + 2.0)

    angles = (1.12, -0.48, -0.85)
    d_mat = _o3_matrix(irreps, layout, angles, parity)

    x = torch.randn(6, irreps.dim, dtype=torch.float64) + 2.5
    out_rot = ln(x @ d_mat.T)
    expected = ln(x) @ d_mat.T
    torch.testing.assert_close(out_rot, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
def test_pseudoscalar_0o_parity_sign_flip_and_no_bias_or_mean_shift(
    layout: cue.IrrepsLayout,
) -> None:
    torch.manual_seed(303)
    irreps = cue.Irreps("O3", "4x0e + 4x0o + 2x1o")
    ln = cuet.EquivariantLayerNorm(
        irreps, layout=layout, dtype=torch.float64, eps=1e-12
    )
    # Bias must exist ONLY on the 4x0e scalars, NEVER on 4x0o or 2x1o
    assert ln.bias.shape == (4,)
    with torch.no_grad():
        ln.weight.fill_(1.5)
        ln.bias.fill_(3.25)

    # 1. At exact zero input, 0e outputs bias (3.25), while 0o and 1o are EXACT 0.0
    x_zero = torch.zeros(3, irreps.dim, dtype=torch.float64)
    out_zero = ln(x_zero)
    torch.testing.assert_close(
        out_zero[:, :4], torch.full((3, 4), 3.25, dtype=torch.float64)
    )
    assert torch.all(out_zero[:, 4:] == 0.0)

    # 2. Constant non-zero 0o channels must NOT be zeroed by mean-centering
    x_const_0o = torch.zeros(2, irreps.dim, dtype=torch.float64)
    x_const_0o[:, 4:8] = 5.0
    out_const_0o = ln(x_const_0o)
    torch.testing.assert_close(
        out_const_0o[:, 4:8],
        torch.full((2, 4), 1.5, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )

    # 3. Under pure parity inversion, 0e is invariant while 0o and 1o flip sign
    inv = cuet.Inversion(irreps, layout=layout, method="naive")
    x = torch.randn(5, irreps.dim, dtype=torch.float64) + 4.0
    y = ln(x)
    y_of_inv = ln(inv(x))
    torch.testing.assert_close(y_of_inv[:, :4], y[:, :4], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(y_of_inv[:, 4:8], -y[:, 4:8], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(y_of_inv[:, 8:], -y[:, 8:], atol=1e-13, rtol=1e-13)


def test_gate_layout_equivalence_fp64() -> None:
    torch.manual_seed(404)
    irreps_s = cue.Irreps("O3", "4x0e")
    irreps_g = cue.Irreps("O3", "7x0e")
    irreps_v = cue.Irreps("O3", "2x0o + 3x1o + 2x2e")

    g_mul = cuet.Gate(
        irreps_s, [F.silu], irreps_g, [torch.sigmoid], irreps_v, layout=cue.mul_ir
    )
    g_ir = cuet.Gate(
        irreps_s, [F.silu], irreps_g, [torch.sigmoid], irreps_v, layout=cue.ir_mul
    )

    to_ir = cuet.TransposeIrrepsLayout(
        g_mul.irreps_in, source=cue.mul_ir, target=cue.ir_mul, use_fallback=True
    )
    to_mul = cuet.TransposeIrrepsLayout(
        g_mul.irreps_out, source=cue.ir_mul, target=cue.mul_ir, use_fallback=True
    )

    x_mul = torch.randn(8, g_mul.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    x_ir = to_ir(x_mul.detach()).clone().requires_grad_(True)

    y_mul = g_mul(x_mul)
    y_ir_as_mul = to_mul(g_ir(x_ir))
    assert (y_mul - y_ir_as_mul).abs().max().item() < 1e-12

    cot_mul = torch.randn_like(y_mul)
    cot_ir = to_ir_out = cuet.TransposeIrrepsLayout(
        g_mul.irreps_out, source=cue.mul_ir, target=cue.ir_mul, use_fallback=True
    )(cot_mul)
    (y_mul * cot_mul).sum().backward()
    (g_ir(x_ir) * cot_ir).sum().backward()

    grad_ir_as_mul = cuet.TransposeIrrepsLayout(
        g_mul.irreps_in, source=cue.ir_mul, target=cue.mul_ir, use_fallback=True
    )(x_ir.grad)
    assert (x_mul.grad - grad_ir_as_mul).abs().max().item() < 1e-12
    _ = to_ir_out


@pytest.mark.parametrize("norm_mode", ["per_irrep", "per_channel", "global"])
@pytest.mark.parametrize("normalization", ["component", "norm"])
@pytest.mark.parametrize("cls_type", ["rms", "rms_centered", "ln"])
def test_norm_layout_equivalence_fp64(
    norm_mode: str, normalization: str, cls_type: str
) -> None:
    torch.manual_seed(505)
    irreps = cue.Irreps("O3", "4x0e + 2x0o + 3x1o + 2x1e + 3x2e")

    if cls_type == "rms":
        m_mul = cuet.EquivariantRMSNorm(
            irreps,
            norm_mode=norm_mode,
            normalization=normalization,
            centering=False,
            layout=cue.mul_ir,
            dtype=torch.float64,
        )
        m_ir = cuet.EquivariantRMSNorm(
            irreps,
            norm_mode=norm_mode,
            normalization=normalization,
            centering=False,
            layout=cue.ir_mul,
            dtype=torch.float64,
        )
    elif cls_type == "rms_centered":
        m_mul = cuet.EquivariantRMSNorm(
            irreps,
            norm_mode=norm_mode,
            normalization=normalization,
            centering=True,
            layout=cue.mul_ir,
            dtype=torch.float64,
        )
        m_ir = cuet.EquivariantRMSNorm(
            irreps,
            norm_mode=norm_mode,
            normalization=normalization,
            centering=True,
            layout=cue.ir_mul,
            dtype=torch.float64,
        )
    else:
        m_mul = cuet.EquivariantLayerNorm(
            irreps,
            norm_mode=norm_mode,
            normalization=normalization,
            layout=cue.mul_ir,
            dtype=torch.float64,
        )
        m_ir = cuet.EquivariantLayerNorm(
            irreps,
            norm_mode=norm_mode,
            normalization=normalization,
            layout=cue.ir_mul,
            dtype=torch.float64,
        )

    with torch.no_grad():
        w = torch.randn_like(m_mul.weight)
        m_mul.weight.copy_(w)
        m_ir.weight.copy_(w)
        if m_mul.bias is not None:
            b = torch.randn_like(m_mul.bias)
            m_mul.bias.copy_(b)
            m_ir.bias.copy_(b)

    to_ir = cuet.TransposeIrrepsLayout(
        irreps, source=cue.mul_ir, target=cue.ir_mul, use_fallback=True
    )
    to_mul = cuet.TransposeIrrepsLayout(
        irreps, source=cue.ir_mul, target=cue.mul_ir, use_fallback=True
    )

    x_mul = torch.randn(9, irreps.dim, dtype=torch.float64, requires_grad=True)
    x_ir = to_ir(x_mul.detach()).clone().requires_grad_(True)

    y_mul = m_mul(x_mul)
    y_ir = m_ir(x_ir)
    assert (y_mul - to_mul(y_ir)).abs().max().item() < 1e-12

    cot_mul = torch.randn_like(y_mul)
    cot_ir = to_ir(cot_mul)
    (y_mul * cot_mul).sum().backward()
    (y_ir * cot_ir).sum().backward()

    assert (x_mul.grad - to_mul(x_ir.grad)).abs().max().item() < 1e-12
    assert (m_mul.weight.grad - m_ir.weight.grad).abs().max().item() < 1e-12
    if m_mul.bias is not None:
        assert (m_mul.bias.grad - m_ir.bias.grad).abs().max().item() < 1e-12


def test_scalar_equivalence_with_pytorch_norms() -> None:
    torch.manual_seed(606)
    irreps = cue.Irreps("O3", "16x0e")
    x = torch.randn(5, 16, dtype=torch.float64) * 3.0 + 1.7

    eq_rms = cuet.EquivariantRMSNorm(
        irreps, eps=1e-6, layout=cue.mul_ir, dtype=torch.float64
    )
    pt_rms = nn.RMSNorm(16, eps=1e-6, dtype=torch.float64)
    with torch.no_grad():
        w = torch.randn(16, dtype=torch.float64)
        eq_rms.weight.copy_(w)
        pt_rms.weight.copy_(w)
    torch.testing.assert_close(eq_rms(x), pt_rms(x), atol=1e-13, rtol=1e-13)

    eq_ln = cuet.EquivariantLayerNorm(
        irreps, eps=1e-6, layout=cue.mul_ir, dtype=torch.float64
    )
    pt_ln = nn.LayerNorm(16, eps=1e-6, dtype=torch.float64)
    with torch.no_grad():
        b = torch.randn(16, dtype=torch.float64)
        eq_ln.weight.copy_(w)
        pt_ln.weight.copy_(w)
        eq_ln.bias.copy_(b)
        pt_ln.bias.copy_(b)
    torch.testing.assert_close(eq_ln(x), pt_ln(x), atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("eps", [0.0, 1e-6])
@pytest.mark.parametrize("norm_mode", ["per_irrep", "per_channel", "global"])
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
def test_zero_vector_double_where_1st_and_2nd_derivatives(
    eps: float, norm_mode: str, layout: cue.IrrepsLayout
) -> None:
    torch.manual_seed(707)
    irreps = cue.Irreps("O3", "3x0e + 2x0o + 2x1o + 2x2e")

    rms = cuet.EquivariantRMSNorm(
        irreps, eps=eps, norm_mode=norm_mode, layout=layout, dtype=torch.float64
    )
    ln = cuet.EquivariantLayerNorm(
        irreps, eps=eps, norm_mode=norm_mode, layout=layout, dtype=torch.float64
    )

    for mod in (rms, ln):
        x_zero = torch.zeros(4, irreps.dim, dtype=torch.float64, requires_grad=True)
        out = mod(x_zero)
        assert torch.isfinite(out).all()
        assert torch.all(out == 0.0)

        cot1 = torch.randn_like(out)
        (g1,) = torch.autograd.grad((out * cot1).sum(), x_zero, create_graph=True)
        assert torch.isfinite(g1).all()
        assert torch.all(g1 == 0.0)

        cot2 = torch.randn_like(g1)
        (g2,) = torch.autograd.grad((g1 * cot2).sum(), x_zero)
        assert torch.isfinite(g2).all()
        assert torch.all(g2 == 0.0)


@pytest.mark.parametrize("eps", [0.0, 1e-5])
@pytest.mark.parametrize("norm_mode", ["per_irrep", "per_channel", "global"])
def test_gradcheck_and_gradgradcheck_including_isolated_zero_atom(
    eps: float, norm_mode: str
) -> None:
    torch.manual_seed(808)
    irreps = cue.Irreps("O3", "3x0e + 2x0o + 2x1o")

    rms = cuet.EquivariantRMSNorm(
        irreps, eps=eps, norm_mode=norm_mode, layout=cue.ir_mul, dtype=torch.float64
    )
    ln = cuet.EquivariantLayerNorm(
        irreps, eps=eps, norm_mode=norm_mode, layout=cue.ir_mul, dtype=torch.float64
    )

    # Atom 0 is an isolated atom with exact zero neighbor sum; atoms 1..2 have non-zero features
    atom_active = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float64)
    z = torch.randn(3, irreps.dim, dtype=torch.float64, requires_grad=True)

    for mod in (rms, ln):

        def fn(inp: torch.Tensor) -> torch.Tensor:
            return mod(inp * atom_active)

        assert torch.autograd.gradcheck(fn, (z,), eps=1e-6, atol=1e-8, rtol=1e-6)
        assert torch.autograd.gradgradcheck(fn, (z,), eps=1e-6, atol=1e-7, rtol=1e-5)

    # Gate gradcheck & gradgradcheck
    gate = cuet.Gate(
        "2x0e",
        [F.silu],
        "4x0e",
        [torch.sigmoid],
        "2x0o + 2x1o",
        group="O3",
        layout=cue.ir_mul,
    )
    zg = torch.randn(3, gate.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(gate, (zg,), eps=1e-6, atol=1e-8, rtol=1e-6)
    assert torch.autograd.gradgradcheck(gate, (zg,), eps=1e-6, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize(
    "low_dtype,scale,tol",
    [
        (torch.bfloat16, 500.0, {"atol": 3e-2, "rtol": 2e-2}),
        (torch.float16, 300.0, {"atol": 5e-3, "rtol": 5e-3}),
    ],
)
@pytest.mark.parametrize("norm_mode", ["per_irrep", "per_channel", "global"])
def test_bf16_and_fp16_large_activation_overflow_stability(
    low_dtype: torch.dtype,
    scale: float,
    tol: dict[str, float],
    norm_mode: str,
) -> None:
    torch.manual_seed(909)
    irreps = cue.Irreps("O3", "8x0e + 4x1o + 4x2e")

    x_fp64 = scale * torch.randn(4, irreps.dim, dtype=torch.float64)
    x_low = x_fp64.to(dtype=low_dtype).detach().requires_grad_(True)
    # Confirm naive squaring in FP16 genuinely overflows > 65504 -> inf
    if low_dtype == torch.float16:
        assert torch.isinf(x_low.detach() ** 2).any()

    x_ref = x_low.detach().to(dtype=torch.float64).requires_grad_(True)

    for cls in (cuet.EquivariantRMSNorm, cuet.EquivariantLayerNorm):
        mod_low = cls(
            irreps, norm_mode=norm_mode, layout=cue.ir_mul, dtype=torch.float32
        )
        mod_ref = cls(
            irreps, norm_mode=norm_mode, layout=cue.ir_mul, dtype=torch.float64
        )
        with torch.no_grad():
            w = torch.randn_like(mod_ref.weight)
            mod_ref.weight.copy_(w)
            mod_low.weight.copy_(w.to(torch.float32))
            if mod_ref.bias is not None:
                b = torch.randn_like(mod_ref.bias)
                mod_ref.bias.copy_(b)
                mod_low.bias.copy_(b.to(torch.float32))

        x_low_in = x_low.detach().clone().requires_grad_(True)
        x_ref_in = x_ref.detach().clone().requires_grad_(True)

        out_low = mod_low(x_low_in)
        out_ref = mod_ref(x_ref_in)

        assert out_low.dtype == low_dtype
        assert torch.isfinite(out_low).all()
        torch.testing.assert_close(out_low.to(torch.float64), out_ref, **tol)

        out_low.sum().backward()
        out_ref.sum().backward()
        assert torch.isfinite(x_low_in.grad).all()
        assert torch.isfinite(mod_low.weight.grad).all()


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
def test_multibatch_mask_isolation_and_compile(layout: cue.IrrepsLayout) -> None:
    torch.manual_seed(1010)
    irreps_s = cue.Irreps("O3", "4x0e")
    irreps_g = cue.Irreps("O3", "5x0e")
    irreps_v = cue.Irreps("O3", "2x0o + 3x1o")

    # Use sigmoid on scalars so unmasked f(0) = 0.5 != 0, testing post-mask zeroing
    gate = cuet.Gate(
        irreps_s,
        [torch.sigmoid],
        irreps_g,
        [torch.sigmoid],
        irreps_v,
        layout=layout,
        dtype=torch.float64,
    )
    ln = cuet.EquivariantLayerNorm(gate.irreps_out, layout=layout, dtype=torch.float64)
    rms = cuet.EquivariantRMSNorm(gate.irreps_out, layout=layout, dtype=torch.float64)
    with torch.no_grad():
        ln.bias.fill_(1.75)

    batch_shape = (2, 4, 5)
    mask = torch.rand(batch_shape) > 0.35
    mask[0, 0, 0] = True
    mask[0, 0, 1] = False

    x_clean = torch.randn(*batch_shape, gate.irreps_in.dim, dtype=torch.float64)
    # Inject NaN and Inf into masked-out padding positions!
    x_dirty = x_clean.clone()
    x_dirty[~mask] = float("nan")
    x_dirty[0, 0, 1, 0] = float("inf")
    x_dirty.requires_grad_(True)

    y_gate = gate(x_dirty, mask=mask)
    assert y_gate.shape == (*batch_shape, gate.irreps_out.dim)
    assert torch.isfinite(y_gate).all()
    assert torch.all(y_gate[~mask] == 0.0)

    # Compare valid positions against running strictly on valid rows
    x_valid = x_clean[mask].detach().clone().requires_grad_(True)
    y_gate_valid = gate(x_valid)
    torch.testing.assert_close(y_gate[mask], y_gate_valid, atol=1e-13, rtol=1e-13)

    y_gate.sum().backward()
    y_gate_valid.sum().backward()
    assert torch.isfinite(x_dirty.grad).all()
    assert torch.all(x_dirty.grad[~mask] == 0.0)
    torch.testing.assert_close(x_dirty.grad[mask], x_valid.grad, atol=1e-13, rtol=1e-13)

    # Mask isolation for EquivariantLayerNorm (with non-zero bias!) and EquivariantRMSNorm
    for norm_mod in (ln, rms):
        norm_mod.zero_grad(set_to_none=True)
        z_clean = torch.randn(*batch_shape, gate.irreps_out.dim, dtype=torch.float64)
        z_dirty = z_clean.clone()
        z_dirty[~mask] = float("nan")
        z_dirty.requires_grad_(True)

        out_masked = norm_mod(z_dirty, mask=mask)
        assert out_masked.shape == (*batch_shape, gate.irreps_out.dim)
        assert torch.isfinite(out_masked).all()
        assert torch.all(out_masked[~mask] == 0.0)

        out_masked.sum().backward()
        assert torch.isfinite(z_dirty.grad).all()
        assert torch.all(z_dirty.grad[~mask] == 0.0)
        w_grad_masked = norm_mod.weight.grad.clone()
        b_grad_masked = (
            norm_mod.bias.grad.clone() if norm_mod.bias is not None else None
        )

        norm_mod.zero_grad(set_to_none=True)
        z_valid = z_clean[mask].detach().clone().requires_grad_(True)
        out_valid = norm_mod(z_valid)
        torch.testing.assert_close(out_masked[mask], out_valid, atol=1e-13, rtol=1e-13)
        out_valid.sum().backward()
        torch.testing.assert_close(
            z_dirty.grad[mask], z_valid.grad, atol=1e-13, rtol=1e-13
        )
        torch.testing.assert_close(
            w_grad_masked, norm_mod.weight.grad, atol=1e-12, rtol=1e-12
        )
        if b_grad_masked is not None:
            torch.testing.assert_close(
                b_grad_masked, norm_mod.bias.grad, atol=1e-12, rtol=1e-12
            )

    # 1D unbatched & 0-sized empty batch support
    x_1d = torch.randn(gate.irreps_in.dim, dtype=torch.float64)
    assert ln(gate(x_1d)).shape == (gate.irreps_out.dim,)
    x_empty = torch.zeros(0, 4, gate.irreps_in.dim, dtype=torch.float64)
    assert ln(gate(x_empty)).shape == (0, 4, gate.irreps_out.dim)

    # torch.compile(backend="aot_eager", fullgraph=True) & torch.jit.script
    class Pipeline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = gate
            self.rms = rms
            self.ln = ln

        def forward(self, inp: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            h = self.gate(inp, mask=m)
            h = self.rms(h, mask=m)
            return self.ln(h, mask=m)

    pipe = Pipeline()
    x_comp = x_clean.clone().requires_grad_(True)
    expected_out = pipe(x_comp, mask)
    expected_out.square().sum().backward()
    expected_grad = x_comp.grad.clone()

    compiled_pipe = torch.compile(pipe, backend="aot_eager", fullgraph=True)
    x_comp2 = x_clean.clone().requires_grad_(True)
    comp_out = compiled_pipe(x_comp2, mask)
    comp_out.square().sum().backward()
    torch.testing.assert_close(comp_out, expected_out, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(x_comp2.grad, expected_grad, atol=1e-13, rtol=1e-13)

    scripted_pipe = torch.jit.script(pipe)
    torch.testing.assert_close(
        scripted_pipe(x_clean, mask), expected_out, atol=1e-13, rtol=1e-13
    )
