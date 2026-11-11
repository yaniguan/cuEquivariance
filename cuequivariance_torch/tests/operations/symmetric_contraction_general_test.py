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
from functools import cache

import numpy as np
import pytest
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _o3_representation_matrix(
    irreps: cue.Irreps,
    layout: cue.IrrepsLayout,
    axis: np.ndarray,
    angle: float,
    parity: int,
    target_device: torch.device,
) -> torch.Tensor:
    blocks = []
    for mul, ir in irreps:
        rot = ir.rotation(axis, angle)
        if parity == -1:
            rot = rot * ir.p
        if layout == cue.mul_ir:
            blocks.append(np.kron(np.eye(mul), rot))
        else:
            blocks.append(np.kron(rot, np.eye(mul)))

    mat = np.zeros((irreps.dim, irreps.dim), dtype=np.float64)
    offset = 0
    for block in blocks:
        n = block.shape[0]
        mat[offset : offset + n, offset : offset + n] = block
        offset += n
    return torch.from_numpy(mat).to(device=target_device, dtype=torch.float64)


@cache
def _cached_tapered_layer(
    irreps_in_str: str,
    irreps_out_str: str,
    layout: cue.IrrepsLayout,
    original_mace: bool,
) -> cuet.SymmetricContraction:
    torch.manual_seed(42)
    return cuet.SymmetricContraction(
        cue.Irreps("O3", irreps_in_str),
        cue.Irreps("O3", irreps_out_str),
        contraction_degree=3,
        num_elements=3,
        layout=layout,
        original_mace=original_mace,
        dtype=torch.float64,
        math_dtype=torch.float64,
        device=device,
    )


@pytest.mark.parametrize(
    "irreps_in_str, irreps_out_str, expected_gcd",
    [
        ("64x0e + 32x1o + 16x2e", "64x0e + 32x1o", 16),
        ("4x0e + 2x1o", "2x0e + 4x1o", 2),
    ],
)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("original_mace", [False, True])
def test_tapered_mace_equivariance_and_uniform_1d(
    irreps_in_str: str,
    irreps_out_str: str,
    expected_gcd: int,
    layout: cue.IrrepsLayout,
    original_mace: bool,
):
    layer = _cached_tapered_layer(irreps_in_str, irreps_out_str, layout, original_mace)
    assert layer.method == "uniform_1d"
    assert layer.mul_gcd == expected_gcd
    assert layer.etp.all_same_segment_shape()

    irreps_in = layer.irreps_in
    irreps_out = layer.irreps_out

    gen = torch.Generator(device="cpu").manual_seed(123)
    x = torch.randn((6, irreps_in.dim), generator=gen, dtype=torch.float64).to(device)
    indices = torch.tensor([0, 1, 2, 0, 2, 1], dtype=torch.int64, device=device)

    axis = np.array([0.37, -0.81, 0.45], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = 1.17

    with torch.no_grad():
        for parity in (1, -1):
            d_in = _o3_representation_matrix(
                irreps_in, layout, axis, angle, parity, device
            )
            d_out = _o3_representation_matrix(
                irreps_out, layout, axis, angle, parity, device
            )

            y_rot_in = layer(x @ d_in.T, indices)
            y_rot_out = layer(x, indices) @ d_out.T
            max_err = (y_rot_in - y_rot_out).abs().max().item()
            assert max_err < 1e-10, (
                f"O(3) equivariance error {max_err} >= 1e-10 for "
                f"{irreps_in_str} -> {irreps_out_str}, layout={layout}, "
                f"original_mace={original_mace}, parity={parity}"
            )


@pytest.mark.parametrize(
    "irreps_in_str, irreps_out_str",
    [
        ("64x0e + 32x1o + 16x2e", "64x0e + 32x1o"),
        ("4x0e + 2x1o", "2x0e + 4x1o"),
    ],
)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_original_mace_spans_same_polynomial_space(
    irreps_in_str: str,
    irreps_out_str: str,
    layout: cue.IrrepsLayout,
):
    m_std = _cached_tapered_layer(irreps_in_str, irreps_out_str, layout, False)
    m_mace = _cached_tapered_layer(irreps_in_str, irreps_out_str, layout, True)

    proj = m_mace.projection
    assert proj is not None
    assert torch.linalg.matrix_rank(proj).item() == proj.shape[1]

    gen = torch.Generator(device="cpu").manual_seed(321)
    x = torch.randn((5, m_std.irreps_in.dim), generator=gen, dtype=torch.float64).to(
        device
    )
    indices = torch.tensor([0, 2, 1, 2, 0], dtype=torch.int64, device=device)

    orig_std_w = m_std.weight.data.clone()
    orig_mace_w = m_mace.weight.data.clone()
    try:
        with torch.no_grad():
            # Direction 1: any original_mace=True polynomial is exactly representable by original_mace=False
            w_mace = torch.randn(
                m_mace.weight.shape, generator=gen, dtype=torch.float64
            ).to(device)
            m_mace.weight.data.copy_(w_mace)
            m_std.weight.data.copy_(torch.einsum("zau,ab->zbu", w_mace, proj))
            torch.testing.assert_close(
                m_mace(x, indices), m_std(x, indices), atol=1e-11, rtol=1e-11
            )

            # Direction 2: any original_mace=False polynomial is exactly representable by original_mace=True
            w_std = torch.randn(
                m_std.weight.shape, generator=gen, dtype=torch.float64
            ).to(device)
            proj_pinv = torch.linalg.pinv(proj)
            m_std.weight.data.copy_(w_std)
            m_mace.weight.data.copy_(torch.einsum("zbu,ba->zau", w_std, proj_pinv))
            torch.testing.assert_close(
                m_mace(x, indices), m_std(x, indices), atol=1e-11, rtol=1e-11
            )
    finally:
        m_std.weight.data.copy_(orig_std_w)
        m_mace.weight.data.copy_(orig_mace_w)


def test_bf16_projection_precision():
    irreps_in = cue.Irreps("O3", "4x0e + 2x1o")
    irreps_out = cue.Irreps("O3", "2x0e + 4x1o")

    torch.manual_seed(7)
    layer_bf16 = cuet.SymmetricContraction(
        irreps_in,
        irreps_out,
        contraction_degree=3,
        num_elements=2,
        layout=cue.mul_ir,
        original_mace=True,
        dtype=torch.bfloat16,
        math_dtype=torch.float32,
        device=device,
    )
    assert layer_bf16.weight.dtype == torch.bfloat16
    assert layer_bf16.projection is not None
    assert layer_bf16.projection.dtype == torch.float32

    layer_fp64 = cuet.SymmetricContraction(
        irreps_in,
        irreps_out,
        contraction_degree=3,
        num_elements=2,
        layout=cue.mul_ir,
        original_mace=True,
        dtype=torch.float64,
        math_dtype=torch.float64,
        device=device,
    )
    assert layer_fp64.projection.dtype == torch.float64
    layer_fp64.weight.data.copy_(layer_bf16.weight.detach().to(torch.float64))

    # Compare FP32-projected BF16 weights vs naively BF16-quantized projection
    exact_proj_w = torch.einsum(
        "...au,ab->...bu",
        layer_bf16.weight.detach().to(torch.float64),
        layer_fp64.projection,
    )
    fp32_proj_w = (
        torch.einsum(
            "...au,ab->...bu",
            layer_bf16.weight.detach().to(torch.float32),
            layer_bf16.projection,
        )
        .to(torch.bfloat16)
        .to(torch.float64)
    )
    naive_bf16_proj_w = torch.einsum(
        "...au,ab->...bu",
        layer_bf16.weight.detach(),
        layer_fp64.projection.to(torch.bfloat16),
    ).to(torch.float64)

    err_fp32_proj = (fp32_proj_w - exact_proj_w).abs().mean().item()
    err_naive_bf16 = (naive_bf16_proj_w - exact_proj_w).abs().mean().item()
    assert err_fp32_proj < err_naive_bf16

    x_bf16 = torch.randn(
        (8, irreps_in.dim), dtype=torch.bfloat16, device=device, requires_grad=True
    )
    indices = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1], dtype=torch.int64, device=device)
    out_bf16 = layer_bf16(x_bf16, indices)
    assert out_bf16.dtype == torch.bfloat16
    out_bf16.sum().backward()
    assert x_bf16.grad is not None
    assert layer_bf16.weight.grad is not None


@pytest.mark.parametrize("original_mace", [False, True])
def test_single_species_indices_none_fast_path(original_mace: bool):
    irreps_in = cue.Irreps("O3", "4x0e + 2x1o")
    irreps_out = cue.Irreps("O3", "2x0e + 4x1o")

    torch.manual_seed(19)
    layer = cuet.SymmetricContraction(
        irreps_in,
        irreps_out,
        contraction_degree=3,
        layout=cue.mul_ir,
        original_mace=original_mace,
        dtype=torch.float64,
        math_dtype=torch.float64,
        device=device,
    )
    assert layer.num_elements == 1

    x1 = torch.randn(
        (7, irreps_in.dim), dtype=torch.float64, device=device, requires_grad=True
    )
    x2 = x1.detach().clone().requires_grad_(True)
    zero_indices = torch.zeros(7, dtype=torch.int64, device=device)

    out_none = layer(x1)
    out_zeros = layer(x2, zero_indices)
    torch.testing.assert_close(out_none, out_zeros, atol=1e-12, rtol=1e-12)

    grad_out = torch.randn_like(out_none)
    layer.zero_grad()
    out_none.backward(grad_out)
    w_grad_none = layer.weight.grad.detach().clone()

    layer.zero_grad()
    out_zeros.backward(grad_out)
    w_grad_zeros = layer.weight.grad.detach().clone()

    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(w_grad_none, w_grad_zeros, atol=1e-12, rtol=1e-12)

    # Multi-element layer without indices must raise ValueError
    layer_multi = cuet.SymmetricContraction(
        irreps_in,
        irreps_out,
        contraction_degree=3,
        num_elements=3,
        layout=cue.mul_ir,
        dtype=torch.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="indices"):
        layer_multi(x1.detach())


@pytest.mark.parametrize("original_mace", [False, True])
def test_multibatch_mask_and_higher_order_autograd(original_mace: bool):
    irreps_in = cue.Irreps("O3", "4x0e + 2x1o")
    irreps_out = cue.Irreps("O3", "2x0e + 4x1o")

    torch.manual_seed(99)
    layer = cuet.SymmetricContraction(
        irreps_in,
        irreps_out,
        contraction_degree=3,
        num_elements=3,
        layout=cue.mul_ir,
        original_mace=original_mace,
        dtype=torch.float64,
        math_dtype=torch.float64,
        device=device,
    )

    mask = torch.tensor(
        [[True, True, True, False, False], [True, False, True, True, False]],
        dtype=torch.bool,
        device=device,
    )
    indices = torch.tensor(
        [[0, 2, 1, -1, -1], [2, -1, 0, 1, -99]],
        dtype=torch.int64,
        device=device,
    )

    x = torch.randn((2, 5, irreps_in.dim), dtype=torch.float64, device=device)
    # Put NaNs on masked padding slots to verify clean masking in forward & backward
    x_with_nan_pad = x.clone()
    x_with_nan_pad[~mask] = float("nan")
    x_with_nan_pad.requires_grad_(True)

    out = layer(x_with_nan_pad, indices, mask=mask)
    assert out.shape == (2, 5, irreps_out.dim)
    assert not torch.isnan(out).any()
    assert torch.all(out[~mask] == 0.0)

    # Compare active atoms against unpadded 2D execution
    out_unpadded = layer(x[mask], indices[mask])
    torch.testing.assert_close(out[mask], out_unpadded, atol=1e-12, rtol=1e-12)

    loss = out.pow(2).sum()
    loss.backward()
    assert x_with_nan_pad.grad is not None
    assert not torch.isnan(x_with_nan_pad.grad).any()
    assert torch.all(x_with_nan_pad.grad[~mask] == 0.0)
    assert layer.weight.grad is not None
    assert not torch.isnan(layer.weight.grad).any()

    # Check TorchScript compatibility with multi-batch, sentinel indices, and mask
    scripted = torch.jit.script(layer)
    x_clean = x.clone()
    torch.testing.assert_close(
        layer(x_clean, indices, mask=mask),
        scripted(x_clean, indices, mask=mask),
        atol=1e-12,
        rtol=1e-12,
    )

    # 1st and 2nd order finite-difference autograd checks (gradcheck & gradgradcheck)
    x_gc = (
        0.5 * torch.randn((2, 5, irreps_in.dim), dtype=torch.float64, device=device)
    ).requires_grad_(True)
    w_gc = layer.weight.detach().clone().requires_grad_(True)

    def func(inp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(layer, {"weight": w}, (inp, indices, mask))

    assert torch.autograd.gradcheck(
        lambda inp: layer(inp, indices, mask=mask),
        (x_gc,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.autograd.gradgradcheck(
        lambda inp: layer(inp, indices, mask=mask),
        (x_gc,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.autograd.gradcheck(
        func, (x_gc, w_gc), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True
    )
    assert torch.autograd.gradgradcheck(
        func, (x_gc, w_gc), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True
    )

    # Shape validation checks
    with pytest.raises(ValueError, match="last dimension"):
        layer(
            torch.randn((2, 5, irreps_in.dim + 1), dtype=torch.float64, device=device),
            indices,
        )
    with pytest.raises(ValueError, match="indices"):
        layer(x_clean, torch.zeros((2, 4), dtype=torch.int64, device=device))
    with pytest.raises(ValueError, match="mask"):
        layer(
            x_clean, indices, mask=torch.ones((2, 4), dtype=torch.bool, device=device)
        )
