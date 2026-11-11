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
import pytest
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet


def test_equivariant() -> None:
    irreps = cue.Irreps("O3", "3x0e + 3x0o + 4x1e")
    m = cuet.layers.BatchNorm(irreps, layout=cue.mul_ir)
    m(torch.randn(16, irreps.dim))
    m(torch.randn(16, irreps.dim))
    m.train()


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("instance", [True, False])
def test_modes(layout, affine, reduce, instance) -> None:
    irreps = cue.Irreps("O3", "10x0e + 5x1e")

    m = cuet.layers.BatchNorm(
        irreps,
        layout=layout,
        affine=affine,
        reduce=reduce,
        instance=instance,
    )
    repr(m)

    m.train()
    m(torch.randn(20, 20, irreps.dim))

    m.eval()
    m(torch.randn(20, 20, irreps.dim))


@pytest.mark.parametrize("instance", [True, False])
def test_normalization(instance) -> None:
    float_tolerance = 1e-5
    sqrt_float_tolerance = float_tolerance**0.5

    batch, n = 20, 20
    irreps = cue.Irreps("O3", "3x0e + 4x1e")

    m = cuet.layers.BatchNorm(irreps, layout=cue.mul_ir, instance=instance)

    x = torch.randn(batch, n, irreps.dim).mul(5.0).add(10.0)
    x = m(x)

    a = x[..., :3]  # [batch, space, mul]
    assert a.mean([0, 1]).abs().max() < float_tolerance
    assert a.pow(2).mean([0, 1]).sub(1).abs().max() < sqrt_float_tolerance

    a = x[..., 3:].reshape(batch, n, 4, 3)  # [batch, space, mul, repr]
    assert a.pow(2).mean(3).mean([0, 1]).sub(1).abs().max() < sqrt_float_tolerance


def test_e3nn_compat():
    try:
        from e3nn import o3
    except ImportError:
        pytest.skip("e3nn not installed")

    irreps = o3.Irreps("3x0e + 4x1e")
    with pytest.warns(UserWarning):
        cuet.layers.BatchNorm(irreps, layout=cue.mul_ir)


def test_O3_default_warning():
    with pytest.warns(UserWarning):
        m = cuet.layers.BatchNorm("32x0e + 16x1o", layout=cue.ir_mul)
        assert m.irreps.irrep_class == cue.O3


def _o3_rep_matrix(
    irreps: cue.Irreps,
    layout: cue.IrrepsLayout,
    axis: list[float],
    angle: float,
    invert: bool = False,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    import numpy as np

    blocks = []
    axis_np = np.asarray(axis, dtype=np.float64)
    for mul, ir in irreps:
        rot = torch.from_numpy(ir.rotation(axis_np, angle)).to(dtype=dtype)
        if invert and ir.p == -1:
            rot = -rot
        eye_mul = torch.eye(mul, dtype=dtype)
        if layout == cue.mul_ir:
            blocks.append(torch.kron(eye_mul, rot))
        else:
            blocks.append(torch.kron(rot, eye_mul))
    return torch.block_diag(*blocks)


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("instance", [True, False])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("reduce", ["mean", "max"])
def test_batchnorm_torchscript(
    layout: cue.IrrepsLayout,
    instance: bool,
    affine: bool,
    include_bias: bool,
    reduce: str,
) -> None:
    irreps = cue.Irreps("O3", "3x0e + 2x0o + 4x1o")
    m = cuet.layers.BatchNorm(
        irreps,
        layout=layout,
        affine=affine,
        include_bias=include_bias,
        reduce=reduce,
        instance=instance,
    )
    scripted = torch.jit.script(m)

    x_3d = torch.randn(4, 7, irreps.dim)
    torch.testing.assert_close(scripted(x_3d), m(x_3d))

    x_2d = torch.randn(11, irreps.dim)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int64)
    mask = torch.tensor(
        [True, True, True, False, True, True, False, True, True, True, False]
    )
    torch.testing.assert_close(
        scripted(x_2d, batch=batch, num_graphs=3, mask=mask),
        m(x_2d, batch=batch, num_graphs=3, mask=mask),
    )

    m.eval()
    scripted.eval()
    torch.testing.assert_close(scripted(x_2d), m(x_2d))


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("invert", [False, True])
def test_batchnorm_2d_instance_preserves_scalars_and_equivariance(
    layout: cue.IrrepsLayout,
    reduce: str,
    invert: bool,
) -> None:
    torch.manual_seed(42)
    irreps = cue.Irreps("O3", "3x0e + 2x0o + 4x1o + 2x2e")
    m = cuet.layers.BatchNorm(
        irreps,
        layout=layout,
        instance=True,
        reduce=reduce,
        affine=True,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        m.weight.copy_(torch.linspace(0.7, 1.5, m.weight.numel(), dtype=torch.float64))
        m.bias.copy_(torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64))

    num_nodes = 16
    x = torch.randn(num_nodes, irreps.dim, dtype=torch.float64) * 3.0 + 5.0
    out_2d = m(x)
    out_3d = m(x.unsqueeze(0)).squeeze(0)

    # 2D (N, dim) with instance=True must match single-graph (1, N, dim) exactly
    torch.testing.assert_close(out_2d, out_3d, atol=1e-12, rtol=1e-12)

    # Centered scalar features must NOT be wiped to bias/zero across nodes
    scalars = out_2d[:, :3]
    assert (scalars - m.bias).abs().max().item() > 0.5
    torch.testing.assert_close(scalars.mean(dim=0), m.bias, atol=1e-12, rtol=1e-12)
    if reduce == "mean":
        expected_std = m.weight[:3]
        actual_std = (scalars - m.bias).pow(2).mean(dim=0).sqrt()
        torch.testing.assert_close(actual_std, expected_std, atol=1e-4, rtol=1e-4)

    # O(3) equivariance (proper rotation + optional inversion)
    D = _o3_rep_matrix(irreps, layout, [0.4, -0.7, 0.5], 1.13, invert=invert)
    out_rot_before = m(x @ D.T)
    out_rot_after = m(x) @ D.T
    torch.testing.assert_close(out_rot_before, out_rot_after, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("batch_dtype", [torch.int64, torch.int32])
def test_batchnorm_pyg_batched_graphs(
    layout: cue.IrrepsLayout,
    reduce: str,
    batch_dtype: torch.dtype,
) -> None:
    torch.manual_seed(7)
    irreps = cue.Irreps("O3", "4x0e + 3x1o + 2x2e")
    m = cuet.layers.BatchNorm(
        irreps,
        layout=layout,
        instance=True,
        reduce=reduce,
        affine=True,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        m.weight.copy_(torch.linspace(0.5, 1.8, m.weight.numel(), dtype=torch.float64))
        m.bias.copy_(torch.tensor([0.3, -0.2, 0.5, -0.1], dtype=torch.float64))

    g0 = (torch.randn(5, irreps.dim, dtype=torch.float64) * 2.0 + 3.0).requires_grad_(
        True
    )
    g1 = (torch.randn(9, irreps.dim, dtype=torch.float64) * 0.5 - 4.0).requires_grad_(
        True
    )
    g3 = (torch.randn(6, irreps.dim, dtype=torch.float64) * 4.0 + 1.0).requires_grad_(
        True
    )
    # Graph index 2 intentionally has 0 nodes to test empty-graph bucket safety
    batch = torch.tensor(
        [0] * 5 + [1] * 9 + [3] * 6,
        dtype=batch_dtype,
    )

    x_cat = torch.cat([g0, g1, g3], dim=0).detach().clone().requires_grad_(True)
    out_batched = m(x_cat, batch=batch, num_graphs=4)
    out_separate = torch.cat([m(g0), m(g1), m(g3)], dim=0)
    torch.testing.assert_close(out_batched, out_separate, atol=1e-12, rtol=1e-12)

    grad_weights = torch.randn_like(out_batched)
    (grad_cat,) = torch.autograd.grad((out_batched * grad_weights).sum(), x_cat)
    grads_sep = torch.autograd.grad((out_separate * grad_weights).sum(), (g0, g1, g3))
    torch.testing.assert_close(
        grad_cat, torch.cat(grads_sep, dim=0), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("instance", [True, False])
@pytest.mark.parametrize("reduce", ["mean", "max"])
def test_batchnorm_mask_padding_isolation(
    layout: cue.IrrepsLayout,
    instance: bool,
    reduce: str,
) -> None:
    torch.manual_seed(19)
    irreps = cue.Irreps("O3", "3x0e + 3x1o")

    # 1. 2D single/unindexed input with padded dummy rows
    m_unpadded = cuet.layers.BatchNorm(
        irreps, layout=layout, instance=instance, reduce=reduce, affine=True
    ).to(dtype=torch.float64)
    m_padded = cuet.layers.BatchNorm(
        irreps, layout=layout, instance=instance, reduce=reduce, affine=True
    ).to(dtype=torch.float64)
    with torch.no_grad():
        w = torch.linspace(0.8, 1.4, m_unpadded.weight.numel(), dtype=torch.float64)
        b = torch.tensor([0.4, -0.25, 0.15], dtype=torch.float64)
        m_unpadded.weight.copy_(w)
        m_unpadded.bias.copy_(b)
        m_padded.weight.copy_(w)
        m_padded.bias.copy_(b)

    x_valid = torch.randn(7, irreps.dim, dtype=torch.float64, requires_grad=True)
    dummy_pad = torch.full((4, irreps.dim), 9999.0, dtype=torch.float64)
    x_pad = torch.cat([x_valid.detach(), dummy_pad], dim=0).requires_grad_(True)
    mask_2d = torch.tensor([True] * 7 + [False] * 4)

    out_valid = m_unpadded(x_valid)
    out_pad = m_padded(x_pad, mask=mask_2d)

    torch.testing.assert_close(out_pad[:7], out_valid, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        out_pad[7:], torch.zeros_like(out_pad[7:]), atol=0.0, rtol=0.0
    )
    if not instance:
        torch.testing.assert_close(
            m_padded.running_mean, m_unpadded.running_mean, atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            m_padded.running_var, m_unpadded.running_var, atol=1e-12, rtol=1e-12
        )

    out_pad.sum().backward()
    out_valid.sum().backward()
    torch.testing.assert_close(x_pad.grad[:7], x_valid.grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        x_pad.grad[7:], torch.zeros_like(x_pad.grad[7:]), atol=0.0, rtol=0.0
    )

    # 2. PyG batched input with padded dummy nodes
    if instance:
        g0_valid = torch.randn(4, irreps.dim, dtype=torch.float64)
        g1_valid = torch.randn(5, irreps.dim, dtype=torch.float64)
        x_pyg_pad = torch.cat(
            [
                g0_valid,
                torch.full((2, irreps.dim), -5000.0, dtype=torch.float64),
                g1_valid,
                torch.full((3, irreps.dim), 8000.0, dtype=torch.float64),
            ],
            dim=0,
        ).requires_grad_(True)
        batch_pyg = torch.tensor([0] * 6 + [1] * 8, dtype=torch.int64)
        mask_pyg = torch.tensor([True] * 4 + [False] * 2 + [True] * 5 + [False] * 3)
        out_pyg = m_padded(x_pyg_pad, batch=batch_pyg, mask=mask_pyg)
        torch.testing.assert_close(
            out_pyg[:4], m_unpadded(g0_valid), atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            out_pyg[6:11], m_unpadded(g1_valid), atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            out_pyg[4:6], torch.zeros_like(out_pyg[4:6]), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            out_pyg[11:], torch.zeros_like(out_pyg[11:]), atol=0.0, rtol=0.0
        )


@pytest.mark.parametrize("low_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("instance", [True, False])
def test_batchnorm_low_precision_large_magnitude_no_overflow(
    low_dtype: torch.dtype,
    instance: bool,
) -> None:
    torch.manual_seed(123)
    irreps = cue.Irreps("O3", "4x0e + 4x1o")
    m_fp32 = cuet.layers.BatchNorm(irreps, layout=cue.mul_ir, instance=instance)
    m_low = cuet.layers.BatchNorm(irreps, layout=cue.mul_ir, instance=instance).to(
        dtype=low_dtype
    )

    # Values around ~500 exceed sqrt(65504) ~= 255.9, which would overflow FP16 pow(2)
    x_fp32 = torch.randn(16, irreps.dim, dtype=torch.float32) * 500.0 + 300.0
    x_low = x_fp32.to(dtype=low_dtype).detach().requires_grad_(True)
    x_ref = x_low.to(dtype=torch.float32).detach().requires_grad_(True)

    out_low = m_low(x_low)
    out_ref = m_fp32(x_ref)

    assert torch.isfinite(out_low).all()
    assert out_low.dtype == low_dtype
    tol = 2e-2 if low_dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out_low.to(torch.float32), out_ref, rtol=tol, atol=tol)

    out_low.sum().backward()
    assert x_low.grad is not None and torch.isfinite(x_low.grad).all()


@pytest.mark.parametrize("instance", [True, False])
@pytest.mark.parametrize("use_batch", [False, True])
def test_batchnorm_gradgradcheck_and_zero_feature_stability(
    instance: bool,
    use_batch: bool,
) -> None:
    torch.manual_seed(31)
    irreps = cue.Irreps("O3", "2x0e + 2x1o")
    m = cuet.layers.BatchNorm(
        irreps, layout=cue.mul_ir, instance=instance, affine=True
    ).to(dtype=torch.float64)

    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64) if use_batch else None

    # 1. Analytic 2nd-order derivative check on random inputs
    x_rand = torch.randn(6, irreps.dim, dtype=torch.float64, requires_grad=True)
    w_param = m.weight.detach().clone().requires_grad_(True)
    b_param = m.bias.detach().clone().requires_grad_(True)

    def fn(x_in: torch.Tensor, w_in: torch.Tensor, b_in: torch.Tensor) -> torch.Tensor:
        mod = cuet.layers.BatchNorm(
            irreps, layout=cue.mul_ir, instance=instance, affine=False
        ).to(dtype=torch.float64)
        out = mod(x_in, batch=batch)
        # Apply affine cleanly for functional gradgradcheck wrt x_in, w_in, b_in
        scalars = out[:, :2] * w_in[:2] + b_in
        vecs = (out[:, 2:].reshape(6, 2, 3) * w_in[2:].reshape(1, 2, 1)).reshape(6, 6)
        return torch.cat([scalars, vecs], dim=-1)

    assert torch.autograd.gradgradcheck(
        fn, (x_rand, w_param, b_param), eps=1e-6, atol=1e-4, rtol=1e-4
    )

    # 2. 2nd-order stability at exact zero features (x == 0)
    x_zero = torch.zeros(6, irreps.dim, dtype=torch.float64, requires_grad=True)
    out_zero = m(x_zero, batch=batch)
    assert torch.isfinite(out_zero).all()
    (g1,) = torch.autograd.grad(out_zero.pow(2).sum(), x_zero, create_graph=True)
    assert torch.isfinite(g1).all()
    (g2,) = torch.autograd.grad(g1.pow(2).sum(), x_zero)
    assert torch.isfinite(g2).all()
    assert torch.autograd.gradgradcheck(
        lambda z: m(z, batch=batch), (x_zero,), eps=1e-6, atol=1e-4, rtol=1e-4
    )
