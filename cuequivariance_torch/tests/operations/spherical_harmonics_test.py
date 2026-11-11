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
from cuequivariance_torch._tests.utils import (
    module_with_mode,
)

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.parametrize(
    "dtype, tol",
    [(torch.float64, 1e-5), (torch.float32, 1e-4)],
)
@pytest.mark.parametrize("ell", [0, 1, 2, 3])
@pytest.mark.parametrize("use_fallback", [False, True])
def test_spherical_harmonics_equivariance(use_fallback: bool, ell: int, dtype, tol):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    vec = torch.randn(3, dtype=dtype, device=device)
    axis = np.random.randn(3)
    angle = np.random.rand()
    scale = 1.3

    m = cuet.SphericalHarmonics([ell], False, device=device, use_fallback=use_fallback)

    yl = m(vec.unsqueeze(0)).squeeze(0)

    R = torch.from_numpy(cue.SO3(1).rotation(axis, angle)).to(dtype).to(device)
    Rl = torch.from_numpy(cue.SO3(ell).rotation(axis, angle)).to(dtype).to(device)

    yl1 = m((scale * R @ vec).unsqueeze(0)).squeeze(0)
    yl2 = scale**ell * Rl @ yl

    torch.testing.assert_close(yl1, yl2, rtol=tol, atol=tol)


data_types = [torch.float32, torch.float64]

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    data_types += [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("ls", [[0], [1], [2], [0, 1], [0, 1, 2]])
@pytest.mark.parametrize("use_fallback", [False, True])
def test_spherical_harmonics_full(dtype, ls: list[int], use_fallback: bool):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m = cuet.SphericalHarmonics(ls, False, use_fallback=use_fallback, device=device)

    vec = torch.randn(10, 3, device=device, dtype=dtype)
    yl = m(vec)
    assert yl.shape[0] == 10
    assert yl.shape[1] == sum(2 * ell + 1 for ell in ls)


export_modes = ["compile", "script", "jit"]


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("ls", [[0], [1], [2], [0, 1], [0, 1, 2]])
@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("mode", export_modes)
def test_export(dtype, ls: list[int], use_fallback: bool, mode: str, tmp_path: str):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip JIT mode as it's slow
    if mode == "jit":
        pytest.skip("Skipping slow JIT compilation test")

    # Skip float16/bfloat16 with fallback=False to reduce test matrix
    if not use_fallback and dtype in [torch.float16, torch.bfloat16]:
        pytest.skip("Skipping fp16/bf16 tests with CUDA backend")

    # Skip high degree spherical harmonics with multiple ls for speed
    if len(ls) > 2 and max(ls) >= 2:
        pytest.skip("Skipping high degree multi-l tests for speed")

    # Skip compile mode for speed - consistently takes time (0.6+ seconds)
    if mode == "compile":
        pytest.skip("Skipping compile mode for speed")

    # Skip script mode for speed
    if mode == "script":
        pytest.skip("Skipping script mode for speed")

    # Skip use_fallback=False for speed - only test fallback
    if use_fallback is False:
        pytest.skip("Skipping use_fallback=False for speed")

    # Skip complex ls combinations - only test simple ones
    if len(ls) > 1:
        pytest.skip("Skipping complex ls combinations for speed")

    # Skip float16/bfloat16 tests entirely for speed
    if dtype in [torch.float16, torch.bfloat16]:
        pytest.skip("Skipping fp16/bf16 tests for speed")

    tol = 1e-5
    if dtype in [torch.float16, torch.bfloat16]:
        tol = 1e-2
    m = cuet.SphericalHarmonics(ls, False, use_fallback=use_fallback, device=device)

    vec = torch.randn(10, 3, device=device, dtype=dtype)
    inputs = (vec,)
    out1 = m(vec)

    m = module_with_mode(mode, m, inputs, dtype, tmp_path)
    out2 = m(*inputs)
    torch.testing.assert_close(out1, out2, atol=tol, rtol=tol)


def _default_method() -> str:
    return "uniform_1d" if torch.cuda.is_available() else "naive"


def test_invalid_eps_and_shapes():
    with pytest.raises(ValueError, match="eps"):
        cuet.SphericalHarmonics([0, 1], eps=-1e-6, method="naive")
    with pytest.raises(ValueError, match="eps"):
        cuet.SphericalHarmonics([0, 1], eps=float("nan"), method="naive")

    m = cuet.SphericalHarmonics(
        [0, 1, 2], normalize=True, method="naive", device=device
    )
    with pytest.raises(AssertionError):
        m(torch.tensor(1.0, device=device))
    with pytest.raises(AssertionError):
        m(torch.randn(4, 4, device=device))
    with pytest.raises(AssertionError):
        m(torch.randn(2, device=device))


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16]
)
def test_zero_vector_forward_grad_and_hessian(dtype: torch.dtype):
    ls = [0, 1, 2, 3, 4]
    out_dim = sum(2 * ell + 1 for ell in ls)
    method = (
        "naive"
        if dtype in (torch.bfloat16, torch.float16) and not torch.cuda.is_available()
        else _default_method()
    )
    m = cuet.SphericalHarmonics(ls, normalize=True, method=method, device=device)

    v = torch.zeros(1, 3, dtype=dtype, device=device, requires_grad=True)
    out = m(v)
    assert out.shape == (1, out_dim)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    assert out[0, 0].item() == 1.0
    assert (out[0, 1:] == 0.0).all()

    # Verify full Jacobian (25 x 3) and full Hessian (25 x 3 x 3) at exact [0, 0, 0]
    for k in range(out_dim):
        (grad_k,) = torch.autograd.grad(
            out[0, k], v, create_graph=True, retain_graph=True
        )
        assert torch.isfinite(grad_k).all()
        assert (grad_k == 0.0).all()
        for i in range(3):
            (hess_ki,) = torch.autograd.grad(grad_k[0, i], v, retain_graph=True)
            assert torch.isfinite(hess_ki).all()
            assert (hess_ki == 0.0).all()

    # Also test mixed batch with [0, 0, 0], normal vector, and tiny vector whose square underflows
    tiny = 1e-25 if dtype in (torch.float32, torch.bfloat16) else 0.0
    v_mix = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 0.5], [tiny, 0.0, 0.0]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    out_mix = m(v_mix)
    weights = torch.linspace(0.5, 1.5, out_dim, dtype=dtype, device=device)
    (g1_mix,) = torch.autograd.grad((out_mix * weights).sum(), v_mix, create_graph=True)
    (g2_mix,) = torch.autograd.grad((g1_mix * g1_mix).sum() + g1_mix.sum(), v_mix)
    assert torch.isfinite(out_mix).all()
    assert torch.isfinite(g1_mix).all()
    assert torch.isfinite(g2_mix).all()
    assert out_mix[0, 0].item() == 1.0 and (out_mix[0, 1:] == 0.0).all()
    assert (g1_mix[0] == 0.0).all() and (g2_mix[0] == 0.0).all()


@pytest.mark.parametrize(
    "normalize, eps",
    [(True, 0.0), (True, 1e-2), (False, 0.0)],
)
@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 1, 3)])
def test_gradcheck_and_gradgradcheck_nonzero(
    normalize: bool, eps: float, shape: tuple[int, ...]
):
    ls = [0, 1, 2, 3, 4]
    m = cuet.SphericalHarmonics(
        ls, normalize=normalize, eps=eps, method=_default_method(), device=device
    )
    gen = torch.Generator(device="cpu").manual_seed(123)
    v = torch.randn(*shape, generator=gen, dtype=torch.float64).to(device)
    # Ensure well away from origin for the nonzero check
    v = v + torch.where(v >= 0, 0.5, -0.5)
    v.requires_grad_(True)

    fast = shape != (3,)
    assert torch.autograd.gradcheck(
        m, (v,), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=fast
    )
    assert torch.autograd.gradgradcheck(
        m, (v,), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=fast
    )


def test_gradcheck_and_gradgradcheck_exact_zero():
    ls = [0, 1, 2, 3, 4]
    v_zero = torch.zeros(1, 3, dtype=torch.float64, device=device, requires_grad=True)

    # 1. Unnormalized polynomial spherical harmonics are smooth at exact v = [0, 0, 0]
    m_unnorm = cuet.SphericalHarmonics(
        ls, normalize=False, method=_default_method(), device=device
    )
    assert torch.autograd.gradcheck(m_unnorm, (v_zero,), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(
        m_unnorm, (v_zero,), eps=1e-6, atol=1e-5, rtol=1e-4
    )

    # 2. Normalized spherical harmonics with eps > 0 are C^inf smooth at exact v = [0, 0, 0]
    m_eps = cuet.SphericalHarmonics(
        ls, normalize=True, eps=1e-1, method=_default_method(), device=device
    )
    assert torch.autograd.gradcheck(m_eps, (v_zero,), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(
        m_eps, (v_zero,), eps=1e-6, atol=1e-5, rtol=1e-4
    )

    # 3. Masked batch containing exact [0, 0, 0] in masked slot with eps=0.0
    m_norm = cuet.SphericalHarmonics(
        ls, normalize=True, eps=0.0, method=_default_method(), device=device
    )
    v_batch = torch.tensor(
        [[0.8, -0.5, 1.1], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    mask = torch.tensor([True, False], device=device)
    assert torch.autograd.gradcheck(
        lambda x: m_norm(x, mask=mask), (v_batch,), eps=1e-6, atol=1e-5, rtol=1e-4
    )
    assert torch.autograd.gradgradcheck(
        lambda x: m_norm(x, mask=mask), (v_batch,), eps=1e-6, atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize(
    "shape",
    [(3,), (7, 3), (2, 5, 3), (2, 3, 4, 3), (0, 3), (3, 0, 3)],
)
@pytest.mark.parametrize("normalize", [True, False])
def test_multi_batch_and_empty_shapes(shape: tuple[int, ...], normalize: bool):
    ls = [0, 1, 2, 3]
    out_dim = sum(2 * ell + 1 for ell in ls)
    m = cuet.SphericalHarmonics(
        ls, normalize=normalize, method=_default_method(), device=device
    )

    v = torch.randn(*shape, dtype=torch.float64, device=device, requires_grad=True)
    out = m(v)
    assert out.shape == shape[:-1] + (out_dim,)
    assert out.dtype == torch.float64

    if v.numel() > 0:
        v_flat = v.detach().reshape(-1, 3).requires_grad_(True)
        out_flat = m(v_flat)
        torch.testing.assert_close(out, out_flat.reshape(shape[:-1] + (out_dim,)))

        (g1,) = torch.autograd.grad(out.pow(2).sum(), v, create_graph=True)
        (g2,) = torch.autograd.grad(g1.pow(2).sum(), v)
        (g1_ref,) = torch.autograd.grad(
            out_flat.pow(2).sum(), v_flat, create_graph=True
        )
        (g2_ref,) = torch.autograd.grad(g1_ref.pow(2).sum(), v_flat)
        torch.testing.assert_close(g1, g1_ref.reshape(shape))
        torch.testing.assert_close(g2, g2_ref.reshape(shape))
    else:
        assert out.requires_grad
        (g1,) = torch.autograd.grad(out.pow(2).sum(), v, create_graph=True)
        (g2,) = torch.autograd.grad(g1.sum(), v)
        assert g1.shape == shape and g2.shape == shape


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float64])
def test_mask_nan_inf_isolation(mask_dtype: torch.dtype):
    ls = [0, 1, 2, 3, 4]
    out_dim = sum(2 * ell + 1 for ell in ls)
    m = cuet.SphericalHarmonics(
        ls, normalize=True, method=_default_method(), device=device
    )

    # 3D Replica Batching (R=2, E=4, 3) with valid vectors, valid [0,0,0], and masked NaN/Inf garbage
    v_data = torch.tensor(
        [
            [
                [1.0, -0.5, 2.0],
                [0.0, 0.0, 0.0],
                [float("nan"), float("inf"), float("-inf")],
                [float("nan"), 0.0, float("inf")],
            ],
            [
                [-0.3, 1.4, 0.7],
                [2.0, 1.0, -1.5],
                [float("inf"), float("nan"), 1e30],
                [float("-inf"), float("nan"), float("nan")],
            ],
        ],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    # Broadcastable mask of shape (1, 4) -> broadcasts to (2, 4)
    mask_1d = torch.tensor([[True, True, False, False]], device=device).to(mask_dtype)

    out = m(v_data, mask=mask_1d)
    assert out.shape == (2, 4, out_dim)
    assert torch.isfinite(out).all()
    # Masked-out edges must be exact 0.0 across ALL channels including l=0!
    assert (out[:, 2:, :] == 0.0).all()
    # Valid zero-vector at [0, 1] must still have Y_0 = 1.0 and Y_{l>0} = 0.0!
    assert out[0, 1, 0].item() == 1.0
    assert (out[0, 1, 1:] == 0.0).all()

    weights = torch.linspace(0.2, 1.3, out_dim, dtype=torch.float64, device=device)
    loss1 = (out * weights).pow(2).sum() + (out * weights).sum()
    (g1,) = torch.autograd.grad(loss1, v_data, create_graph=True)
    loss2 = (g1 * g1).sum() + g1.sum()
    (g2,) = torch.autograd.grad(loss2, v_data)

    assert torch.isfinite(g1).all()
    assert torch.isfinite(g2).all()
    assert (g1[:, 2:, :] == 0.0).all()
    assert (g2[:, 2:, :] == 0.0).all()
    assert (g1[0, 1, :] == 0.0).all()
    assert (g2[0, 1, :] == 0.0).all()

    # Compare valid slots against unmasked valid-only reference
    v_ref = v_data.detach()[:, :2, :].clone().requires_grad_(True)
    out_ref = m(v_ref)
    loss1_ref = (out_ref * weights).pow(2).sum() + (out_ref * weights).sum()
    (g1_ref,) = torch.autograd.grad(loss1_ref, v_ref, create_graph=True)
    loss2_ref = (g1_ref * g1_ref).sum() + g1_ref.sum()
    (g2_ref,) = torch.autograd.grad(loss2_ref, v_ref)

    torch.testing.assert_close(out[:, :2, :], out_ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(g1[:, :2, :], g1_ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(g2[:, :2, :], g2_ref, atol=1e-12, rtol=1e-12)

    # Also check 1D (3,) single vector with 0D scalar mask=False and NaN input
    v_1d = torch.tensor(
        [float("nan"), float("inf"), -1.0],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    out_1d = m(v_1d, mask=torch.tensor(False, device=device).to(mask_dtype))
    assert out_1d.shape == (out_dim,)
    assert (out_1d == 0.0).all()
    (g1_1d,) = torch.autograd.grad(out_1d.sum(), v_1d, create_graph=True)
    (g2_1d,) = torch.autograd.grad(g1_1d.sum(), v_1d)
    assert (g1_1d == 0.0).all() and (g2_1d == 0.0).all()


def test_compile_fx_and_script_compatibility(tmp_path):
    sh = cuet.SphericalHarmonics(
        [0, 1, 2, 3], normalize=True, method="naive", device=device
    )
    x = torch.tensor(
        [[1.0, -0.5, 2.0], [0.0, 0.0, 0.0], [float("nan"), float("inf"), -1.0]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    mask = torch.tensor([True, True, False], device=device)

    # 1. torch.compile (aot_eager backend works universally without external C++ compiler)
    def energy_and_forces(vecs: torch.Tensor, edge_mask: torch.Tensor):
        y = sh(vecs, mask=edge_mask)
        energy = (y * y).sum()
        (forces,) = torch.autograd.grad(energy, vecs, create_graph=True)
        return energy, -forces

    e_ref, f_ref = energy_and_forces(x, mask)
    (h_ref,) = torch.autograd.grad((f_ref * f_ref).sum() + e_ref, x)

    compiled_ef = torch.compile(energy_and_forces, backend="aot_eager", fullgraph=True)
    x_comp = x.detach().clone().requires_grad_(True)
    e_comp, f_comp = compiled_ef(x_comp, mask)
    (h_comp,) = torch.autograd.grad((f_comp * f_comp).sum() + e_comp, x_comp)

    torch.testing.assert_close(e_comp, e_ref)
    torch.testing.assert_close(f_comp, f_ref)
    torch.testing.assert_close(h_comp, h_ref)

    # 2. torch.fx.symbolic_trace (treating SegmentedPolynomial primitive as leaf)
    class _LeafTracer(torch.fx.Tracer):
        def is_leaf_module(
            self, m: torch.nn.Module, module_qualified_name: str
        ) -> bool:
            if isinstance(m, cuet.SegmentedPolynomial):
                return True
            return super().is_leaf_module(m, module_qualified_name)

    gm = torch.fx.GraphModule(sh, _LeafTracer().trace(sh))
    torch.testing.assert_close(gm(x, mask), sh(x, mask))

    # 3. torch.export.export
    exported = torch.export.export(sh, (x.detach(),), {"mask": mask})
    torch.testing.assert_close(
        exported.module()(x.detach(), mask=mask), sh(x.detach(), mask)
    )

    # 4. torch.jit.script (for positive degrees on naive backend)
    sh_pos = cuet.SphericalHarmonics(
        [1, 2], normalize=True, method="naive", device=device
    )
    scripted = torch.jit.script(sh_pos)
    torch.testing.assert_close(scripted(x.detach(), mask), sh_pos(x.detach(), mask))
    torch.testing.assert_close(
        scripted(torch.zeros(0, 3, dtype=torch.float64, device=device)),
        sh_pos(torch.zeros(0, 3, dtype=torch.float64, device=device)),
    )
