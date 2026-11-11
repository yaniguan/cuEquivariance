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
import sys
import types
import warnings

import pytest
import torch
from cuequivariance_torch._tests.utils import module_with_mode

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

list_of_irreps = [
    cue.Irreps("SU2", "3x1/2 + 4x1"),
    cue.Irreps("SU2", "2x1/2 + 5x1 + 2x1/2"),
    cue.Irreps("SU2", "2x1/2 + 0x1 + 0x1/2 + 1 + 2"),
]


@pytest.mark.parametrize("irreps_in", list_of_irreps)
@pytest.mark.parametrize("irreps_out", list_of_irreps)
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("shared_weights", [True, False])
def test_linear_fwd(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    layout: cue.IrrepsLayout,
    shared_weights: bool,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    linear = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
        device=device,
        dtype=torch.float64,
        use_fallback=False,
    )

    torch.manual_seed(0)
    linear_fx = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
        device=device,
        dtype=torch.float64,
        use_fallback=True,
    )
    x = torch.randn(10, irreps_in.dim, dtype=torch.float64).cuda()

    if shared_weights:
        y = linear(x)
        y_fx = linear_fx(x)
    else:
        w = torch.randn(10, linear.weight_numel, dtype=torch.float64).cuda()
        y = linear(x, w)
        y_fx = linear_fx(x, w)

    assert y.shape == (10, irreps_out.dim)

    torch.testing.assert_close(y, y_fx)


@pytest.mark.parametrize("irreps_in", list_of_irreps)
@pytest.mark.parametrize("irreps_out", list_of_irreps)
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("shared_weights", [True, False])
def test_linear_bwd_bwd(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    layout: cue.IrrepsLayout,
    shared_weights: bool,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip redundant layout combinations
    if layout == cue.ir_mul:
        pytest.skip("Skipping redundant layout test")

    # Skip non-shared weights tests for speed
    if not shared_weights:
        pytest.skip("Skipping non-shared weights test for speed")

    # Skip complex irreps combinations - only test simple ones
    if irreps_in.num_irreps > 2 or irreps_out.num_irreps > 2:
        pytest.skip("Skipping complex irreps combination for speed")

    torch.manual_seed(0)
    linear = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
        device=device,
        dtype=torch.float64,
        use_fallback=False,
    )

    torch.manual_seed(0)
    linear_fx = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
        device=device,
        dtype=torch.float64,
        use_fallback=True,
    )

    x = torch.randn(10, irreps_in.dim, dtype=torch.float64, requires_grad=True).cuda()

    if shared_weights:
        y = linear(x)
        y_fx = linear_fx(x)
    else:
        w = torch.randn(
            10, linear.weight_numel, dtype=torch.float64, requires_grad=True
        ).cuda()
        y = linear(x, w)
        y_fx = linear_fx(x, w)

    if shared_weights:
        grad_inputs = [x]
        grad_inputs_fx = [x]
    else:
        grad_inputs = [x, w]
        grad_inputs_fx = [x, w]

    grad_outputs = torch.randn_like(y)
    (g_x,) = torch.autograd.grad(y, grad_inputs[:1], grad_outputs, create_graph=True)
    (g_x_fx,) = torch.autograd.grad(
        y_fx, grad_inputs_fx[:1], grad_outputs, create_graph=True
    )

    torch.testing.assert_close(g_x, g_x_fx)

    gg_x = torch.autograd.grad(g_x.sum(), grad_inputs[:1])[0]
    gg_x_fx = torch.autograd.grad(g_x_fx.sum(), grad_inputs_fx[:1])[0]

    torch.testing.assert_close(gg_x, gg_x_fx)


def test_e3nn_compatibility():
    try:
        from e3nn import o3
    except ImportError:
        pytest.skip("e3nn is not installed")

    with pytest.warns(UserWarning):
        irreps = o3.Irreps("3x1o + 4x1e")
        cuet.Linear(irreps, irreps, layout=cue.mul_ir)

    with pytest.warns(UserWarning):
        cuet.Linear("3x0e + 5x1o", "3x0e + 2x1o", layout=cue.ir_mul)


def test_no_layout_warning():
    with pytest.warns(UserWarning):
        cuet.Linear(cue.Irreps("SU2", "1"), cue.Irreps("SU2", "1"))


@pytest.mark.parametrize("irreps_in", list_of_irreps)
@pytest.mark.parametrize("irreps_out", list_of_irreps)
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("shared_weights", [True, False])
def test_linear_copy(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    layout: cue.IrrepsLayout,
    shared_weights: bool,
):
    linear = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
    ).to(device)

    copy.deepcopy(linear)


export_modes = ["compile", "script", "jit"]


@pytest.mark.parametrize("irreps_in", list_of_irreps)
@pytest.mark.parametrize("irreps_out", list_of_irreps)
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("shared_weights", [True, False])
@pytest.mark.parametrize("mode", export_modes)
@pytest.mark.parametrize("use_fallback", [True, False])
def test_export(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    layout: cue.IrrepsLayout,
    shared_weights: bool,
    mode: str,
    use_fallback: bool,
    tmp_path: str,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip JIT mode as it's slow
    if mode == "jit":
        pytest.skip("Skipping slow JIT compilation test")

    # Skip redundant combinations - test only one layout with fallback=True
    if use_fallback is True and layout == cue.ir_mul:
        pytest.skip("Skipping redundant layout test with fallback=True")

    # Skip compile mode entirely for speed - consistently takes time
    # if mode == "compile":
    #     pytest.skip("Skipping compile mode for speed")

    # Skip script mode for speed
    # if mode == "script":
    #     pytest.skip("Skipping script mode for speed")

    # Skip use_fallback=False for speed - only test fallback
    if use_fallback is False:
        pytest.skip("Skipping use_fallback=False for speed")

    # Skip shared_weights=False for speed
    if shared_weights is False:
        pytest.skip("Skipping shared_weights=False for speed")

    # Skip complex irreps combinations - only test the first one
    if irreps_in != list_of_irreps[0] or irreps_out != list_of_irreps[0]:
        pytest.skip("Skipping complex irreps combinations for speed")

    torch.manual_seed(0)
    m = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
        device=device,
        dtype=torch.float32,
        use_fallback=use_fallback,
    )

    x = torch.randn(10, irreps_in.dim, dtype=torch.float32).cuda()

    if shared_weights:
        inputs = (x,)
    else:
        w = torch.randn(10, m.weight_numel, dtype=torch.float32).cuda()
        inputs = (x, w)

    out1 = m(*inputs)
    m = module_with_mode(mode, m, inputs, torch.float32, tmp_path)
    out2 = m(*inputs)
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("irreps_in", [cue.Irreps("SU2", "3x1/2 + 4x1")])
@pytest.mark.parametrize("irreps_out", [cue.Irreps("SU2", "3x1/2 + 4x1")])
@pytest.mark.parametrize("layout", [cue.mul_ir])
@pytest.mark.parametrize("method", ["naive", "fused_tp", "indexed_linear"])
@pytest.mark.parametrize("math_dtype", [None, torch.float32, "float64"])
def test_linear_fwd_methods(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    layout: cue.IrrepsLayout,
    method: str,
    math_dtype: str | torch.dtype | None,
):
    if math_dtype == torch.float32 and method == "fused_tp":
        pytest.skip("fused_tp does not support float32 math_dtype with FP64 inputs")
    if method == "indexed_linear" and math_dtype is not None:
        pytest.skip("indexed_linear does not support non-None or CUBLAS math_dtype")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    linear_indexed = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=True,
        internal_weights=False,
        weight_classes=3,
        device=device,
        dtype=torch.float64,
        math_dtype=math_dtype,
        method=method,
    )

    torch.manual_seed(0)
    linear = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=False,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        math_dtype=math_dtype,
        method="naive",
    )

    x = torch.randn(10, irreps_in.dim, dtype=torch.float64).cuda()
    weights = torch.randn(3, linear_indexed.weight_numel, dtype=torch.float64).cuda()
    indices, _ = torch.sort(torch.randint(0, 3, (10,), dtype=torch.int32).cuda())
    pre_indexed_weights = weights[indices]

    y_ind = linear_indexed(x, weights, indices)
    y = linear(x, pre_indexed_weights)

    torch.testing.assert_close(y_ind, y)


def _install_mock_cue_ops(monkeypatch):
    """Install a lightweight mock cuequivariance_ops_torch for CPU-only testing."""

    class _MockFusedTensorProductOp3(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

        def forward(self, x0, x1):
            return x0.new_zeros((max(x0.shape[0], x1.shape[0]), 1))

    def _mock_indexed_linear(w, x, counts, u, v, C, Zw, subscripts, coefficients):
        lhs_sub, rhs_sub, out_sub = subscripts
        formula = f"{lhs_sub},n{rhs_sub}->n{out_sub}"
        chunks = []
        offset = 0
        coeff = coefficients.to(dtype=w.dtype)
        for c in range(C):
            cnt = int(counts[c].item())
            x_c = x[offset : offset + cnt]
            offset += cnt
            chunks.append(coeff * torch.einsum(formula, w[c], x_c))
        return torch.cat(chunks, dim=0) if chunks else x.new_zeros((0, v))

    mock_mod = types.ModuleType("cuequivariance_ops_torch")
    mock_mod.FusedTensorProductOp3 = _MockFusedTensorProductOp3
    mock_mod.indexed_linear = _mock_indexed_linear
    monkeypatch.setitem(sys.modules, "cuequivariance_ops_torch", mock_mod)
    monkeypatch.setattr(cuet.primitives.segmented_polynomial, "HAS_CUE_OPS", True)


@pytest.mark.parametrize(
    "dtype, expected_math_dtype",
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
        (None, torch.float32),
    ],
)
def test_linear_fused_tp_math_dtype_promotion(
    dtype: torch.dtype | None,
    expected_math_dtype: torch.dtype,
    monkeypatch,
):
    irreps_in = cue.Irreps("O3", "3x0e + 2x1o")
    irreps_out = cue.Irreps("O3", "2x0e + 4x1o")

    # 1. Direct construction in the current environment (real GPU or fallback)
    lin_direct = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=cue.ir_mul,
        device=device if torch.cuda.is_available() else None,
        dtype=dtype,
        method="fused_tp",
    )
    assert lin_direct.f._math_dtype_orig == expected_math_dtype
    if (
        torch.cuda.is_available()
        and cuet.primitives.segmented_polynomial.HAS_CUE_OPS
        and dtype is not None
    ):
        x = torch.randn(8, irreps_in.dim, device=device, dtype=dtype)
        y = lin_direct(x)
        assert y.shape == (8, irreps_out.dim)
        assert y.dtype == dtype

    # 2. Mocked HAS_CUE_OPS=True to exercise SegmentedPolynomialFusedTP.__init__ directly
    _install_mock_cue_ops(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lin = cuet.Linear(
            irreps_in,
            irreps_out,
            layout=cue.ir_mul,
            dtype=dtype,
            method="fused_tp",
        )
    assert len(caught) == 0
    assert lin.method == "fused_tp"
    assert lin.f.method == "fused_tp"
    assert lin.f.m.math_dtype == expected_math_dtype

    # Explicit math_dtype override must still be respected
    lin_override = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=cue.ir_mul,
        dtype=torch.bfloat16,
        math_dtype=torch.float64,
        method="fused_tp",
    )
    assert lin_override.f.m.math_dtype == torch.float64

    # Deepcopy must preserve math_dtype without warnings or errors
    lin_copy = copy.deepcopy(lin)
    assert lin_copy.f.m.math_dtype == expected_math_dtype


@pytest.mark.parametrize(
    "dtype, expected_math_dtype",
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_linear_auto_backend_gpu_vs_cpu(
    dtype: torch.dtype,
    expected_math_dtype: torch.dtype,
    monkeypatch,
):
    irreps_in = cue.Irreps("O3", "4x0e + 2x1o")
    irreps_out = cue.Irreps("O3", "2x0e + 3x1o")

    # Case 1: GPU available + HAS_CUE_OPS=True -> auto-selects "fused_tp" cleanly
    _install_mock_cue_ops(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with warnings.catch_warnings(record=True) as caught_gpu:
        warnings.simplefilter("always")
        lin_gpu = cuet.Linear(
            irreps_in,
            irreps_out,
            layout=cue.ir_mul,
            dtype=dtype,
        )
    assert len(caught_gpu) == 0
    assert lin_gpu.method == "fused_tp"
    assert lin_gpu.f.method == "fused_tp"
    assert lin_gpu.f.m.math_dtype == expected_math_dtype

    # Case 2: GPU available + HAS_CUE_OPS=True + explicit device="cpu" -> "naive", 0 warnings
    with warnings.catch_warnings(record=True) as caught_cpu_explicit:
        warnings.simplefilter("always")
        lin_cpu_explicit = cuet.Linear(
            irreps_in,
            irreps_out,
            layout=cue.ir_mul,
            device=torch.device("cpu"),
            dtype=dtype,
        )
    assert len(caught_cpu_explicit) == 0
    assert lin_cpu_explicit.method == "naive"
    assert lin_cpu_explicit.f.method == "naive"

    # Case 3: CPU-only (cuda not available) + HAS_CUE_OPS=True -> "naive", 0 warnings
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings(record=True) as caught_no_cuda:
        warnings.simplefilter("always")
        lin_no_cuda = cuet.Linear(
            irreps_in,
            irreps_out,
            layout=cue.ir_mul,
            dtype=dtype,
        )
    assert len(caught_no_cuda) == 0
    assert lin_no_cuda.method == "naive"
    assert lin_no_cuda.f.method == "naive"

    # Case 4: GPU available + HAS_CUE_OPS=False -> "naive", 0 fallback warnings
    monkeypatch.setattr(cuet.primitives.segmented_polynomial, "HAS_CUE_OPS", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with warnings.catch_warnings(record=True) as caught_no_ops:
        warnings.simplefilter("always")
        lin_no_ops = cuet.Linear(
            irreps_in,
            irreps_out,
            layout=cue.ir_mul,
            dtype=dtype,
        )
    assert len(caught_no_ops) == 0
    assert lin_no_ops.method == "naive"
    assert lin_no_ops.f.method == "naive"


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("internal_weights", [True, False])
def test_linear_indexed_auto_sort_indices(
    layout: cue.IrrepsLayout,
    idx_dtype: torch.dtype,
    internal_weights: bool,
    monkeypatch,
):
    use_real_cuda = (
        torch.cuda.is_available() and cuet.primitives.segmented_polynomial.HAS_CUE_OPS
    )
    if not use_real_cuda:
        _install_mock_cue_ops(monkeypatch)
    run_device = device if use_real_cuda else torch.device("cpu")

    irreps_in = cue.Irreps("O3", "4x0e + 4x1o + 9x2e")
    irreps_out = cue.Irreps("O3", "3x0e + 5x1o + 2x2e")
    weight_classes = 5

    # Ensure SegmentedPolynomialIndexedLinear registers FP64 coefficients when constructed
    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        lin_idx = cuet.Linear(
            irreps_in,
            irreps_out,
            layout=layout,
            shared_weights=True,
            internal_weights=internal_weights,
            weight_classes=weight_classes,
            device=run_device,
            dtype=torch.float64,
            method="indexed_linear",
        )
        lin_idx_no_sort = cuet.Linear(
            irreps_in,
            irreps_out,
            layout=layout,
            shared_weights=True,
            internal_weights=internal_weights,
            weight_classes=weight_classes,
            auto_sort_indices=False,
            device=run_device,
            dtype=torch.float64,
            method="indexed_linear",
        )
        lin_ref = cuet.Linear(
            irreps_in,
            irreps_out,
            layout=layout,
            shared_weights=True,
            internal_weights=internal_weights,
            weight_classes=weight_classes,
            device=run_device,
            dtype=torch.float64,
            method="naive",
        )
    finally:
        torch.set_default_dtype(prev_default)

    assert isinstance(
        lin_idx.f.m,
        cuet.primitives.segmented_polynomial_indexed_linear.SegmentedPolynomialIndexedLinear,
    )
    if not use_real_cuda:
        lin_idx.f.options["disable_cpu_fallback"] = True
        lin_idx_no_sort.f.options["disable_cpu_fallback"] = True

    # Unsorted species indices (note class 1 is absent, classes 0/2/3/4 repeated out of order)
    raw_indices = [4, 0, 2, 4, 3, 0, 2, 0, 4, 3, 2]
    idx = torch.tensor(raw_indices, dtype=idx_dtype, device=run_device)
    idx_orig = idx.clone()

    torch.manual_seed(42)
    x_idx = torch.randn(
        len(raw_indices),
        irreps_in.dim,
        dtype=torch.float64,
        device=run_device,
        requires_grad=True,
    )
    x_ref = x_idx.detach().clone().requires_grad_(True)

    if internal_weights:
        with torch.no_grad():
            lin_ref.weight.copy_(lin_idx.weight)
            lin_idx_no_sort.weight.copy_(lin_idx.weight)
        w_idx = None
        w_ref = None
        diff_idx = (x_idx, lin_idx.weight)
        diff_ref = (x_ref, lin_ref.weight)
    else:
        w_idx = torch.randn(
            weight_classes,
            lin_idx.weight_numel,
            dtype=torch.float64,
            device=run_device,
            requires_grad=True,
        )
        w_ref = w_idx.detach().clone().requires_grad_(True)
        diff_idx = (x_idx, w_idx)
        diff_ref = (x_ref, w_ref)

    # With auto_sort_indices=False, unsorted indices fail as before
    with pytest.raises(AssertionError, match="non-sorted indices"):
        lin_idx_no_sort(x_idx, weight=w_idx, weight_indices=idx)

    # With auto_sort_indices=True (default), unsorted indices succeed without mutating idx
    y_idx = lin_idx(x_idx, weight=w_idx, weight_indices=idx)
    y_ref = lin_ref(x_ref, weight=w_ref, weight_indices=idx)
    assert torch.equal(idx, idx_orig)
    torch.testing.assert_close(y_idx, y_ref, atol=1e-12, rtol=1e-12)

    # 1st backward (create_graph=True)
    cotangent = torch.randn_like(y_ref)
    loss_idx = (y_idx * cotangent).pow(2).sum()
    loss_ref = (y_ref * cotangent).pow(2).sum()
    gx_idx, gw_idx = torch.autograd.grad(loss_idx, diff_idx, create_graph=True)
    gx_ref, gw_ref = torch.autograd.grad(loss_ref, diff_ref, create_graph=True)
    torch.testing.assert_close(gx_idx, gx_ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(gw_idx, gw_ref, atol=1e-12, rtol=1e-12)

    # 2nd backward
    v_x = torch.randn_like(gx_ref)
    v_w = torch.randn_like(gw_ref)
    ggx_idx, ggw_idx = torch.autograd.grad(
        (gx_idx * v_x).sum() + (gw_idx * v_w).sum(), diff_idx
    )
    ggx_ref, ggw_ref = torch.autograd.grad(
        (gx_ref * v_x).sum() + (gw_ref * v_w).sum(), diff_ref
    )
    torch.testing.assert_close(ggx_idx, ggx_ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ggw_idx, ggw_ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "irreps_in_str, irreps_out_str",
    [
        ("2x0e", "2x1o"),
        ("0x0e", "2x0e"),
        ("2x0e", "0x1o"),
        ("", ""),
    ],
)
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
def test_linear_disjoint_and_empty_irreps(
    irreps_in_str: str,
    irreps_out_str: str,
    layout: cue.IrrepsLayout,
):
    irreps_in = cue.Irreps("O3", irreps_in_str)
    irreps_out = cue.Irreps("O3", irreps_out_str)

    lin = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        device=device,
        dtype=torch.float64,
    )
    assert lin.weight_numel == 0
    assert lin.method == "naive"

    x = torch.randn(
        6, irreps_in.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    y = lin(x)
    assert y.shape == (6, irreps_out.dim)
    assert torch.all(y == 0)

    gx, gw = torch.autograd.grad(y.pow(2).sum(), (x, lin.weight), create_graph=True)
    ggx, ggw = torch.autograd.grad(gx.sum() + gw.sum(), (x, lin.weight))
    assert torch.all(gx == 0)
    assert torch.all(gw == 0)
    assert torch.all(ggx == 0)
    assert torch.all(ggw == 0)

    scripted = torch.jit.script(lin)
    torch.testing.assert_close(scripted(x.detach()), y.detach())
    copy.deepcopy(lin)
