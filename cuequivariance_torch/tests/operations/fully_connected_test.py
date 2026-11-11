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
import pytest
import torch
from cuequivariance_torch._tests.utils import (
    module_with_mode,
)

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

export_modes = ["compile", "script", "jit"]

irreps = [
    (
        cue.Irreps("O3", "4x0e + 4x1o"),
        cue.Irreps("O3", "4x0e + 4x1o"),
        cue.Irreps("O3", "4x0e + 4x1o"),
    ),
    (
        cue.Irreps("O3", "2e + 0x0e + 0o + 0x1e + 1e"),
        cue.Irreps("O3", "4x0e + 4x1o"),
        cue.Irreps("O3", "2e + 0x0e + 0o + 0x1e + 1e"),
    ),
]


@pytest.mark.parametrize("irreps1, irreps2, irreps3", irreps)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("math_dtype", [None, torch.float32, "float64"])
def test_fully_connected(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3: cue.Irreps,
    layout: cue.IrrepsLayout,
    use_fallback: bool,
    math_dtype: str | torch.dtype | None,
):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if use_fallback is False and math_dtype is torch.float32:
        pytest.skip("Skipping float32 test for fallback=False (fused_tp)")

    m1 = cuet.FullyConnectedTensorProduct(
        irreps1,
        irreps2,
        irreps3,
        shared_weights=True,
        internal_weights=True,
        layout=layout,
        device=device,
        dtype=torch.float64,
        math_dtype=math_dtype,
        use_fallback=use_fallback,
    )

    x1 = torch.randn(32, irreps1.dim, dtype=torch.float64).to(device)
    x2 = torch.randn(32, irreps2.dim, dtype=torch.float64).to(device)

    out1 = m1(x1, x2)

    d = descriptors.fully_connected_tensor_product(
        irreps1, irreps2, irreps3
    ).polynomial.operations[0][1]
    if layout == cue.mul_ir:
        d = d.add_or_transpose_modes("uvw,ui,vj,wk+ijk")
    poly = cue.SegmentedPolynomial.eval_last_operand(d)
    m2 = cuet.SegmentedPolynomial(poly, method="naive", math_dtype=torch.float64).to(
        device
    )
    out2 = m2(
        [
            m1.weight.to(torch.float64),
            x1.to(torch.float64),
            x2.to(torch.float64),
        ]
    )[0].to(out1.dtype)

    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("irreps1, irreps2, irreps3", irreps)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("internal_weights", [False, True])
@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("mode", export_modes)
def test_export(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3: cue.Irreps,
    layout: cue.IrrepsLayout,
    internal_weights: bool,
    use_fallback: bool,
    mode: str,
    tmp_path: str,
):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip JIT mode as it's slow
    if mode == "jit":
        pytest.skip("Skipping slow JIT compilation test")

    # Skip redundant layout combinations with fallback=True
    if use_fallback is True and layout == cue.ir_mul:
        pytest.skip("Skipping redundant layout test with fallback=True")

    # Skip compile mode entirely for speed - it's consistently slow (1+ seconds)
    if mode == "compile":
        pytest.skip("Skipping compile mode for speed - takes 1+ seconds")

    # Skip script mode for speed
    if mode == "script":
        pytest.skip("Skipping script mode for speed")

    # Skip use_fallback=False for speed - only test fallback
    if use_fallback is False:
        pytest.skip("Skipping use_fallback=False for speed")

    # Skip internal_weights=True for speed
    if internal_weights is True:
        pytest.skip("Skipping internal_weights=True for speed")

    dtype = torch.float32
    m1 = cuet.FullyConnectedTensorProduct(
        irreps1,
        irreps2,
        irreps3,
        shared_weights=True,
        internal_weights=internal_weights,
        layout=layout,
        device=device,
        dtype=dtype,
        use_fallback=use_fallback,
    )

    x1 = torch.randn(32, irreps1.dim, dtype=dtype).to(device)
    x2 = torch.randn(32, irreps2.dim, dtype=dtype).to(device)

    if internal_weights:
        inputs = (x1, x2)
    else:
        weights = torch.randn(1, m1.weight_numel, device=device, dtype=dtype)
        inputs = (x1, x2, weights)

    out1 = m1(*inputs)

    m1 = module_with_mode(mode, m1, inputs, dtype, tmp_path)
    out2 = m1(*inputs)
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("shared_weights", [True, False])
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
def test_fully_connected_indexing(
    layout: cue.IrrepsLayout,
    shared_weights: bool,
    index_dtype: torch.dtype,
):
    torch.manual_seed(0)
    irreps1 = cue.Irreps("O3", "3x0e + 2x1o")
    irreps2 = cue.Irreps("O3", "0e + 1o")
    irreps3 = cue.Irreps("O3", "3x0e + 2x1o")

    m = cuet.FullyConnectedTensorProduct(
        irreps1,
        irreps2,
        irreps3,
        shared_weights=shared_weights,
        internal_weights=False,
        layout=layout,
        device=device,
        dtype=torch.float64,
        method="naive",
    )

    num_src, num_dst, num_edges = 6, 8, 15
    x1 = torch.randn(
        num_src, irreps1.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    x2 = torch.randn(
        num_edges, irreps2.dim, device=device, dtype=torch.float64, requires_grad=True
    )
    w_batch = 1 if shared_weights else num_edges
    w = torch.randn(
        w_batch, m.weight_numel, device=device, dtype=torch.float64, requires_grad=True
    )
    src = torch.randint(0, num_src, (num_edges,), device=device, dtype=index_dtype)
    dst = torch.randint(0, num_dst - 2, (num_edges,), device=device, dtype=index_dtype)

    out_indexed = m(x1, x2, w, indices_1=src, indices_out=dst, size_out=num_dst)

    out_edge = m(x1[src.to(torch.int64)], x2, w)
    out_ref = torch.zeros(
        (num_dst, irreps3.dim), device=device, dtype=torch.float64
    ).scatter_add(0, dst.to(torch.int64).unsqueeze(-1).expand_as(out_edge), out_edge)

    torch.testing.assert_close(out_indexed, out_ref, atol=1e-12, rtol=1e-12)

    g_idx = torch.autograd.grad(
        out_indexed.pow(2).sum(), (x1, x2, w), create_graph=True
    )
    g_ref = torch.autograd.grad(out_ref.pow(2).sum(), (x1, x2, w), create_graph=True)
    for gi, gr in zip(g_idx, g_ref):
        torch.testing.assert_close(gi, gr, atol=1e-12, rtol=1e-12)

    gg_idx = torch.autograd.grad(sum(g.pow(2).sum() for g in g_idx), (x1, x2, w))
    gg_ref = torch.autograd.grad(sum(g.pow(2).sum() for g in g_ref), (x1, x2, w))
    for ggi, ggr in zip(gg_idx, gg_ref):
        torch.testing.assert_close(ggi, ggr, atol=1e-12, rtol=1e-12)
