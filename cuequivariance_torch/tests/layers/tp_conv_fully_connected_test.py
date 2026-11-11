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
from cuequivariance_torch.layers.tp_conv_fully_connected import scatter_reduce
from torch import nn

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize(
    "mlp_channels, mlp_activation, scalar_sizes",
    [
        [(30, 8, 8), nn.Sequential(nn.Dropout(0.3), nn.ReLU()), (15, 15, 0)],
        [(7,), nn.GELU(), (2, 3, 2)],
        [None, None, None],
    ],
)
def test_tensor_product_conv_equivariance(
    mlp_channels, mlp_activation, scalar_sizes, batch_norm, layout
):
    # Skip redundant layout combinations for speed
    if layout == cue.ir_mul:
        pytest.skip("Skipping redundant layout test")

    # Skip complex mlp configurations with batch_norm=False for speed
    if not batch_norm and mlp_channels is not None and len(mlp_channels) > 1:
        pytest.skip("Skipping complex MLP test without batch_norm")

    torch.manual_seed(12345)

    in_irreps = cue.Irreps("O3", "10x0e + 10x1o + 5x2e")
    out_irreps = cue.Irreps("O3", "20x0e + 5x1o + 5x2e")
    sh_irreps = cue.Irreps("O3", "0e + 1o")

    tp_conv = cuet.layers.FullyConnectedTensorProductConv(
        in_irreps=in_irreps,
        sh_irreps=sh_irreps,
        out_irreps=out_irreps,
        mlp_channels=mlp_channels,
        mlp_activation=mlp_activation,
        batch_norm=batch_norm,
        layout=layout,
        use_fallback=not torch.cuda.is_available(),
    ).to(device)

    num_src_nodes, num_dst_nodes = 9, 7
    num_edges = 40
    src = torch.randint(num_src_nodes, (num_edges,), device=device)
    dst = torch.randint(num_dst_nodes, (num_edges,), device=device)
    edge_index = torch.vstack((src, dst))

    src_pos = torch.randn(num_src_nodes, 3, device=device)
    dst_pos = torch.randn(num_dst_nodes, 3, device=device)
    edge_vec = dst_pos[dst] - src_pos[src]
    edge_sh = torch.concatenate(
        [
            torch.ones(num_edges, 1, device=device),
            edge_vec / edge_vec.norm(dim=1, keepdim=True),
        ],
        dim=1,
    )
    src_features = torch.randn(num_src_nodes, in_irreps.dim, device=device)

    def D(irreps, axis, angle):
        return torch.block_diag(
            *[
                torch.from_numpy(ir.rotation(axis, angle)).to(device, torch.float32)
                for mul, ir in irreps
                for _ in range(mul)
            ]
        )

    axis, angle = np.array([0.6, 0.3, -0.1]), 0.52
    D_in = D(in_irreps, axis, angle)
    D_sh = D(sh_irreps, axis, angle)
    D_out = D(out_irreps, axis, angle)

    if mlp_channels is None:
        edge_emb = torch.randn(num_edges, tp_conv.tp.weight_numel, device=device)
        src_scalars = dst_scalars = None
    else:
        if scalar_sizes:
            edge_emb = torch.randn(num_edges, scalar_sizes[0], device=device)
            src_scalars = (
                None
                if scalar_sizes[1] == 0
                else torch.randn(num_src_nodes, scalar_sizes[1], device=device)
            )
            dst_scalars = (
                None
                if scalar_sizes[2] == 0
                else torch.randn(num_dst_nodes, scalar_sizes[2], device=device)
            )
        else:
            edge_emb = torch.randn(num_edges, tp_conv.mlp[0].in_features, device=device)
            src_scalars = dst_scalars = None

    # rotate before
    out_before = tp_conv(
        src_features=src_features @ D_in.T,
        edge_sh=edge_sh @ D_sh.T,
        edge_emb=edge_emb,
        graph=(edge_index, (num_src_nodes, num_dst_nodes)),
        src_scalars=src_scalars,
        dst_scalars=dst_scalars,
    )

    # rotate after
    out_after = (
        tp_conv(
            src_features=src_features,
            edge_sh=edge_sh,
            edge_emb=edge_emb,
            graph=(edge_index, (num_src_nodes, num_dst_nodes)),
            src_scalars=src_scalars,
            dst_scalars=dst_scalars,
        )
        @ D_out.T
    )

    torch.allclose(out_before, out_after, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduce", ["sum", "mean", "prod", "amax", "amin"])
def test_scatter_reduce(reduce: str):
    src = torch.Tensor([3, 1, 0, 1, 1, 2])
    index = torch.Tensor([0, 1, 2, 2, 3, 1])

    src = src.to(device)
    index = index.to(device)

    out = scatter_reduce(src, index, dim=0, dim_size=None, reduce=reduce)

    out_true = {
        "sum": torch.Tensor([3.0, 3.0, 1.0, 1.0]),
        "mean": torch.Tensor([3.0, 1.5, 0.5, 1.0]),
        "prod": torch.Tensor([3.0, 2.0, 0.0, 1.0]),
        "amax": torch.Tensor([3.0, 2.0, 1.0, 1.0]),
        "amin": torch.Tensor([3.0, 1.0, 0.0, 1.0]),
    }
    assert torch.allclose(out.cpu(), out_true[reduce])


def test_scatter_reduce_empty():
    src, index = torch.empty((0, 41)), torch.empty((0,))
    src = src.to(device)
    index = index.to(device)

    out = scatter_reduce(src, index, dim=0, dim_size=None)

    assert out.numel() == 0
    assert out.size(1) == src.size(1)


def _reference_tp_conv(
    tp_conv: cuet.layers.FullyConnectedTensorProductConv,
    src_features: torch.Tensor,
    edge_sh: torch.Tensor,
    edge_emb: torch.Tensor,
    graph: tuple[torch.Tensor, tuple[int, int]],
    src_scalars: torch.Tensor | None = None,
    dst_scalars: torch.Tensor | None = None,
    reduce: str = "mean",
    edge_envelope: torch.Tensor | None = None,
) -> torch.Tensor:
    (src, dst), (_, num_dst_nodes) = graph
    src_i64 = src.to(torch.int64)
    dst_i64 = dst.to(torch.int64)

    if tp_conv.mlp is not None:
        if src_scalars is None and dst_scalars is None:
            tp_weights = tp_conv.mlp(edge_emb)
        else:
            edge_emb_size = edge_emb.size(-1)
            src_scalars_size = 0 if src_scalars is None else src_scalars.size(-1)
            dst_scalars_size = 0 if dst_scalars is None else dst_scalars.size(-1)
            w_edge, w_src, w_dst = torch.split(
                tp_conv.mlp[0].weight,
                (edge_emb_size, src_scalars_size, dst_scalars_size),
                dim=-1,
            )
            tp_weights = edge_emb @ w_edge.T + tp_conv.mlp[0].bias
            if src_scalars is not None:
                tp_weights = tp_weights + (src_scalars @ w_src.T)[src_i64]
            if dst_scalars is not None:
                tp_weights = tp_weights + (dst_scalars @ w_dst.T)[dst_i64]
            tp_weights = tp_conv.mlp[1:](tp_weights)
    else:
        tp_weights = edge_emb

    edge_out = tp_conv.tp(src_features[src_i64], edge_sh, tp_weights)
    if edge_envelope is not None:
        edge_out = edge_out * edge_envelope.view(-1, 1)

    out = scatter_reduce(
        edge_out, dst_i64, dim=0, dim_size=num_dst_nodes, reduce=reduce
    )
    if tp_conv.batch_norm is not None:
        out = tp_conv.batch_norm(out)
    return out


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("use_envelope", [False, True])
@pytest.mark.parametrize("mlp_mode", ["none", "edge_only", "split_scalars"])
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
def test_tp_conv_fused_gather_scatter_and_envelope_exact_fp64(
    layout: cue.IrrepsLayout,
    reduce: str,
    use_envelope: bool,
    mlp_mode: str,
    index_dtype: torch.dtype,
) -> None:
    torch.manual_seed(2025)
    in_irreps = cue.Irreps("O3", "3x0e + 2x1o + 1x2e")
    sh_irreps = cue.Irreps("O3", "0e + 1o")
    out_irreps = cue.Irreps("O3", "3x0e + 2x1o")

    if mlp_mode == "none":
        mlp_channels = None
    elif mlp_mode == "edge_only":
        mlp_channels = [6, 8]
    else:
        mlp_channels = [9, 8]

    tp_conv = cuet.layers.FullyConnectedTensorProductConv(
        in_irreps=in_irreps,
        sh_irreps=sh_irreps,
        out_irreps=out_irreps,
        batch_norm=False,
        mlp_channels=mlp_channels,
        mlp_activation=nn.SiLU(),
        layout=layout,
        method="naive",
    ).to(device=device, dtype=torch.float64)

    # Include isolated destination nodes (e.g., node 2 and trailing node 6 have 0 incoming edges)
    num_src_nodes, num_dst_nodes = 6, 7
    src = torch.tensor([0, 1, 3, 4, 5, 2, 0, 3, 1, 5], device=device, dtype=index_dtype)
    dst = torch.tensor([0, 1, 1, 3, 3, 4, 4, 4, 5, 0], device=device, dtype=index_dtype)
    edge_index = torch.vstack((src, dst))
    num_edges = src.numel()

    src_features = torch.randn(
        num_src_nodes,
        in_irreps.dim,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_sh = torch.randn(
        num_edges,
        sh_irreps.dim,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )

    src_scalars = None
    dst_scalars = None
    if mlp_mode == "none":
        edge_emb = torch.randn(
            num_edges,
            tp_conv.tp.weight_numel,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
    elif mlp_mode == "edge_only":
        edge_emb = torch.randn(
            num_edges, 6, device=device, dtype=torch.float64, requires_grad=True
        )
    else:
        edge_emb = torch.randn(
            num_edges, 4, device=device, dtype=torch.float64, requires_grad=True
        )
        src_scalars = torch.randn(
            num_src_nodes, 3, device=device, dtype=torch.float64, requires_grad=True
        )
        dst_scalars = torch.randn(
            num_dst_nodes, 2, device=device, dtype=torch.float64, requires_grad=True
        )

    edge_emb_snapshot = edge_emb.detach().clone()
    edge_envelope = (
        torch.rand(num_edges, device=device, dtype=torch.float64, requires_grad=True)
        if use_envelope
        else None
    )

    out_fused = tp_conv(
        src_features=src_features,
        edge_sh=edge_sh,
        edge_emb=edge_emb,
        graph=(edge_index, (num_src_nodes, num_dst_nodes)),
        src_scalars=src_scalars,
        dst_scalars=dst_scalars,
        reduce=reduce,
        edge_envelope=edge_envelope,
    )
    # Caller's edge_emb must never be mutated in-place
    torch.testing.assert_close(edge_emb.detach(), edge_emb_snapshot, atol=0.0, rtol=0.0)

    out_ref = _reference_tp_conv(
        tp_conv=tp_conv,
        src_features=src_features,
        edge_sh=edge_sh,
        edge_emb=edge_emb,
        graph=(edge_index, (num_src_nodes, num_dst_nodes)),
        src_scalars=src_scalars,
        dst_scalars=dst_scalars,
        reduce=reduce,
        edge_envelope=edge_envelope,
    )

    # Forward agreement to 1e-12 in FP64 (including 0.0 on isolated nodes 2 and 6)
    torch.testing.assert_close(out_fused, out_ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        out_fused[[2, 6]], torch.zeros_like(out_fused[[2, 6]]), atol=0.0, rtol=0.0
    )

    # 1st-order & 2nd-order backward agreement to 1e-12 in FP64
    diff_tensors = [src_features, edge_sh, edge_emb]
    if src_scalars is not None:
        diff_tensors.append(src_scalars)
    if dst_scalars is not None:
        diff_tensors.append(dst_scalars)
    if edge_envelope is not None:
        diff_tensors.append(edge_envelope)
    diff_tensors.extend(list(tp_conv.parameters()))

    cotangent = torch.randn_like(out_fused)
    grads_fused = torch.autograd.grad(
        (out_fused * cotangent).sum(), diff_tensors, create_graph=True
    )
    grads_ref = torch.autograd.grad(
        (out_ref * cotangent).sum(), diff_tensors, create_graph=True
    )
    for gf, gr in zip(grads_fused, grads_ref):
        assert torch.isfinite(gf).all()
        torch.testing.assert_close(gf, gr, atol=1e-12, rtol=1e-12)

    grad2_cotangents = [torch.randn_like(gf) for gf in grads_fused]
    ggrads_fused = torch.autograd.grad(
        sum((gf * c).sum() for gf, c in zip(grads_fused, grad2_cotangents)),
        diff_tensors,
        allow_unused=True,
    )
    ggrads_ref = torch.autograd.grad(
        sum((gr * c).sum() for gr, c in zip(grads_ref, grad2_cotangents)),
        diff_tensors,
        allow_unused=True,
    )
    for ggf, ggr in zip(ggrads_fused, ggrads_ref):
        if ggf is None and ggr is None:
            continue
        assert ggf is not None and ggr is not None
        assert torch.isfinite(ggf).all()
        torch.testing.assert_close(ggf, ggr, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("batch_norm", [False, True])
@pytest.mark.parametrize("num_dst_nodes", [5, 0])
def test_tp_conv_zero_edge_empty_graph(
    reduce: str,
    batch_norm: bool,
    num_dst_nodes: int,
) -> None:
    in_irreps = cue.Irreps("O3", "3x0e + 2x1o")
    sh_irreps = cue.Irreps("O3", "0e + 1o")
    out_irreps = cue.Irreps("O3", "3x0e + 2x1o")

    tp_conv = cuet.layers.FullyConnectedTensorProductConv(
        in_irreps=in_irreps,
        sh_irreps=sh_irreps,
        out_irreps=out_irreps,
        batch_norm=batch_norm,
        mlp_channels=[7, 8],
        mlp_activation=nn.SiLU(),
        layout=cue.mul_ir,
        method="naive",
    ).to(device=device, dtype=torch.float64)

    num_src_nodes = 4 if num_dst_nodes > 0 else 0
    src = torch.empty((0,), device=device, dtype=torch.int32)
    dst = torch.empty((0,), device=device, dtype=torch.int32)
    edge_index = torch.vstack((src, dst))

    src_features = torch.randn(
        num_src_nodes,
        in_irreps.dim,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_sh = torch.empty(
        (0, sh_irreps.dim), device=device, dtype=torch.float64, requires_grad=True
    )
    edge_emb = torch.empty(
        (0, 3), device=device, dtype=torch.float64, requires_grad=True
    )
    src_scalars = torch.randn(
        num_src_nodes, 2, device=device, dtype=torch.float64, requires_grad=True
    )
    dst_scalars = torch.randn(
        num_dst_nodes, 2, device=device, dtype=torch.float64, requires_grad=True
    )
    edge_envelope = torch.empty(
        (0,), device=device, dtype=torch.float64, requires_grad=True
    )

    out = tp_conv(
        src_features=src_features,
        edge_sh=edge_sh,
        edge_emb=edge_emb,
        graph=(edge_index, (num_src_nodes, num_dst_nodes)),
        src_scalars=src_scalars,
        dst_scalars=dst_scalars,
        reduce=reduce,
        edge_envelope=edge_envelope,
    )
    assert out.shape == (num_dst_nodes, out_irreps.dim)
    assert torch.isfinite(out).all()
    if not batch_norm:
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0.0, rtol=0.0)

    grads = torch.autograd.grad(
        out.pow(2).sum(),
        (src_features, edge_sh, edge_emb),
        create_graph=True,
    )
    for g in grads:
        assert torch.isfinite(g).all()
    ggrads = torch.autograd.grad(
        sum(g.pow(2).sum() for g in grads),
        (src_features, edge_sh, edge_emb),
    )
    for gg in ggrads:
        assert torch.isfinite(gg).all()
