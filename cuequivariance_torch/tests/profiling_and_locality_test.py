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
"""Tests for GPU Roofline analyzer and 3D Morton graph locality reordering."""

from __future__ import annotations

import warnings

import pytest
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet

# ---------------------------------------------------------------------------
# 1. GPU Specs & Roofline / Warp-Alignment Analyzer Tests
# ---------------------------------------------------------------------------


def test_exports_and_gpu_specs_registry():
    assert hasattr(cuet, "profiling")
    assert hasattr(cuet, "graph_locality")
    assert cuet.analyze_roofline is cuet.profiling.analyze_roofline
    assert cuet.morton_code_3d is cuet.graph_locality.morton_code_3d
    assert (
        cuet.reorder_graph_for_locality
        is cuet.graph_locality.reorder_graph_for_locality
    )

    required_gpus = {
        "A100_SXM_80GB": (2039.0, 19.5, 312.0, 40 * 1024 * 1024),
        "A100_PCIE_80GB": (1935.0, 19.5, 312.0, 40 * 1024 * 1024),
        "H100_SXM_80GB": (3350.0, 67.0, 989.0, 50 * 1024 * 1024),
        "H100_PCIE_80GB": (2000.0, 51.0, 756.0, 50 * 1024 * 1024),
    }
    for name, (bw, fp32, bf16, l2_bytes) in required_gpus.items():
        spec = cuet.GPU_SPECS[name]
        assert spec.bandwidth_gbps == pytest.approx(bw)
        assert spec.fp32_tflops == pytest.approx(fp32)
        assert spec.bf16_tflops == pytest.approx(bf16)
        assert spec.l2_cache_bytes == l2_bytes

    h100 = cuet.GPU_SPECS["H100_SXM_80GB"]
    assert h100.tf32_tflops == pytest.approx(494.5)
    assert h100.fp8_tflops == pytest.approx(1979.0)
    assert h100.peak_tflops_for_dtype(
        torch.float32, use_tensor_cores=True
    ) == pytest.approx(494.5)

    a100 = cuet.GPU_SPECS["A100_SXM_80GB"]
    assert a100.fp64_tflops == pytest.approx(9.7)
    assert a100.fp64_tc_tflops == pytest.approx(19.5)
    if hasattr(torch, "float8_e4m3fn"):
        with pytest.raises(ValueError, match="FP8"):
            a100.peak_tflops_for_dtype(torch.float8_e4m3fn)


@pytest.mark.parametrize("gpu_name", ["A100_SXM_80GB", "H100_SXM_80GB"])
def test_analyze_roofline_linear_regimes_and_exact_flops(gpu_name: str):
    # Small multiplicity / small batch -> memory-bound (weights dominate HBM reads).
    irreps_small = cue.Irreps("O3", "32x0e + 32x1o")
    lin_small = cuet.Linear(
        irreps_small,
        irreps_small,
        layout=cue.ir_mul,
        shared_weights=True,
        method="naive",
    )
    poly_small = lin_small.f._polynomial_orig
    rep_mem = cuet.analyze_roofline(
        lin_small, batch_sizes=8, gpu=gpu_name, dtype=torch.float32
    )
    assert rep_mem.flops == poly_small.flop(8)
    expected_bytes = poly_small.memory([1, 8, 8]) * 4
    assert rep_mem.hbm_bytes == expected_bytes
    assert rep_mem.regime == "memory-bound"
    assert rep_mem.arithmetic_intensity < rep_mem.ridge_point

    # High multiplicity + large batch + shared weights -> compute-bound in FP32 SIMT.
    irreps_large = cue.Irreps("O3", "256x0e + 256x1o")
    lin_large = cuet.Linear(
        irreps_large,
        irreps_large,
        layout=cue.ir_mul,
        shared_weights=True,
        method="naive",
    )
    poly_large = lin_large.f._polynomial_orig
    rep_cmp = cuet.analyze_roofline(
        lin_large, batch_sizes=4096, gpu=gpu_name, dtype=torch.float32
    )
    assert rep_cmp.flops == poly_large.flop(4096)
    assert rep_cmp.regime == "compute-bound"
    assert rep_cmp.arithmetic_intensity > rep_cmp.ridge_point
    assert any("TF32" in r or "BF16" in r for r in rep_cmp.recommendations)

    # Switching to BF16 Tensor Cores raises ridge point ~15x -> becomes memory-bound again.
    rep_bf16 = cuet.analyze_roofline(
        lin_large, batch_sizes=4096, gpu=gpu_name, dtype=torch.bfloat16
    )
    assert rep_bf16.flops == poly_large.flop(4096)
    assert rep_bf16.hbm_bytes == rep_cmp.hbm_bytes // 2
    assert rep_bf16.regime == "memory-bound"


def test_analyze_roofline_channelwise_gather_scatter_and_l2_boundary():
    irreps_node = cue.Irreps("O3", "64x0e + 64x1o + 64x2e")  # dim = 64 * 9 = 576
    irreps_sh = cue.Irreps("O3", "0e + 1o + 2e")  # dim = 9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cwtp = cuet.ChannelWiseTensorProduct(
            irreps_node, irreps_sh, layout=cue.ir_mul, shared_weights=True
        )
    poly = cwtp.f._polynomial_orig

    num_edges = 12_000
    num_nodes = 2_000
    rep_h100 = cuet.analyze_roofline(
        cwtp,
        batch_sizes=num_edges,
        gpu="H100_SXM_80GB",
        dtype=torch.float32,
        indexed_operands={1: num_nodes, 3: num_nodes},
    )
    assert rep_h100.flops == poly.flop(num_edges)
    assert rep_h100.fits_in_l2 is True
    assert rep_h100.hbm_bytes < rep_h100.cold_hbm_bytes
    assert rep_h100.regime == "memory-bound"
    assert any("reorder_graph_for_locality" in r for r in rep_h100.recommendations)

    # Register accumulation reduces output scatter traffic by 1 * num_nodes * out_bytes.
    rep_reg = cuet.analyze_roofline(
        cwtp,
        batch_sizes=num_edges,
        gpu="H100_SXM_80GB",
        dtype=torch.float32,
        indexed_operands={1: num_nodes, 3: num_nodes},
        scatter_mode="register",
    )
    out_op_bytes = poly.outputs[0].size * 4
    assert rep_h100.hbm_bytes - rep_reg.hbm_bytes == num_nodes * out_op_bytes

    # Working set between 40 MiB (A100 L2) and 50 MiB (H100 L2):
    # Check exact L2 fit divergence between A100_SXM_80GB and H100_SXM_80GB!
    per_node_ws_bytes = (poly.inputs[1].size + poly.outputs[0].size) * 4
    target_ws_bytes = 45 * 1024 * 1024
    n_boundary_nodes = target_ws_bytes // per_node_ws_bytes
    e_boundary = n_boundary_nodes * 16

    rep_a100_bound = cuet.analyze_roofline(
        cwtp,
        batch_sizes=e_boundary,
        gpu="A100_SXM_80GB",
        dtype=torch.float32,
        indexed_operands={1: n_boundary_nodes, 3: n_boundary_nodes},
    )
    rep_h100_bound = cuet.analyze_roofline(
        cwtp,
        batch_sizes=e_boundary,
        gpu="H100_SXM_80GB",
        dtype=torch.float32,
        indexed_operands={1: n_boundary_nodes, 3: n_boundary_nodes},
    )
    assert rep_a100_bound.fits_in_l2 is False
    assert rep_h100_bound.fits_in_l2 is True
    assert rep_a100_bound.hbm_bytes == rep_a100_bound.cold_hbm_bytes
    assert rep_h100_bound.hbm_bytes < rep_a100_bound.hbm_bytes


def test_warp_alignment_diagnostics_mul32_vs_mul17():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc_aligned = cuet.SymmetricContraction(
            cue.Irreps("O3", "32x0e + 32x1o"),
            cue.Irreps("O3", "32x0e"),
            contraction_degree=3,
            num_elements=5,
            layout=cue.ir_mul,
        )
        sc_unaligned = cuet.SymmetricContraction(
            cue.Irreps("O3", "17x0e + 17x1o"),
            cue.Irreps("O3", "17x0e"),
            contraction_degree=3,
            num_elements=5,
            layout=cue.ir_mul,
        )

    rep_32 = cuet.analyze_roofline(sc_aligned, batch_sizes=512, gpu="H100_SXM_80GB")
    rep_17 = cuet.analyze_roofline(sc_unaligned, batch_sizes=512, gpu="H100_SXM_80GB")

    assert rep_32.flops == sc_aligned.etp.polynomial.flop(512)
    assert rep_32.warp_alignment_efficiency == pytest.approx(1.0)
    assert rep_32.unaligned_operands == []

    assert rep_17.flops == sc_unaligned.etp.polynomial.flop(512)
    assert rep_17.warp_alignment_efficiency == pytest.approx(17.0 / 32.0)
    assert rep_17.unaligned_operands == [0, 1, 2]
    assert any("17 -> 32" in r for r in rep_17.recommendations)


def test_backward_and_second_backward_and_other_modules():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cwtp = cuet.ChannelWiseTensorProduct(
            cue.Irreps("O3", "32x0e + 32x1o"),
            cue.Irreps("O3", "0e + 1o"),
            layout=cue.ir_mul,
        )
        fctp = cuet.FullyConnectedTensorProduct(
            cue.Irreps("O3", "16x0e + 16x1o"),
            cue.Irreps("O3", "0e + 1o"),
            cue.Irreps("O3", "16x0e + 16x1o"),
            layout=cue.ir_mul,
            method="naive",
        )
        sh = cuet.SphericalHarmonics(ls=[0, 1, 2], method="naive")
        rot = cuet.Rotation(
            cue.Irreps("O3", "32x0e + 32x1o"), layout=cue.ir_mul, method="naive"
        )

    poly = cwtp.f._polynomial_orig
    bwd_poly, _ = poly.backward(poly.used_inputs(), poly.used_outputs())
    bwd2_poly, _ = bwd_poly.backward(bwd_poly.used_inputs(), bwd_poly.used_outputs())

    B = 256
    r_fwd = cuet.analyze_roofline(cwtp, B, gpu="A100_SXM_80GB")
    r_bwd = cuet.analyze_roofline(cwtp, B, gpu="A100_SXM_80GB", include_backward=True)
    r_bwd2 = cuet.analyze_roofline(
        cwtp, B, gpu="A100_SXM_80GB", include_second_backward=True
    )

    assert r_fwd.flops == poly.flop(B)
    assert r_bwd.flops == poly.flop(B) + bwd_poly.flop(B)
    assert r_bwd2.flops == poly.flop(B) + bwd_poly.flop(B) + bwd2_poly.flop(B)
    assert r_bwd2.hbm_bytes > r_bwd.hbm_bytes > r_fwd.hbm_bytes

    table = r_bwd2.format_table()
    assert "A100_SXM_80GB" in table
    assert "FLOP Breakdown" in table
    assert "Arithmetic Intensity" in table

    # Check extraction from EquivariantPolynomial, SegmentedPolynomial, FCTP, SH, Rotation.
    etp = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2])
    assert cuet.analyze_roofline(etp, 128).flops == etp.flop(128)
    assert cuet.analyze_roofline(etp.polynomial, 128).flops == etp.flop(128)
    assert cuet.analyze_roofline(fctp, 128).flops == fctp.f._polynomial_orig.flop(128)
    assert cuet.analyze_roofline(sh, 128).flops == sh.f._polynomial_orig.flop(128)
    assert cuet.analyze_roofline(rot, 128).flops == rot.f._polynomial_orig.flop(128)


def test_benchmark_roofline_cpu():
    lin = cuet.Linear(
        cue.Irreps("O3", "16x0e + 16x1o"),
        cue.Irreps("O3", "16x0e + 16x1o"),
        layout=cue.ir_mul,
        dtype=torch.float32,
        method="naive",
    )
    x = torch.randn(64, lin.irreps_in.dim, dtype=torch.float32)

    bench_fwd = cuet.profiling.benchmark_roofline(
        lin, x, gpu="H100_PCIE_80GB", num_warmup=1, num_iters=3
    )
    assert bench_fwd.elapsed_time_ms > 0.0
    assert bench_fwd.achieved_tflops > 0.0
    assert bench_fwd.achieved_bandwidth_gbps > 0.0
    assert bench_fwd.pct_of_roofline_peak > 0.0
    assert "Empirical Benchmark" in bench_fwd.format_table()

    bench_bwd2 = cuet.profiling.benchmark_roofline(
        lin,
        x,
        gpu="A100_PCIE_80GB",
        include_second_backward=True,
        num_warmup=1,
        num_iters=2,
    )
    assert bench_bwd2.report.second_backward_flops > 0
    assert bench_bwd2.elapsed_time_ms > 0.0


# ---------------------------------------------------------------------------
# 2. 3D Morton Code & Graph Locality Reordering Tests
# ---------------------------------------------------------------------------


def test_morton_code_3d_edge_cases_and_spatial_prefixes():
    # N = 0 and N = 1
    empty = cuet.morton_code_3d(torch.empty((0, 3), dtype=torch.float64))
    assert empty.shape == (0,)
    assert empty.dtype == torch.int64

    single = cuet.morton_code_3d(torch.tensor([[3.5, -1.2, 9.0]], dtype=torch.float64))
    assert torch.equal(single, torch.zeros((1,), dtype=torch.int64))

    # All identical points (fully degenerate bounding box)
    identical = cuet.morton_code_3d(torch.ones((10, 3), dtype=torch.float32) * 7.0)
    assert torch.equal(identical, torch.zeros((10,), dtype=torch.int64))

    # Collinear 1D line along X (Y and Z degenerate)
    t = torch.linspace(-5.0, 5.0, 17, dtype=torch.float64)
    collinear = torch.stack(
        [t, torch.full_like(t, 2.0), torch.full_like(t, -3.0)], dim=1
    )
    codes_1d = cuet.morton_code_3d(collinear, bits=21)
    assert codes_1d.dtype == torch.int64
    assert bool((codes_1d >= 0).all())
    assert bool((codes_1d[1:] > codes_1d[:-1]).all())

    # Full 21-bit corners -> 0 and 2^63 - 1 (non-negative int64 max!)
    corners = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    corner_codes = cuet.morton_code_3d(corners, bits=21)
    assert int(corner_codes[0].item()) == 0
    assert int(corner_codes[1].item()) == (1 << 63) - 1
    assert int(corner_codes[1].item()) > 0

    # Octree prefix property: points in the same depth-3 octree cell [0.25, 0.35]^3
    # share the top 3 * 3 = 9 MSB bits, whereas a distant cell [0.80, 0.90]^3 differs.
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # anchor min
            [1.0, 1.0, 1.0],  # anchor max
            [0.26, 0.27, 0.28],  # cell (2, 2, 2) at depth 3 (since [0.25, 0.375))
            [0.36, 0.35, 0.34],  # also in cell (2, 2, 2) at depth 3
            [0.85, 0.85, 0.85],  # distant octant (6, 6, 6)
        ],
        dtype=torch.float64,
    )
    c = cuet.morton_code_3d(pts, bits=21)
    shift_depth3 = 3 * (21 - 3)
    assert int((c[2] >> shift_depth3).item()) == int((c[3] >> shift_depth3).item())
    assert int((c[2] >> shift_depth3).item()) != int((c[4] >> shift_depth3).item())


def _build_3d_cubic_graph(
    nx: int, spacing: float, cutoff: float, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a randomly numbered 3D liquid-like perturbed crystal graph."""
    g = torch.Generator().manual_seed(seed)
    grid_1d = torch.arange(nx, dtype=torch.float64) * spacing
    gx, gy, gz = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
    pos = pos + 0.12 * spacing * torch.randn(
        pos.shape, generator=g, dtype=torch.float64
    )

    num_nodes = pos.shape[0]
    species = torch.randint(0, 4, (num_nodes,), generator=g, dtype=torch.int64)

    # Shuffle node ordering randomly to simulate un-ordered MD / crystal input.
    rand_node_order = torch.randperm(num_nodes, generator=g)
    pos = pos[rand_node_order]
    species = species[rand_node_order]

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 3): pos[dst] - pos[src]
    dist = torch.linalg.norm(diff, dim=-1)
    mask = (dist > 1e-8) & (dist < cutoff)
    src, dst = torch.where(mask)
    edge_index = torch.stack([src, dst], dim=0)

    # Shuffle edges randomly as well.
    rand_edge_order = torch.randperm(edge_index.shape[1], generator=g)
    edge_index = edge_index[:, rand_edge_order]
    return pos, edge_index, species


def test_reorder_graph_invariants_and_warp_cache_locality():
    pos, edge_index, species = _build_3d_cubic_graph(
        nx=8, spacing=1.5, cutoff=3.3, seed=123
    )
    num_nodes = pos.shape[0]
    num_edges = edge_index.shape[1]
    assert num_nodes == 512
    assert num_edges > 4000

    metrics_before = cuet.estimate_edge_bandwidth_metrics(
        edge_index, num_nodes=num_nodes, window_size=32
    )

    pos_r, edge_r, species_r, perm = cuet.reorder_graph_for_locality(
        pos, edge_index, species, sort_by_species_first=True, primary_edge_key="dst"
    )

    # Check permutation bijections and species monotonicity.
    assert species_r is not None
    assert perm.species_counts is not None
    assert bool((species_r[1:] >= species_r[:-1]).all())
    assert torch.equal(perm.species_counts, torch.bincount(species_r))
    assert torch.equal(
        perm.node_perm[perm.inv_node_perm], torch.arange(num_nodes, dtype=torch.int64)
    )
    assert torch.equal(
        perm.edge_perm[perm.inv_edge_perm], torch.arange(num_edges, dtype=torch.int64)
    )
    assert torch.equal(cuet.restore_node_order(pos_r, perm), pos)

    # Check lexicographic (dst, src) edge sorting.
    dst_r, src_r = edge_r[1], edge_r[0]
    assert bool((dst_r[1:] >= dst_r[:-1]).all())
    same_dst = dst_r[1:] == dst_r[:-1]
    assert bool((src_r[1:][same_dst] >= src_r[:-1][same_dst]).all())

    # Compare warp cache-line touches before vs after Morton + (dst, src) reordering!
    metrics_after_species_morton = cuet.estimate_edge_bandwidth_metrics(
        edge_r, num_nodes=num_nodes, window_size=32
    )
    _, edge_pure_morton, _, _ = cuet.reorder_graph_for_locality(
        pos, edge_index, species=None, primary_edge_key="dst"
    )
    metrics_after_pure_morton = cuet.estimate_edge_bandwidth_metrics(
        edge_pure_morton, num_nodes=num_nodes, window_size=32
    )

    for m_after in (metrics_after_species_morton, metrics_after_pure_morton):
        assert (
            m_after["unique_dst_per_window"] < metrics_before["unique_dst_per_window"]
        )
        assert (
            m_after["unique_src_per_window"] < metrics_before["unique_src_per_window"]
        )
        assert (
            m_after["unique_dst_cache_lines_per_window"]
            < metrics_before["unique_dst_cache_lines_per_window"]
        )
        assert (
            m_after["unique_src_cache_lines_per_window"]
            < metrics_before["unique_src_cache_lines_per_window"]
        )
        assert m_after["mean_edge_span"] < metrics_before["mean_edge_span"]


# ---------------------------------------------------------------------------
# 3. End-to-End Equivariant MD Force Conservation & Permutation Invariance
# ---------------------------------------------------------------------------


class _MiniEquivariantMDPotential(torch.nn.Module):
    """Two-layer O(3)-equivariant NequIP-style message-passing potential E(R, Z)."""

    def __init__(self, num_species: int, cutoff: float, linear_method: str):
        super().__init__()
        self.num_species = num_species
        self.cutoff = cutoff

        self.irreps_scalar = cue.Irreps("O3", "8x0e")
        self.irreps_hidden = cue.Irreps("O3", "8x0e + 8x1o + 8x2e")
        self.irreps_sh = cue.Irreps("O3", "0e + 1o + 2e")
        self.irreps_scalar_out = cue.Irreps("O3", "1x0e")

        g = torch.Generator().manual_seed(2026)
        # Atomic species embedding is strictly O(3)-scalar (0e).
        self.species_embed = torch.nn.Parameter(
            torch.randn(
                num_species,
                self.irreps_scalar.dim,
                generator=g,
                dtype=torch.float64,
            )
            * 0.25
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sh = cuet.SphericalHarmonics(
                ls=[0, 1, 2], normalize=True, math_dtype=torch.float64, method="naive"
            )
            self.tp1 = cuet.ChannelWiseTensorProduct(
                self.irreps_scalar,
                self.irreps_sh,
                layout=cue.ir_mul,
                shared_weights=False,
                internal_weights=False,
                dtype=torch.float64,
                math_dtype=torch.float64,
                method="naive",
            )
            self.linear1 = cuet.Linear(
                self.tp1.irreps_out,
                self.irreps_hidden,
                layout=cue.ir_mul,
                shared_weights=True,
                internal_weights=True,
                weight_classes=num_species,
                dtype=torch.float64,
                method=linear_method,
            )
            self.tp2 = cuet.ChannelWiseTensorProduct(
                self.irreps_hidden,
                self.irreps_sh,
                layout=cue.ir_mul,
                shared_weights=False,
                internal_weights=False,
                dtype=torch.float64,
                math_dtype=torch.float64,
                method="naive",
            )
            self.linear2 = cuet.Linear(
                self.tp2.irreps_out,
                self.irreps_hidden,
                layout=cue.ir_mul,
                shared_weights=True,
                internal_weights=True,
                weight_classes=num_species,
                dtype=torch.float64,
                method=linear_method,
            )
            self.energy_readout = cuet.Linear(
                self.irreps_hidden,
                self.irreps_scalar_out,
                layout=cue.ir_mul,
                shared_weights=True,
                internal_weights=True,
                weight_classes=num_species,
                dtype=torch.float64,
                method=linear_method,
            )
            self.radial_w1 = torch.nn.Parameter(
                torch.randn(8, 16, generator=g, dtype=torch.float64) * 0.3
            )
            self.radial_out1 = torch.nn.Parameter(
                torch.randn(16, self.tp1.weight_numel, generator=g, dtype=torch.float64)
                * 0.2
            )
            self.radial_out2 = torch.nn.Parameter(
                torch.randn(16, self.tp2.weight_numel, generator=g, dtype=torch.float64)
                * 0.2
            )

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        species: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = positions.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Edge displacement and smooth polynomial cutoff envelope
        edge_vec = positions[dst] - positions[src]  # (E, 3)
        dist = torch.linalg.norm(edge_vec, dim=-1)  # (E,)
        u = torch.clamp(dist / self.cutoff, max=1.0)
        envelope = 1.0 - 3.0 * (u**2) + 2.0 * (u**3)  # C^1 smooth cutoff in [0, 1]

        # Radial basis -> per-layer TP edge weights
        freqs = torch.arange(1, 9, dtype=positions.dtype, device=positions.device)
        rbf = torch.sin(freqs * torch.pi * u.unsqueeze(-1)) * envelope.unsqueeze(-1)
        radial_hidden = torch.tanh(rbf @ self.radial_w1)
        w1 = radial_hidden @ self.radial_out1
        w2 = radial_hidden @ self.radial_out2

        edge_sh = self.sh(edge_vec)  # (E, 9)
        h0 = self.species_embed[species]  # (N, irreps_scalar.dim)

        # Layer 1: lift scalar species features into l=0,1,2 geometric node features
        msg1 = self.tp1(
            h0,
            edge_sh,
            weight=w1,
            indices_1=src,
            indices_out=dst,
            size_out=num_nodes,
        )
        h1 = self.linear1(msg1, weight_indices=species)

        # Layer 2: full l=0,1,2 tensor-product message passing + residual
        msg2 = self.tp2(
            h1,
            edge_sh,
            weight=w2,
            indices_1=src,
            indices_out=dst,
            size_out=num_nodes,
        )
        h2 = h1 + self.linear2(msg2, weight_indices=species)

        # Species-indexed O(3)-invariant atomic energy readout
        atomic_energies = self.energy_readout(h2, weight_indices=species).squeeze(-1)
        return atomic_energies.sum()


def test_end_to_end_md_energy_and_force_permutation_invariance_fp64():
    cutoff = 3.2
    pos_orig, edge_orig, species_orig = _build_3d_cubic_graph(
        nx=4, spacing=1.45, cutoff=cutoff, seed=777
    )
    num_species = 4

    # Reference model on random-order graph (uses method="naive" since species_orig is unsorted)
    model_ref = _MiniEquivariantMDPotential(
        num_species=num_species, cutoff=cutoff, linear_method="naive"
    )
    # Reordered model using method="indexed_linear" directly on sorted species_reordered!
    model_indexed = _MiniEquivariantMDPotential(
        num_species=num_species, cutoff=cutoff, linear_method="indexed_linear"
    )
    model_indexed.load_state_dict(model_ref.state_dict())

    # 1. Evaluate E and conservative forces F = -dE/dR on un-reordered random graph
    pos_in_orig = pos_orig.detach().clone().requires_grad_(True)
    energy_orig = model_ref(pos_in_orig, edge_orig, species_orig)
    (grad_orig,) = torch.autograd.grad(energy_orig, pos_in_orig)
    forces_orig = -grad_orig

    # 2. Reorder graph via Morton space-filling curve within species + (dst, src) edge sort
    pos_reord, edge_reord, species_reord, perm = cuet.reorder_graph_for_locality(
        pos_orig,
        edge_orig,
        species_orig,
        sort_by_species_first=True,
        primary_edge_key="dst",
    )
    assert species_reord is not None
    assert bool((species_reord[1:] >= species_reord[:-1]).all())

    pos_in_reord = pos_reord.detach().clone().requires_grad_(True)
    energy_reord = model_indexed(pos_in_reord, edge_reord, species_reord)
    (grad_reord,) = torch.autograd.grad(energy_reord, pos_in_reord)
    forces_reord = -grad_reord
    forces_restored = cuet.restore_node_order(forces_reord, perm)

    # 3. Exact total energy and per-atom conservative force invariance (< 1e-12 in FP64!)
    energy_diff = float(torch.abs(energy_reord - energy_orig).item())
    max_force_diff = float(torch.max(torch.abs(forces_restored - forces_orig)).item())
    assert energy_diff < 1e-12, f"Energy mismatch: {energy_diff:.3e}"
    assert max_force_diff < 1e-12, f"Max force mismatch: {max_force_diff:.3e}"

    # 4. Physical conservation laws: net force == 0 (translation invariance) & O(3) equivariance
    net_force = float(torch.max(torch.abs(forces_restored.sum(dim=0))).item())
    assert net_force < 1e-12, f"Non-zero net force: {net_force:.3e}"

    # Apply a proper 3D rotation Q in SO(3) and verify F(R @ Q^T) == F(R) @ Q^T
    skew = torch.tensor(
        [[0.0, -0.7, 0.4], [0.7, 0.0, -0.5], [-0.4, 0.5, 0.0]], dtype=torch.float64
    )
    rot_q = torch.matrix_exp(skew)
    pos_rot = (pos_reord @ rot_q.T).detach().requires_grad_(True)
    e_rot = model_indexed(pos_rot, edge_reord, species_reord)
    (g_rot,) = torch.autograd.grad(e_rot, pos_rot)
    forces_rot = -g_rot

    assert float(torch.abs(e_rot - energy_reord).item()) < 1e-12
    rot_force_err = float(
        torch.max(torch.abs(forces_rot - (forces_reord @ rot_q.T))).item()
    )
    assert rot_force_err < 1e-12, f"SO(3) force equivariance error: {rot_force_err:.3e}"

    # Improper O(3) parity inversion R -> -R: E(-R) == E(R) and F(-R) == -F(R)
    pos_inv = (-pos_reord).detach().requires_grad_(True)
    e_inv = model_indexed(pos_inv, edge_reord, species_reord)
    (g_inv,) = torch.autograd.grad(e_inv, pos_inv)
    forces_inv = -g_inv
    assert float(torch.abs(e_inv - energy_reord).item()) < 1e-12
    assert float(torch.max(torch.abs(forces_inv + forces_reord)).item()) < 1e-12

    # Central finite-difference check confirming F_i = -dE/dR_i is strictly conservative
    eps = 1e-6
    atom_idx = 0
    fd_force = torch.zeros(3, dtype=torch.float64)
    with torch.no_grad():
        for alpha in range(3):
            p_plus = pos_reord.clone()
            p_minus = pos_reord.clone()
            p_plus[atom_idx, alpha] += eps
            p_minus[atom_idx, alpha] -= eps
            e_plus = model_indexed(p_plus, edge_reord, species_reord)
            e_minus = model_indexed(p_minus, edge_reord, species_reord)
            fd_force[alpha] = -(e_plus - e_minus) / (2.0 * eps)
    assert float(torch.max(torch.abs(fd_force - forces_reord[atom_idx])).item()) < 1e-8


def test_empty_graphs_src_sorting_and_validation_errors():
    # Empty graph (N=0, E=0) and edgeless graph (N=5, E=0)
    p0, e0, s0, perm0 = cuet.reorder_graph_for_locality(
        torch.empty((0, 3), dtype=torch.float64),
        torch.empty((2, 0), dtype=torch.int64),
        torch.empty((0,), dtype=torch.int64),
    )
    assert p0.shape == (0, 3)
    assert e0.shape == (2, 0)
    assert s0 is not None and s0.shape == (0,)
    assert perm0.species_counts is not None and perm0.species_counts.numel() == 0

    m0 = cuet.estimate_edge_bandwidth_metrics(e0, num_nodes=0)
    assert m0["unique_src_per_window"] == 0.0
    assert m0["unique_dst_per_window"] == 0.0

    # primary_edge_key="src" + restore_edge_order
    pos, edge_index, species = _build_3d_cubic_graph(
        nx=3, spacing=1.5, cutoff=3.0, seed=99
    )
    edge_attr = torch.randn(edge_index.shape[1], 4, dtype=torch.float64)
    _, edge_src_sorted, _, perm_src = cuet.reorder_graph_for_locality(
        pos,
        edge_index,
        species,
        sort_by_species_first=False,
        primary_edge_key="src",
    )
    src_s, dst_s = edge_src_sorted[0], edge_src_sorted[1]
    assert bool((src_s[1:] >= src_s[:-1]).all())
    same_src = src_s[1:] == src_s[:-1]
    assert bool((dst_s[1:][same_src] >= dst_s[:-1][same_src]).all())

    edge_attr_reord = edge_attr[perm_src.edge_perm]
    assert torch.equal(cuet.restore_edge_order(edge_attr_reord, perm_src), edge_attr)

    # Validation checks
    with pytest.raises(ValueError, match="bits"):
        cuet.morton_code_3d(pos, bits=22)
    with pytest.raises(ValueError, match="shape"):
        cuet.morton_code_3d(torch.zeros((4, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        cuet.reorder_graph_for_locality(
            pos, edge_index, torch.full((pos.shape[0],), -1, dtype=torch.int64)
        )
    with pytest.raises(ValueError, match="primary_edge_key"):
        cuet.reorder_graph_for_locality(pos, edge_index, primary_edge_key="invalid")


def test_roofline_aliases_sequence_batches_and_conv_layer():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv = cuet.layers.FullyConnectedTensorProductConv(
            in_irreps=cue.Irreps("O3", "8x0e + 8x1o"),
            sh_irreps=cue.Irreps("O3", "0e + 1o"),
            out_irreps=cue.Irreps("O3", "8x0e + 8x1o"),
            mlp_channels=[6, 16],
            layout=cue.ir_mul,
            use_fallback=True,
        )
        inv = cuet.Inversion(
            cue.Irreps("O3", "32x0e + 32x1o"), layout=cue.ir_mul, method="naive"
        )

    # Alias lookup + negative indexed_operands key + explicit per-operand batch_sizes
    rep_conv = cuet.analyze_roofline(
        conv,
        batch_sizes=[1000, 100, 1000, 100],
        gpu="a100-sxm",
        dtype="float64",
        indexed_operands={1: 100, -1: 100},
    )
    assert rep_conv.gpu_name == "A100_SXM_80GB"
    assert rep_conv.flops == conv.tp.f._polynomial_orig.flop(1000)

    rep_inv = cuet.analyze_roofline(inv, batch_sizes=256, gpu="h100")
    assert rep_inv.gpu_name == "H100_SXM_80GB"
    assert rep_inv.flops == inv.f._polynomial_orig.flop(256)
