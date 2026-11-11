# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Architecture Note 07: Equivariant Operators on NVIDIA A100 vs H100 Roofline Model.

Demonstrates:
1. Analytical Roofline comparison between A100-SXM4-80GB (HBM2e 2.039 TB/s) and
   H100-SXM5-80GB (HBM3 3.35 TB/s) across 6 canonical equivariant workloads using
   exact `poly.flop(batch)` and `poly.memory(batch_sizes)` + RMW scatter traffic:
   - Proves why SphericalHarmonics, ChannelWiseTP, EdgeRotation (3-angle vs 2-angle),
     and MACE SymmetricContraction are strictly Memory-Bandwidth / SIMT Bound (making
     fused gather/scatter, `ir_mul` warp coalescing, and `yx_rotation` decisive).
   - Proves why high-channel `Linear` / `eSCN` SO(2) and `TriangleMultiplicativeUpdate`
     cross into the Tensor-Core Compute-Bound regime where H100 TF32/BF16/FP8 shines.
2. GPU warp alignment (`segment_extent % 32 == 0`) & 128-byte cache-sector audit.
3. Hardware `torch.float8_e4m3fn` SO(3) equivariance audit: why per-component scaling/clipping
   shatters equivariance, how radial irrep-norm block scaling preserves exact O(3) symmetry
   of the scale/clip envelope, and why 3-bit E4M3 mantissa ripple restricts FP8 to l=0 MLPs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.spatial.transform
import torch
from cuequivariance.group_theory.experimental import escn

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors

logging.getLogger(
    "cuequivariance_torch.primitives.segmented_polynomial_naive"
).setLevel(logging.ERROR)


def section(title: str) -> None:
    print(f"\n{'=' * 86}\n{title}\n{'=' * 86}")


@dataclass(frozen=True)
class GPUSpec:
    name: str
    hbm_tb_s: float
    fp64_tf: float
    fp32_cuda_tf: float
    tf32_tf: float
    bf16_tf: float
    fp8_tf: Optional[float]

    def ridge_point(self, peak_tf: float) -> float:
        """FLOP/Byte required to saturate compute units at peak HBM bandwidth."""
        return peak_tf / self.hbm_tb_s

    def attainable_tflops(self, arith_intensity: float, peak_tf: float) -> float:
        return min(peak_tf, arith_intensity * self.hbm_tb_s)


A100 = GPUSpec(
    name="A100-SXM4-80GB",
    hbm_tb_s=2.039,
    fp64_tf=19.5,
    fp32_cuda_tf=19.5,
    tf32_tf=156.0,
    bf16_tf=312.0,
    fp8_tf=None,
)

H100 = GPUSpec(
    name="H100-SXM5-80GB",
    hbm_tb_s=3.350,
    fp64_tf=67.0,
    fp32_cuda_tf=67.0,
    tf32_tf=494.5,
    bf16_tf=989.0,
    fp8_tf=1979.0,
)


@dataclass(frozen=True)
class WorkloadMetrics:
    name: str
    engine: str  # "SIMT" (uniform_1d / poly) vs "TensorCore" (GEMM / Triton)
    flops: int
    bytes_fp32: int
    bytes_bf16: int

    @property
    def ai_fp32(self) -> float:
        return self.flops / self.bytes_fp32

    @property
    def ai_bf16(self) -> float:
        return self.flops / self.bytes_bf16


def build_six_canonical_workloads() -> list[WorkloadMetrics]:
    n_atoms = 4096
    z_coord = 30
    n_edges = n_atoms * z_coord  # 122,880 directed edges
    z_species = 10

    # 1. SphericalHarmonics (l_max = 3, E edges)
    sh_poly = descriptors.spherical_harmonics(cue.O3(1, -1), [0, 1, 2, 3]).polynomial
    sh_flops = sh_poly.flop(n_edges)
    sh_elems = sh_poly.memory([n_edges, n_edges])

    # 2a & 2b. ChannelWiseTP (C=128, l_max=2, E edges): Unfused vs Fused gather/scatter
    ir_cw = cue.Irreps("O3", "128x0e + 128x1o + 128x2e")
    ir_sh2 = cue.Irreps("O3", "0e + 1o + 2e")
    cw_poly = (
        descriptors.channelwise_tensor_product(
            ir_cw,
            ir_sh2,
            [cue.O3(0, 1), cue.O3(1, -1), cue.O3(2, 1)],
            simplify_irreps3=True,
        )
        .flatten_coefficient_modes()
        .squeeze_modes()
        .polynomial
    )
    cw_flops = cw_poly.flop(n_edges)
    # Unfused: gather h[src] (read+write E*D), TP (read w, h_src, sh, write msg), scatter (read msg + RMW 2*N*D)
    d_in = cw_poly.inputs[1].size
    d_out = cw_poly.outputs[0].size
    cw_unfused_elems = (
        2 * n_edges * d_in
        + cw_poly.memory([n_edges, n_edges, n_edges, n_edges])
        + n_edges * d_out
        + 2 * n_atoms * d_out
    )
    # Fused uniform_1d: stream/L2-gather h_src, read w & sh, register-accumulate & RMW scatter 2*N*D (zero E*D msg write!)
    cw_fused_elems = cw_poly.memory([n_edges, n_edges, n_edges, 2 * n_atoms])

    # 3a & 3b. EdgeRotation (C=128, l_max=4, E edges): 3-angle yxy vs 2-angle yx
    ir_rot = cue.Irreps("SO3", "128x0 + 128x1 + 128x2 + 128x3 + 128x4")
    yxy_poly = descriptors.yxy_rotation(ir_rot).polynomial
    yx_poly = descriptors.yx_rotation(ir_rot).polynomial
    yxy_flops = yxy_poly.flop(n_edges)
    yx_flops = yx_poly.flop(n_edges)
    yxy_elems = yxy_poly.memory([n_edges] * yxy_poly.num_operands)
    yx_elems = yx_poly.memory([n_edges] * yx_poly.num_operands)

    # 4. MACE SymmetricContraction (nu=1,2,3, C=128, l_max=2, N atoms, Z=10 species)
    sc_poly = descriptors.symmetric_contraction(ir_cw, ir_cw, (1, 2, 3)).polynomial
    sc_flops = sc_poly.flop(n_atoms)
    sc_naive_elems = sc_poly.memory(
        [2 * n_atoms, n_atoms, n_atoms]
    )  # W[species] gather
    sc_fused_elems = sc_poly.memory(
        [z_species, n_atoms, n_atoms]
    )  # L2-resident Z table

    # 5. High-Channel Linear / eSCN SO(2) Compact Block GEMM (C=512, l_max=3, shared weights)
    ir_lin = cue.Irreps("SO3", "512x0 + 512x1 + 512x2 + 512x3")
    lin_poly = descriptors.linear(ir_lin, ir_lin).polynomial
    n_batch_lin = 16384
    lin_flops = lin_poly.flop(n_batch_lin)
    lin_elems = lin_poly.memory([1, n_batch_lin, n_batch_lin])

    escn_cmp = escn.escn_tp_compact(
        cue.Irreps("SO3", "256x0 + 256x1 + 256x2 + 256x3 + 256x4"),
        cue.Irreps("SO3", "256x0 + 256x1 + 256x2 + 256x3 + 256x4"),
        m_max=2,
    )
    assert escn_cmp.operations[0][1].num_paths == 9

    # 6. Triangle Multiplicative Update (B=1, N_seq=1536, C=D=128)
    n_seq, c_pair, d_tri = 1536, 128, 128
    tri_in_gemm = 2 * (n_seq * n_seq) * c_pair * (4 * d_tri)
    tri_contract = 2 * d_tri * (n_seq**3)
    tri_out_gemm = 2 * (n_seq * n_seq) * (d_tri * c_pair + c_pair * c_pair)
    tri_flops = tri_in_gemm + tri_contract + tri_out_gemm
    # Read x_in + write/read a,b + write/read tri + write x_out + weights
    tri_elems = (
        (2 + 2 + 2 + 1) * (n_seq * n_seq * d_tri)
        + 4 * d_tri * c_pair
        + 2 * c_pair * d_tri
    )

    return [
        WorkloadMetrics(
            "1. SphericalHarmonics (l<=3)", "SIMT", sh_flops, sh_elems * 4, sh_elems * 2
        ),
        WorkloadMetrics(
            "2a. ChannelWiseTP Unfused",
            "SIMT",
            cw_flops,
            cw_unfused_elems * 4,
            cw_unfused_elems * 2,
        ),
        WorkloadMetrics(
            "2b. ChannelWiseTP Fused u1d",
            "SIMT",
            cw_flops,
            cw_fused_elems * 4,
            cw_fused_elems * 2,
        ),
        WorkloadMetrics(
            "3a. EdgeRot 3-Angle (yxy, l<=4)",
            "SIMT",
            yxy_flops,
            yxy_elems * 4,
            yxy_elems * 2,
        ),
        WorkloadMetrics(
            "3b. EdgeRot 2-Angle (yx,  l<=4)",
            "SIMT",
            yx_flops,
            yx_elems * 4,
            yx_elems * 2,
        ),
        WorkloadMetrics(
            "4a. MACE SymCtr Naive Gather",
            "SIMT",
            sc_flops,
            sc_naive_elems * 4,
            sc_naive_elems * 2,
        ),
        WorkloadMetrics(
            "4b. MACE SymCtr Fused u1d",
            "SIMT",
            sc_flops,
            sc_fused_elems * 4,
            sc_fused_elems * 2,
        ),
        WorkloadMetrics(
            "5. Linear / SO(2) GEMM (C=512)",
            "TensorCore",
            lin_flops,
            lin_elems * 4,
            lin_elems * 2,
        ),
        WorkloadMetrics(
            "6. TriangleMulUpdate (N=1536)",
            "TensorCore",
            tri_flops,
            tri_elems * 4,
            tri_elems * 2,
        ),
    ]


def audit_a100_vs_h100_roofline() -> None:
    section("1. NVIDIA A100-SXM4 vs H100-SXM5 Analytical Roofline Audit (6 Workloads)")
    for gpu in (A100, H100):
        fp8_str = (
            f"{gpu.ridge_point(gpu.fp8_tf):6.1f} FLOP/B"
            if gpu.fp8_tf is not None
            else "   N/A       "
        )
        print(
            f"  {gpu.name:<15s} | HBM={gpu.hbm_tb_s:5.3f} TB/s | Ridge Points -> "
            f"FP64={gpu.ridge_point(gpu.fp64_tf):5.1f}, "
            f"TF32={gpu.ridge_point(gpu.tf32_tf):5.1f}, "
            f"BF16={gpu.ridge_point(gpu.bf16_tf):5.1f}, FP8={fp8_str}"
        )

    workloads = build_six_canonical_workloads()
    print(
        f"\n  {'Workload':<31s} | {'AI(BF16)':>8s} | {'A100 TFLOP/s':>12s} | "
        f"{'H100 TFLOP/s':>12s} | {'H100/A100':>9s} | {'Regime on H100 (BF16 TC)':<24s}"
    )
    print(
        f"  {'-' * 31}-+-{'-' * 8}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 9}-+-{'-' * 24}"
    )

    h100_bf16_ridge = H100.ridge_point(H100.bf16_tf)
    a100_bf16_ridge = A100.ridge_point(A100.bf16_tf)

    by_name = {w.name: w for w in workloads}
    for w in workloads:
        # Even if BF16 Tensor Cores were available everywhere, check AI vs BF16 TC Roofline
        t_a100 = A100.attainable_tflops(w.ai_bf16, A100.bf16_tf)
        t_h100 = H100.attainable_tflops(w.ai_bf16, H100.bf16_tf)
        speedup = t_h100 / t_a100
        regime = (
            "Tensor-Core Compute-Bound"
            if w.ai_bf16 >= h100_bf16_ridge
            else (
                "Compute(A100)->BW(H100)"
                if w.ai_bf16 >= a100_bf16_ridge
                else "Memory-Bandwidth Bound"
            )
        )
        print(
            f"  {w.name:<31s} | {w.ai_bf16:8.1f} | {t_a100:12.1f} | "
            f"{t_h100:12.1f} | {speedup:8.2f}x | {regime:<24s}"
        )

    # Key architectural assertions
    # (1) CG Tensor Products & Spherical Harmonics are strictly below BF16 Tensor Core ridge points
    assert by_name["1. SphericalHarmonics (l<=3)"].ai_bf16 < a100_bf16_ridge
    assert by_name["2b. ChannelWiseTP Fused u1d"].ai_bf16 < a100_bf16_ridge
    # (2) Fusing gather/scatter in ChannelWiseTP boosts Arithmetic Intensity by > 2.4x!
    cw_gain = (
        by_name["2b. ChannelWiseTP Fused u1d"].ai_bf16
        / by_name["2a. ChannelWiseTP Unfused"].ai_bf16
    )
    assert cw_gain > 2.4
    # (3) 2-Angle yx_rotation cuts total wall-time FLOPs by > 45% vs 3-Angle yxy_rotation
    rot_saving = 1.0 - (
        by_name["3b. EdgeRot 2-Angle (yx,  l<=4)"].flops
        / by_name["3a. EdgeRot 3-Angle (yxy, l<=4)"].flops
    )
    assert rot_saving > 0.45
    # (4) High-C Linear crosses A100 TF32 & BF16 ridge points; Triangle crosses BOTH A100 & H100 BF16 + FP8 ridge points!
    assert by_name["5. Linear / SO(2) GEMM (C=512)"].ai_bf16 > a100_bf16_ridge
    assert by_name["5. Linear / SO(2) GEMM (C=512)"].ai_fp32 > A100.ridge_point(
        A100.tf32_tf
    )
    assert by_name["6. TriangleMulUpdate (N=1536)"].ai_bf16 > h100_bf16_ridge

    # Also evaluate H100 FP8 speedup on TriangleMulUpdate (1 B/elem -> 2x AI!)
    tri_ai_fp8 = by_name["6. TriangleMulUpdate (N=1536)"].ai_bf16 * 2.0
    assert H100.fp8_tf is not None
    assert tri_ai_fp8 > H100.ridge_point(H100.fp8_tf)
    tri_h100_fp8_tf = H100.attainable_tflops(tri_ai_fp8, H100.fp8_tf)
    tri_a100_bf16_tf = A100.attainable_tflops(
        by_name["6. TriangleMulUpdate (N=1536)"].ai_bf16, A100.bf16_tf
    )
    print(
        f"\n  [Fusion & Tensor Core Highlights]\n"
        f"    ChannelWiseTP fused gather/scatter AI boost : {cw_gain:.2f}x less HBM traffic per FLOP\n"
        f"    EdgeRot 2-angle (yx) vs 3-angle (yxy) FLOPs : -{100.0 * rot_saving:.1f}% FLOPs\n"
        f"    TriangleMulUpdate H100 FP8 vs A100 BF16     : {tri_h100_fp8_tf:.0f} vs {tri_a100_bf16_tf:.0f} TFLOP/s "
        f"({tri_h100_fp8_tf / tri_a100_bf16_tf:.2f}x Roofline speedup!)"
    )


def audit_warp_alignment_and_cache_sectors() -> None:
    section(
        "2. GPU Warp Alignment (`segment_extent % 32 == 0`) & 128-Byte Cache Sectors"
    )
    warp_lanes = 32
    cache_line_bytes = 128  # 32 x FP32 = 128 B sector transaction

    print(
        f"  {'mul (u)':>7s} | {'u % 32':>6s} | {'Warps/Seg':>9s} | {'Lane Util':>9s} | {'FP32 Seg Bytes':>14s} | {'128B Aligned?':>13s}"
    )
    print(f"  {'-' * 7}-+-{'-' * 6}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 14}-+-{'-' * 13}")
    for u in (30, 32, 64, 96, 100, 128, 256):
        warps = math.ceil(u / warp_lanes)
        lane_util = 100.0 * u / (warps * warp_lanes)
        seg_bytes = u * 4
        aligned = (seg_bytes % cache_line_bytes) == 0
        print(
            f"  {u:7d} | {u % 32:6d} | {warps:9d} | {lane_util:8.1f}% | {seg_bytes:12d} B | {str(aligned):>13s}"
        )
        if u % 32 == 0:
            assert lane_util == 100.0 and aligned
        else:
            assert lane_util < 100.0 and not aligned

    # Verify GCD split_mode("u", 32) on tapered irreps guarantees warp-aligned uniform segments
    tapered = cue.Irreps("O3", "128x0e + 64x1o + 32x2e")
    poly = (
        descriptors.channelwise_tensor_product(
            tapered, cue.Irreps("O3", "0e + 1o + 2e"), tapered
        )
        .polynomial.flatten_coefficient_modes()
        .squeeze_modes()
        .split_mode("u", 32)
    )
    assert poly.all_same_segment_shape()
    assert poly.operands[1].segment_shape == (32,)
    print(
        f"  Tapered '{tapered}' + split_mode('u', 32) -> uniform segment={poly.operands[1].segment_shape} (100% warp occupancy!)"
    )


def quantize_per_component_fp8(
    x: torch.Tensor, clip_quantile: float = 0.90, use_fp8_grid: bool = True
) -> torch.Tensor:
    """Naive Transformer-style per-column scaling/clipping across magnetic index m."""
    fmax = 448.0  # torch.float8_e4m3fn max normal
    # Separate scale per feature column -> anisotropic across m in {-l, ..., +l}!
    col_scale = torch.quantile(x.abs(), clip_quantile, dim=0, keepdim=True).clamp(
        min=1e-12
    )
    scaled = (x / col_scale * fmax).clamp(-fmax, fmax)
    if use_fp8_grid:
        scaled = scaled.to(torch.float8_e4m3fn).to(x.dtype)
    return scaled * (col_scale / fmax)


def quantize_irrep_norm_fp8(
    x: torch.Tensor,
    irreps: cue.Irreps,
    clip_quantile: float = 0.90,
    use_fp8_grid: bool = True,
) -> torch.Tensor:
    """O(3)-aware block scaling using Euclidean irrep norms ||x_{b, l, c}||_2 (ir_mul layout)."""
    fmax = 448.0
    batch = x.shape[0]
    chunks = []
    offset = 0
    for mul, ir in irreps:
        seg = x[:, offset : offset + ir.dim * mul].reshape(batch, ir.dim, mul)
        offset += ir.dim * mul
        # ||seg||_2 over magnetic axis m (dim=1) is strictly O(3)-invariant!
        norms = torch.linalg.norm(seg, dim=1, keepdim=True)  # (B, 1, mul)
        rad_clip = torch.quantile(norms, clip_quantile, dim=0, keepdim=True).clamp(
            min=1e-12
        )  # (1, 1, mul)
        # Radial saturation preserves exact 3D vector/tensor orientation!
        factor = torch.clamp(rad_clip / norms.clamp(min=1e-12), max=1.0)
        seg_clipped = seg * factor
        scaled = seg_clipped * (fmax / rad_clip)
        if use_fp8_grid:
            scaled = scaled.to(torch.float8_e4m3fn).to(x.dtype)
        chunks.append((scaled * (rad_clip / fmax)).reshape(batch, ir.dim * mul))
    return torch.cat(chunks, dim=-1)


def audit_fp8_so3_equivariance() -> None:
    section(
        "3. FP8 (`float8_e4m3fn`) SO(3) Equivariance Audit: Per-Component vs Irrep-Norm"
    )
    torch.manual_seed(2026)
    irreps = cue.Irreps("O3", "32x1o + 32x2e")
    ir_layout = cue.IrrepsAndLayout(irreps, cue.ir_mul)
    lin = cuet.Linear(
        irreps,
        irreps,
        layout=cue.ir_mul,
        method="naive",
        dtype=torch.float64,
        math_dtype=torch.float64,
    )

    # Simulate slab/interface MD batch with strong z-axis anisotropy (8x larger along m=0)
    batch = 64
    x = torch.randn(batch, irreps.dim, dtype=torch.float64)
    x[:, 32:64] *= 8.0  # Boost l=1, m=0 component
    x[:, 96 + 64 : 96 + 96] *= 8.0  # Boost l=2, m=0 component

    rng = np.random.default_rng(55)
    rotvec = scipy.spatial.transform.Rotation.random(random_state=rng).as_rotvec()
    d_mat = torch.tensor(ir_layout.exp_map(rotvec, np.array([0])), dtype=torch.float64)

    def rel_equiv_err(fn) -> float:
        lhs = lin(fn(x @ d_mat.T)).detach()
        rhs = (lin(fn(x)) @ d_mat.T).detach()
        return float(torch.linalg.norm(lhs - rhs) / torch.linalg.norm(rhs))

    # (A) Scale + saturation envelope alone (isolating group-commutation symmetry)
    err_env_comp = rel_equiv_err(
        lambda t: quantize_per_component_fp8(t, use_fp8_grid=False)
    )
    err_env_norm = rel_equiv_err(
        lambda t: quantize_irrep_norm_fp8(t, irreps, use_fp8_grid=False)
    )

    # (B) Actual hardware floating-point representations: E4M3 vs BF16 vs FP32
    err_fp8_comp = rel_equiv_err(
        lambda t: quantize_per_component_fp8(t, use_fp8_grid=True)
    )
    err_fp8_norm = rel_equiv_err(
        lambda t: quantize_irrep_norm_fp8(t, irreps, use_fp8_grid=True)
    )
    err_bf16 = rel_equiv_err(lambda t: t.to(torch.bfloat16).to(torch.float64))
    err_fp32 = rel_equiv_err(lambda t: t.to(torch.float32).to(torch.float64))

    print(
        f"  [Scale/Clip Envelope Only] Per-component amax : rel SO(3) defect = {err_env_comp:.3e} (BROKEN symmetry!)"
    )
    print(
        f"  [Scale/Clip Envelope Only] Irrep-norm radial  : rel SO(3) defect = {err_env_norm:.3e} (EXACT O(3) symmetry!)"
    )
    print(
        f"  [Hardware E4M3 Quantized]  Per-component FP8  : rel SO(3) defect = {err_fp8_comp:.3e}"
    )
    print(
        f"  [Hardware E4M3 Quantized]  Irrep-norm FP8     : rel SO(3) defect = {err_fp8_norm:.3e} (~3-bit mantissa floor)"
    )
    print(
        f"  [Hardware BF16 / FP32]     BF16={err_bf16:.3e} (7-bit mantissa), FP32={err_fp32:.3e} (23-bit mantissa)"
    )

    assert err_env_norm < 1e-14
    assert err_env_comp > 0.05
    assert err_fp8_comp > 2.0 * err_fp8_norm
    assert err_fp32 < err_bf16 < err_fp8_norm


if __name__ == "__main__":
    audit_a100_vs_h100_roofline()
    audit_warp_alignment_and_cache_sectors()
    audit_fp8_so3_equivariance()
    print(
        "\n[PASS] 07_gpu_roofline_a100_h100_audit.py completed all checks successfully."
    )
