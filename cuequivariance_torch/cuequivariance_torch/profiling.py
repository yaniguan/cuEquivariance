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
"""A100/H100 GPU Roofline, L2 working-set, and warp-alignment analyzer for cuEquivariance."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

import cuequivariance as cue
from cuequivariance_torch.primitives.segmented_polynomial import (
    SegmentedPolynomial as TorchSegmentedPolynomial,
)
from cuequivariance_torch.primitives.segmented_polynomial_indexed_linear import (
    SUBDICT as _INDEXED_LINEAR_SUBDICT,
)

_MIB = 1024 * 1024


@dataclasses.dataclass(frozen=True)
class GPUSpec:
    """Hardware Roofline and cache specification for an NVIDIA data-center GPU.

    Attributes:
        name: Canonical identifier (e.g. ``"H100_SXM_80GB"``).
        bandwidth_gbps: Peak HBM memory bandwidth in decimal GB/s (``10**9`` B/s).
        fp64_tflops: Peak FP64 SIMT (CUDA core) throughput in TFLOP/s (``10**12`` FLOP/s).
        fp64_tc_tflops: Peak FP64 Tensor Core throughput in TFLOP/s.
        fp32_tflops: Peak FP32 SIMT (CUDA core) throughput in TFLOP/s.
        tf32_tflops: Peak dense TF32 Tensor Core throughput in TFLOP/s.
        bf16_tflops: Peak dense BF16 Tensor Core throughput in TFLOP/s.
        fp16_tflops: Peak dense FP16 Tensor Core throughput in TFLOP/s.
        fp8_tflops: Peak dense FP8 Tensor Core throughput in TFLOP/s, or ``None``
            when unsupported by the architecture.
        l2_cache_bytes: Total on-die L2 cache capacity in bytes.
        warp_size: Threads per GPU warp (32 on NVIDIA GPUs).
        cache_line_bytes: L1/L2 cache-line size in bytes (128 bytes).
    """

    name: str
    bandwidth_gbps: float
    fp64_tflops: float
    fp64_tc_tflops: float
    fp32_tflops: float
    tf32_tflops: float
    bf16_tflops: float
    fp16_tflops: float
    fp8_tflops: Optional[float] = None
    l2_cache_bytes: int = 40 * _MIB
    warp_size: int = 32
    cache_line_bytes: int = 128

    def peak_tflops_for_dtype(
        self, dtype: torch.dtype, *, use_tensor_cores: bool = False
    ) -> float:
        """Return peak dense TFLOP/s for ``dtype`` on this GPU."""
        fp8_dtypes = tuple(
            dt
            for attr in (
                "float8_e4m3fn",
                "float8_e5m2",
                "float8_e4m3fnuz",
                "float8_e5m2fnuz",
            )
            if (dt := getattr(torch, attr, None)) is not None
        )
        if dtype == torch.float64:
            return self.fp64_tc_tflops if use_tensor_cores else self.fp64_tflops
        if dtype == torch.float32:
            return self.tf32_tflops if use_tensor_cores else self.fp32_tflops
        if dtype == torch.bfloat16:
            return self.bf16_tflops
        if dtype == torch.float16:
            return self.fp16_tflops
        if dtype in fp8_dtypes:
            if self.fp8_tflops is None:
                raise ValueError(
                    f"GPU {self.name} does not natively support FP8 Tensor Cores."
                )
            return self.fp8_tflops
        raise ValueError(f"Unsupported dtype for Roofline analysis: {dtype}.")


GPU_SPECS: dict[str, GPUSpec] = {
    "A100_SXM_80GB": GPUSpec(
        name="A100_SXM_80GB",
        bandwidth_gbps=2039.0,
        fp64_tflops=9.7,
        fp64_tc_tflops=19.5,
        fp32_tflops=19.5,
        tf32_tflops=156.0,
        bf16_tflops=312.0,
        fp16_tflops=312.0,
        fp8_tflops=None,
        l2_cache_bytes=40 * _MIB,
    ),
    "A100_PCIE_80GB": GPUSpec(
        name="A100_PCIE_80GB",
        bandwidth_gbps=1935.0,
        fp64_tflops=9.7,
        fp64_tc_tflops=19.5,
        fp32_tflops=19.5,
        tf32_tflops=156.0,
        bf16_tflops=312.0,
        fp16_tflops=312.0,
        fp8_tflops=None,
        l2_cache_bytes=40 * _MIB,
    ),
    "A100_SXM_40GB": GPUSpec(
        name="A100_SXM_40GB",
        bandwidth_gbps=1555.0,
        fp64_tflops=9.7,
        fp64_tc_tflops=19.5,
        fp32_tflops=19.5,
        tf32_tflops=156.0,
        bf16_tflops=312.0,
        fp16_tflops=312.0,
        fp8_tflops=None,
        l2_cache_bytes=40 * _MIB,
    ),
    "H100_SXM_80GB": GPUSpec(
        name="H100_SXM_80GB",
        bandwidth_gbps=3350.0,
        fp64_tflops=34.0,
        fp64_tc_tflops=67.0,
        fp32_tflops=67.0,
        tf32_tflops=494.5,
        bf16_tflops=989.0,
        fp16_tflops=989.0,
        fp8_tflops=1979.0,
        l2_cache_bytes=50 * _MIB,
    ),
    "H100_PCIE_80GB": GPUSpec(
        name="H100_PCIE_80GB",
        bandwidth_gbps=2000.0,
        fp64_tflops=26.0,
        fp64_tc_tflops=51.0,
        fp32_tflops=51.0,
        tf32_tflops=378.0,
        bf16_tflops=756.0,
        fp16_tflops=756.0,
        fp8_tflops=1513.0,
        l2_cache_bytes=50 * _MIB,
    ),
}

_GPU_ALIASES: dict[str, str] = {
    "A100": "A100_SXM_80GB",
    "A100_SXM": "A100_SXM_80GB",
    "A100_PCIE": "A100_PCIE_80GB",
    "H100": "H100_SXM_80GB",
    "H100_SXM": "H100_SXM_80GB",
    "H100_PCIE": "H100_PCIE_80GB",
}


def resolve_gpu_spec(gpu: str | GPUSpec) -> GPUSpec:
    """Resolve a GPU name or alias to a :class:`GPUSpec`."""
    if isinstance(gpu, GPUSpec):
        return gpu
    if not isinstance(gpu, str):
        raise TypeError(f"gpu must be a str or GPUSpec, got {type(gpu)}.")
    key = gpu.strip().upper().replace("-", "_").replace(" ", "_")
    key = _GPU_ALIASES.get(key, key)
    if key not in GPU_SPECS:
        available = ", ".join(sorted(GPU_SPECS.keys()))
        raise KeyError(f"Unknown GPU spec {gpu!r}. Available specs: {available}.")
    return GPU_SPECS[key]


@dataclasses.dataclass(frozen=True)
class RooflineReport:
    """Analytical GPU Roofline, L2 working-set, and warp-alignment report."""

    flops: int
    hbm_bytes: int
    arithmetic_intensity: float
    ridge_point: float
    regime: str
    theoretical_min_time_ms: float
    peak_tflops: float
    bandwidth_gbps: float
    warp_alignment_efficiency: float
    unaligned_operands: list[int]
    l2_working_set_bytes: int
    fits_in_l2: bool
    recommendations: list[str]
    gpu_name: str = ""
    dtype: torch.dtype = torch.float32
    method: Optional[str] = None
    cold_hbm_bytes: int = 0
    forward_flops: int = 0
    backward_flops: int = 0
    second_backward_flops: int = 0

    def format_table(self) -> str:
        """Format the Roofline analysis as a crisp human-readable summary table."""
        rows: list[tuple[str, str]] = [
            ("GPU Target", self.gpu_name or "Custom GPUSpec"),
            ("Precision / Backend", f"{self.dtype} (method={self.method or 'auto'})"),
            ("Total FLOPs", f"{self.flops:,}"),
        ]
        if self.backward_flops > 0 or self.second_backward_flops > 0:
            rows.append(
                (
                    "  FLOP Breakdown (fwd/bwd/bwd2)",
                    f"{self.forward_flops:,} / {self.backward_flops:,} / {self.second_backward_flops:,}",
                )
            )
        rows.extend(
            [
                (
                    "Effective HBM Traffic",
                    f"{self.hbm_bytes:,} B ({self.hbm_bytes / _MIB:.3f} MiB)",
                ),
                (
                    "Cold HBM Traffic (no L2 reuse)",
                    f"{self.cold_hbm_bytes:,} B ({self.cold_hbm_bytes / _MIB:.3f} MiB)",
                ),
                (
                    "Arithmetic Intensity",
                    f"{self.arithmetic_intensity:.2f} FLOP/Byte",
                ),
                ("Hardware Ridge Point", f"{self.ridge_point:.2f} FLOP/Byte"),
                ("Roofline Regime", self.regime),
                ("Peak Compute", f"{self.peak_tflops:.1f} TFLOP/s"),
                ("Peak HBM Bandwidth", f"{self.bandwidth_gbps:.1f} GB/s"),
                ("Theoretical Min Time", f"{self.theoretical_min_time_ms:.4f} ms"),
                (
                    "Warp Alignment Efficiency",
                    f"{100.0 * self.warp_alignment_efficiency:.1f}% (unaligned={self.unaligned_operands})",
                ),
                (
                    "L2 Working Set",
                    f"{self.l2_working_set_bytes / _MIB:.3f} MiB (fits_in_l2={self.fits_in_l2})",
                ),
            ]
        )

        key_width = max(len(k) for k, _ in rows)
        val_width = max(len(v) for _, v in rows)
        sep = f"+-{'-' * key_width}-+-{'-' * val_width}-+"
        lines = [sep]
        for k, v in rows:
            lines.append(f"| {k.ljust(key_width)} | {v.ljust(val_width)} |")
        lines.append(sep)
        if self.recommendations:
            lines.append("Optimization Recommendations:")
            for idx, rec in enumerate(self.recommendations, start=1):
                lines.append(f"  {idx}. {rec}")
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class BenchmarkRooflineResult:
    """Empirical wall-clock timing paired with analytical :class:`RooflineReport`."""

    report: RooflineReport
    elapsed_time_ms: float
    achieved_tflops: float
    achieved_bandwidth_gbps: float
    pct_of_roofline_peak: float
    pct_of_peak_compute: float
    pct_of_peak_bandwidth: float
    device: str
    num_warmup: int
    num_iters: int

    def format_table(self) -> str:
        """Format both analytical Roofline bounds and empirical timing metrics."""
        base = self.report.format_table()
        emp_lines = [
            "",
            f"Empirical Benchmark ({self.device}, warmup={self.num_warmup}, iters={self.num_iters}):",
            f"  Elapsed Time         : {self.elapsed_time_ms:.4f} ms",
            f"  Achieved Compute     : {self.achieved_tflops:.3f} TFLOP/s ({self.pct_of_peak_compute:.1f}% of compute peak)",
            f"  Achieved Bandwidth   : {self.achieved_bandwidth_gbps:.2f} GB/s ({self.pct_of_peak_bandwidth:.1f}% of HBM peak)",
            f"  % of Roofline Ceiling: {self.pct_of_roofline_peak:.1f}%",
        ]
        return base + "\n" + "\n".join(emp_lines)


@dataclasses.dataclass(frozen=True)
class _OperandTrafficSpec:
    buffer_rows: int
    indexed_unique: Optional[int] = None


def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        if isinstance(resolved, torch.dtype):
            return resolved
    raise ValueError(f"Invalid torch.dtype: {dtype!r}.")


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def extract_segmented_polynomial(
    poly_or_module: cue.SegmentedPolynomial
    | cue.EquivariantPolynomial
    | torch.nn.Module,
) -> tuple[cue.SegmentedPolynomial, Optional[str]]:
    """Extract ``(cue.SegmentedPolynomial, method)`` from a polynomial or ``cuet`` module."""
    if isinstance(poly_or_module, cue.SegmentedPolynomial):
        return poly_or_module, None
    if isinstance(poly_or_module, cue.EquivariantPolynomial):
        return poly_or_module.polynomial, None
    if isinstance(poly_or_module, TorchSegmentedPolynomial):
        return poly_or_module._polynomial_orig, poly_or_module.method

    if isinstance(poly_or_module, torch.nn.Module):
        outer_method: Optional[str] = getattr(poly_or_module, "method", None)

        f_attr = getattr(poly_or_module, "f", None)
        if isinstance(f_attr, TorchSegmentedPolynomial):
            return f_attr._polynomial_orig, outer_method or f_attr.method

        etp_attr = getattr(poly_or_module, "etp", None)
        if isinstance(etp_attr, cue.EquivariantPolynomial):
            return etp_attr.polynomial, outer_method
        if isinstance(etp_attr, cue.SegmentedPolynomial):
            return etp_attr, outer_method

        poly_attr = getattr(poly_or_module, "polynomial", None)
        if isinstance(poly_attr, cue.SegmentedPolynomial):
            return poly_attr, outer_method

        tp_attr = getattr(poly_or_module, "tp", None)
        if isinstance(tp_attr, torch.nn.Module) and tp_attr is not poly_or_module:
            poly, sub_method = extract_segmented_polynomial(tp_attr)
            return poly, outer_method or sub_method

        sp_submodules = [
            m
            for m in poly_or_module.modules()
            if isinstance(m, TorchSegmentedPolynomial)
        ]
        if len(sp_submodules) == 1:
            return (
                sp_submodules[0]._polynomial_orig,
                outer_method or sp_submodules[0].method,
            )

    raise TypeError(
        "Expected cue.SegmentedPolynomial, cue.EquivariantPolynomial, or a "
        f"cuequivariance_torch nn.Module containing one, got {type(poly_or_module)}."
    )


def _infer_operand_specs(
    poly_or_module: Any,
    poly: cue.SegmentedPolynomial,
    batch_sizes: Sequence[int] | int,
    indexed_operands: Mapping[int, int] | None,
) -> tuple[list[_OperandTrafficSpec], int]:
    num_operands = poly.num_operands
    canonical_indexed: dict[int, int] = {}
    if indexed_operands is not None:
        for raw_idx, n_unique in indexed_operands.items():
            idx = raw_idx + num_operands if raw_idx < 0 else raw_idx
            if not (0 <= idx < num_operands):
                raise ValueError(
                    f"indexed_operands key {raw_idx} out of range for {num_operands} operands."
                )
            if n_unique < 0:
                raise ValueError(
                    f"indexed_operands[{raw_idx}] must be >= 0, got {n_unique}."
                )
            canonical_indexed[idx] = int(n_unique)

    if isinstance(batch_sizes, int):
        if batch_sizes < 0:
            raise ValueError(f"batch_sizes must be >= 0, got {batch_sizes}.")
        exec_batch = int(batch_sizes)
        rows = [exec_batch] * num_operands

        # Inspect module metadata for shared/species-indexed weight operand 0.
        if isinstance(poly_or_module, torch.nn.Module) and poly.num_inputs >= 1:
            shared_weights = getattr(poly_or_module, "shared_weights", None)
            weight_classes = getattr(poly_or_module, "weight_classes", None)
            num_elements = getattr(poly_or_module, "num_elements", None)

            if num_elements is not None:
                rows[0] = int(num_elements)
                if 0 not in canonical_indexed and exec_batch > 0:
                    canonical_indexed[0] = min(exec_batch, int(num_elements))
            elif shared_weights is True:
                wc = int(weight_classes) if weight_classes is not None else 1
                rows[0] = wc
                if wc > 1 and 0 not in canonical_indexed and exec_batch > 0:
                    canonical_indexed[0] = min(exec_batch, wc)

        for idx, n_unique in canonical_indexed.items():
            rows[idx] = n_unique
    else:
        rows = [int(b) for b in batch_sizes]
        if len(rows) != num_operands:
            raise ValueError(
                f"Expected batch_sizes of length {num_operands}, got {len(rows)}."
            )
        if any(b < 0 for b in rows):
            raise ValueError(f"All batch_sizes must be >= 0, got {rows}.")
        non_indexed = [
            b for i, b in enumerate(rows) if i not in canonical_indexed and b > 1
        ]
        if non_indexed:
            exec_batch = max(non_indexed)
        elif rows:
            exec_batch = max(rows)
        else:
            exec_batch = 0

    specs = [
        _OperandTrafficSpec(
            buffer_rows=rows[i],
            indexed_unique=canonical_indexed.get(i),
        )
        for i in range(num_operands)
    ]
    return specs, exec_batch


def _compute_l2_working_set_bytes(
    poly: cue.SegmentedPolynomial,
    specs: Sequence[_OperandTrafficSpec],
    exec_batch: int,
    elem_bytes: int,
    poly_or_module: Any,
) -> int:
    shared_weight_op0 = bool(
        isinstance(poly_or_module, torch.nn.Module)
        and getattr(poly_or_module, "shared_weights", False)
    )
    working_set = 0
    for oid, (ope, spec) in enumerate(zip(poly.operands, specs)):
        op_bytes = ope.size * elem_bytes
        if spec.indexed_unique is not None:
            working_set += min(exec_batch, spec.indexed_unique) * op_bytes
        elif spec.buffer_rows < exec_batch or (oid == 0 and shared_weight_op0):
            working_set += spec.buffer_rows * op_bytes
    return working_set


def _compute_stage_hbm_bytes(
    stage_poly: cue.SegmentedPolynomial,
    in_specs: Sequence[_OperandTrafficSpec],
    out_specs: Sequence[_OperandTrafficSpec],
    exec_batch: int,
    elem_bytes: int,
    fits_in_l2: bool,
    scatter_mode: str,
) -> tuple[int, int]:
    """Return ``(effective_hbm_bytes, cold_hbm_bytes)`` for one polynomial pass."""
    if scatter_mode not in ("atomic", "register"):
        raise ValueError(
            f"scatter_mode must be 'atomic' or 'register', got {scatter_mode!r}."
        )
    rmw_factor = 1 if scatter_mode == "register" else 2

    effective = 0
    cold = 0

    for used, ope, spec in zip(stage_poly.used_inputs(), stage_poly.inputs, in_specs):
        if not used:
            continue
        op_bytes = ope.size * elem_bytes
        if spec.indexed_unique is not None:
            cold += exec_batch * op_bytes
            active_rows = (
                min(exec_batch, spec.indexed_unique) if fits_in_l2 else exec_batch
            )
            effective += active_rows * op_bytes
        else:
            cold += spec.buffer_rows * op_bytes
            effective += spec.buffer_rows * op_bytes

    for used, ope, spec in zip(
        stage_poly.used_outputs(), stage_poly.outputs, out_specs
    ):
        if not used:
            continue
        op_bytes = ope.size * elem_bytes
        if spec.indexed_unique is not None:
            cold += 2 * exec_batch * op_bytes
            if fits_in_l2:
                effective += (
                    rmw_factor * min(exec_batch, spec.indexed_unique) * op_bytes
                )
            else:
                effective += 2 * exec_batch * op_bytes
        else:
            cold += spec.buffer_rows * op_bytes
            effective += spec.buffer_rows * op_bytes

    return effective, cold


def _analyze_warp_alignment(
    poly: cue.SegmentedPolynomial,
    specs: Sequence[_OperandTrafficSpec],
    exec_batch: int,
    gpu_spec: GPUSpec,
) -> tuple[float, list[int], set[int]]:
    """Analyze inner segment extent alignment against ``gpu_spec.warp_size``."""
    try:
        squeezed = poly.flatten_coefficient_modes().squeeze_modes()
    except ValueError:
        squeezed = poly.squeeze_modes()

    warp_size = gpu_spec.warp_size
    unaligned_operands: list[int] = []
    unaligned_extents: set[int] = set()

    weighted_eff_sum = 0.0
    total_weight = 0.0
    has_segmented_mode = False

    for oid, (ope, spec) in enumerate(zip(squeezed.operands, specs)):
        if ope.ndim == 0 or ope.num_segments == 0:
            continue
        has_segmented_mode = True
        operand_unaligned = False
        rows = max(spec.buffer_rows, 1)
        for seg in ope.segments:
            extent = seg[-1]
            if extent <= 0:
                continue
            padded = math.ceil(extent / warp_size) * warp_size
            seg_eff = extent / padded
            seg_weight = float(rows * math.prod(seg))
            weighted_eff_sum += seg_eff * seg_weight
            total_weight += seg_weight
            if extent % warp_size != 0:
                operand_unaligned = True
                unaligned_extents.add(extent)
        if operand_unaligned:
            unaligned_operands.append(oid)

    if not has_segmented_mode:
        if exec_batch <= 0:
            return 1.0, [], set()
        padded_batch = math.ceil(exec_batch / warp_size) * warp_size
        return exec_batch / padded_batch, [], set()

    efficiency = weighted_eff_sum / total_weight if total_weight > 0.0 else 1.0
    return efficiency, unaligned_operands, unaligned_extents


def _is_uniform_1d_compatible(poly: cue.SegmentedPolynomial) -> bool:
    try:
        p = poly.flatten_coefficient_modes().squeeze_modes()
    except ValueError:
        return False
    extent: Optional[int] = None
    for ope in p.operands:
        if ope.ndim not in (0, 1) or not ope.all_same_segment_shape():
            return False
        if ope.ndim == 1 and ope.num_segments > 0:
            (seg_ext,) = ope.segment_shape
            if extent is None:
                extent = seg_ext
            elif extent != seg_ext:
                return False
    return all(stp.coefficient_subscripts == "" for _, stp in p.operations)


def _is_indexed_linear_compatible(poly: cue.SegmentedPolynomial) -> bool:
    if len(poly.operations) == 0:
        return False
    for ope, stp in poly.operations:
        oid_out, _ = ope.output_operand_buffer(poly.num_inputs)
        subscripts = stp.move_operand_last(oid_out).canonicalize_subscripts().subscripts
        if subscripts not in _INDEXED_LINEAR_SUBDICT:
            return False
    return True


def _build_recommendations(
    poly: cue.SegmentedPolynomial,
    specs: Sequence[_OperandTrafficSpec],
    gpu_spec: GPUSpec,
    dtype: torch.dtype,
    method: Optional[str],
    regime: str,
    arithmetic_intensity: float,
    ridge_point: float,
    warp_eff: float,
    unaligned_operands: list[int],
    unaligned_extents: set[int],
    l2_working_set_bytes: int,
    fits_in_l2: bool,
    use_tensor_cores: bool,
) -> list[str]:
    recs: list[str] = []
    elem_bytes = _dtype_nbytes(dtype)
    warp_size = gpu_spec.warp_size

    if unaligned_operands:
        examples = ", ".join(
            f"{ext} -> {math.ceil(ext / warp_size) * warp_size}"
            for ext in sorted(unaligned_extents)[:3]
        )
        recs.append(
            f"Pad channel/segment multiplicity ({examples}) to a multiple of "
            f"{warp_size} for full warp utilization and {gpu_spec.cache_line_bytes}-byte "
            f"cache-line coalescing (current warp efficiency: {100.0 * warp_eff:.1f}%, "
            f"unaligned operands: {unaligned_operands})."
        )
    elif elem_bytes <= 2:
        try:
            squeezed = poly.flatten_coefficient_modes().squeeze_modes()
        except ValueError:
            squeezed = poly.squeeze_modes()
        sub_line_exts = sorted(
            {
                seg[-1]
                for ope in squeezed.operands
                if ope.ndim >= 1
                for seg in ope.segments
                if (seg[-1] * elem_bytes) % gpu_spec.cache_line_bytes != 0
            }
        )
        if sub_line_exts:
            target_ext = gpu_spec.cache_line_bytes // elem_bytes
            recs.append(
                f"Segment extent {sub_line_exts[0]} fills a 32-thread warp in {dtype} "
                f"({sub_line_exts[0] * elem_bytes} B) but is smaller than a full "
                f"{gpu_spec.cache_line_bytes}-byte cache line; consider multiplicity "
                f"{target_ext} if register pressure allows."
            )

    has_indexed = any(s.indexed_unique is not None for s in specs)
    if method in (None, "naive"):
        if (
            specs
            and specs[0].indexed_unique is not None
            and _is_indexed_linear_compatible(poly)
        ):
            recs.append(
                "Switch from method='naive' to method='indexed_linear' with species-sorted "
                "atoms (cuet.reorder_graph_for_locality) for grouped Tensor Core GEMMs."
            )
        elif _is_uniform_1d_compatible(poly):
            recs.append(
                "Switch from method='naive' to method='uniform_1d' to fuse segmented "
                "tensor-product paths in registers and avoid intermediate HBM traffic."
            )
        elif any(len(ope.buffers) in (3, 4) for ope, _ in poly.operations):
            recs.append(
                "Switch from method='naive' to method='fused_tp' (or unify irrep "
                "multiplicities to unlock method='uniform_1d')."
            )
    elif method == "indexed_linear":
        recs.append(
            "Pre-sort node species once per neighbor-list build via "
            "cuet.reorder_graph_for_locality(..., sort_by_species_first=True) so "
            "indexed_linear runs with zero runtime argsort/permute overhead."
        )

    if has_indexed:
        ws_mb = l2_working_set_bytes / _MIB
        l2_mb = gpu_spec.l2_cache_bytes / _MIB
        if fits_in_l2:
            recs.append(
                f"Indexed gather/scatter working set ({ws_mb:.2f} MiB) fits in "
                f"{gpu_spec.name} L2 cache ({l2_mb:.0f} MiB): apply 3D Morton + (dst, src) "
                "ordering via cuet.reorder_graph_for_locality() to sustain L2 hit rates "
                "and minimize atomicAdd contention."
            )
        else:
            recs.append(
                f"Indexed gather/scatter working set ({ws_mb:.2f} MiB) exceeds "
                f"{gpu_spec.name} L2 cache ({l2_mb:.0f} MiB): apply 3D Morton + (dst, src) "
                "spatial tiling via cuet.reorder_graph_for_locality() to prevent cold HBM "
                "thrashing."
            )

    if regime == "compute-bound" and dtype == torch.float32 and not use_tensor_cores:
        recs.append(
            f"Workload is compute-bound on FP32 SIMT (AI={arithmetic_intensity:.1f} >= "
            f"ridge={ridge_point:.1f} FLOP/B): enable TF32 ({gpu_spec.tf32_tflops:.1f} "
            f"TFLOP/s) or BF16 ({gpu_spec.bf16_tflops:.1f} TFLOP/s) Tensor Cores if "
            "force/energy accuracy permits."
        )
    elif regime == "memory-bound" and dtype in (torch.float32, torch.float64):
        next_prec = "float32/bfloat16" if dtype == torch.float64 else "bfloat16/float16"
        recs.append(
            f"Workload is memory-bound (AI={arithmetic_intensity:.2f} < "
            f"ridge={ridge_point:.2f} FLOP/B): consider {next_prec} operand storage or "
            "fusing surrounding activation/transpose ops to reduce HBM bandwidth demand."
        )

    return recs


def analyze_roofline(
    poly_or_module: cue.SegmentedPolynomial
    | cue.EquivariantPolynomial
    | torch.nn.Module,
    batch_sizes: Sequence[int] | int,
    *,
    gpu: str | GPUSpec = "H100_SXM_80GB",
    dtype: torch.dtype = torch.float32,
    indexed_operands: Mapping[int, int] | None = None,
    include_backward: bool = False,
    include_second_backward: bool = False,
    use_tensor_cores: bool = False,
    scatter_mode: str = "atomic",
) -> RooflineReport:
    """Perform analytical GPU Roofline, L2 cache, and warp-alignment analysis.

    Args:
        poly_or_module: A :class:`cue.SegmentedPolynomial`,
            :class:`cue.EquivariantPolynomial`, :class:`cuet.SegmentedPolynomial`,
            or higher-level ``cuet`` module (``Linear``,
            ``ChannelWiseTensorProduct``, ``FullyConnectedTensorProduct``,
            ``SymmetricContraction``, ``SphericalHarmonics``, ``Rotation``,
            ``Inversion``).
        batch_sizes: Either a single execution batch size ``B`` (e.g. number of
            nodes or edges) or a per-operand sequence of length
            ``poly.num_operands``.
        gpu: Target GPU name in :data:`GPU_SPECS` or a :class:`GPUSpec` instance.
        dtype: Element data type for bandwidth and peak compute lookup.
        indexed_operands: Optional mapping ``{operand_id: num_unique_rows}``
            identifying gathered input operands and/or scatter-added output
            operands across ``E`` edges.
        include_backward: Include first-order reverse-mode AD (``poly.backward``)
            FLOPs and HBM traffic.
        include_second_backward: Include both first-order and second-order AD
            (``jvp`` + ``backward`` through ``poly.backward``, as used in
            conservative MD force training and Force-QAT).
        use_tensor_cores: Use Tensor Core peak TFLOP/s for FP32 (TF32) / FP64.
        scatter_mode: ``"atomic"`` (2x RMW traffic per unique/edge scatter) or
            ``"register"`` (1x writeback when edges are grouped by ``dst``).

    Returns:
        Populated :class:`RooflineReport`.
    """
    gpu_spec = resolve_gpu_spec(gpu)
    resolved_dtype = _resolve_dtype(dtype)
    elem_bytes = _dtype_nbytes(resolved_dtype)
    poly, method = extract_segmented_polynomial(poly_or_module)

    specs, exec_batch = _infer_operand_specs(
        poly_or_module, poly, batch_sizes, indexed_operands
    )
    in_specs = specs[: poly.num_inputs]
    out_specs = specs[poly.num_inputs :]

    l2_working_set_bytes = _compute_l2_working_set_bytes(
        poly, specs, exec_batch, elem_bytes, poly_or_module
    )
    fits_in_l2 = l2_working_set_bytes <= gpu_spec.l2_cache_bytes

    fwd_flops = poly.flop(exec_batch)
    fwd_eff_bytes, fwd_cold_bytes = _compute_stage_hbm_bytes(
        poly, in_specs, out_specs, exec_batch, elem_bytes, fits_in_l2, scatter_mode
    )

    bwd_flops = 0
    bwd_eff_bytes = 0
    bwd_cold_bytes = 0
    bwd2_flops = 0
    bwd2_eff_bytes = 0
    bwd2_cold_bytes = 0

    if include_backward or include_second_backward:
        bwd_poly, bwd_map = poly.backward(poly.used_inputs(), poly.used_outputs())
        bwd_in_specs, bwd_out_specs = bwd_map((in_specs, out_specs))
        bwd_flops = bwd_poly.flop(exec_batch)
        bwd_eff_bytes, bwd_cold_bytes = _compute_stage_hbm_bytes(
            bwd_poly,
            bwd_in_specs,
            bwd_out_specs,
            exec_batch,
            elem_bytes,
            fits_in_l2,
            scatter_mode,
        )

        if include_second_backward:
            bwd2_poly, bwd2_map = bwd_poly.backward(
                bwd_poly.used_inputs(), bwd_poly.used_outputs()
            )
            bwd2_in_specs, bwd2_out_specs = bwd2_map((bwd_in_specs, bwd_out_specs))
            bwd2_flops = bwd2_poly.flop(exec_batch)
            bwd2_eff_bytes, bwd2_cold_bytes = _compute_stage_hbm_bytes(
                bwd2_poly,
                bwd2_in_specs,
                bwd2_out_specs,
                exec_batch,
                elem_bytes,
                fits_in_l2,
                scatter_mode,
            )

    total_flops = fwd_flops + bwd_flops + bwd2_flops
    total_hbm_bytes = fwd_eff_bytes + bwd_eff_bytes + bwd2_eff_bytes
    total_cold_bytes = fwd_cold_bytes + bwd_cold_bytes + bwd2_cold_bytes

    peak_tflops = gpu_spec.peak_tflops_for_dtype(
        resolved_dtype, use_tensor_cores=use_tensor_cores
    )
    bandwidth_gbps = gpu_spec.bandwidth_gbps
    ridge_point = (peak_tflops * 1e3) / bandwidth_gbps
    arithmetic_intensity = total_flops / total_hbm_bytes if total_hbm_bytes > 0 else 0.0
    regime = "compute-bound" if arithmetic_intensity >= ridge_point else "memory-bound"

    compute_time_s = total_flops / (peak_tflops * 1e12) if peak_tflops > 0 else 0.0
    memory_time_s = (
        total_hbm_bytes / (bandwidth_gbps * 1e9) if bandwidth_gbps > 0 else 0.0
    )
    theoretical_min_time_ms = max(compute_time_s, memory_time_s) * 1e3

    warp_eff, unaligned_operands, unaligned_extents = _analyze_warp_alignment(
        poly, specs, exec_batch, gpu_spec
    )
    recommendations = _build_recommendations(
        poly=poly,
        specs=specs,
        gpu_spec=gpu_spec,
        dtype=resolved_dtype,
        method=method,
        regime=regime,
        arithmetic_intensity=arithmetic_intensity,
        ridge_point=ridge_point,
        warp_eff=warp_eff,
        unaligned_operands=unaligned_operands,
        unaligned_extents=unaligned_extents,
        l2_working_set_bytes=l2_working_set_bytes,
        fits_in_l2=fits_in_l2,
        use_tensor_cores=use_tensor_cores,
    )

    return RooflineReport(
        flops=total_flops,
        hbm_bytes=total_hbm_bytes,
        arithmetic_intensity=arithmetic_intensity,
        ridge_point=ridge_point,
        regime=regime,
        theoretical_min_time_ms=theoretical_min_time_ms,
        peak_tflops=peak_tflops,
        bandwidth_gbps=bandwidth_gbps,
        warp_alignment_efficiency=warp_eff,
        unaligned_operands=unaligned_operands,
        l2_working_set_bytes=l2_working_set_bytes,
        fits_in_l2=fits_in_l2,
        recommendations=recommendations,
        gpu_name=gpu_spec.name,
        dtype=resolved_dtype,
        method=method,
        cold_hbm_bytes=total_cold_bytes,
        forward_flops=fwd_flops,
        backward_flops=bwd_flops,
        second_backward_flops=bwd2_flops,
    )


def _extract_primary_output_tensor(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Callable did not return a torch.Tensor, got {type(out)}.")


def benchmark_roofline(
    poly_or_module: torch.nn.Module | Callable[..., Any],
    inputs: Sequence[torch.Tensor] | torch.Tensor,
    *,
    forward_kwargs: Mapping[str, Any] | None = None,
    batch_sizes: Sequence[int] | int | None = None,
    gpu: str | GPUSpec = "H100_SXM_80GB",
    dtype: torch.dtype | None = None,
    indexed_operands: Mapping[int, int] | None = None,
    include_backward: bool = False,
    include_second_backward: bool = False,
    use_tensor_cores: bool = False,
    scatter_mode: str = "atomic",
    num_warmup: int = 5,
    num_iters: int = 20,
    polynomial: cue.SegmentedPolynomial | cue.EquivariantPolynomial | None = None,
) -> BenchmarkRooflineResult:
    """Empirically time a module/callable and compare against its GPU Roofline.

    Uses ``torch.cuda.Event(enable_timing=True)`` when tensors reside on a CUDA
    device and ``time.perf_counter`` on CPU.

    Args:
        poly_or_module: Executable ``cuet`` module (or arbitrary callable when
            ``polynomial`` is explicitly provided).
        inputs: Single input tensor or sequence of input tensors.
        forward_kwargs: Optional keyword arguments forwarded to ``poly_or_module``.
        batch_sizes: Execution batch size(s) passed to :func:`analyze_roofline`.
            Inferred from ``inputs`` when omitted.
        gpu: Target GPU spec name or :class:`GPUSpec` for Roofline comparison.
        dtype: Data type override (inferred from floating-point ``inputs`` if ``None``).
        indexed_operands: Optional gather/scatter unique counts for :func:`analyze_roofline`.
        include_backward: Time and analyze forward + backward pass.
        include_second_backward: Time and analyze forward + first + second backward pass.
        use_tensor_cores: Use Tensor Core TFLOP/s ceiling for FP32/FP64.
        scatter_mode: Scatter traffic mode forwarded to :func:`analyze_roofline`.
        num_warmup: Number of untimed warmup iterations (``>= 0``).
        num_iters: Number of timed iterations (``>= 1``).
        polynomial: Optional explicit polynomial descriptor when ``poly_or_module``
            is a plain Python function wrapping a polynomial.

    Returns:
        :class:`BenchmarkRooflineResult` containing achieved TFLOP/s, GB/s, and
        percentage of the Roofline ceiling.
    """
    if num_warmup < 0:
        raise ValueError(f"num_warmup must be >= 0, got {num_warmup}.")
    if num_iters < 1:
        raise ValueError(f"num_iters must be >= 1, got {num_iters}.")

    tensor_list = [inputs] if isinstance(inputs, torch.Tensor) else list(inputs)
    if not tensor_list:
        raise ValueError("inputs must contain at least one torch.Tensor.")

    device = tensor_list[0].device
    if dtype is None:
        float_tensors = [t for t in tensor_list if t.is_floating_point()]
        dtype = float_tensors[0].dtype if float_tensors else torch.float32

    if batch_sizes is None:
        non_unit = [t.shape[0] for t in tensor_list if t.ndim >= 1 and t.shape[0] > 1]
        batch_sizes = max(non_unit) if non_unit else tensor_list[0].shape[0]

    target_for_analysis = polynomial if polynomial is not None else poly_or_module
    report = analyze_roofline(
        target_for_analysis,
        batch_sizes,
        gpu=gpu,
        dtype=dtype,
        indexed_operands=indexed_operands,
        include_backward=include_backward,
        include_second_backward=include_second_backward,
        use_tensor_cores=use_tensor_cores,
        scatter_mode=scatter_mode,
    )

    fwd_kwargs = dict(forward_kwargs) if forward_kwargs is not None else {}
    run_bwd = include_backward or include_second_backward

    if run_bwd:
        prepared_tensors = [
            t.detach().clone().requires_grad_(True) if t.is_floating_point() else t
            for t in tensor_list
        ]
        diff_inputs = [t for t in prepared_tensors if t.requires_grad]
        if isinstance(poly_or_module, torch.nn.Module):
            diff_params = [p for p in poly_or_module.parameters() if p.requires_grad]
        else:
            diff_params = []
        grad_targets = diff_inputs + diff_params
        if not grad_targets:
            raise ValueError(
                "include_backward=True requires at least one floating-point input or parameter."
            )
    else:
        prepared_tensors = tensor_list
        grad_targets = []

    def _run_forward() -> Any:
        if isinstance(poly_or_module, TorchSegmentedPolynomial):
            return poly_or_module(prepared_tensors, **fwd_kwargs)
        return poly_or_module(*prepared_tensors, **fwd_kwargs)

    def _step() -> None:
        out = _run_forward()
        if run_bwd:
            out_t = _extract_primary_output_tensor(out)
            grads = torch.autograd.grad(
                out_t.sum(),
                grad_targets,
                create_graph=include_second_backward,
                allow_unused=True,
            )
            if include_second_backward:
                active_grads = [g for g in grads if g is not None and g.requires_grad]
                if active_grads:
                    second_loss = sum(g.sum() for g in active_grads)
                    torch.autograd.grad(
                        second_loss,
                        grad_targets,
                        allow_unused=True,
                    )

    for _ in range(num_warmup):
        _step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iters):
            _step()
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed_time_ms = float(start_event.elapsed_time(end_event)) / num_iters
    else:
        t0 = time.perf_counter()
        for _ in range(num_iters):
            _step()
        t1 = time.perf_counter()
        elapsed_time_ms = ((t1 - t0) * 1e3) / num_iters

    elapsed_s = max(elapsed_time_ms * 1e-3, 1e-15)
    achieved_tflops = (report.flops / elapsed_s) / 1e12
    achieved_bandwidth_gbps = (report.hbm_bytes / elapsed_s) / 1e9

    pct_of_peak_compute = (
        100.0 * (achieved_tflops / report.peak_tflops)
        if report.peak_tflops > 0
        else 0.0
    )
    pct_of_peak_bandwidth = (
        100.0 * (achieved_bandwidth_gbps / report.bandwidth_gbps)
        if report.bandwidth_gbps > 0
        else 0.0
    )
    pct_of_roofline_peak = (
        100.0 * (report.theoretical_min_time_ms / elapsed_time_ms)
        if elapsed_time_ms > 0
        else 0.0
    )

    return BenchmarkRooflineResult(
        report=report,
        elapsed_time_ms=elapsed_time_ms,
        achieved_tflops=achieved_tflops,
        achieved_bandwidth_gbps=achieved_bandwidth_gbps,
        pct_of_roofline_peak=pct_of_roofline_peak,
        pct_of_peak_compute=pct_of_peak_compute,
        pct_of_peak_bandwidth=pct_of_peak_bandwidth,
        device=str(device),
        num_warmup=num_warmup,
        num_iters=num_iters,
    )
