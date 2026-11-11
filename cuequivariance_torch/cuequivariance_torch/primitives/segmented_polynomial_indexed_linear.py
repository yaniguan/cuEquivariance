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

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.fx
import torch.nn as nn

import cuequivariance as cue


def _compute_counts(v: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.int32, device=v.device)
    ones = torch.ones(v.shape, dtype=torch.int32, device=v.device)
    return counts.scatter_add(0, v.to(dtype=torch.int64), ones)


class IndexedLinear(nn.Module):
    def __init__(
        self,
        d: cue.SegmentedTensorProduct,
        subscripts: Tuple[str],
        dim_indices: List[List[int]],
        math_dtype: Optional[str] = None,
    ):
        super().__init__()
        self.repr = d.__repr__()
        self.subscripts = subscripts
        self.Zind = dim_indices[0]
        self.Cind = dim_indices[1]
        self.uind = dim_indices[2]
        self.vind = dim_indices[3]
        self.wind = dim_indices[4]
        self.reshape_inputs = dim_indices[5]
        self.math_dtype = math_dtype

        d = d.sort_paths(-1)
        pids = d.compressed_path_segment(-1)
        self.slices = [operand.segment_slices() for operand in d.operands]
        self.out_segment_sizes = [math.prod(seg) for seg in d.operands[-1].segments]
        # Assuming all paths have the same 2 inputs
        self.indices = []
        self.segment_shapes = []
        coefficients = []
        self.coefficients_indices = []
        n = 0
        for pid_start, pid_end in zip(pids[:-1], pids[1:]):
            seg_inds = []
            seg_shapes = []
            coeffs = []
            for path in d.paths[pid_start:pid_end]:
                shape0 = d.get_segment_shape(0, path)
                shape1 = d.get_segment_shape(1, path)
                if math.prod(shape0) == 0 or math.prod(shape1) == 0:
                    continue
                seg_inds.append([path.indices[0], path.indices[1]])
                seg_shapes.append([shape0, shape1])
                coefficients.append(path.coefficients.item())
                coeffs.append(n)
                n += 1
            self.indices.append(seg_inds)
            self.segment_shapes.append(seg_shapes)
            self.coefficients_indices.append(coeffs)
        self.register_buffer(
            "coefficients", torch.tensor(coefficients, dtype=torch.float64)
        )

    def __repr__(self):
        return self.repr

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        out_shape: Tuple[int, int],
        counts: torch.Tensor,
    ):
        shape_list = [input1.shape, input2.shape, out_shape, [1]]
        Z = shape_list[self.Zind[0]][self.Zind[1]]
        C = shape_list[self.Cind[0]][self.Cind[1]]
        tp_out_batch = C if self.Cind[0] == 2 else Z
        coef_dtype = torch.float64 if input1.dtype == torch.float64 else torch.float32
        coefs_cast = self.coefficients.to(dtype=coef_dtype, device=input1.device)

        has_any_path = False
        segment_outputs = []
        for ii, shapes, coefs, seg_size in zip(
            self.indices,
            self.segment_shapes,
            self.coefficients_indices,
            self.out_segment_sizes,
        ):
            if len(ii) == 0 or seg_size == 0:
                segment_outputs.append(input1.new_zeros((tp_out_batch, seg_size)))
            else:
                has_any_path = True
                parts = [
                    apply_linear(
                        input1[
                            :, self.slices[0][i[0]].start : self.slices[0][i[0]].stop
                        ].reshape((input1.shape[0],) + shape[0]),
                        input2[
                            :, self.slices[1][i[1]].start : self.slices[1][i[1]].stop
                        ].reshape((input2.shape[0],) + shape[1]),
                        Z,
                        C,
                        self.uind,
                        self.vind,
                        self.wind,
                        self.reshape_inputs,
                        counts,
                        self.subscripts,
                        coefs_cast[coef],
                        self.math_dtype,
                    ).reshape(tp_out_batch, seg_size)
                    for i, shape, coef in zip(ii, shapes, coefs)
                ]
                seg_sum = parts[0]
                for part in parts[1:]:
                    seg_sum = seg_sum + part
                segment_outputs.append(seg_sum)

        if len(segment_outputs) == 0:
            res = input1.new_zeros((tp_out_batch, 0))
        else:
            res = torch.cat(segment_outputs, dim=-1)

        if not has_any_path and (input1.requires_grad or input2.requires_grad):
            res = res + (input1.sum() + input2.sum()) * 0.0
        return res


def apply_linear(
    input1: torch.Tensor,
    input2: torch.Tensor,
    Z: int,
    C: int,
    uind: List[int],
    vind: List[int],
    wind: List[int],
    reshape_inputs: List[int],
    counts: torch.Tensor,
    subscripts: Tuple[str],
    coefficients: torch.Tensor,
    math_dtype: Optional[str] = None,
):
    from cuequivariance_ops_torch import indexed_linear

    shape_list = [input1.shape, input2.shape, [], [1]]
    u = shape_list[uind[0]][uind[1]]
    v = shape_list[vind[0]][vind[1]]
    w = shape_list[wind[0]][wind[1]]
    outsh = None
    if reshape_inputs[0] and not reshape_inputs[1]:
        outsh = input1.shape[:-1]
    elif reshape_inputs[1] and not reshape_inputs[0]:
        outsh = input2.shape[:-1]
    if reshape_inputs[0]:
        input1 = input1.reshape(-1, input1.shape[-1])
    if reshape_inputs[1]:
        input2 = input2.reshape(-1, input2.shape[-1])
    output = indexed_linear(
        input1,
        input2,
        counts * w,
        u,
        v,
        C,
        Z * w,
        subscripts,
        coefficients,  # math_dtype
    )
    if outsh is not None:
        output = output.reshape(outsh + output.shape[1:])
    return output


class SegmentedPolynomialIndexedLinear(nn.Module):
    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        math_dtype: Optional[str] = None,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
        options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._polynomial_orig = polynomial
        self._output_dtype_map_orig = output_dtype_map
        self._math_dtype_orig = math_dtype
        self._options_orig = options
        opts = options if options is not None else {}
        self.sort_indices = bool(
            opts.get("sort_indices", opts.get("sort_unaligned_indices", False))
        )
        self.skip_sorted_check = bool(opts.get("skip_sorted_check", False))
        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs
        self.input_sizes = [o.size for o in polynomial.inputs]
        self.name = name
        if math_dtype is not None:
            warnings.warn(
                "`indexed_linear` does not support explicit `math_dtype`."
                "This will be ignored."
            )
        self.out_size = [o.size for o in polynomial.outputs]
        default_dtype_map = [
            0 if polynomial.num_inputs >= 1 else -1
        ] * polynomial.num_outputs
        self.dtypes = list(range(self.num_inputs)) + (
            default_dtype_map if output_dtype_map is None else output_dtype_map
        )

        # Build the TPs
        self.tps = torch.nn.ModuleList()
        self.b_outs = []
        self.input_inds = []
        self.indexed_input = [None for _ in range(self.num_inputs + self.num_outputs)]
        for operation, d in polynomial.operations:
            ope_out, b_out = operation.output_operand_buffer(self.num_inputs)
            self.b_outs.append(b_out - self.num_inputs)
            self.input_inds.append(operation.input_buffers(self.num_inputs))
            d = d.move_operand_last(ope_out)
            subscripts = d.canonicalize_subscripts().subscripts

            if subscripts not in SUBDICT.keys():
                raise NotImplementedError(
                    f"Indexed_linear does not support the operation {subscripts}."
                )

            signature = SUBDICT[subscripts]
            # Each input has to be either indexed or not
            C_index = signature[2][0]
            if C_index < 2:
                C_index = self.input_inds[-1][C_index]
            else:
                C_index = b_out
            if self.indexed_input[C_index] is None:
                self.indexed_input[C_index] = True
            else:
                assert self.indexed_input[C_index], (
                    f"Buffer {C_index} has multiple indexed values."
                )
            Z_index = signature[1][0]
            if Z_index < 2:
                Z_index = self.input_inds[-1][Z_index]
            else:
                Z_index = b_out
            if self.indexed_input[Z_index] is None:
                self.indexed_input[Z_index] = False
            else:
                assert not self.indexed_input[Z_index], (
                    f"Buffer {Z_index} has multiple indexed values."
                )
            self.tps.append(IndexedLinear(d, signature[0], signature[1:], math_dtype))

    def __reduce__(self):
        from cuequivariance_torch.primitives.segmented_polynomial import (
            SegmentedPolynomial,
        )

        return (
            SegmentedPolynomial,
            (
                self._polynomial_orig,
                "indexed_linear",
                self._math_dtype_orig,
                self._output_dtype_map_orig,
                self.name,
                self._options_orig,
            ),
        )

    def _should_check_sorted(self, v: torch.Tensor) -> bool:
        if self.sort_indices or self.skip_sorted_check:
            return False
        if v.numel() <= 1:
            return False
        if torch.jit.is_scripting():
            return False
        if torch.jit.is_tracing() or torch.compiler.is_compiling():
            return False
        if v.is_cuda and torch.cuda.is_current_stream_capturing():
            return False
        fx = getattr(torch, "fx", None)
        if fx is not None:
            symbolic_trace = getattr(fx, "_symbolic_trace", None)
            if (
                symbolic_trace is not None
                and getattr(symbolic_trace, "is_fx_symbolic_tracing", bool)()
            ):
                return False
        return True

    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Optional[Dict[int, torch.Tensor]] = None,
        output_shapes: Optional[Dict[int, torch.Tensor]] = None,
        output_indices: Optional[Dict[int, torch.Tensor]] = None,
    ):
        empty_dict: Dict[int, torch.Tensor] = {}
        if input_indices is None:
            input_indices = dict(empty_dict)
        if output_shapes is None:
            output_shapes = dict(empty_dict)
        if output_indices is None:
            output_indices = dict(empty_dict)
        inputs = list(inputs)

        for i, x, size in zip(range(self.num_inputs), inputs, self.input_sizes):
            if x.ndim == 0:
                raise ValueError(f"Input {i} has no dimensions")
            if x.shape[-1] != size:
                raise ValueError(
                    f"Input {i} has shape {x.shape} but expected shape {size}."
                )

        # Input indexing
        count_indices: Dict[int, torch.Tensor] = {}
        input_perms: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for k, v in input_indices.items():
            if not self.indexed_input[k]:
                inputs[k] = inputs[k][v]
            else:
                if self._should_check_sorted(v):
                    assert torch.all(v[1:] - v[:-1] >= 0), (
                        "Indexed_linear does not support non-sorted indices."
                    )
                if self.sort_indices and v.numel() > 1:
                    perm = torch.argsort(v, stable=True)
                    arange = torch.arange(
                        perm.shape[0], dtype=perm.dtype, device=perm.device
                    )
                    inv_perm = torch.zeros_like(perm).scatter(0, perm, arange)
                    input_perms[k] = (perm, inv_perm)
                count_indices[k] = _compute_counts(v, inputs[k].shape[0])

        # Output indices:
        out_indices = [torch.empty(0) for _ in range(self.num_outputs)]
        output_perms: Dict[int, torch.Tensor] = {}
        for k, v in output_indices.items():
            assert k in output_shapes.keys(), (
                "output shapes must be provided for output indices"
            )
            if not self.indexed_input[k + self.num_inputs]:
                out_indices[k] = v
            else:
                if self._should_check_sorted(v):
                    assert torch.all(v[1:] - v[:-1] >= 0), (
                        "Indexed_linear does not support non-sorted indices."
                    )
                if self.sort_indices and v.numel() > 1:
                    output_perms[k] = torch.argsort(v, stable=True)
                out_indices[k] = _compute_counts(v, output_shapes[k].shape[0])

        input_batch_sizes = [
            input_indices[k].shape[0] if k in input_indices else t.shape[0]
            for k, t in enumerate(inputs)
        ]
        batch_size = 1
        for k, size in enumerate(input_batch_sizes):
            if not self.indexed_input[k] or k in input_indices:
                if size != 1:
                    assert batch_size in [1, size]
                    batch_size = size

        for k, v in input_indices.items():
            if k in count_indices and v.shape[0] == 1 and batch_size != 1:
                count_indices[k] = count_indices[k] * batch_size

        outputs_dims = [
            (batch_size, shape)
            if i not in output_shapes.keys()
            else (output_shapes[i].shape[0], shape)
            for i, shape in enumerate(self.out_size)
        ]
        out_buffers = [
            torch.zeros(
                out_shape,
                dtype=inputs[self.dtypes[self.num_inputs + i]].dtype,
                device=inputs[0].device,
            )
            for i, out_shape in enumerate(outputs_dims)
        ]

        # Apply TPs
        for i, tp in enumerate(self.tps):
            b_out = self.b_outs[i]
            in0 = inputs[self.input_inds[i][0]]
            in1 = inputs[self.input_inds[i][1]]
            out_perm_inv: Optional[torch.Tensor] = None

            if tp.Cind[0] < 2:
                c_buf = self.input_inds[i][tp.Cind[0]]
                if c_buf in count_indices:
                    counts = count_indices[c_buf]
                elif inputs[c_buf].shape[0] == 1:
                    counts = torch.full(
                        (1,),
                        batch_size,
                        dtype=torch.int32,
                        device=inputs[c_buf].device,
                    )
                elif inputs[c_buf].shape[0] == batch_size:
                    counts = torch.ones(
                        (batch_size,),
                        dtype=torch.int32,
                        device=inputs[c_buf].device,
                    )
                else:
                    raise ValueError(
                        f"Missing input_indices for indexed buffer {c_buf} with shape {inputs[c_buf].shape}."
                    )

                if tp.Cind[0] == 0:
                    if in1.shape[0] == 1 and batch_size != 1:
                        in1 = in1.expand(batch_size, -1)
                    if c_buf in input_perms:
                        perm, out_perm_inv = input_perms[c_buf]
                        in1 = in1[perm]
                else:
                    if in0.shape[0] == 1 and batch_size != 1:
                        in0 = in0.expand(batch_size, -1)
                    if c_buf in input_perms:
                        perm, out_perm_inv = input_perms[c_buf]
                        in0 = in0[perm]
            else:
                counts = out_indices[b_out]
                target_z = (
                    output_indices[b_out].shape[0]
                    if b_out in output_indices
                    else batch_size
                )
                if in0.shape[0] == 1 and target_z != 1:
                    in0 = in0.expand(target_z, -1)
                if in1.shape[0] == 1 and target_z != 1:
                    in1 = in1.expand(target_z, -1)
                if b_out in output_perms:
                    perm = output_perms[b_out]
                    in0 = in0[perm]
                    in1 = in1[perm]

            out = tp(in0, in1, outputs_dims[b_out], counts)
            if out_perm_inv is not None:
                out = out[out_perm_inv]

            if self.indexed_input[b_out + self.num_inputs]:
                out_buffers[b_out] = out_buffers[b_out] + out.reshape(
                    out_buffers[b_out].shape[0], -1
                )
            elif out_indices[b_out].size() != torch.Size([0]):
                # In case we need to replicate before scattering:
                if out.shape[0] == 1 and out_indices[b_out].shape[0] > 1:
                    out = out.expand(out_indices[b_out].shape[0], out.shape[1])
                inds = (
                    out_indices[b_out]
                    .to(dtype=torch.int64)
                    .unsqueeze(-1)
                    .expand_as(out)
                )
                out_buffers[b_out] = out_buffers[b_out].scatter_add(0, inds, out)
            else:
                if out_buffers[b_out].shape[0] == out.shape[0]:
                    out_buffers[b_out] = out_buffers[b_out] + out
                elif out_buffers[b_out].shape[0] == 1:
                    out_buffers[b_out] = out_buffers[b_out] + out.sum(
                        dim=0, keepdim=True
                    )
                elif out.shape[0] == 1:
                    out_buffers[b_out] = out_buffers[b_out] + out.expand_as(
                        out_buffers[b_out]
                    )
                else:
                    raise ValueError(
                        f"Input/output batch size mismatch {out_buffers[b_out].shape} vs {out.shape}."
                    )

        return out_buffers


# Dict of subscripts to
# [canonicalized subscripts, indices of Z, C, u, v, w, whether to reshape each input]
# indices in a list [input1.shape, input2.shape, output.shape, [1]]
SUBDICT = {
    "uv,u,v": [("uv", "u", "v"), [1, 0], [0, 0], [0, 1], [0, 2], [3, 0], [0, 0]],
    "uv,v,u": [("uv", "v", "u"), [1, 0], [0, 0], [0, 1], [0, 2], [3, 0], [0, 0]],
    "uv,wu,wv": [("uv", "u", "v"), [1, 0], [0, 0], [0, 1], [0, 2], [1, 1], [0, 1]],
    "uv,wv,wu": [("uv", "v", "u"), [1, 0], [0, 0], [0, 1], [0, 2], [1, 1], [0, 1]],
    "u,uv,v": [("u", "uv", "v"), [0, 0], [1, 0], [0, 1], [1, 2], [3, 0], [0, 0]],
    "u,vu,v": [
        ("u", "vu", "v"),
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [3, 0],
        [0, 0],
    ],
    "uv,wv,uw": [
        ("u", "vu", "v"),
        [0, 0],
        [1, 0],
        [0, 2],
        [1, 1],
        [0, 1],
        [1, 0],
    ],
    "uv,vw,uw": [
        ("u", "uv", "v"),
        [0, 0],
        [1, 0],
        [0, 2],
        [1, 2],
        [0, 1],
        [1, 0],
    ],
    "u,v,vu": [
        ("u", "v", "vu"),
        [0, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [3, 0],
        [0, 0],
    ],
    "u,v,uv": [
        ("u", "v", "uv"),
        [0, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [3, 0],
        [0, 0],
    ],
    "uv,uw,wv": [
        ("u", "v", "vu"),
        [0, 0],
        [2, 0],
        [0, 2],
        [1, 2],
        [0, 1],
        [1, 1],
    ],
    "uv,uw,vw": [
        ("u", "v", "uv"),
        [0, 0],
        [2, 0],
        [0, 2],
        [1, 2],
        [0, 1],
        [1, 1],
    ],
}
