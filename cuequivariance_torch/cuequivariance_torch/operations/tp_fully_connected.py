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
import warnings
from typing import Dict, Optional

import torch
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
    default_layout,
)

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors


class FullyConnectedTensorProduct(torch.nn.Module):
    """
    Fully connected tensor product layer.

    Args:
        irreps_in1 (Irreps): Input irreps for the first operand.
        irreps_in2 (Irreps): Input irreps for the second operand.
        irreps_out (Irreps): Output irreps.
        layout (IrrepsLayout, optional): The layout of the input and output irreps. Default is ``cue.mul_ir`` which is the layout corresponding to e3nn.
        layout_in1 (IrrepsLayout, optional): The layout of the first input irreducible representations, by default ``layout``.
        layout_in2 (IrrepsLayout, optional): The layout of the second input irreducible representations, by default ``layout``.
        layout_out (IrrepsLayout, optional): The layout of the output irreducible representations, by default ``layout``.
        shared_weights (bool, optional): Whether to share weights across the batch dimension. Default is True.
        internal_weights (bool, optional): Whether to create module parameters for weights. Default is None.
        device (torch.device, optional): The device to use for the operation.
        dtype (torch.dtype, optional): The dtype to use for the operation weights, by default the torch default dtype.
        math_dtype (torch.dtype or string, optional): The dtype to use for the math operations, by default it follows the dtype of the input tensors,
            if possible, or the torch default dtype (see SegmentedPolynomial for more details).
        method (str, optional): The method to use for the linear layer, by default "fused_tp" (using a CUDA kernel).
        use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
            If `True`, the "naive" method is used.
            If `False` or `None` (default), the "fused_tp" method is used.

    Note:
        In e3nn there was a irrep_normalization and path_normalization parameters.
        This module currently only supports "component" irrep normalization and "element" path normalization.
    """

    def __init__(
        self,
        irreps_in1: cue.Irreps,
        irreps_in2: cue.Irreps,
        irreps_out: cue.Irreps,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in1: Optional[cue.IrrepsLayout] = None,
        layout_in2: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        shared_weights: bool = True,
        internal_weights: bool = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
    ):
        super().__init__()
        irreps_in1, irreps_in2, irreps_out = default_irreps(
            irreps_in1, irreps_in2, irreps_out
        )
        assert_same_group(irreps_in1, irreps_in2, irreps_out)

        if dtype is None:
            dtype = torch.get_default_dtype()

        e = descriptors.fully_connected_tensor_product(
            irreps_in1, irreps_in2, irreps_out
        )
        assert e.polynomial.operations[0][1].subscripts == "uvw,iu,jv,kw+ijk"

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        self.weight_numel = e.polynomial.operations[0][1].operands[0].size

        self.shared_weights = shared_weights
        self.internal_weights = (
            internal_weights if internal_weights is not None else shared_weights
        )

        if self.internal_weights:
            if not self.shared_weights:
                raise ValueError("Internal weights should be shared")
            self.weight = torch.nn.Parameter(
                torch.randn(1, self.weight_numel, device=device, dtype=dtype)
            )
        else:
            self.weight = None

        layout_in1 = default_layout(layout_in1 or layout)
        self.transpose_in1 = cuet.TransposeIrrepsLayout(
            e.inputs[1].irreps,
            source=layout_in1,
            target=e.inputs[1].layout,
            device=device,
            use_fallback=use_fallback,
        )

        layout_in2 = default_layout(layout_in2 or layout)
        self.transpose_in2 = cuet.TransposeIrrepsLayout(
            e.inputs[2].irreps,
            source=layout_in2,
            target=e.inputs[2].layout,
            device=device,
            use_fallback=use_fallback,
        )

        layout_out = default_layout(layout_out or layout)
        self.transpose_out = cuet.TransposeIrrepsLayout(
            e.outputs[0].irreps,
            source=e.outputs[0].layout,
            target=layout_out,
            device=device,
            use_fallback=use_fallback,
        )

        if method is None:
            if use_fallback is None:
                # No warning here as it's the default behavior
                self.method = "fused_tp"
            else:
                warnings.warn(
                    "`use_fallback` is deprecated, please use `method` instead",
                    DeprecationWarning,
                )
                self.method = "naive" if use_fallback else "fused_tp"
        else:
            self.method = method

        if self.method == "fused_tp" and math_dtype is None:
            math_dtype = dtype

        self.f = cuet.SegmentedPolynomial(
            e.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

    def extra_repr(self) -> str:
        return (
            f"shared_weights={self.shared_weights}"
            f", internal_weights={self.internal_weights}"
            f", weight_numel={self.weight_numel}"
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        indices_1: Optional[torch.Tensor] = None,
        indices_2: Optional[torch.Tensor] = None,
        indices_out: Optional[torch.Tensor] = None,
        size_out: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the fully connected tensor product operation.

        Args:
            x1 (torch.Tensor): Input tensor for the first operand. It should have the shape (batch_size, irreps_in1.dim)
                or (num_nodes_1, irreps_in1.dim) when ``indices_1`` is provided.
            x2 (torch.Tensor): Input tensor for the second operand. It should have the shape (batch_size, irreps_in2.dim)
                or (num_nodes_2, irreps_in2.dim) when ``indices_2`` is provided.
            weight (torch.Tensor, optional): Weights for the tensor product. It should have the shape (batch_size, weight_numel)
                if shared_weights is False, or (1, weight_numel) / (weight_numel,) if shared_weights is True.
                If None, the internal weights are used.
            indices_1 (torch.Tensor, optional): 1D integer indices to gather elements from ``x1``.
            indices_2 (torch.Tensor, optional): 1D integer indices to gather elements from ``x2``.
            indices_out (torch.Tensor, optional): 1D integer indices to sum-scatter elements into the output.
            size_out (int, optional): First dimension size of the output. Required when ``indices_out`` is provided.
            mask (torch.Tensor, optional): Optional boolean mask indicating valid edge/batch entries
                for fixed-capacity execution.

        Returns:
            torch.Tensor:
                Output tensor resulting from the fully connected tensor product operation.
                It will have the shape (size_out, irreps_out.dim) if ``indices_out`` is provided,
                otherwise (batch_size, irreps_out.dim).

        Raises:
            ValueError: If internal weights are used and weight is not None,
                or if internal weights are not used and weight is None,
                or if ``indices_out`` is provided without ``size_out``.
        """
        x1 = self.transpose_in1(x1)
        x2 = self.transpose_in2(x2)

        need_i64 = self.f.method in ("naive", "fused_tp") or x1.device.type == "cpu"

        indices_in: Dict[int, torch.Tensor] = {}
        if indices_1 is not None:
            if indices_1.dtype not in (torch.int32, torch.int64):
                raise ValueError("indices_1 must be int32 or int64")
            indices_in[1] = indices_1.to(torch.int64) if need_i64 else indices_1
        if indices_2 is not None:
            if indices_2.dtype not in (torch.int32, torch.int64):
                raise ValueError("indices_2 must be int32 or int64")
            indices_in[2] = indices_2.to(torch.int64) if need_i64 else indices_2

        dict_indices_out: Dict[int, torch.Tensor] = {}
        sizes_out: Dict[int, torch.Tensor] = {}
        if indices_out is not None:
            if size_out is None:
                raise ValueError(
                    "size_out should be provided if indices_out is provided"
                )
            if indices_out.dtype not in (torch.int32, torch.int64):
                raise ValueError("indices_out must be int32 or int64")
            idx_out = indices_out.to(torch.int64) if need_i64 else indices_out
            dict_indices_out = {0: idx_out}
            sizes_out = {
                0: torch.empty((1, 1), device=x1.device, dtype=x1.dtype).expand(
                    size_out, 1
                )
            }

        if self.weight is not None:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")
            weight_tensor = self.weight
        else:
            if weight is None:
                raise ValueError(
                    "Internal weights are not used, weight should not be None"
                )
            weight_tensor = weight

        if (
            indices_out is not None
            and size_out is not None
            and indices_out.numel() == 0
        ):
            for idx_t in (indices_1, indices_2, indices_out):
                if idx_t is not None:
                    if idx_t.dim() != 1:
                        raise ValueError("Index tensor must be one-dimensional")
                    if idx_t.dtype not in (torch.int32, torch.int64):
                        raise ValueError("Index tensor must be int32 or int64")
            if weight_tensor.shape[-1] != self.weight_numel:
                raise ValueError(
                    f"Expected weight last dim {self.weight_numel}, got {weight_tensor.shape[-1]}"
                )
            if x1.shape[-1] != self.irreps_in1.dim:
                raise ValueError(
                    f"Expected x1 last dim {self.irreps_in1.dim}, got {x1.shape[-1]}"
                )
            if x2.shape[-1] != self.irreps_in2.dim:
                raise ValueError(
                    f"Expected x2 last dim {self.irreps_in2.dim}, got {x2.shape[-1]}"
                )
            zero_anchor = (
                (weight_tensor * weight_tensor).sum()
                + (x1 * x1).sum().to(weight_tensor.dtype)
                + (x2 * x2).sum().to(weight_tensor.dtype)
            ) * 0.0
            return (
                torch.zeros(
                    (size_out, self.irreps_out.dim),
                    dtype=weight_tensor.dtype,
                    device=x1.device,
                )
                + zero_anchor
            )

        if (
            x1.ndim == 1
            and x2.ndim == 1
            and len(indices_in) == 0
            and len(dict_indices_out) == 0
            and self.shared_weights
            and weight_tensor.ndim == 2
            and weight_tensor.shape[0] == 1
        ):
            weight_tensor = weight_tensor.squeeze(0)

        has_indices = len(indices_in) > 0 or len(dict_indices_out) > 0
        if not has_indices and mask is None:
            output = self.f([weight_tensor, x1, x2])
        elif mask is None:
            output = self.f(
                [weight_tensor, x1, x2],
                input_indices=indices_in,
                output_shapes=sizes_out,
                output_indices=dict_indices_out,
            )
        else:
            output = self.f(
                [weight_tensor, x1, x2],
                input_indices=indices_in,
                output_shapes=sizes_out,
                output_indices=dict_indices_out,
                mask=mask,
            )

        return self.transpose_out(output[0])
