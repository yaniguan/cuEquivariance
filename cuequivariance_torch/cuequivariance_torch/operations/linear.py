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
from typing import Dict, List, Optional

import torch
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
    default_layout,
)

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors


class _ZeroLinear(torch.nn.Module):
    """Fallback module for linear layers with zero effective contraction paths."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Optional[Dict[int, torch.Tensor]] = None,
        output_shapes: Optional[Dict[int, torch.Tensor]] = None,
        output_indices: Optional[Dict[int, torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        weight = inputs[0]
        x = inputs[1]
        if x.ndim == 0 or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Input 1 has shape {x.shape} but expected shape {self.in_dim}."
            )
        if weight.ndim == 0 or weight.shape[-1] != 0:
            raise ValueError(f"Input 0 has shape {weight.shape} but expected shape 0.")
        w_batch = weight.shape[0]
        if input_indices is not None:
            if 0 in input_indices:
                w_batch = input_indices[0].shape[0]
        x_batch = x.shape[0]
        if x_batch == 1 and w_batch != 1:
            batch_size = w_batch
        elif w_batch in (1, x_batch):
            batch_size = x_batch
        else:
            raise ValueError(
                f"Input/output batch size mismatch ({w_batch}, {x_batch})."
            )
        zero = (x.narrow(-1, 0, 0).sum() * weight.narrow(-1, 0, 0).sum()).to(
            dtype=weight.dtype
        )
        out = (
            torch.zeros((batch_size, self.out_dim), dtype=weight.dtype, device=x.device)
            + zero
        )
        return [out]


class Linear(torch.nn.Module):
    """
    A class that represents an equivariant linear layer.

    Args:
        irreps_in (Irreps): The input irreducible representations.
        irreps_out (Irreps): The output irreducible representations.
        layout (IrrepsLayout, optional): The layout of the irreducible representations, by default ``cue.mul_ir``. This is the layout used in the e3nn library.
        layout_in (IrrepsLayout, optional): The layout of the input irreducible representations, by default ``layout``.
        layout_out (IrrepsLayout, optional): The layout of the output irreducible representations, by default ``layout``.
        shared_weights (bool, optional): Whether to use shared weights, by default True.
        internal_weights (bool, optional): Whether to use internal weights, by default True if shared_weights is True, otherwise False.
        weight_classes (int, optional): If provided, the weight tensor will have this as a batch dimension (expected if using external weights).
            If this is specified and >1, at forward time each batch element will use a slice of the weight tensor as indexed by the weight_indices.
        auto_sort_indices (bool, optional): When ``method="indexed_linear"`` and ``weight_classes > 1``, whether to automatically sort
            ``weight_indices`` (permuting inputs and inverse-permuting outputs) so unsorted indices are supported, by default True.
        device (torch.device, optional): The device to use for the linear layer.
        dtype (torch.dtype, optional): The dtype to use for the linear layer weights, by default ``torch.float32``.
        math_dtype (torch.dtype or string, optional): The dtype to use for the math operations, by default it follows the dtype of the input tensors,
            if possible, or the torch default dtype (see SegmentedPolynomial for more details).
        method (str, optional): The method to use for the linear layer, by default ``"fused_tp"`` when CUDA and ``cuequivariance_ops_torch``
            are available (unless ``device`` is CPU), otherwise ``"naive"``.
        use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
            If `True`, the "naive" method is used.
            If `False`, the "fused_tp" method is used.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        shared_weights: bool = True,
        internal_weights: bool = None,
        weight_classes: Optional[int] = 1,
        auto_sort_indices: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
    ):
        super().__init__()
        irreps_in, irreps_out = default_irreps(irreps_in, irreps_out)
        assert_same_group(irreps_in, irreps_out)

        e = descriptors.linear(irreps_in, irreps_out)
        assert e.polynomial.operations[0][1].subscripts == "uv,iu,iv"

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.weight_numel = e.inputs[0].dim

        if weight_classes is None:
            weight_classes = 1
        self.weight_classes = weight_classes
        self.auto_sort_indices = auto_sort_indices
        self.shared_weights = shared_weights
        if weight_classes > 1 and not shared_weights:
            raise ValueError("Weight classes require shared weights.")
        self.internal_weights = (
            internal_weights if internal_weights is not None else shared_weights
        )

        if self.internal_weights:
            if not self.shared_weights:
                raise ValueError("Internal weights should be shared")
            self.weight = torch.nn.Parameter(
                torch.randn(
                    weight_classes, self.weight_numel, device=device, dtype=dtype
                )
            )
        else:
            self.weight = None

        layout_in = default_layout(layout_in or layout)
        self.transpose_in = cuet.TransposeIrrepsLayout(
            e.inputs[1].irreps,
            source=layout_in,
            target=e.inputs[1].layout,
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
                # Auto-select fast CUDA kernel when available and not targeting CPU;
                # otherwise select "naive" without emitting fallback warnings.
                is_cuda = (
                    torch.device(device).type == "cuda"
                    if device is not None
                    else torch.cuda.is_available()
                )
                if (
                    self.weight_numel > 0
                    and cuet.primitives.segmented_polynomial.HAS_CUE_OPS
                    and is_cuda
                ):
                    self.method = "fused_tp"
                else:
                    self.method = "naive"
            else:
                warnings.warn(
                    "`use_fallback` is deprecated, please use `method` instead",
                    DeprecationWarning,
                )
                self.method = "naive" if use_fallback else "fused_tp"
        else:
            self.method = method

        # For fused_tp we have to specify a supported math_dtype (float32 or float64);
        # promote reduced-precision dtypes (e.g. float16, bfloat16) to float32.
        if self.method == "fused_tp" and math_dtype is None:
            effective_dtype = dtype if dtype is not None else torch.get_default_dtype()
            math_dtype = (
                effective_dtype
                if effective_dtype in (torch.float32, torch.float64)
                else torch.float32
            )

        if self.weight_numel == 0:
            if self.method not in (
                "naive",
                "fused_tp",
                "indexed_linear",
                "uniform_1d",
                "",
            ):
                raise ValueError(f"Invalid method: {self.method}")
            self.f = _ZeroLinear(e.inputs[1].dim, e.outputs[0].dim).to(device)
        else:
            self.f = cuet.SegmentedPolynomial(
                e.polynomial,
                method=self.method,
                math_dtype=math_dtype,
            ).to(device)

    def extra_repr(self) -> str:
        return f"shared_weights={self.shared_weights}, internal_weights={self.internal_weights}, weight_numel={self.weight_numel}"

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the linear layer.

        Args:
            x (torch.Tensor): The input tensor.
            weight (torch.Tensor, optional): The weight tensor. If None, the internal weight tensor is used, otherwise:
               If weights are not shared, this should be a tensor of shape (batch_size, weight_numel).
               If weights are shared, this should be a tensor of shape (weight_classes, weight_numel)
               (where weight_classes is 1 if unspecified).
            weight_indices (torch.Tensor, optional): The indices of the weight tensor:
               if weight_classes > 1, this is an integer tensor of shape (batch_size,),
               indicating which weight slice to use for each batch element.

        Returns:
            torch.Tensor: The output tensor after applying the linear transformation.

        Raises:
            ValueError: If internal weights are used and weight is not None,
                or if shared weights are used and weight is not a 1D tensor,
                or if shared weights are not used and weight is not a 2D tensor.
        """
        if self.internal_weights:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")
            weight = self.weight

        inv_perm: Optional[torch.Tensor] = None
        input_indices: Dict[int, torch.Tensor] = {}
        if self.weight_classes > 1:
            if weight_indices is None:
                raise ValueError(
                    "weight_indices should be provided if weight_classes > 1"
                )
            else:
                if (
                    self.method == "indexed_linear"
                    and self.auto_sort_indices
                    and weight_indices.ndim == 1
                    and weight_indices.numel() > 1
                ):
                    perm = torch.argsort(weight_indices, stable=True)
                    weight_indices = weight_indices[perm]
                    if x.shape[0] > 1:
                        x = x[perm]
                    inv_perm = torch.empty_like(perm)
                    inv_perm[perm] = torch.arange(
                        perm.shape[0], dtype=perm.dtype, device=perm.device
                    )
                input_indices[0] = weight_indices

        if weight is None:
            raise ValueError("Weights should not be None")

        output = self.f([weight, self.transpose_in(x)], input_indices=input_indices)
        y = self.transpose_out(output[0])
        if inv_perm is not None:
            y = y[inv_perm]
        return y
