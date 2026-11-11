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
import math
import warnings
from typing import Dict, Optional

import torch
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction,
)
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
    default_layout,
)

import cuequivariance as cue
import cuequivariance_torch as cuet


class SymmetricContraction(torch.nn.Module):
    """
    Accelerated implementation of the symmetric contraction operation introduced in https://arxiv.org/abs/2206.07697.

    Args:
        irreps_in (Irreps): The input irreps. Both uniform and non-uniform (tapered)
            multiplicities are supported via greatest-common-divisor channel grouping.
        irreps_out (Irreps): The output irreps. Both uniform and non-uniform (tapered)
            multiplicities are supported.
        contraction_degree (int): The degree of the symmetric contraction, specifying the maximum degree of the
            polynomial in the symmetric contraction.
        num_elements (int, optional): The number of elements for the weight tensor, by default ``1``.
        layout (IrrepsLayout, optional): The layout of the input and output irreps. If not provided, a default layout is used.
        layout_in (IrrepsLayout, optional): The layout of the input irreducible representations, by default ``layout``.
        layout_out (IrrepsLayout, optional): The layout of the output irreducible representations, by default ``layout``.
        device (torch.device, optional): The device to use for the operation.
        dtype (torch.dtype, optional): The dtype to use for the operation weights, by default ``torch.float32``.
        math_dtype (str or torch.dtype, optional): The dtype to use for the math operations, by default it follows the dtype of the input tensors,
            if possible, or the torch default dtype (see SegmentedPolynomial for more details).
        original_mace (bool, optional): Whether to use the original MACE implementation, by default False.
        method (str, optional): The method to use for the operation, by default "uniform_1d" (using a CUDA kernel)
            if all segments have the same shape, otherwise "naive" (using a PyTorch implementation).
        use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
            If `True` the "naive" method is used.
            If `False` the "uniform_1d" method is used (make sure all segments have the same shape).

    Examples:
        >>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        >>> irreps_in = cue.Irreps("O3", "32x0e + 32x1o")
        >>> irreps_out = cue.Irreps("O3", "32x0e")
        >>> layer = SymmetricContraction(irreps_in, irreps_out, contraction_degree=3, num_elements=5, layout=cue.ir_mul, dtype=torch.float32, device=device)

        The argument `original_mace` can be set to `True` to emulate the original MACE implementation.

        >>> feats_irreps = cue.Irreps("O3", "32x0e + 32x1o + 32x2e")
        >>> target_irreps = cue.Irreps("O3", "32x0e + 32x1o")
        >>> # OLD FUNCTION DEFINITION:
        >>> # symmetric_contractions_old = SymmetricContraction(
        >>> #     irreps_in=feats_irreps,
        >>> #     irreps_out=target_irreps,
        >>> #     correlation=3,
        >>> #     num_elements=10,
        >>> # )
        >>> # NEW FUNCTION DEFINITION:
        >>> symmetric_contractions_new = cuet.SymmetricContraction(
        ...     irreps_in=feats_irreps,
        ...     irreps_out=target_irreps,
        ...     contraction_degree=3,
        ...     num_elements=10,
        ...     layout_in=cue.ir_mul,
        ...     layout_out=cue.mul_ir,
        ...     original_mace=True,
        ...     dtype=torch.float64,
        ...     device=device,
        ... )

        Then the execution is as follows:

        >>> node_feats = torch.randn(128, 32, feats_irreps.dim // 32, dtype=torch.float64, device=device)
        >>> # with node_attrs_index being the index version of node_attrs, sth like:
        >>> # node_attrs_index = torch.nonzero(node_attrs)[:, 1].int()
        >>> node_attrs_index = torch.randint(0, 10, (128,), dtype=torch.int32, device=device)
        >>> # OLD CALL:
        >>> # symmetric_contractions_old(node_feats, node_attrs)
        >>> # NEW CALL:
        >>> node_feats = torch.transpose(node_feats, 1, 2).flatten(1)
        >>> symmetric_contractions_new(node_feats, node_attrs_index)
        tensor([[...)

    Note:
        The term 'mul' refers to the multiplicity of an irrep, indicating how many times it appears
        in the representation. Multiplicities across input and output irreps are grouped by their
        greatest common divisor ``mul_gcd = math.gcd(*irreps_in.muls, *irreps_out.muls)`` so every
        polynomial segment shares uniform extent ``mul_gcd``.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        contraction_degree: int,
        num_elements: int = 1,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        original_mace: bool = False,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
    ):
        super().__init__()

        if num_elements < 1:
            raise ValueError(f"num_elements must be >= 1, got {num_elements}")

        irreps_in, irreps_out = default_irreps(irreps_in, irreps_out)
        assert_same_group(irreps_in, irreps_out)
        self.contraction_degree = contraction_degree

        all_muls = [*irreps_in.muls, *irreps_out.muls]
        mul_gcd = math.gcd(*all_muls) if all_muls else 1
        if mul_gcd == 0:
            mul_gcd = 1
        self.mul_gcd = mul_gcd

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_in_dim: int = irreps_in.dim
        self.irreps_out_dim: int = irreps_out.dim

        self.etp, p = symmetric_contraction(
            irreps_in, irreps_out, range(1, contraction_degree + 1)
        )
        same_shape = self.etp.all_same_segment_shape()

        effective_dtype = dtype if dtype is not None else torch.get_default_dtype()
        if original_mace:
            proj_dtype = (
                torch.float64 if effective_dtype == torch.float64 else torch.float32
            )
            self.register_buffer(
                "projection", torch.tensor(p, dtype=proj_dtype, device=device)
            )
            self.weight_shape = (p.shape[0], mul_gcd)
        else:
            self.projection = None
            self.weight_shape = (self.etp.inputs[0].dim // mul_gcd, mul_gcd)

        self.num_elements = num_elements
        self.weight = torch.nn.Parameter(
            torch.randn(
                self.num_elements, *self.weight_shape, device=device, dtype=dtype
            )
        )

        layout_in = default_layout(layout_in or layout)
        self.transpose_in = cuet.TransposeIrrepsLayout(
            self.etp.inputs[1].irreps,
            source=layout_in,
            target=self.etp.inputs[1].layout,
            device=device,
            use_fallback=use_fallback,
        )

        layout_out = default_layout(layout_out or layout)
        self.transpose_out = cuet.TransposeIrrepsLayout(
            self.etp.outputs[0].irreps,
            source=self.etp.outputs[0].layout,
            target=layout_out,
            device=device,
            use_fallback=use_fallback,
        )

        if method is None:
            if use_fallback is None:
                if same_shape:
                    # No warning here as it's the default behavior
                    self.method = "uniform_1d"
                else:
                    warnings.warn(
                        "Segments are not the same shape, falling back to `naive` method\n"
                        "You can consider making the segments uniform in the descriptor."
                    )
                    self.method = "naive"
            else:
                warnings.warn(
                    "`use_fallback` is deprecated, please use `method` instead",
                    DeprecationWarning,
                )
                if not same_shape and not use_fallback:
                    raise ValueError(
                        "`uniform_1d` method requires segments to be the same shape\n"
                        "You can consider making the segments uniform in the descriptor."
                    )
                self.method = "naive" if use_fallback else "uniform_1d"
        else:
            if method == "uniform_1d" and not same_shape:
                raise ValueError(
                    "`uniform_1d` method requires segments to be the same shape\n"
                    "You can consider making the segments uniform in the descriptor."
                )
            self.method = method

        self.f = cuet.SegmentedPolynomial(
            self.etp.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return (
            f"contraction_degree={self.contraction_degree}"
            f", weight_shape={self.weight_shape}"
        )

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the symmetric contraction operation.

        Args:
            x (torch.Tensor): The input tensor of shape ``(*batch, irreps_in.dim)``.
            indices (torch.Tensor, optional): The index of the weight to use for each batch element,
                of shape ``(*batch,)``. Optional when ``num_elements == 1``.
            mask (torch.Tensor, optional): Boolean mask of shape ``(*batch,)`` where ``True`` marks
                valid elements and ``False`` marks padded/masked elements whose outputs are zeroed.

        Returns:
            torch.Tensor: The output tensor of shape ``(*batch, irreps_out.dim)``.
        """
        if x.ndim < 1 or x.shape[-1] != self.irreps_in_dim:
            raise ValueError(
                f"Expected x to have last dimension {self.irreps_in_dim}, got shape {x.shape}"
            )

        batch_shape = list(x.shape[:-1])
        mask_bool: Optional[torch.Tensor] = None
        if mask is not None:
            if list(mask.shape) != batch_shape:
                raise ValueError(
                    f"Expected mask to have shape {batch_shape}, got {list(mask.shape)}"
                )
            mask_bool = mask if mask.dtype == torch.bool else mask.to(dtype=torch.bool)
            x = torch.where(mask_bool.unsqueeze(-1), x, torch.zeros_like(x))

        if self.projection is not None:
            proj_dtype = (
                torch.float64
                if (
                    self.weight.dtype == torch.float64
                    or self.projection.dtype == torch.float64
                )
                else torch.float32
            )
            weight = torch.einsum(
                "...au,ab->...bu",
                self.weight.to(dtype=proj_dtype),
                self.projection.to(dtype=proj_dtype),
            ).to(dtype=self.weight.dtype)
        else:
            weight = self.weight
        weight = weight.flatten(1)

        x_flat = self.transpose_in(x.reshape(-1, self.irreps_in_dim))

        input_indices: Dict[int, torch.Tensor] = {}
        if indices is None:
            if self.num_elements != 1:
                raise ValueError(
                    f"indices must be provided when num_elements != 1 (got num_elements={self.num_elements})"
                )
        else:
            if list(indices.shape) != batch_shape:
                raise ValueError(
                    f"Expected indices to have shape {batch_shape}, got {list(indices.shape)}"
                )
            if mask_bool is not None:
                indices = torch.where(mask_bool, indices, torch.zeros_like(indices))
            input_indices[0] = indices.reshape(-1)

        output = self.f([weight, x_flat], input_indices=input_indices)
        out = self.transpose_out(output[0]).reshape(batch_shape + [self.irreps_out_dim])
        if mask_bool is not None:
            out = torch.where(mask_bool.unsqueeze(-1), out, torch.zeros_like(out))
        return out
