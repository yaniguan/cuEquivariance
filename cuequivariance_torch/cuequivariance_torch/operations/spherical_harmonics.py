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
from typing import Optional

import torch
import torch.nn as nn

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors


class SphericalHarmonics(nn.Module):
    r"""Compute the spherical harmonics of the input vectors as a torch module."""

    def __init__(
        self,
        ls: list[int],
        normalize: bool = True,
        device: Optional[torch.device] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
        eps: float = 0.0,
    ):
        """
        Args:
            ls (list of int): List of spherical harmonic degrees.
            normalize (bool, optional): Whether to normalize the input vectors. Defaults to True.
            device (torch.device, optional): The device to use for the operation.
            math_dtype (torch.dtype or string, optional): The dtype to use for the operation, by default it follows the dtype of the input tensors,
                if possible, or the torch default dtype (see SegmentedPolynomial for more details).
            method (str, optional): The method to use for the operation, by default "uniform_1d" (using a CUDA kernel).
            use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
                If `True` the "naive" method is used.
                If `False` or None (default) the "uniform_1d" method is used.
            eps (float, optional): Non-negative squared-norm regularization added when ``normalize=True``.
                Defaults to ``0.0`` (exact zero-safe normalization with zero 1st and 2nd derivatives at the origin).
        """
        super().__init__()
        self.ls = ls if isinstance(ls, list) else [ls]
        assert self.ls == sorted(set(self.ls))
        if not (eps >= 0.0):
            raise ValueError(f"`eps` must be non-negative, got {eps}")
        self.normalize = normalize
        self.eps = float(eps)

        e = descriptors.spherical_harmonics(cue.SO3(1), self.ls)
        self.out_dim: int = e.outputs[0].dim

        if method is None:
            if use_fallback is None:
                # No warning here as it's the default behavior
                self.method = "uniform_1d"
            else:
                warnings.warn(
                    "`use_fallback` is deprecated, please use `method` instead",
                    DeprecationWarning,
                )
                self.method = "naive" if use_fallback else "uniform_1d"
        else:
            self.method = method

        self.f = cuet.SegmentedPolynomial(
            e.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

    def forward(
        self,
        vectors: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vectors (torch.Tensor): Input vectors of shape ``(*batch_dims, 3)``.
            mask (torch.Tensor, optional): Optional boolean or floating mask broadcastable
                to ``vectors.shape[:-1]``. Masked-out slots (``False`` or ``0``) are sanitized
                before normalization and polynomial evaluation and zeroed across all output
                channels (including ``l = 0``).

        Returns:
            torch.Tensor: The spherical harmonics of the input vectors of shape ``(*batch_dims, dim)``,
            where ``dim`` is the sum of ``2*l+1`` for ``l`` in ``ls``.
        """
        is_fx_tracing = False
        if not torch.jit.is_scripting():
            fx = getattr(torch, "fx", None)
            if fx is not None:
                symbolic_trace = getattr(fx, "_symbolic_trace", None)
                if symbolic_trace is not None:
                    is_fx_tracing = getattr(
                        symbolic_trace, "is_fx_symbolic_tracing", bool
                    )()

        if not is_fx_tracing:
            torch._assert(
                vectors.ndim >= 1,
                "Input vectors must have shape (*batch_dims, 3)",
            )
            torch._assert(
                vectors.shape[-1] == 3,
                "Input vectors last dimension must have size 3",
            )

        mask_bool: Optional[torch.Tensor] = None
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool).expand(vectors.shape[:-1])
            vectors = torch.where(
                mask_bool.unsqueeze(-1), vectors, torch.zeros_like(vectors)
            )

        if self.normalize:
            sq_norm = torch.sum(vectors * vectors, dim=-1, keepdim=True)
            if self.eps > 0.0:
                sq_norm = sq_norm + self.eps
            is_zero = sq_norm == 0.0
            safe_sq = torch.where(is_zero, torch.ones_like(sq_norm), sq_norm)
            inv_norm = torch.rsqrt(safe_sq)
            vectors = torch.where(
                is_zero, torch.zeros_like(vectors), vectors * inv_norm
            )

        if torch.jit.is_scripting():
            out_shape = vectors.shape[:-1] + [self.out_dim]
        else:
            out_shape = vectors.shape[:-1] + (self.out_dim,)

        vectors_2d = vectors.reshape(-1, 3)
        if not is_fx_tracing and vectors_2d.shape[0] == 0:
            output_2d = vectors_2d.sum(dim=-1, keepdim=True) * vectors_2d.new_zeros(
                (0, self.out_dim)
            )
        else:
            output_2d = self.f([vectors_2d])[0]

        output = output_2d.reshape(out_shape)

        if mask_bool is not None:
            output = torch.where(
                mask_bool.unsqueeze(-1), output, torch.zeros_like(output)
            )

        return output
