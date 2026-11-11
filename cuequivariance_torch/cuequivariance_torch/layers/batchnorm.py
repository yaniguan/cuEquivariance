# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import torch
from cuequivariance.group_theory.irreps_array.misc_ui import (
    default_irreps,
    default_layout,
)

import cuequivariance as cue


# This implementation is an adaptation of https://github.com/e3nn/e3nn/blob/ef93f876c9985b3816aefb2982b3cf4325df6ba4/e3nn/nn/_batchnorm.py
class BatchNorm(torch.nn.Module):
    """Batch normalization for orthonormal representations.

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.

    Args:
        irreps (Irreps): Input irreps.
        layout (IrrepsLayout, optional): Layout of the input tensor, by default `IrrepsLayout.mul_ir`.
        eps (float, optional): Epsilon value for numerical stability, by default 1e-5.
        momentum (float, optional): Momentum for the running mean and variance, by default 0.1.
        affine (bool, optional): Whether to apply an affine transformation, by default True.
        reduce (str, optional): How to reduce the norm of the representations, by default "mean".
        instance (bool, optional): Whether to use instance normalization, by default False.
        include_bias (bool, optional): Whether to include a bias term, by default True.
    """

    __constants__ = [
        "instance",
        "reduce",
        "irs",
        "affine",
        "include_bias",
        "_is_mul_ir",
        "_dim",
    ]
    features: list[int]

    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        layout: cue.IrrepsLayout = None,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        reduce: str = "mean",
        instance: bool = False,
        include_bias: bool = True,
    ):
        super().__init__()
        self.layout = default_layout(layout)
        if self.layout not in (cue.mul_ir, cue.ir_mul):
            raise ValueError(f"Invalid layout option {self.layout}")
        self._is_mul_ir: bool = self.layout == cue.mul_ir
        (self.irreps,) = default_irreps(irreps)
        self._dim: int = self.irreps.dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.instance = instance
        self.include_bias = include_bias

        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps
        self.features = []

        if self.instance:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
        else:
            self.register_buffer("running_mean", torch.zeros(num_scalar))
            self.register_buffer("running_var", torch.ones(num_features))

        if affine:
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            if self.include_bias:
                self.bias = torch.nn.Parameter(torch.zeros(num_scalar))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce
        irs: list[tuple[int, int, bool]] = []
        for mul, ir in self.irreps:
            if mul > 0:
                irs.append((mul, ir.dim, ir.is_scalar()))
        self.irs = irs

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return f"{self.irreps}, layout={self.layout}, eps={self.eps}, momentum={self.momentum}"

    def _roll_avg(self, curr: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return (1 - self.momentum) * curr + self.momentum * update.detach().to(
            curr.dtype
        )

    def _run_indexed(
        self,
        input: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: Optional[int],
        mask_bool: Optional[torch.Tensor],
        acc_dtype: torch.dtype,
    ) -> torch.Tensor:
        num_nodes = input.shape[0]
        batch_idx = batch.to(torch.int64)
        if num_graphs is not None:
            n_graphs = num_graphs
        else:
            n_graphs = int(batch_idx.max().item()) + 1

        if mask_bool is not None:
            valid_f = mask_bool.to(acc_dtype).unsqueeze(-1)
        else:
            valid_f = torch.ones((num_nodes, 1), dtype=acc_dtype, device=input.device)

        counts = torch.zeros(
            (n_graphs, 1), dtype=acc_dtype, device=input.device
        ).scatter_add(0, batch_idx.unsqueeze(-1), valid_f)
        safe_counts = counts.clamp_min(1.0)
        total_valid = counts.sum()
        safe_total_valid = total_valid.clamp_min(1.0)

        new_means: list[torch.Tensor] = []
        new_vars: list[torch.Tensor] = []
        fields: list[torch.Tensor] = []

        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for mul, d, is_scalar in self.irs:
            field = input[:, ix : ix + mul * d]
            ix += mul * d

            if self._is_mul_ir:
                field = field.reshape(num_nodes, mul, d)
            else:
                field = field.reshape(num_nodes, d, mul).transpose(1, 2)

            field_acc = field.to(acc_dtype)
            if mask_bool is not None:
                field_acc = torch.where(
                    mask_bool.view(num_nodes, 1, 1),
                    field_acc,
                    torch.zeros_like(field_acc),
                )

            idx_mul = batch_idx.unsqueeze(-1).expand(num_nodes, mul)

            if is_scalar:
                if self.training or self.instance:
                    if self.instance:
                        graph_sum = torch.zeros(
                            (n_graphs, mul), dtype=acc_dtype, device=input.device
                        ).scatter_add(0, idx_mul, field_acc.reshape(num_nodes, mul))
                        graph_mean = graph_sum / safe_counts
                        field_acc = field_acc - graph_mean[batch_idx].reshape(
                            num_nodes, mul, 1
                        )
                    else:
                        field_mean = (
                            field_acc.reshape(num_nodes, mul).sum(dim=0)
                            / safe_total_valid
                        )
                        assert self.running_mean is not None
                        curr_rm = self.running_mean[irm : irm + mul]
                        upd_rm = self._roll_avg(curr_rm, field_mean)
                        if mask_bool is not None:
                            upd_rm = torch.where(total_valid > 0, upd_rm, curr_rm)
                        new_means.append(upd_rm)
                        field_acc = field_acc - field_mean.reshape(1, mul, 1)
                else:
                    assert self.running_mean is not None
                    field_mean = self.running_mean[irm : irm + mul].to(acc_dtype)
                    field_acc = field_acc - field_mean.reshape(1, mul, 1)
                irm += mul

                if mask_bool is not None:
                    field_acc = torch.where(
                        mask_bool.view(num_nodes, 1, 1),
                        field_acc,
                        torch.zeros_like(field_acc),
                    )

            if self.training or self.instance:
                sq = (field_acc * field_acc).mean(dim=2)  # [num_nodes, mul]

                if self.reduce == "mean":
                    if self.instance:
                        graph_sq = torch.zeros(
                            (n_graphs, mul), dtype=acc_dtype, device=input.device
                        ).scatter_add(0, idx_mul, sq)
                        graph_var = graph_sq / safe_counts
                        inv_std = torch.rsqrt(graph_var + self.eps)[batch_idx]
                    else:
                        field_norm = sq.sum(dim=0) / safe_total_valid
                        assert self.running_var is not None
                        curr_rv = self.running_var[irv : irv + mul]
                        upd_rv = self._roll_avg(curr_rv, field_norm)
                        if mask_bool is not None:
                            upd_rv = torch.where(total_valid > 0, upd_rv, curr_rv)
                        new_vars.append(upd_rv)
                        inv_std = torch.rsqrt(field_norm + self.eps).reshape(1, mul)
                elif self.reduce == "max":
                    graph_max = torch.zeros(
                        (n_graphs, mul), dtype=acc_dtype, device=input.device
                    ).scatter_reduce(0, idx_mul, sq, reduce="amax", include_self=False)
                    if self.instance:
                        inv_std = torch.rsqrt(graph_max + self.eps)[batch_idx]
                    else:
                        graph_valid = (counts > 0).to(acc_dtype)
                        num_valid_graphs = graph_valid.sum().clamp_min(1.0)
                        field_norm = (graph_max * graph_valid).sum(
                            dim=0
                        ) / num_valid_graphs
                        assert self.running_var is not None
                        curr_rv = self.running_var[irv : irv + mul]
                        upd_rv = self._roll_avg(curr_rv, field_norm)
                        if mask_bool is not None:
                            upd_rv = torch.where(total_valid > 0, upd_rv, curr_rv)
                        new_vars.append(upd_rv)
                        inv_std = torch.rsqrt(field_norm + self.eps).reshape(1, mul)
                else:
                    raise ValueError(f"Invalid reduce option {self.reduce}")
            else:
                assert self.running_var is not None
                field_norm = self.running_var[irv : irv + mul].to(acc_dtype)
                inv_std = torch.rsqrt(field_norm + self.eps).reshape(1, mul)
            irv += mul

            if self.affine:
                assert self.weight is not None
                weight = self.weight[iw : iw + mul].to(acc_dtype)
                iw += mul
                inv_std = inv_std * weight

            field_acc = field_acc * inv_std.unsqueeze(-1)

            if self.affine and self.include_bias and is_scalar:
                assert self.bias is not None
                bias = self.bias[ib : ib + mul].to(acc_dtype)
                ib += mul
                field_acc = field_acc + bias.reshape(1, mul, 1)

            if mask_bool is not None:
                field_acc = torch.where(
                    mask_bool.view(num_nodes, 1, 1),
                    field_acc,
                    torch.zeros_like(field_acc),
                )

            field_out = field_acc.to(input.dtype)
            if not self._is_mul_ir:
                field_out = field_out.transpose(1, 2)
            fields.append(field_out.reshape(num_nodes, mul * d))

        torch._assert(
            ix == self._dim,
            f"`ix` should have reached input.size(-1) ({self._dim}), but it ended at {ix}",
        )

        if self.training and not self.instance:
            assert self.running_mean is not None
            assert self.running_var is not None
            torch._assert(
                irm == self.running_mean.numel(), "irm == self.running_mean.numel()"
            )
            torch._assert(
                irv == self.running_var.size(0), "irv == self.running_var.size(0)"
            )
            if len(new_means) > 0:
                self.running_mean.copy_(torch.cat(new_means))
            if len(new_vars) > 0:
                self.running_var.copy_(torch.cat(new_vars))

        if self.affine:
            assert self.weight is not None
            torch._assert(iw == self.weight.size(0), "iw == self.weight.size(0)")
            if self.include_bias:
                assert self.bias is not None
                torch._assert(ib == self.bias.numel(), "ib == self.bias.numel()")

        return torch.cat(fields, dim=1)

    def forward(
        self,
        input: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Normalize the input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape ``(num_nodes, irreps.dim)`` or
                ``(batch, ..., irreps.dim)``. The last dimension must match ``irreps.dim``.
            batch (torch.Tensor, optional): 1D integer graph assignment vector of shape
                ``(num_nodes,)`` for batched molecular/PyG graphs with 2D ``input``.
            num_graphs (int, optional): Total number of graphs when ``batch`` is provided.
                If ``None``, inferred as ``int(batch.max()) + 1``.
            mask (torch.Tensor, optional): Boolean or 0/1 mask of shape ``input.shape[:-1]``
                (``True`` / ``1`` for valid nodes, ``False`` / ``0`` for padded dummy nodes).
                Masked positions are excluded from mean/variance statistics and zeroed in the output.

        Returns:
            torch.Tensor: Normalized tensor with the same shape and dtype as ``input``.
        """
        if input.dim() < 2:
            raise ValueError(
                f"Expected input with at least 2 dimensions, got {input.dim()}D"
            )

        orig_shape = list(input.shape)
        dim = orig_shape[-1]
        if dim != self._dim:
            raise ValueError(
                f"Expected input last dimension to be {self._dim}, got {dim}"
            )

        mask_bool: Optional[torch.Tensor] = None
        if mask is not None:
            if mask.dim() == input.dim() and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            if list(mask.shape) != orig_shape[:-1]:
                raise ValueError(
                    f"Expected mask shape {orig_shape[:-1]}, got {list(mask.shape)}"
                )
            mask_bool = mask.to(dtype=torch.bool)

        if batch is not None:
            if input.dim() != 2:
                raise ValueError(
                    f"Expected 2D input (num_nodes, dim) when batch is provided, got {input.dim()}D"
                )
            if batch.dim() != 1 or batch.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Expected 1D batch tensor of length {input.shape[0]}, got {list(batch.shape)}"
                )

        if input.numel() == 0:
            out_empty = input * 0.0
            if self.affine and self.weight is not None:
                out_empty = (
                    out_empty + (self.weight * self.weight).sum().to(input.dtype) * 0.0
                )
                if self.include_bias and self.bias is not None:
                    out_empty = (
                        out_empty + (self.bias * self.bias).sum().to(input.dtype) * 0.0
                    )
            return out_empty

        acc_dtype = (
            torch.float32
            if input.dtype in (torch.float16, torch.bfloat16)
            else input.dtype
        )

        if batch is not None:
            return self._run_indexed(input, batch, num_graphs, mask_bool, acc_dtype)

        if self.instance and input.dim() == 2:
            batch_size = 1
            sample_size = orig_shape[0]
        else:
            batch_size = orig_shape[0]
            sample_size = 1
            for s in orig_shape[1:-1]:
                sample_size *= s

        x = input.reshape(
            batch_size, sample_size, dim
        )  # [batch, sample, stacked features]

        mask_2d: Optional[torch.Tensor] = None
        safe_row_counts = torch.empty(0, dtype=acc_dtype, device=input.device)
        row_counts = torch.empty(0, dtype=acc_dtype, device=input.device)
        total_valid = torch.empty(0, dtype=acc_dtype, device=input.device)
        safe_total_valid = torch.empty(0, dtype=acc_dtype, device=input.device)
        if mask_bool is not None:
            mask_2d = mask_bool.reshape(batch_size, sample_size)
            mask_f = mask_2d.to(acc_dtype)
            row_counts = mask_f.sum(dim=1, keepdim=True)
            safe_row_counts = row_counts.clamp_min(1.0)
            total_valid = row_counts.sum()
            safe_total_valid = total_valid.clamp_min(1.0)

        new_means: list[torch.Tensor] = []
        new_vars: list[torch.Tensor] = []
        fields: list[torch.Tensor] = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for mul, d, is_scalar in self.irs:
            field = x[:, :, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d

            if self._is_mul_ir:
                # [batch, sample, mul, repr]
                field = field.reshape(batch_size, sample_size, mul, d)
            else:
                # [batch, sample, repr, mul] -> [batch, sample, mul, repr]
                field = field.reshape(batch_size, sample_size, d, mul).transpose(2, 3)

            field_acc = field.to(acc_dtype)
            if mask_2d is not None:
                field_acc = torch.where(
                    mask_2d.view(batch_size, sample_size, 1, 1),
                    field_acc,
                    torch.zeros_like(field_acc),
                )

            if is_scalar:
                if self.training or self.instance:
                    if self.instance:
                        if mask_2d is not None:
                            field_mean = (
                                field_acc.sum(dim=1).reshape(batch_size, mul)
                                / safe_row_counts
                            )
                        else:
                            field_mean = field_acc.mean(dim=1).reshape(
                                batch_size, mul
                            )  # [batch, mul]
                    else:
                        if mask_2d is not None:
                            field_mean = (
                                field_acc.sum(dim=[0, 1]).reshape(mul)
                                / safe_total_valid
                            )
                        else:
                            field_mean = field_acc.mean(dim=[0, 1]).reshape(
                                mul
                            )  # [mul]
                        assert self.running_mean is not None
                        curr_rm = self.running_mean[irm : irm + mul]
                        upd_rm = self._roll_avg(curr_rm, field_mean)
                        if mask_2d is not None:
                            upd_rm = torch.where(total_valid > 0, upd_rm, curr_rm)
                        new_means.append(upd_rm)
                else:
                    assert self.running_mean is not None
                    field_mean = self.running_mean[irm : irm + mul].to(acc_dtype)
                irm += mul

                # [batch, sample, mul, repr]
                field_acc = field_acc - field_mean.reshape(-1, 1, mul, 1)
                if mask_2d is not None:
                    field_acc = torch.where(
                        mask_2d.view(batch_size, sample_size, 1, 1),
                        field_acc,
                        torch.zeros_like(field_acc),
                    )

            if self.training or self.instance:
                sq = (field_acc * field_acc).mean(dim=3)  # [batch, sample, mul]

                if self.reduce == "mean":
                    if mask_2d is not None:
                        if self.instance:
                            field_norm = sq.sum(dim=1) / safe_row_counts
                        else:
                            field_norm = sq.sum(dim=[0, 1]) / safe_total_valid
                    else:
                        field_norm = sq.mean(dim=1)  # [batch, mul]
                        if not self.instance:
                            field_norm = field_norm.mean(dim=0)  # [mul]
                elif self.reduce == "max":
                    field_norm = sq.max(dim=1).values  # [batch, mul]
                    if not self.instance:
                        if mask_2d is not None:
                            row_valid = (row_counts > 0).to(acc_dtype)
                            field_norm = (field_norm * row_valid).sum(
                                dim=0
                            ) / row_valid.sum().clamp_min(1.0)
                        else:
                            field_norm = field_norm.mean(dim=0)  # [mul]
                else:
                    raise ValueError(f"Invalid reduce option {self.reduce}")

                if not self.instance:
                    assert self.running_var is not None
                    curr_rv = self.running_var[irv : irv + mul]
                    upd_rv = self._roll_avg(curr_rv, field_norm)
                    if mask_2d is not None:
                        upd_rv = torch.where(total_valid > 0, upd_rv, curr_rv)
                    new_vars.append(upd_rv)
            else:
                assert self.running_var is not None
                field_norm = self.running_var[irv : irv + mul].to(acc_dtype)
            irv += mul

            inv_std = torch.rsqrt(field_norm + self.eps)  # [(batch,) mul]

            if self.affine:
                assert self.weight is not None
                weight = self.weight[iw : iw + mul].to(acc_dtype)  # [mul]
                iw += mul

                inv_std = inv_std * weight  # [(batch,) mul]

            field_acc = field_acc * inv_std.reshape(
                -1, 1, mul, 1
            )  # [batch, sample, mul, repr]

            if self.affine and self.include_bias and is_scalar:
                assert self.bias is not None
                bias = self.bias[ib : ib + mul].to(acc_dtype)  # [mul]
                ib += mul
                field_acc = field_acc + bias.reshape(
                    1, 1, mul, 1
                )  # [batch, sample, mul, repr]

            if mask_2d is not None:
                field_acc = torch.where(
                    mask_2d.view(batch_size, sample_size, 1, 1),
                    field_acc,
                    torch.zeros_like(field_acc),
                )

            field_out = field_acc.to(input.dtype)
            if not self._is_mul_ir:
                field_out = field_out.transpose(2, 3)

            fields.append(
                field_out.reshape(batch_size, sample_size, mul * d)
            )  # [batch, sample, mul * repr]

        torch._assert(
            ix == dim,
            f"`ix` should have reached input.size(-1) ({dim}), but it ended at {ix}",
        )

        if self.training and not self.instance:
            assert self.running_mean is not None
            assert self.running_var is not None
            torch._assert(
                irm == self.running_mean.numel(), "irm == self.running_mean.numel()"
            )
            torch._assert(
                irv == self.running_var.size(0), "irv == self.running_var.size(0)"
            )
            if len(new_means) > 0:
                self.running_mean.copy_(torch.cat(new_means))
            if len(new_vars) > 0:
                self.running_var.copy_(torch.cat(new_vars))

        if self.affine:
            assert self.weight is not None
            torch._assert(iw == self.weight.size(0), "iw == self.weight.size(0)")
            if self.include_bias:
                assert self.bias is not None
                torch._assert(ib == self.bias.numel(), "ib == self.bias.numel()")

        output = torch.cat(fields, dim=2)  # [batch, sample, stacked features]
        return output.reshape(orig_shape)
