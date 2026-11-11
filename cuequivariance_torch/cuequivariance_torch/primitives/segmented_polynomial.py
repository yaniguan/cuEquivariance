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

import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.fx
import torch.nn as nn
from cuequivariance_torch.primitives.segmented_polynomial_fused_tp import (
    SegmentedPolynomialFusedTP,
)
from cuequivariance_torch.primitives.segmented_polynomial_indexed_linear import (
    SegmentedPolynomialIndexedLinear,
)
from cuequivariance_torch.primitives.segmented_polynomial_naive import (
    SegmentedPolynomialNaive,
)
from cuequivariance_torch.primitives.segmented_polynomial_uniform_1d import (
    SegmentedPolynomialFromUniform1dJit,
)

import cuequivariance as cue

try:
    import cuequivariance_ops_torch  # noqa: F401

    HAS_CUE_OPS = True
except ImportError:
    HAS_CUE_OPS = False


def _prod(shape: List[int]) -> int:
    result = 1
    for s in shape:
        result *= s
    return result


def _broadcast_batch_shapes(shapes: List[List[int]]) -> List[int]:
    max_ndim = 0
    for s in shapes:
        if len(s) > max_ndim:
            max_ndim = len(s)
    result: List[int] = []
    for d in range(max_ndim):
        dim = 1
        has_zero = False
        for s in shapes:
            idx = len(s) - max_ndim + d
            if idx >= 0:
                val = s[idx]
                if val == 0:
                    if dim != 1 and dim != 0:
                        raise RuntimeError(
                            f"The size of tensor batch dimension ({val}) must match the existing size ({dim})"
                        )
                    dim = 0
                    has_zero = True
                elif val != 1:
                    if has_zero or (dim != 1 and dim != val):
                        raise RuntimeError(
                            f"The size of tensor batch dimension ({val}) must match the existing size ({dim})"
                        )
                    dim = val
        result.append(dim)
    return result


@torch.jit.unused
def _broadcast_shapes_eager(shapes: List[List[int]]) -> List[int]:
    return list(torch.broadcast_shapes(*shapes))


def _zero_touch(inputs: List[torch.Tensor]) -> torch.Tensor:
    z = inputs[0].reshape(-1).sum() * 0.0
    for inp in inputs[1:]:
        z = z + inp.reshape(-1).sum() * 0.0
    return z


class SegmentedPolynomial(nn.Module):
    """PyTorch module that computes a segmented polynomial.

    Args:
        polynomial: The segmented polynomial to compute, an instance of
            `cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`.
        method: Specifies the implementation method to use. Options are:

            - ``"naive"``: Uses a naive PyTorch implementation. It always works but is not optimized.
            - ``"uniform_1d"``: Uses a CUDA implementation for polynomials with a single uniform mode.
            - ``"fused_tp"``: Uses a CUDA implementation for polynomials with 3- and 4-operand contractions.
            - ``"indexed_linear"``: Uses a CUDA implementation for linear layers with indexed weights.

        math_dtype: Optional data type for computational operations.
            If specified, internal buffers will be of this dtype,
            and operands will be converted to this type for all computations.

            Values can be specified as a string corresponding to a torch.dtype,
            or as a torch.dtype.
            For some methods, special values can be used:

            - For method ``"naive"``: Any torch.dtype or corresponding string.
            - For method ``"uniform_1d"``: ``torch.float32`` or ``torch.float64`` or corresponding strings.
            - For method ``"fused_tp"``: ``torch.float32`` or ``torch.float64`` or corresponding strings.
            - For method ``"indexed_linear"``: this is not supported and will be ignored.

            .. note::
               This will not be affected by changes to the module dtype,
               and not all methods support all dtypes.

            If ``math_dtype`` is not specified:

            - For method ``"naive"``, the dtype of the input tensors will be used.
            - For method ``"uniform_1d"``, the dtype of the input tensors will be used if allowed
              (FP32 or FP64), otherwise float32 will be used.
            - For method ``"fused_tp"``, the default dtype (FP32) will be used.
            - For method ``"indexed_linear"``, the dtype of the input tensors will be used.

        output_dtype_map: Optional list that, for each output buffer, specifies
            the index of the input buffer from which it inherits its data type.
            -1 means the math_dtype is used.
            Default is 0 if there are input tensors, otherwise -1.
        name: Optional name for the operation. Defaults to "segmented_polynomial".
        options: Optional dictionary of method-specific options.

    Examples:
        Basic usage with spherical harmonics:

        >>> import torch
        >>> import cuequivariance as cue
        >>> from cuequivariance_torch import SegmentedPolynomial
        >>>
        >>> # Create spherical harmonics polynomial
        >>> poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2]).polynomial
        >>> sp = SegmentedPolynomial(poly, method="naive")
        >>>
        >>> # Compute spherical harmonics for unit vector along y-axis
        >>> x = torch.tensor([[0.0, 1.0, 0.0]])
        >>> result = sp([x])
        >>> print(result[0].shape)
        torch.Size([1, 9])

        Example with a linear layer:

        >>> # Create a linear transformation
        >>> input_irreps = cue.Irreps(cue.O3, "5x0e + 3x1o")
        >>> output_irreps = cue.Irreps(cue.O3, "4x0e + 2x1o")
        >>> poly = cue.descriptors.linear(input_irreps, output_irreps).polynomial
        >>>
        >>> # Create the module
        >>> linear = SegmentedPolynomial(poly, method="naive")
        >>>
        >>> # Create random weights and input
        >>> weights = torch.randn(1, poly.inputs[0].size)
        >>> x = torch.randn(10, poly.inputs[1].size)
        >>>
        >>> # Forward pass
        >>> result = linear([weights, x])
        >>> print(result[0].shape)
        torch.Size([10, 10])

        Example with indexed operations:

        >>> # Create indexed weights for different elements
        >>> weights = torch.randn(3, poly.inputs[0].size)  # 3 different weight sets
        >>> x = torch.randn(5, poly.inputs[1].size)        # 5 input vectors
        >>>
        >>> # Index tensor specifying which weights to use for each input
        >>> weight_indices = torch.tensor([0, 1, 0, 2, 1])  # Use weights 0,1,0,2,1
        >>>
        >>> result = linear([weights, x],
        ...                input_indices={0: weight_indices})
        >>> print(result[0].shape)
        torch.Size([5, 10])
    """

    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        method: str = "",
        math_dtype: str | torch.dtype = None,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
        options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self._polynomial_orig = polynomial
        self._math_dtype_orig = math_dtype
        self._output_dtype_map_orig = output_dtype_map
        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs
        self.input_sizes: List[int] = [o.size for o in polynomial.inputs]
        self.output_sizes: List[int] = [o.size for o in polynomial.outputs]
        default_dtype_map = [
            0 if polynomial.num_inputs >= 1 else -1
        ] * polynomial.num_outputs
        self._output_dtype_map: List[int] = (
            default_dtype_map if output_dtype_map is None else list(output_dtype_map)
        )
        resolved_math_dtype: Optional[torch.dtype] = None
        if isinstance(math_dtype, torch.dtype):
            resolved_math_dtype = math_dtype
        elif isinstance(math_dtype, str) and hasattr(torch, math_dtype):
            attr = getattr(torch, math_dtype)
            if isinstance(attr, torch.dtype):
                resolved_math_dtype = attr
        self._resolved_math_dtype: Optional[torch.dtype] = resolved_math_dtype
        self.method = method
        self.repr = polynomial.__repr__()
        self.options = options if options is not None else {}

        if method == "":
            warnings.warn(
                "Hello! It looks like you're using code that was written for an older version of this library.\n"
                "Starting in v0.6.0, the `method` argument is suggested when using `SegmentedPolynomial()`.\n"
                "This change helps ensure you get optimal performance by explicitly choosing the computation method.\n"
                "For the moment, we will default to the 'uniform_1d' method.\n\n"
                "To remove this warning, add a `method` parameter to your function call. Here are the available options:\n"
                "• 'naive' - Works everywhere but not optimized (good for testing)\n"
                "• 'uniform_1d' - Fast CUDA implementation for single uniform mode polynomials\n"
                "• 'fused_tp' - A more general CUDA implementation, supporting many 3 and 4 operands contractions.\n"
                "• 'indexed_linear' - A CUDA implementation for linear layers with indexed weights.\n"
            )
            method = "uniform_1d"

        if not isinstance(polynomial, cue.SegmentedPolynomial):
            raise ValueError(
                f"The polynomial is not a cue.SegmentedPolynomial, but a {type(polynomial)}",
                "Did you forget to call `.polynomial` on the descriptor?",
            )

        if method != "naive" and not HAS_CUE_OPS:
            method = "naive"
            warnings.warn(
                "cuequivariance_ops_torch is not available. Falling back to naive implementation."
            )

        if method == "uniform_1d":
            try:
                self.m = SegmentedPolynomialFromUniform1dJit(
                    polynomial, math_dtype, output_dtype_map, name, self.options
                )
            except ImportError:
                method = "naive"
                warnings.warn(
                    "uniform_1d backend is not available. "
                    "Falling back to naive implementation."
                )
                self.m = SegmentedPolynomialNaive(
                    polynomial, math_dtype, output_dtype_map, name
                )
        elif method == "naive":
            self.m = SegmentedPolynomialNaive(
                polynomial, math_dtype, output_dtype_map, name
            )
        elif method == "fused_tp":
            try:
                self.m = SegmentedPolynomialFusedTP(
                    polynomial, math_dtype, output_dtype_map, name
                )
            except ImportError:
                method = "naive"
                warnings.warn(
                    "fused_tp backend is not available. "
                    "Falling back to naive implementation."
                )
                self.m = SegmentedPolynomialNaive(
                    polynomial, math_dtype, output_dtype_map, name
                )
        elif method == "indexed_linear":
            try:
                self.m = SegmentedPolynomialIndexedLinear(
                    polynomial, math_dtype, output_dtype_map, name, self.options
                )
            except ImportError:
                method = "naive"
                warnings.warn(
                    "indexed_linear backend is not available. "
                    "Falling back to naive implementation."
                )
                self.m = SegmentedPolynomialNaive(
                    polynomial, math_dtype, output_dtype_map, name
                )
        else:
            raise ValueError(f"Invalid method: {method}")
        self.method = method

    def __reduce__(self):
        return (
            SegmentedPolynomial,
            (
                self._polynomial_orig,
                self.method,
                self._math_dtype_orig,
                self._output_dtype_map_orig,
                "segmented_polynomial",
                self.options,
            ),
        )

    def __repr__(self):
        return self.repr + f"\n{super().__repr__()}"

    def _get_output_dtype(self, k: int, inputs: List[torch.Tensor]) -> torch.dtype:
        dtype_idx = self._output_dtype_map[k]
        if dtype_idx >= 0 and dtype_idx < len(inputs):
            return inputs[dtype_idx].dtype
        math_dt = self._resolved_math_dtype
        if math_dt is not None:
            return math_dt
        if len(inputs) > 0:
            return inputs[0].dtype
        return torch.float32

    def _call_backend(
        self,
        inputs: List[torch.Tensor],
        input_indices: Dict[int, torch.Tensor],
        output_shapes: Dict[int, torch.Tensor],
        output_indices: Dict[int, torch.Tensor],
    ) -> List[torch.Tensor]:
        if not torch.jit.is_scripting():
            is_fx_tracing = False
            fx = getattr(torch, "fx", None)
            if fx is not None:
                symbolic_trace = getattr(fx, "_symbolic_trace", None)
                if symbolic_trace is not None:
                    is_fx_tracing = getattr(
                        symbolic_trace, "is_fx_symbolic_tracing", bool
                    )()

            if (
                not torch.jit.is_tracing()
                and not torch.compiler.is_compiling()
                and not is_fx_tracing
                and not self.options.get("disable_cpu_fallback", False)
                and len(inputs) > 0
                and inputs[0].device == torch.device("cpu")
                and self.method in ("fused_tp", "indexed_linear")
            ):
                method_label = (
                    "Fused TP" if self.method == "fused_tp" else "indexed_linear"
                )
                warnings.warn(
                    f"{method_label} is not supported on CPU. Falling back to naive implementation."
                )
                if not hasattr(self, "_cpu_fallback"):
                    fallback_math_dtype = (
                        None
                        if self.method == "indexed_linear"
                        else self._math_dtype_orig
                    )
                    object.__setattr__(
                        self,
                        "_cpu_fallback",
                        SegmentedPolynomialNaive(
                            self._polynomial_orig,
                            fallback_math_dtype,
                            self._output_dtype_map_orig,
                        ),
                    )
                return self._cpu_fallback(
                    inputs, input_indices, output_shapes, output_indices
                )

        return self.m(inputs, input_indices, output_shapes, output_indices)

    # For torch.jit.trace, we cannot pass explicit optionals,
    # so these must be passed as kwargs then.
    # List[Optional[Tensor]] does not work for similar reasons, hence, Dict
    # is the only option.
    # Also, shapes cannot be passed as integers, so they are passed via a
    # (potentially small-strided) tensor with the right shape.
    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Optional[Dict[int, torch.Tensor]] = None,
        output_shapes: Optional[Dict[int, torch.Tensor]] = None,
        output_indices: Optional[Dict[int, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Compute the segmented polynomial based on the specified descriptor.

        Args:
            inputs: The input tensors. The number of input tensors must match
                the number of input buffers in the descriptor.
                Each input tensor should have a shape of ``(*batch, operand_size)``
                (including 1D ``(operand_size,)`` or 2D ``(batch, operand_size)`` /
                ``(1, operand_size)``) in the unindexed case, or
                ``(index, operand_size)`` in the indexed case.
                Here, ``operand_size`` is the size of each operand as defined in
                the descriptor.
            input_indices: A dictionary that contains an optional indexing tensor
                for each input tensor. The key is the index into the inputs.
                If a key is not present, no indexing takes place.
                Negative indices (such as ``-1`` padding sentinels in fixed-capacity
                neighbor lists) are safely treated as masked-out invalid entries.

                .. note::
                   Method ``"indexed_linear"`` requires the indices to be sorted
                   unless ``options={"sort_indices": True}`` is specified.

            output_shapes: A dictionary specifying the size of the output batch
                dimensions using Tensors. We only read ``shape_tensor.shape[:-1]``
                (``shape_tensor.shape[0]`` in the 2D / indexed case).
                This is mandatory if the output tensor is indexed. Otherwise,
                the default shape is ``(*broadcast_batch, operand_size)``.
            output_indices: A dictionary that contains an optional indexing tensor
                for each output tensor. See ``input_indices`` for details.
            mask: An optional boolean tensor indicating valid batch/edge entries
                for fixed-capacity execution. Invalid entries contribute zero to
                both forward outputs and gradients of all orders.

        Returns:
            The output tensors resulting from the segmented polynomial.
            Their shapes are specified just like the inputs.
        """

        # General checks
        empty_dict: Dict[int, torch.Tensor] = {}
        if input_indices is None:
            input_indices = dict(empty_dict)
        if output_shapes is None:
            output_shapes = dict(empty_dict)
        if output_indices is None:
            output_indices = dict(empty_dict)

        inputs = list(inputs)
        is_indexed = len(input_indices) > 0 or len(output_indices) > 0

        if not torch.jit.is_scripting():
            is_fx_tracing = False
            fx = getattr(torch, "fx", None)
            if fx is not None:
                symbolic_trace = getattr(fx, "_symbolic_trace", None)
                if symbolic_trace is not None:
                    is_fx_tracing = getattr(
                        symbolic_trace, "is_fx_symbolic_tracing", bool
                    )()

            if (
                not torch.jit.is_tracing()
                and not torch.compiler.is_compiling()
                and not is_fx_tracing
            ):
                torch._assert(
                    len(inputs) == self.num_inputs,
                    "the number of inputs must match the number of inputs of the polynomial",
                )

                expected_idx_len: Optional[int] = None
                for k, v in input_indices.items():
                    torch._assert(
                        0 <= k < self.num_inputs, "input index must be in range"
                    )
                    torch._assert(v.ndim == 1, "input index must be one-dimensional")
                    torch._assert(
                        v.dtype in [torch.int32, torch.int64],
                        "input index must be integral",
                    )
                    if expected_idx_len is None:
                        expected_idx_len = v.shape[0]
                    else:
                        torch._assert(
                            v.shape[0] == expected_idx_len,
                            "all index tensors must have the same length",
                        )
                for k, v in output_indices.items():
                    torch._assert(
                        0 <= k < self.num_outputs, "output index must be in range"
                    )
                    torch._assert(v.ndim == 1, "input index must be one-dimensional")
                    torch._assert(
                        v.dtype in [torch.int32, torch.int64],
                        "input index must be integral",
                    )
                    if expected_idx_len is None:
                        expected_idx_len = v.shape[0]
                    else:
                        torch._assert(
                            v.shape[0] == expected_idx_len,
                            "all index tensors must have the same length",
                        )

                all_inputs_2d = all(inp.ndim == 2 for inp in inputs)
                for k, v in output_shapes.items():
                    torch._assert(
                        0 <= k < self.num_outputs, "output index must be in range"
                    )
                    if is_indexed or all_inputs_2d:
                        torch._assert(
                            v.ndim == 2, "output shape must be two-dimensional"
                        )
                    else:
                        torch._assert(
                            v.ndim >= 1,
                            "output shape must have at least one dimension",
                        )

                if mask is not None:
                    if is_indexed:
                        torch._assert(mask.ndim == 1, "mask must be one-dimensional")
                        if expected_idx_len is not None:
                            torch._assert(
                                mask.shape[0] == expected_idx_len,
                                "mask length must match index length",
                            )
                    elif (
                        len(inputs) > 0
                        and all_inputs_2d
                        and all(v.ndim == 2 for v in output_shapes.values())
                    ):
                        torch._assert(mask.ndim == 1, "mask must be one-dimensional")

        for i in range(len(inputs)):
            x = inputs[i]
            if x.ndim == 0:
                raise ValueError(f"Input {i} has no dimensions")
            if x.shape[-1] != self.input_sizes[i]:
                raise ValueError(
                    f"Input {i} has shape {x.shape} but expected shape {self.input_sizes[i]}."
                )

        # 1. Indexed execution (with safe gather/scatter for masks and negative sentinels)
        if is_indexed:
            safe_inputs = list(inputs)
            for i in range(len(safe_inputs)):
                if i not in input_indices and safe_inputs[i].ndim == 1:
                    safe_inputs[i] = safe_inputs[i].unsqueeze(0)
                if safe_inputs[i].ndim != 2:
                    raise ValueError(
                        f"Input {i} must be 2D in indexed mode, got shape {safe_inputs[i].shape}."
                    )

            for k in output_indices.keys():
                if k not in output_shapes:
                    raise ValueError(
                        "output shapes must be provided for output indices"
                    )

            masks_list: List[torch.Tensor] = []
            if mask is not None:
                masks_list.append(mask.to(dtype=torch.bool))
            for v in input_indices.values():
                masks_list.append(v >= 0)
            for v in output_indices.values():
                masks_list.append(v >= 0)
            valid = masks_list[0]
            for m_t in masks_list[1:]:
                valid = valid & m_t

            e_len = valid.shape[0]
            if e_len == 0:
                dev = safe_inputs[0].device if len(safe_inputs) > 0 else valid.device
                zt: Optional[torch.Tensor] = (
                    _zero_touch(safe_inputs) if len(safe_inputs) > 0 else None
                )
                outs_empty: List[torch.Tensor] = []
                for k in range(self.num_outputs):
                    out_rows = output_shapes[k].shape[0] if k in output_shapes else 0
                    out_dt = self._get_output_dtype(k, safe_inputs)
                    z_out = torch.zeros(
                        (out_rows, self.output_sizes[k]), dtype=out_dt, device=dev
                    )
                    if zt is not None and zt.requires_grad:
                        z_out = z_out + zt.to(dtype=out_dt)
                    outs_empty.append(z_out)
                return outs_empty

            safe_input_indices: Dict[int, torch.Tensor] = {}
            for k, in_idx in input_indices.items():
                if safe_inputs[k].shape[0] == 0:
                    pad_row = torch.zeros(
                        (1, self.input_sizes[k]),
                        dtype=safe_inputs[k].dtype,
                        device=safe_inputs[k].device,
                    )
                    if safe_inputs[k].requires_grad:
                        pad_row = pad_row + safe_inputs[k].reshape(-1).sum() * 0.0
                    safe_inputs[k] = pad_row
                safe_input_indices[k] = torch.clamp_min(in_idx, 0)

            safe_output_indices: Dict[int, torch.Tensor] = {}
            safe_output_shapes: Dict[int, torch.Tensor] = {}
            for k, shape_stub in output_shapes.items():
                if k in output_indices:
                    n_out = shape_stub.shape[0]
                    out_idx = output_indices[k]
                    safe_output_indices[k] = torch.where(
                        valid, out_idx, torch.full_like(out_idx, n_out)
                    )
                    safe_output_shapes[k] = torch.empty(
                        (1, 1), device=shape_stub.device, dtype=shape_stub.dtype
                    ).expand(n_out + 1, 1)
                else:
                    if shape_stub.shape[0] == 1 and e_len != 1:
                        # Compute unreduced (e_len, out_size), mask, then sum below
                        pass
                    else:
                        safe_output_shapes[k] = shape_stub

            outs = self._call_backend(
                safe_inputs,
                safe_input_indices,
                safe_output_shapes,
                safe_output_indices,
            )

            valid_col = valid.unsqueeze(-1)
            for k in range(self.num_outputs):
                if k in output_indices:
                    outs[k] = outs[k][: output_shapes[k].shape[0]]
                else:
                    out_k = outs[k]
                    if out_k.shape[0] == 1 and e_len > 1:
                        out_k = out_k.expand(e_len, self.output_sizes[k])
                    out_k = torch.where(valid_col, out_k, torch.zeros_like(out_k))
                    if (
                        k in output_shapes
                        and output_shapes[k].shape[0] == 1
                        and e_len != 1
                    ):
                        out_k = out_k.sum(dim=0, keepdim=True)
                    outs[k] = out_k
            return outs

        # 2. Zero-input polynomials
        if len(inputs) == 0:
            outs = self._call_backend(
                inputs, input_indices, output_shapes, output_indices
            )
            if mask is not None:
                v_col = mask.to(dtype=torch.bool).unsqueeze(-1)
                outs = [torch.where(v_col, out, torch.zeros_like(out)) for out in outs]
            return outs

        # 3. Fast path for standard unmasked 2D execution
        is_all_2d = True
        for inp in inputs:
            if inp.ndim != 2:
                is_all_2d = False
        for v in output_shapes.values():
            if v.ndim != 2:
                is_all_2d = False

        if is_all_2d and mask is None:
            has_zero_batch = False
            inferred_2d_batch = 1
            for inp in inputs:
                s0 = inp.shape[0]
                if s0 == 0:
                    has_zero_batch = True
                elif s0 != 1:
                    if inferred_2d_batch != 1 and inferred_2d_batch != s0:
                        raise ValueError(
                            f"Input batch size mismatch: {inferred_2d_batch} vs {s0}."
                        )
                    inferred_2d_batch = s0
            if has_zero_batch and inferred_2d_batch != 1:
                raise ValueError(
                    f"Input batch size mismatch: 0 vs {inferred_2d_batch}."
                )
            if has_zero_batch and len(output_shapes) == 0:
                zt_2d = _zero_touch(inputs)
                outs_zero_2d: List[torch.Tensor] = []
                for k in range(self.num_outputs):
                    out_dt = self._get_output_dtype(k, inputs)
                    z_out = torch.zeros(
                        (0, self.output_sizes[k]),
                        dtype=out_dt,
                        device=inputs[0].device,
                    )
                    if zt_2d.requires_grad:
                        z_out = z_out + zt_2d.to(dtype=out_dt)
                    outs_zero_2d.append(z_out)
                return outs_zero_2d

            return self._call_backend(
                inputs, input_indices, output_shapes, output_indices
            )

        # 4. General unindexed execution: 1D, multi-batch (*batch, dim), broadcasting, and/or mask
        max_batch_ndim = 0
        batch_shapes: List[List[int]] = []
        for inp in inputs:
            b_sh = list(inp.shape[:-1])
            if len(b_sh) > max_batch_ndim:
                max_batch_ndim = len(b_sh)
            batch_shapes.append(b_sh)

        for v in output_shapes.values():
            v_sh = list(v.shape[:-1])
            if len(v_sh) > max_batch_ndim:
                max_batch_ndim = len(v_sh)
            if _prod(v_sh) != 1:
                batch_shapes.append(v_sh)

        if mask is not None:
            if mask.ndim > max_batch_ndim:
                raise ValueError(
                    f"mask has {mask.ndim} dimensions, exceeding batch dimensions ({max_batch_ndim})."
                )
            batch_shapes.append(list(mask.shape))

        if torch.jit.is_scripting():
            broadcast_batch = _broadcast_batch_shapes(batch_shapes)
        else:
            broadcast_batch = _broadcast_shapes_eager(batch_shapes)

        # Validate output_shapes against broadcast_batch
        for k, v in output_shapes.items():
            v_batch = list(v.shape[:-1])
            if _prod(v_batch) != 1 and v_batch != broadcast_batch:
                raise ValueError(
                    f"Output {k} batch shape {v_batch} does not match broadcast batch shape {broadcast_batch}."
                )

        # 4a. Pure 1D inputs -> 1D outputs
        if len(broadcast_batch) == 0:
            inputs_1d_as_2d = [
                inp.reshape(1, self.input_sizes[i]) for i, inp in enumerate(inputs)
            ]
            out_shapes_1d: Dict[int, torch.Tensor] = {}
            for k, v in output_shapes.items():
                out_shapes_1d[k] = torch.empty((1, 1), device=v.device, dtype=v.dtype)
            outs_1d = self._call_backend(
                inputs_1d_as_2d, input_indices, out_shapes_1d, output_indices
            )
            if mask is not None:
                v_bool = mask.to(dtype=torch.bool).reshape(1, 1)
                outs_1d = [
                    torch.where(v_bool, out, torch.zeros_like(out)) for out in outs_1d
                ]
            return [out.squeeze(0) for out in outs_1d]

        # 4b. Multi-dimensional batch (*broadcast_batch, dim)
        b_flat = _prod(broadcast_batch)
        if b_flat == 0:
            zt_mb = _zero_touch(inputs)
            outs_zero_mb: List[torch.Tensor] = []
            for k in range(self.num_outputs):
                out_size = self.output_sizes[k]
                out_dt = self._get_output_dtype(k, inputs)
                target_batch = (
                    list(output_shapes[k].shape[:-1])
                    if k in output_shapes
                    else broadcast_batch
                )
                z_out = torch.zeros(
                    target_batch + [out_size], dtype=out_dt, device=inputs[0].device
                )
                if zt_mb.requires_grad:
                    z_out = z_out + zt_mb.to(dtype=out_dt)
                outs_zero_mb.append(z_out)
            return outs_zero_mb

        inputs_2d: List[torch.Tensor] = []
        for i, inp in enumerate(inputs):
            op_size = self.input_sizes[i]
            in_batch = list(inp.shape[:-1])
            in_numel = _prod(in_batch)
            if in_numel == 1:
                # Bandwidth-preserving shared operand: keep as (1, op_size) without .expand()
                inputs_2d.append(inp.reshape(1, op_size))
            elif in_batch == broadcast_batch:
                inputs_2d.append(inp.reshape(b_flat, op_size))
            else:
                inputs_2d.append(
                    inp.expand(broadcast_batch + [op_size])
                    .contiguous()
                    .reshape(b_flat, op_size)
                )

        out_shapes_2d: Dict[int, torch.Tensor] = {}
        for k, v in output_shapes.items():
            v_numel = _prod(list(v.shape[:-1]))
            if v_numel == 1:
                if mask is None:
                    out_shapes_2d[k] = torch.empty(
                        (1, 1), device=v.device, dtype=v.dtype
                    )
            else:
                out_shapes_2d[k] = torch.empty(
                    (1, 1), device=v.device, dtype=v.dtype
                ).expand(b_flat, 1)

        outs_mb = self._call_backend(
            inputs_2d, input_indices, out_shapes_2d, output_indices
        )

        if mask is not None:
            valid_flat = (
                mask.to(dtype=torch.bool).expand(broadcast_batch).reshape(b_flat)
            )
            valid_flat_col = valid_flat.unsqueeze(-1)
            for k in range(self.num_outputs):
                out_size = self.output_sizes[k]
                out_k = outs_mb[k]
                if out_k.shape[0] == 1 and b_flat > 1:
                    out_k = out_k.expand(b_flat, out_size)
                out_k = torch.where(valid_flat_col, out_k, torch.zeros_like(out_k))
                if k in output_shapes and _prod(list(output_shapes[k].shape[:-1])) == 1:
                    out_k = out_k.sum(dim=0, keepdim=True)
                outs_mb[k] = out_k

        for k in range(self.num_outputs):
            out_size = self.output_sizes[k]
            target_batch = (
                list(output_shapes[k].shape[:-1])
                if k in output_shapes
                else broadcast_batch
            )
            target_flat = _prod(target_batch)
            outs_mb[k] = (
                outs_mb[k]
                .expand(target_flat, out_size)
                .reshape(target_batch + [out_size])
            )

        return outs_mb
