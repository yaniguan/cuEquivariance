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
import numpy as np
import pytest
from cuequivariance.group_theory.experimental.mace import symmetric_contraction
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction_ir_dict,
)

import cuequivariance as cue


def test_symmetric_contraction():
    e, p = symmetric_contraction(
        cue.Irreps("O3", "32x0e + 32x1o"),
        cue.Irreps("O3", "32x0e + 32x1o"),
        degrees=[1, 2, 3],
    )
    assert e.all_same_segment_shape()
    assert np.linalg.matrix_rank(p) == p.shape[1]


@pytest.mark.parametrize(
    "irreps_in_str, irreps_out_str, expected_gcd",
    [
        ("64x0e + 32x1o + 16x2e", "64x0e + 32x1o", 16),
        ("4x0e + 2x1o", "2x0e + 4x1o", 2),
        ("3x0e + 2x1o", "2x0e + 1x1o", 1),
    ],
)
def test_tapered_mace_symmetric_contraction(
    irreps_in_str: str, irreps_out_str: str, expected_gcd: int
):
    irreps_in = cue.Irreps("O3", irreps_in_str)
    irreps_out = cue.Irreps("O3", irreps_out_str)

    e_std = cue.descriptors.symmetric_contraction(irreps_in, irreps_out, (0, 1, 2))
    assert e_std.all_same_segment_shape()
    expected_seg_shape = () if expected_gcd == 1 else (expected_gcd,)
    for ope in e_std.polynomial.operands:
        assert ope.segment_shape == expected_seg_shape

    ir_dict_std = cue.descriptors.symmetric_contraction_ir_dict(
        irreps_in, irreps_out, (0, 1, 2)
    )
    assert (
        ir_dict_std.polynomial
        == e_std.split_operand_by_irrep(1).split_operand_by_irrep(-1).polynomial
    )

    e_mace, p = symmetric_contraction(irreps_in, irreps_out, degrees=[1, 2, 3])
    assert e_mace.all_same_segment_shape()
    assert np.linalg.matrix_rank(p) == p.shape[1]

    ir_dict_mace, p_ir_dict = symmetric_contraction_ir_dict(
        irreps_in, irreps_out, degrees=[1, 2, 3]
    )
    assert (
        ir_dict_mace.polynomial
        == e_mace.split_operand_by_irrep(1).split_operand_by_irrep(-1).polynomial
    )
    np.testing.assert_array_equal(p, p_ir_dict)
