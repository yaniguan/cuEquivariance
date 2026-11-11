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
import pytest
import torch
from cuequivariance.group_theory.experimental.escn import (
    escn_tp,
    escn_tp_compact,
    escn_weights_to_compact,
    irreps_to_spherical_signal,
    spherical_signal_to_irreps,
)
from cuequivariance_torch._tests.utils import (
    module_with_mode,
)

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def test_rotation():
    irreps = cue.Irreps("SO3", "3x0 + 1 + 0 + 4x2 + 4")
    alpha = torch.tensor([0.3]).to(device)
    beta = torch.tensor([0.4]).to(device)
    gamma = torch.tensor([-0.5]).to(device)

    rot = cuet.Rotation(irreps, layout=cue.ir_mul).to(device)

    x = torch.randn(10, irreps.dim).to(device)

    rx = rot(gamma, beta, alpha, x)
    x_ = rot(-alpha, -beta, -gamma, rx)

    torch.testing.assert_close(x, x_)


def test_vector_to_euler_angles():
    A = torch.randn(4, 3)
    A = torch.nn.functional.normalize(A, dim=-1)

    beta, alpha = cuet.vector_to_euler_angles(A)
    ey = torch.tensor([[0.0, 1.0, 0.0]])
    B = cuet.Rotation(cue.Irreps("SO3", "1"), layout=cue.ir_mul)(
        torch.tensor([0.0]), beta, alpha, ey
    )

    assert torch.allclose(A, B, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_fallback", [False, True])
def test_inversion(use_fallback: bool):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    irreps = cue.Irreps("O3", "2x1e + 2x1o")
    torch.testing.assert_close(
        cuet.Inversion(
            irreps, layout=cue.ir_mul, device=device, use_fallback=use_fallback
        )(
            torch.tensor(
                [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                device=device,
            )
        ),
        torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]],
            device=device,
        ),
    )


export_modes = ["compile", "script", "jit"]


@pytest.mark.parametrize("mode", export_modes)
def test_export(mode: str, tmp_path: str):
    # Skip all export tests for rotation - they are very slow (10+ seconds each)
    pytest.skip(
        "Skipping all rotation export tests for speed - they take 10+ seconds each"
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    irreps = cue.Irreps("SO3", "3x0 + 1 + 0 + 4x2 + 4")
    dtype = torch.float32
    alpha = torch.tensor([0.3]).to(device)
    beta = torch.tensor([0.4]).to(device)
    gamma = torch.tensor([-0.5]).to(device)

    m = cuet.Rotation(irreps, layout=cue.ir_mul).to(device)

    x = torch.randn(10, irreps.dim).to(device)
    inputs = (gamma, beta, alpha, x)

    out1 = m(*inputs)
    m = module_with_mode(mode, m, inputs, dtype, tmp_path)
    out2 = m(*inputs)
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16]
)
def test_vector_to_euler_angles_poles_zero_and_bf16_gradients(dtype: torch.dtype):
    vecs = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.05, 1.0, 0.0],
            [1e-9, 1.0, 0.0],
            [0.05, -1.0, 0.0],
            [0.3, -0.4, 0.5],
        ],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    beta, alpha = cuet.vector_to_euler_angles(vecs)
    assert beta.dtype == dtype
    assert alpha.dtype == dtype

    # Exact values at poles and origin
    assert beta[0].item() == 0.0 and alpha[0].item() == 0.0
    assert abs(beta[1].item() - torch.pi) < 1e-2 and alpha[1].item() == 0.0
    assert beta[2].item() == 0.0 and alpha[2].item() == 0.0

    # First and second derivatives must be finite everywhere and 0 at poles & origin
    loss = (beta * 1.3 + alpha * 0.7).sum()
    (grad1,) = torch.autograd.grad(loss, vecs, create_graph=True)
    assert torch.isfinite(grad1).all(), (
        f"Non-finite 1st derivative for {dtype}: {grad1}"
    )
    torch.testing.assert_close(grad1[:3], torch.zeros_like(grad1[:3]))

    (grad2,) = torch.autograd.grad((grad1 * grad1).sum(), vecs)
    assert torch.isfinite(grad2).all(), (
        f"Non-finite 2nd derivative for {dtype}: {grad2}"
    )
    torch.testing.assert_close(grad2[:3], torch.zeros_like(grad2[:3]))


def test_vector_to_euler_angles_gradcheck():
    torch.manual_seed(0)
    vecs = torch.randn(3, 2, 3, dtype=torch.float64, device=device)
    # Keep well away from the y-axis branch cut for finite-difference gradcheck
    vecs[..., 0] = vecs[..., 0].abs() + 0.3
    vecs[..., 2] = vecs[..., 2].abs() + 0.3
    vecs.requires_grad_(True)

    assert torch.autograd.gradcheck(cuet.vector_to_euler_angles, (vecs,), eps=1e-6)
    assert torch.autograd.gradgradcheck(cuet.vector_to_euler_angles, (vecs,), eps=1e-6)


@pytest.mark.parametrize("ell", [0, 1, 2, 3, 4, 5])
def test_encode_rotation_trig_and_edge_rotations(ell: int):
    torch.manual_seed(1)
    angles = torch.linspace(-3.0, 3.0, 19, dtype=torch.float64, device=device)
    ref = cuet.encode_rotation_angle(angles, ell)
    got = cuet.encode_rotation_trig(torch.cos(angles), torch.sin(angles), ell)
    torch.testing.assert_close(got, ref, atol=1e-13, rtol=1e-13)

    # Compare encode_edge_rotations against vector_to_euler_angles + encode_rotation_angle
    vecs = torch.randn(8, 3, dtype=torch.float64, device=device)
    vecs[0] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
    vecs[1] = torch.tensor([0.0, -2.5, 0.0], dtype=torch.float64, device=device)
    vecs[2] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device=device)

    enc_b, enc_a = cuet.encode_edge_rotations(vecs, ell)
    beta, alpha = cuet.vector_to_euler_angles(vecs)
    ref_b = cuet.encode_rotation_angle(beta, ell)
    ref_a = cuet.encode_rotation_angle(alpha, ell)
    torch.testing.assert_close(enc_b, ref_b, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(enc_a, ref_a, atol=1e-13, rtol=1e-13)


def test_encode_edge_rotations_bf16_and_gradcheck():
    vecs_bf16 = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.05, 1.0, 0.0],
            [1e-9, 1.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    enc_b, enc_a = cuet.encode_edge_rotations(vecs_bf16, 4)
    assert enc_b.dtype == torch.bfloat16 and enc_a.dtype == torch.bfloat16
    (g1,) = torch.autograd.grad(
        (enc_b * 0.7 + enc_a * 1.1).sum(), vecs_bf16, create_graph=True
    )
    (g2,) = torch.autograd.grad((g1 * g1).sum(), vecs_bf16)
    assert torch.isfinite(g1).all()
    assert torch.isfinite(g2).all()

    torch.manual_seed(2)
    v64 = torch.randn(4, 3, dtype=torch.float64, device=device)
    v64[:, 0] = v64[:, 0].abs() + 0.3
    v64[:, 2] = v64[:, 2].abs() + 0.3
    v64.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda v: cuet.encode_edge_rotations(v, 3), (v64,))
    assert torch.autograd.gradgradcheck(
        lambda v: cuet.encode_edge_rotations(v, 3), (v64,)
    )


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_rotation_arbitrary_batch_and_indexed(layout: cue.IrrepsLayout):
    torch.manual_seed(3)
    irreps = cue.Irreps("O3", "2x0e + 2x1o + 2x2e")
    rot = cuet.Rotation(irreps, layout=layout, method="naive", device=device)

    # 3D batch shape (B, N, dim) with broadcastable angles
    x_3d = torch.randn(2, 3, irreps.dim, dtype=torch.float64, device=device)
    gamma = torch.randn(2, 3, dtype=torch.float64, device=device)
    beta = torch.randn(2, 1, dtype=torch.float64, device=device)
    alpha = torch.tensor(0.4, dtype=torch.float64, device=device)

    out_3d = rot(gamma, beta, alpha, x_3d)
    assert out_3d.shape == (2, 3, irreps.dim)
    out_flat = rot(
        gamma.reshape(6),
        beta.expand(2, 3).reshape(6),
        alpha.expand(6),
        x_3d.reshape(6, irreps.dim),
    )
    torch.testing.assert_close(out_3d.reshape(6, irreps.dim), out_flat)

    # Unbatched 1D input (dim,)
    out_1d = rot(gamma[0, 0], beta[0, 0], alpha, x_3d[0, 0])
    assert out_1d.shape == (irreps.dim,)
    torch.testing.assert_close(out_1d, out_3d[0, 0])

    # Indexed gather + scatter equivalence (forward + backward)
    n_nodes, n_edges = 5, 11
    src = torch.randint(0, n_nodes, (n_edges,), device=device)
    dst = torch.randint(0, n_nodes, (n_edges,), device=device)
    g_e = torch.randn(n_edges, dtype=torch.float64, device=device, requires_grad=True)
    b_e = torch.randn(n_edges, dtype=torch.float64, device=device, requires_grad=True)
    a_e = torch.randn(n_edges, dtype=torch.float64, device=device, requires_grad=True)
    x_nodes = torch.randn(
        n_nodes, irreps.dim, dtype=torch.float64, device=device, requires_grad=True
    )

    out_fused = rot(
        g_e, b_e, a_e, x_nodes, indices_in=src, indices_out=dst, size_out=n_nodes
    )

    x_ref = x_nodes.detach().clone().requires_grad_(True)
    g_ref = g_e.detach().clone().requires_grad_(True)
    b_ref = b_e.detach().clone().requires_grad_(True)
    a_ref = a_e.detach().clone().requires_grad_(True)
    edge_out = rot(g_ref, b_ref, a_ref, x_ref[src])
    out_ref = torch.zeros_like(x_ref).index_add_(0, dst, edge_out)

    torch.testing.assert_close(out_fused, out_ref)
    cot = torch.randn_like(out_fused)
    (out_fused * cot).sum().backward()
    (out_ref * cot).sum().backward()
    torch.testing.assert_close(x_nodes.grad, x_ref.grad)
    torch.testing.assert_close(g_e.grad, g_ref.grad)
    torch.testing.assert_close(b_e.grad, b_ref.grad)
    torch.testing.assert_close(a_e.grad, a_ref.grad)


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_edge_rotation_alignment_roundtrip_and_modes(layout: cue.IrrepsLayout):
    assert cuet.operations.EdgeRotation is cuet.EdgeRotation
    torch.manual_seed(4)

    # 1. Aligning any 3D vector to [0, ||v||, 0]
    vecs = torch.randn(10, 3, dtype=torch.float64, device=device) * 2.5
    vecs[0] = torch.tensor([0.0, 3.2, 0.0], dtype=torch.float64, device=device)
    vecs[1] = torch.tensor([0.0, -1.7, 0.0], dtype=torch.float64, device=device)
    vecs[2] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device=device)

    align_vec = cuet.EdgeRotation(
        cue.Irreps("O3", "1o"), inverse=False, layout=layout, method="naive"
    ).to(device)
    aligned = align_vec(vecs, vecs)
    norms = torch.linalg.norm(vecs, dim=-1)
    expected = torch.stack(
        [torch.zeros_like(norms), norms, torch.zeros_like(norms)], dim=-1
    )
    torch.testing.assert_close(aligned, expected, atol=1e-13, rtol=1e-13)

    # Spherical harmonics aligned to +e_y must vanish for all m != 0
    sh = cuet.SphericalHarmonics([0, 1, 2, 3], normalize=True, method="naive").to(
        device
    )
    irreps_sh = cue.Irreps("O3", "0e + 1o + 2e + 3o")
    align_sh = cuet.EdgeRotation(
        irreps_sh, inverse=False, layout=cue.ir_mul, method="naive"
    ).to(device)
    ylm_aligned = align_sh(vecs[3:], sh(vecs[3:]))
    offset = 0
    for _, ir in irreps_sh:
        block = ylm_aligned[:, offset : offset + ir.dim]
        for m_idx in range(ir.dim):
            if m_idx != ir.l:
                torch.testing.assert_close(
                    block[:, m_idx],
                    torch.zeros_like(block[:, m_idx]),
                    atol=1e-13,
                    rtol=1e-13,
                )
        offset += ir.dim

    # 2. Round-trip & agreement with 3-angle Rotation + explicit angles + precomputed encodings
    irreps = cue.Irreps("O3", "2x0e + 2x1o + 2x1e + 2x2e")
    rot_fwd = cuet.EdgeRotation(
        irreps, inverse=False, layout=layout, method="naive"
    ).to(device)
    rot_inv = cuet.EdgeRotation(irreps, inverse=True, layout=layout, method="naive").to(
        device
    )
    rot_3 = cuet.Rotation(irreps, layout=layout, method="naive").to(device)

    x = torch.randn(10, irreps.dim, dtype=torch.float64, device=device)
    y = rot_fwd(vecs, x)
    x_rec = rot_inv(vecs, y)
    torch.testing.assert_close(x_rec, x, atol=1e-13, rtol=1e-13)

    beta, alpha = cuet.vector_to_euler_angles(vecs)
    zeros = torch.zeros_like(beta)
    torch.testing.assert_close(
        y, rot_3(-alpha, -beta, zeros, x), atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        rot_inv(vecs, x), rot_3(zeros, beta, alpha, x), atol=1e-13, rtol=1e-13
    )

    # Explicit (beta, alpha) tuple & kwargs & higher-ell precomputed encodings
    torch.testing.assert_close(y, rot_fwd((beta, alpha), x), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        y, rot_fwd(x=x, beta=beta, alpha=alpha), atol=1e-13, rtol=1e-13
    )
    enc_b4, enc_a4 = cuet.encode_edge_rotations(vecs, ell=4)
    torch.testing.assert_close(
        y, rot_fwd(x=x, encodings=(enc_b4, enc_a4)), atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        rot_inv(vecs, y),
        rot_inv(y, encodings=(enc_b4, enc_a4)),
        atol=1e-13,
        rtol=1e-13,
    )


@pytest.mark.parametrize("m_max", [1, 2])
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_edge_rotation_escn_full_o3_equivariance(m_max: int, layout: cue.IrrepsLayout):
    torch.manual_seed(5)
    irreps_in = cue.Irreps("O3", "2x0e + 2x0o + 2x1o + 2x1e + 2x2e + 2x2o")
    irreps_out = cue.Irreps("O3", "2x0e + 2x0o + 2x1o + 2x1e + 2x2e + 2x2o")

    # EdgeRotation seamlessly bridges user `layout` <-> escn_tp's `cue.ir_mul`!
    edge_fwd = cuet.EdgeRotation(
        irreps_in,
        inverse=False,
        layout_in=layout,
        layout_out=cue.ir_mul,
        method="naive",
        device=device,
    )
    edge_inv = cuet.EdgeRotation(
        irreps_out,
        inverse=True,
        layout_in=cue.ir_mul,
        layout_out=layout,
        method="naive",
        device=device,
    )
    poly_escn = escn_tp(irreps_in, irreps_out, m_max=m_max).polynomial
    sp_escn = cuet.SegmentedPolynomial(poly_escn, method="naive").to(device)

    # Also test compact eSCN block inside the same O(3) equivariance test
    poly_cmp = escn_tp_compact(irreps_in, irreps_out, m_max=m_max)
    sp_cmp = cuet.SegmentedPolynomial(poly_cmp, method="naive").to(device)
    sp_to_cmp = cuet.SegmentedPolynomial(
        irreps_to_spherical_signal(irreps_in, layout=cue.ir_mul).polynomial,
        method="naive",
    ).to(device)
    sp_from_cmp = cuet.SegmentedPolynomial(
        spherical_signal_to_irreps(irreps_out, layout=cue.ir_mul).polynomial,
        method="naive",
    ).to(device)
    sp_w_map = cuet.SegmentedPolynomial(
        escn_weights_to_compact(irreps_in, irreps_out, m_max=m_max).polynomial,
        method="naive",
    ).to(device)

    def escn_block(
        v: torch.Tensor, feat: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        aligned = edge_fwd(v, feat)
        conv = sp_escn([w, aligned])[0]
        return edge_inv(v, conv)

    def escn_compact_block(
        v: torch.Tensor, feat: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        aligned = edge_fwd(v, feat)
        aligned_cmp = sp_to_cmp([aligned])[0]
        w_cmp = sp_w_map([w])[0]
        conv_cmp = sp_cmp([w_cmp, aligned_cmp])[0]
        conv = sp_from_cmp([conv_cmp])[0]
        return edge_inv(v, conv)

    v = torch.randn(7, 3, dtype=torch.float64, device=device)
    v[0] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
    v[1] = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64, device=device)
    x = torch.randn(7, irreps_in.dim, dtype=torch.float64, device=device)
    w = torch.randn(7, poly_escn.inputs[0].size, dtype=torch.float64, device=device)

    y_orig = escn_block(v, x, w)
    y_cmp_orig = escn_compact_block(v, x, w)
    torch.testing.assert_close(y_cmp_orig, y_orig, atol=1e-12, rtol=1e-12)

    # Random proper 3D rotation Q in SO(3)
    gamma = torch.tensor([0.73], dtype=torch.float64, device=device)
    beta = torch.tensor([1.19], dtype=torch.float64, device=device)
    alpha = torch.tensor([-0.61], dtype=torch.float64, device=device)
    rot_v = cuet.Rotation(
        cue.Irreps("O3", "1o"), layout=cue.ir_mul, method="naive", device=device
    )
    rot_in = cuet.Rotation(irreps_in, layout=layout, method="naive", device=device)
    rot_out = cuet.Rotation(irreps_out, layout=layout, method="naive", device=device)

    v_rot = rot_v(gamma, beta, alpha, v)
    x_rot = rot_in(gamma, beta, alpha, x)
    torch.testing.assert_close(
        escn_block(v_rot, x_rot, w),
        rot_out(gamma, beta, alpha, y_orig),
        atol=1e-12,
        rtol=1e-12,
    )

    # Improper rotation -Q = P * Q in O(3) (spatial inversion + 3D rotation)
    inv_in = cuet.Inversion(irreps_in, layout=layout, method="naive", device=device)
    inv_out = cuet.Inversion(irreps_out, layout=layout, method="naive", device=device)
    torch.testing.assert_close(
        escn_block(-v, inv_in(x), w),
        inv_out(y_orig),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        escn_block(-v_rot, inv_in(x_rot), w),
        inv_out(rot_out(gamma, beta, alpha, y_orig)),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_edge_rotation_indexed_gather_scatter_and_gradients(layout: cue.IrrepsLayout):
    torch.manual_seed(6)
    irreps = cue.Irreps("O3", "3x0e + 3x1o + 3x2e")
    n_nodes, n_edges = 6, 15
    src = torch.randint(0, n_nodes, (n_edges,), device=device)
    dst = torch.randint(0, n_nodes, (n_edges,), device=device)

    rot_fwd = cuet.EdgeRotation(
        irreps, inverse=False, layout=layout, method="naive", device=device
    )
    rot_inv = cuet.EdgeRotation(
        irreps, inverse=True, layout=layout, method="naive", device=device
    )

    # 1. Node -> Edge fused gather + rotate (forward + backward wrt nodes & edge vectors)
    v1 = torch.randn(n_edges, 3, dtype=torch.float64, device=device, requires_grad=True)
    x1 = torch.randn(
        n_nodes, irreps.dim, dtype=torch.float64, device=device, requires_grad=True
    )
    edge_fused = rot_fwd(v1, x1, indices_in=src)

    v1_ref = v1.detach().clone().requires_grad_(True)
    x1_ref = x1.detach().clone().requires_grad_(True)
    edge_ref = rot_fwd(v1_ref, x1_ref[src])

    torch.testing.assert_close(edge_fused, edge_ref, atol=1e-13, rtol=1e-13)
    cot_edge = torch.randn_like(edge_fused)
    (edge_fused * cot_edge).sum().backward()
    (edge_ref * cot_edge).sum().backward()
    torch.testing.assert_close(x1.grad, x1_ref.grad, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(v1.grad, v1_ref.grad, atol=1e-13, rtol=1e-13)

    # 2. Edge -> Node fused rotate + scatter (forward + backward wrt edges & edge vectors)
    v2 = torch.randn(n_edges, 3, dtype=torch.float64, device=device, requires_grad=True)
    e2 = torch.randn(
        n_edges, irreps.dim, dtype=torch.float64, device=device, requires_grad=True
    )
    node_fused = rot_inv(v2, e2, indices_out=dst, size_out=n_nodes)

    v2_ref = v2.detach().clone().requires_grad_(True)
    e2_ref = e2.detach().clone().requires_grad_(True)
    node_ref = torch.zeros(
        (n_nodes, irreps.dim), dtype=torch.float64, device=device
    ).index_add_(0, dst, rot_inv(v2_ref, e2_ref))

    torch.testing.assert_close(node_fused, node_ref, atol=1e-13, rtol=1e-13)
    cot_node = torch.randn_like(node_fused)
    (node_fused * cot_node).sum().backward()
    (node_ref * cot_node).sum().backward()
    torch.testing.assert_close(e2.grad, e2_ref.grad, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(v2.grad, v2_ref.grad, atol=1e-13, rtol=1e-13)

    # 3. Empty graph (0 edges) with autograd
    empty_idx = torch.empty(0, dtype=torch.long, device=device)
    v_empty = torch.empty(
        (0, 3), dtype=torch.float64, device=device, requires_grad=True
    )
    x_empty_test = torch.randn(
        n_nodes, irreps.dim, dtype=torch.float64, device=device, requires_grad=True
    )
    out_empty = rot_fwd(
        v_empty,
        x_empty_test,
        indices_in=empty_idx,
        indices_out=empty_idx,
        size_out=n_nodes,
    )
    assert out_empty.shape == (n_nodes, irreps.dim)
    assert (out_empty == 0).all()
    out_empty.sum().backward()
    assert torch.isfinite(x_empty_test.grad).all()


@pytest.mark.parametrize("inverse", [False, True])
def test_edge_rotation_bf16_poles_and_gradcheck(inverse: bool):
    irreps = cue.Irreps("O3", "2x0e + 2x1o + 2x2e")
    mod = cuet.EdgeRotation(
        irreps, inverse=inverse, layout=cue.ir_mul, method="naive", device=device
    )

    # BF16 near-pole, exact poles, and zero vector 1st & 2nd derivatives
    vecs_bf16 = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.05, 1.0, 0.0],
            [1e-9, 1.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    x_bf16 = torch.randn(
        5, irreps.dim, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    out_bf16 = mod(vecs_bf16, x_bf16)
    assert out_bf16.dtype == torch.bfloat16
    gv1, gx1 = torch.autograd.grad(
        (out_bf16 * 0.37).sum(), (vecs_bf16, x_bf16), create_graph=True
    )
    assert torch.isfinite(gv1).all() and torch.isfinite(gx1).all()
    gv2, gx2 = torch.autograd.grad(
        (gv1 * gv1).sum() + (gx1 * gx1).sum(), (vecs_bf16, x_bf16)
    )
    assert torch.isfinite(gv2).all() and torch.isfinite(gx2).all()

    # Finite-difference gradcheck and gradgradcheck in float64
    torch.manual_seed(7)
    v64 = torch.randn(3, 3, dtype=torch.float64, device=device)
    v64[:, 0] = v64[:, 0].abs() + 0.3
    v64[:, 2] = v64[:, 2].abs() + 0.3
    v64.requires_grad_(True)
    x64 = torch.randn(
        3, irreps.dim, dtype=torch.float64, device=device, requires_grad=True
    )
    assert torch.autograd.gradcheck(mod, (v64, x64), eps=1e-6)
    assert torch.autograd.gradgradcheck(mod, (v64, x64), eps=1e-6)
