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
"""Periodic boundary condition (PBC) conservative virial, stress, and strain operations."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch


def apply_cell_strain(
    positions: torch.Tensor,
    cell: torch.Tensor,
    strain: torch.Tensor,
    *,
    batch: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    edge_shifts: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""Apply affine deformation :math:`\mathbf{F} = \mathbf{I} + \boldsymbol{\varepsilon}` to positions, cell, and PBC edges.

    Transforms atomic coordinates and lattice row vectors via:
    .. math::

        \mathbf{r}'_i = \mathbf{r}_i \mathbf{F}^\top, \qquad
        \mathbf{h}' = \mathbf{h} \mathbf{F}^\top

    When ``edge_index`` (and optional integer periodic image shifts ``edge_shifts``) are provided,
    also computes the strained periodic displacement vectors
    :math:`\mathbf{r}'_{ij} = \mathbf{r}'_j - \mathbf{r}'_i + \mathbf{S}_{ij} \mathbf{h}'`.

    Args:
        positions: Atomic coordinates of shape ``(N, 3)``.
        cell: Lattice matrix of shape ``(3, 3)`` (single system) or ``(B, 3, 3)`` (batched),
            with lattice vectors stored as rows.
        strain: Deformation/strain tensor :math:`\boldsymbol{\varepsilon}` of shape ``(3, 3)``
            or ``(B, 3, 3)``.
        batch: Optional graph assignment index tensor of shape ``(N,)`` for batched ``cell``
            of shape ``(B, 3, 3)``.
        edge_index: Optional directed graph edge indices of shape ``(2, E)`` (``[src, dst]``)
            or ``(E, 2)``.
        edge_shifts: Optional integer lattice shift vectors :math:`\mathbf{S}_{ij}` of shape
            ``(E, 3)`` for periodic boundary conditions.

    Returns:
        Tuple ``(strained_positions, strained_cell, edge_vectors)`` where ``edge_vectors`` has
        shape ``(E, 3)`` if ``edge_index`` is provided, otherwise ``None``.
    """
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(
            f"Expected positions of shape (N, 3), got {tuple(positions.shape)}."
        )

    eye = torch.eye(3, dtype=positions.dtype, device=positions.device)

    if cell.ndim == 2:
        if cell.shape != (3, 3):
            raise ValueError(f"Expected cell of shape (3, 3), got {tuple(cell.shape)}.")
        strain_2d = (
            strain.squeeze(0) if (strain.ndim == 3 and strain.shape[0] == 1) else strain
        )
        if strain_2d.shape != (3, 3):
            raise ValueError(
                f"Expected strain of shape (3, 3) for single cell, got {tuple(strain.shape)}."
            )
        deform_t = (eye + strain_2d).transpose(-1, -2)
        strained_positions = positions @ deform_t
        strained_cell = cell @ deform_t
        is_batched = False
    elif cell.ndim == 3:
        num_graphs = cell.shape[0]
        if cell.shape[1:] != (3, 3):
            raise ValueError(
                f"Expected batched cell of shape (B, 3, 3), got {tuple(cell.shape)}."
            )
        if batch is None:
            if num_graphs == 1:
                batch = torch.zeros(
                    positions.shape[0], dtype=torch.long, device=positions.device
                )
            else:
                raise ValueError(
                    "batch index tensor of shape (N,) is required when cell has shape (B, 3, 3) with B > 1."
                )
        strain_3d = (
            strain.unsqueeze(0).expand(num_graphs, 3, 3) if strain.ndim == 2 else strain
        )
        if strain_3d.shape != (num_graphs, 3, 3):
            raise ValueError(
                f"Expected strain of shape ({num_graphs}, 3, 3), got {tuple(strain.shape)}."
            )
        deform = eye.unsqueeze(0) + strain_3d
        deform_t = deform.transpose(-1, -2)
        strained_positions = torch.bmm(positions.unsqueeze(1), deform_t[batch]).squeeze(
            1
        )
        strained_cell = torch.bmm(cell, deform_t)
        is_batched = True
    else:
        raise ValueError(
            f"Expected cell of shape (3, 3) or (B, 3, 3), got {tuple(cell.shape)}."
        )

    if edge_index is None:
        return strained_positions, strained_cell, None

    if edge_index.ndim != 2:
        raise ValueError(
            f"Expected 2D edge_index of shape (2, E) or (E, 2), got {tuple(edge_index.shape)}."
        )
    if edge_index.shape[0] == 2:
        src, dst = edge_index[0], edge_index[1]
    elif edge_index.shape[1] == 2:
        src, dst = edge_index[:, 0], edge_index[:, 1]
    else:
        raise ValueError(
            f"Expected edge_index of shape (2, E) or (E, 2), got {tuple(edge_index.shape)}."
        )

    edge_vectors = strained_positions[dst] - strained_positions[src]

    if edge_shifts is not None:
        shifts = edge_shifts.to(dtype=strained_cell.dtype, device=strained_cell.device)
        if shifts.ndim != 2 or shifts.shape != (src.shape[0], 3):
            raise ValueError(
                f"Expected edge_shifts of shape ({src.shape[0]}, 3), got {tuple(shifts.shape)}."
            )
        if not is_batched:
            edge_vectors = edge_vectors + shifts @ strained_cell
        else:
            assert batch is not None
            edge_batch = batch[src]
            edge_vectors = edge_vectors + torch.bmm(
                shifts.unsqueeze(1), strained_cell[edge_batch]
            ).squeeze(1)

    return strained_positions, strained_cell, edge_vectors


def compute_forces_and_virial(
    energy: torch.Tensor,
    positions: torch.Tensor,
    *,
    strain: Optional[torch.Tensor] = None,
    edge_vectors: Optional[torch.Tensor] = None,
    cell: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    edge_batch: Optional[torch.Tensor] = None,
    create_graph: bool = False,
) -> dict[str, torch.Tensor]:
    r"""Compute conservative forces, virial, Cauchy stress, pressure, and shear stress.

    Supports both canonical periodic boundary condition (PBC) formulations:
    1. **Strain-derivative formulation** (``strain`` provided):
       :math:`\mathbf{F}_i = -\partial E / \partial \mathbf{r}_i`,
       :math:`W_{\alpha\beta} = -\partial E / \partial \varepsilon_{\alpha\beta}`.
    2. **Pairwise / edge-vector formulation** (``edge_vectors`` provided):
       :math:`\mathbf{f}_e = \partial E / \partial \mathbf{r}_e`,
       :math:`W_{\alpha\beta} = -\sum_e r_{e,\alpha} f_{e,\beta}` without deforming the cell in the
       forward graph.

    When ``cell`` is provided, also computes cell volume :math:`\Omega = |\det(\mathbf{h})|`,
    symmetrized Cauchy stress :math:`\boldsymbol{\sigma} = -\frac{1}{2\Omega}(\mathbf{W} + \mathbf{W}^\top)`,
    hydrostatic pressure :math:`P = -\frac{1}{3}\operatorname{Tr}(\boldsymbol{\sigma})` (``0e`` invariant),
    and deviatoric traceless shear stress
    :math:`\boldsymbol{\sigma}_{\text{dev}} = \boldsymbol{\sigma} - \frac{1}{3}\operatorname{Tr}(\boldsymbol{\sigma})\mathbf{I}`
    (``2e`` quadrupole).

    Args:
        energy: Total or per-graph potential energy tensor.
        positions: Atomic coordinates of shape ``(N, 3)`` with ``requires_grad=True``.
        strain: Optional strain tensor of shape ``(3, 3)`` or ``(B, 3, 3)`` with ``requires_grad=True``.
        edge_vectors: Optional edge displacement vectors :math:`\mathbf{r}_e` of shape ``(E, 3)`` used
            to compute ``energy``.
        cell: Optional lattice matrix of shape ``(3, 3)`` or ``(B, 3, 3)`` to normalize Cauchy stress
            by volume :math:`\Omega`.
        batch: Optional batch index tensor of shape ``(N,)`` (or ``(E,)`` for pairwise mode).
        mask: Optional boolean/float mask of shape ``(E,)`` (padded edges) or ``(N,)`` (padded atoms).
        edge_batch: Optional explicit edge-to-graph assignment tensor of shape ``(E,)`` for batched
            pairwise virial calculation.
        create_graph: Retain and build higher-order autograd graph for Force+Stress QAT training.

    Returns:
        Dictionary containing ``"forces"``, ``"virial"`` (plus ``"edge_forces"`` in pairwise mode,
        and ``"stress"``, ``"pressure"``, ``"shear_stress"``, ``"volume"`` when ``cell`` is provided).
    """
    total_energy = energy.sum()
    result: dict[str, torch.Tensor] = {}

    if strain is not None:
        grad_pos, grad_strain = torch.autograd.grad(
            outputs=[total_energy],
            inputs=[positions, strain],
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
        forces = -grad_pos if grad_pos is not None else torch.zeros_like(positions)
        virial = -grad_strain if grad_strain is not None else torch.zeros_like(strain)
        if mask is not None:
            if mask.shape[0] != positions.shape[0]:
                raise ValueError(
                    f"In strain mode, mask length ({mask.shape[0]}) must match positions ({positions.shape[0]})."
                )
            forces = forces * mask.to(dtype=forces.dtype).reshape(positions.shape[0], 1)
    elif edge_vectors is not None:
        if positions.requires_grad and positions is not edge_vectors:
            grad_pos, grad_edge = torch.autograd.grad(
                outputs=[total_energy],
                inputs=[positions, edge_vectors],
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )
            forces = -grad_pos if grad_pos is not None else torch.zeros_like(positions)
        else:
            (grad_edge,) = torch.autograd.grad(
                outputs=[total_energy],
                inputs=[edge_vectors],
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )
            forces = torch.zeros_like(positions)

        edge_forces = (
            grad_edge if grad_edge is not None else torch.zeros_like(edge_vectors)
        )
        r_eff = edge_vectors

        if mask is not None:
            if mask.shape[0] == edge_vectors.shape[0]:
                m_edge = mask.to(dtype=edge_forces.dtype).reshape(
                    edge_vectors.shape[0], 1
                )
                edge_forces = edge_forces * m_edge
                r_eff = edge_vectors * m_edge
            elif mask.shape[0] == positions.shape[0]:
                m_atom = mask.to(dtype=forces.dtype).reshape(positions.shape[0], 1)
                forces = forces * m_atom
            else:
                raise ValueError(
                    f"mask length ({mask.shape[0]}) must match either edge_vectors ({edge_vectors.shape[0]}) "
                    f"or positions ({positions.shape[0]})."
                )

        is_batched = (
            (cell is not None and cell.ndim == 3)
            or edge_batch is not None
            or (batch is not None)
        )
        if not is_batched:
            virial = -torch.einsum("ea,eb->ab", r_eff, edge_forces)
        else:
            if edge_batch is not None:
                e_batch = edge_batch.long()
            elif batch is not None and batch.shape[0] == edge_vectors.shape[0]:
                e_batch = batch.long()
            elif cell is not None and cell.ndim == 3 and cell.shape[0] == 1:
                e_batch = torch.zeros(
                    edge_vectors.shape[0], dtype=torch.long, device=edge_vectors.device
                )
            else:
                raise ValueError(
                    "For batched pairwise virial with B > 1, pass edge_batch of shape (E,) "
                    "or batch of shape (E,)."
                )
            if cell is not None and cell.ndim == 3:
                num_graphs = cell.shape[0]
            elif e_batch.numel() > 0:
                num_graphs = int(e_batch.max().item()) + 1
            else:
                num_graphs = 0
            outer = -(r_eff.unsqueeze(-1) * edge_forces.unsqueeze(-2))
            virial = edge_vectors.new_zeros((num_graphs, 3, 3)).index_add(
                0, e_batch, outer
            )

        result["edge_forces"] = edge_forces
    else:
        raise ValueError(
            "Either `strain` or `edge_vectors` must be provided to compute virial."
        )

    result["forces"] = forces
    result["virial"] = virial

    if cell is not None:
        if cell.ndim == 2 and virial.ndim == 3 and virial.shape[0] == 1:
            virial_matched = virial.squeeze(0)
        elif cell.ndim == 3 and virial.ndim == 2 and cell.shape[0] == 1:
            virial_matched = virial.unsqueeze(0)
        else:
            virial_matched = virial

        volume = torch.abs(torch.linalg.det(cell))
        vol_bcast = volume.unsqueeze(-1).unsqueeze(-1) if cell.ndim == 3 else volume
        raw_stress = -virial_matched / vol_bcast
        stress = 0.5 * (raw_stress + raw_stress.transpose(-1, -2))
        trace = torch.diagonal(stress, dim1=-2, dim2=-1).sum(dim=-1)
        pressure = -trace / 3.0
        eye = torch.eye(3, dtype=stress.dtype, device=stress.device)
        shear_stress = stress - (trace / 3.0).unsqueeze(-1).unsqueeze(-1) * eye

        result["stress"] = stress
        result["pressure"] = pressure
        result["shear_stress"] = shear_stress
        result["volume"] = volume

    return result


def decompose_stress_irreps(
    stress: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decompose a ``(*batch, 3, 3)`` stress or virial tensor into exact :math:`O(3)` irreps.

    Uses the orthonormal Clebsch-Gordan decomposition
    :math:`1o \otimes 1o = 0e \oplus 1e \oplus 2e` in the ``cuequivariance`` :math:`O(3)` basis:
    - ``isotropic_trace_0e`` (shape ``(*batch, 1)``): scalar trace :math:`\frac{1}{\sqrt{3}}\operatorname{Tr}(\mathbf{T})`.
    - ``antisymmetric_torque_1e`` (shape ``(*batch, 3)``): axial pseudovector torque
      :math:`\frac{1}{\sqrt{2}}(T_{yz}-T_{zy}, T_{zx}-T_{xz}, T_{xy}-T_{yx})`. For any
      :math:`O(3)`-invariant potential energy :math:`E`, conservation of angular momentum guarantees
      that this ``1e`` component vanishes identically.
    - ``symmetric_traceless_shear_2e`` (shape ``(*batch, 5)``): quadrupole deviatoric shear
      components transforming under :math:`D^{(2e)}(g)`.

    The mapping is an exact 9D orthogonal isometry:
    :math:`\|\mathbf{T}\|_F^2 = \|y_{0e}\|_2^2 + \|y_{1e}\|_2^2 + \|y_{2e}\|_2^2`.

    Args:
        stress: Rank-2 Cartesian tensor of shape ``(*batch, 3, 3)``.

    Returns:
        Tuple ``(isotropic_trace_0e, antisymmetric_torque_1e, symmetric_traceless_shear_2e)``
        with shapes ``((*batch, 1), (*batch, 3), (*batch, 5))``.
    """
    if stress.ndim < 2 or stress.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected stress tensor of shape (*batch, 3, 3), got {tuple(stress.shape)}."
        )

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    inv_sqrt6 = 1.0 / math.sqrt(6.0)

    s00 = stress[..., 0, 0]
    s01 = stress[..., 0, 1]
    s02 = stress[..., 0, 2]
    s10 = stress[..., 1, 0]
    s11 = stress[..., 1, 1]
    s12 = stress[..., 1, 2]
    s20 = stress[..., 2, 0]
    s21 = stress[..., 2, 1]
    s22 = stress[..., 2, 2]

    isotropic_trace_0e = ((s00 + s11 + s22) * inv_sqrt3).unsqueeze(-1)

    antisymmetric_torque_1e = torch.stack(
        [
            (s12 - s21) * inv_sqrt2,
            (s20 - s02) * inv_sqrt2,
            (s01 - s10) * inv_sqrt2,
        ],
        dim=-1,
    )

    symmetric_traceless_shear_2e = torch.stack(
        [
            (s02 + s20) * inv_sqrt2,
            (s01 + s10) * inv_sqrt2,
            (2.0 * s11 - s00 - s22) * inv_sqrt6,
            (s12 + s21) * inv_sqrt2,
            (s22 - s00) * inv_sqrt2,
        ],
        dim=-1,
    )

    return isotropic_trace_0e, antisymmetric_torque_1e, symmetric_traceless_shear_2e


class ConservativeVirialStress(torch.nn.Module):
    """Differentiable ``nn.Module`` wrapper computing conservative energy, forces, and PBC virial stress."""

    def __init__(
        self,
        energy_fn: Callable[..., torch.Tensor],
        *,
        mode: str = "strain",
    ) -> None:
        super().__init__()
        if mode not in ("strain", "edge_vectors"):
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected 'strain' or 'edge_vectors'."
            )
        if isinstance(energy_fn, torch.nn.Module):
            self.energy_fn = energy_fn
        else:
            self._callable_fn = energy_fn
            self.energy_fn = None  # type: ignore[assignment]
        self.mode = mode

    def _eval_energy(self, *args, **kwargs) -> torch.Tensor:
        if self.energy_fn is not None:
            return self.energy_fn(*args, **kwargs)
        return self._callable_fn(*args, **kwargs)

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shifts: Optional[torch.Tensor] = None,
        *,
        batch: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        create_graph: Optional[bool] = None,
        **model_kwargs,
    ) -> dict[str, torch.Tensor]:
        retain = self.training if create_graph is None else create_graph
        pos_req = (
            positions
            if positions.requires_grad
            else positions.detach().requires_grad_(True)
        )

        if self.mode == "strain":
            strain_shape = (3, 3) if cell.ndim == 2 else (cell.shape[0], 3, 3)
            strain = torch.zeros(
                strain_shape,
                dtype=positions.dtype,
                device=positions.device,
                requires_grad=True,
            )
            strained_pos, strained_cell, edge_vecs = apply_cell_strain(
                pos_req,
                cell,
                strain,
                batch=batch,
                edge_index=edge_index,
                edge_shifts=edge_shifts,
            )
            energy = self._eval_energy(
                strained_pos,
                strained_cell,
                edge_vecs,
                edge_index=edge_index,
                batch=batch,
                **model_kwargs,
            )
            out = compute_forces_and_virial(
                energy,
                pos_req,
                strain=strain,
                cell=cell,
                batch=batch,
                mask=mask,
                create_graph=retain,
            )
        else:
            src, dst = (
                (edge_index[0], edge_index[1])
                if edge_index.shape[0] == 2
                else (edge_index[:, 0], edge_index[:, 1])
            )
            edge_vecs = pos_req[dst] - pos_req[src]
            if edge_shifts is not None:
                shifts = edge_shifts.to(dtype=cell.dtype, device=cell.device)
                if cell.ndim == 2:
                    edge_vecs = edge_vecs + shifts @ cell
                else:
                    e_batch = (
                        batch[src]
                        if batch is not None
                        else torch.zeros_like(src, dtype=torch.long)
                    )
                    edge_vecs = edge_vecs + torch.bmm(
                        shifts.unsqueeze(1), cell[e_batch]
                    ).squeeze(1)
            e_batch_opt = batch[src] if (batch is not None and cell.ndim == 3) else None
            energy = self._eval_energy(
                pos_req,
                cell,
                edge_vecs,
                edge_index=edge_index,
                batch=batch,
                **model_kwargs,
            )
            out = compute_forces_and_virial(
                energy,
                pos_req,
                edge_vectors=edge_vecs,
                cell=cell,
                batch=batch,
                edge_batch=e_batch_opt,
                mask=mask,
                create_graph=retain,
            )

        out["energy"] = energy
        return out


__all__ = [
    "apply_cell_strain",
    "compute_forces_and_virial",
    "decompose_stress_irreps",
    "ConservativeVirialStress",
]
