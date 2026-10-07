"""SIGReg, the sketched isotropic-Gaussian regulariser of LeJEPA, in torch.
The main idea is that the embeddings are projected on random unit directions and each projection's empirical characteristic function is compared with that of N(0, 1)
through the Epps-Pulley statistic."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

SIGREG_KNOTS = 17
SIGREG_T_MAX = 3.0


def _quadrature(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the trapezoid rule on [0, 3] and the Gaussian characteristic function at its knots.

    Inputs: target device and dtype.
    Outputs: knots (K,), phi (K,) and weights (K,) tensors.
    """
    knots = torch.linspace(0.0, SIGREG_T_MAX, SIGREG_KNOTS, device=device, dtype=dtype)
    step = SIGREG_T_MAX / (SIGREG_KNOTS - 1)
    weights = torch.full((SIGREG_KNOTS,), 2.0 * step, device=device, dtype=dtype)
    weights[0] = step
    weights[-1] = step
    phi = torch.exp(-knots.square() / 2.0)
    return knots, phi, weights * phi


def sigreg(
    embeddings: torch.Tensor,
    *,
    projections: int = 256,
    generator: torch.Generator | None = None,
    chunk: int = 32,
) -> torch.Tensor:
    """Compute the mean integrated squared characteristic-function error of the rows against N(0, 1) over random sketches.

    Inputs: embeddings (rows, dim); number of random projections; optional generator for the directions; chunk of projections per checkpointed block.
    Outputs: a scalar tensor, exactly zero for zero rows.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (rows, dim)")
    if projections < 1:
        raise ValueError("projections must be positive")
    rows, dim = embeddings.shape
    if rows == 0:
        return embeddings.new_zeros(())
    directions = torch.randn(dim, projections, device=embeddings.device, dtype=embeddings.dtype, generator=generator)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    knots, phi, weights = _quadrature(embeddings.device, embeddings.dtype)

    def sketch(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Sum the integrated squared characteristic-function error over one block of directions.

        Inputs: embeddings (rows, dim); directions (dim, m).
        Outputs: scalar sum over the m sketches.
        """
        projected = (x @ d).unsqueeze(-1) * knots
        error = (torch.cos(projected).mean(dim=0) - phi).square() + torch.sin(projected).mean(dim=0).square()
        return (error * weights).sum(dim=-1).sum()

    total = embeddings.new_zeros(())
    for start in range(0, projections, max(int(chunk), 1)):
        block = directions[:, start : start + chunk]
        if torch.is_grad_enabled() and embeddings.requires_grad:
            total = total + checkpoint(sketch, embeddings, block, use_reentrant=False)
        else:
            total = total + sketch(embeddings, block)
    return total / projections


__all__ = ["SIGREG_KNOTS", "SIGREG_T_MAX", "sigreg"]
