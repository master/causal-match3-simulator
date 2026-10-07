"""The five adjustment contexts ("arms"). Each maps the decision-time base context B_i (tier one-hot, level one-hot, mastery entering the target
attempt) plus arm-specific information to a context tensor (N, S, width) consumed by the policy and the retention head, and returns a per-row
regularisation penalty (beta * KL for the variational Z, lambda * ||Z||^2 for the deterministic Z, 0 otherwise).

Naive: B_i.  Handcrafted: B_i + standardised history features.  Deterministic Z: B_i + point encoding (4).  Variational Z (CaLA): B_i + S posterior
draws, optionally supplied as standard-normal ``eps`` so a player's draws are fixed across the e grid and reused by every estimator.
Oracle K: B_i + the simulator's true skill. The module refuses to run without an active ``EvaluationOnly`` token.
"""

from __future__ import annotations

from enum import Enum

import torch
from torch import nn

from match3_simulator.experiments.data_base import BASE_CONTEXT_WIDTH
from match3_simulator.experiments.encoder_base import PrefixEncoder, PrefixEncoderConfig
from match3_simulator.experiments.guard import GuardViolation, _oracle_token


class Arm(str, Enum):
    NAIVE = "naive"
    HANDCRAFTED = "handcrafted"
    DET_Z = "det_z"
    VAR_Z = "var_z"
    ORACLE_K = "oracle_k"

    @property
    def deployable(self) -> bool:
        return self is not Arm.ORACLE_K

    @property
    def uses_encoder(self) -> bool:
        return self in (Arm.DET_Z, Arm.VAR_Z)


class ContextModule(nn.Module):
    """Base class. ``forward(prefix, base, extra, *, S, generator, eps) -> (context (N, S, width), penalty (N,), info)``."""

    arm: Arm
    width: int
    latent_dimensions: int = 0

    def forward(self, *, prefix: dict[str, torch.Tensor] | None, base: torch.Tensor, extra: torch.Tensor | None = None, S: int = 1, generator: torch.Generator | None = None,
                eps: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def _check_base(self, base: torch.Tensor) -> None:
        if base.ndim != 2 or base.shape[1] != BASE_CONTEXT_WIDTH:
            raise ValueError(f"base context must have shape (N, {BASE_CONTEXT_WIDTH})")


class NaiveContext(ContextModule):
    arm = Arm.NAIVE

    def __init__(self):
        super().__init__()
        self.width = BASE_CONTEXT_WIDTH

    def forward(self, *, prefix=None, base, extra=None, S=1, generator=None, eps=None):
        self._check_base(base)
        return base.unsqueeze(1).expand(-1, S, -1), base.new_zeros(base.shape[0]), {}


class HandcraftedContext(ContextModule):
    arm = Arm.HANDCRAFTED

    def __init__(self, feature_width: int):
        super().__init__()
        if feature_width < 1:
            raise ValueError("feature_width must be positive")
        self.feature_width = int(feature_width)
        self.width = BASE_CONTEXT_WIDTH + self.feature_width

    def forward(self, *, prefix=None, base, extra=None, S=1, generator=None, eps=None):
        self._check_base(base)
        if extra is None or extra.shape != (base.shape[0], self.feature_width):
            raise ValueError(f"handcrafted features must have shape (N, {self.feature_width})")
        ctx = torch.cat((base, extra.to(base.dtype)), dim=-1)
        return ctx.unsqueeze(1).expand(-1, S, -1), base.new_zeros(base.shape[0]), {}


class DeterministicZContext(ContextModule):
    arm = Arm.DET_Z

    def __init__(self, encoder_config: PrefixEncoderConfig = PrefixEncoderConfig(), *, l2_weight: float = 1e-3):
        super().__init__()
        if l2_weight < 0:
            raise ValueError("l2_weight must be non-negative")
        self.encoder = PrefixEncoder(encoder_config)
        self.l2_weight = float(l2_weight)
        self.latent_dimensions = encoder_config.latent_dimensions
        self.width = BASE_CONTEXT_WIDTH + self.latent_dimensions

    def forward(self, *, prefix, base, extra=None, S=1, generator=None, eps=None):
        self._check_base(base)
        if prefix is None:
            raise ValueError("the deterministic-Z arm needs the prefix batch")
        mean, _ = self.encoder(**prefix)
        ctx = torch.cat((base, mean.to(base.dtype)), dim=-1)
        return ctx.unsqueeze(1).expand(-1, S, -1), self.l2_weight * mean.square().sum(dim=-1), {"mean": mean}


class VariationalZContext(ContextModule):
    arm = Arm.VAR_Z

    def __init__(self, encoder_config: PrefixEncoderConfig = PrefixEncoderConfig(), *, kl_weight: float = 1.0):
        super().__init__()
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        self.encoder = PrefixEncoder(encoder_config)
        self.kl_weight = float(kl_weight)
        self.latent_dimensions = encoder_config.latent_dimensions
        self.width = BASE_CONTEXT_WIDTH + self.latent_dimensions

    def posterior(self, prefix: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(**prefix)

    def forward(self, *, prefix, base, extra=None, S=1, generator=None, eps=None):
        self._check_base(base)
        if prefix is None:
            raise ValueError("the variational-Z arm needs the prefix batch")
        mean, log_scale = self.encoder(**prefix)
        n, d = mean.shape
        if eps is None:
            eps = torch.randn((n, S, d), device=mean.device, dtype=mean.dtype, generator=generator)
        elif eps.shape != (n, S, d):
            raise ValueError(f"eps must have shape {(n, S, d)}")
        z = mean.unsqueeze(1) + eps.to(mean.dtype) * log_scale.exp().unsqueeze(1)
        ctx = torch.cat((base.unsqueeze(1).expand(-1, S, -1), z.to(base.dtype)), dim=-1)
        return ctx, self.kl_weight * PrefixEncoder.kl_standard_normal(mean, log_scale), {"mean": mean, "log_scale": log_scale}


class OracleKContext(ContextModule):
    arm = Arm.ORACLE_K

    def __init__(self, skill_dimensions: int = 4):
        super().__init__()
        self.skill_dimensions = int(skill_dimensions)
        self.width = BASE_CONTEXT_WIDTH + self.skill_dimensions

    def forward(self, *, prefix=None, base, extra=None, S=1, generator=None, eps=None):
        self._check_base(base)
        if _oracle_token.get() is None:
            raise GuardViolation("the oracle-K context runs only under an EvaluationOnly token")
        if extra is None or extra.shape != (base.shape[0], self.skill_dimensions):
            raise ValueError(f"oracle skill must have shape (N, {self.skill_dimensions})")
        ctx = torch.cat((base, extra.to(base.dtype)), dim=-1)
        return ctx.unsqueeze(1).expand(-1, S, -1), base.new_zeros(base.shape[0]), {}


def build_context(arm: Arm | str, *, feature_width: int = 0, encoder_config: PrefixEncoderConfig = PrefixEncoderConfig(), kl_weight: float = 1.0, l2_weight: float = 1e-3) -> ContextModule:
    """Inputs: arm; handcrafted feature width; encoder config; KL / L2 weights. Outputs: the context module."""
    arm = Arm(arm)
    if arm is Arm.NAIVE:
        return NaiveContext()
    if arm is Arm.HANDCRAFTED:
        return HandcraftedContext(feature_width)
    if arm is Arm.DET_Z:
        return DeterministicZContext(encoder_config, l2_weight=l2_weight)
    if arm is Arm.VAR_Z:
        return VariationalZContext(encoder_config, kl_weight=kl_weight)
    return OracleKContext(encoder_config.latent_dimensions if encoder_config.latent_dimensions == 4 else 4)


__all__ = ["Arm", "ContextModule", "DeterministicZContext", "HandcraftedContext", "NaiveContext", "OracleKContext", "VariationalZContext", "build_context"]
