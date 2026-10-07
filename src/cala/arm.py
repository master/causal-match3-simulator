"""
This is the frozen version of the CalaArm (context + shared legal-action policy + learned retention head + auxiliary anchoring heads) with two orthogonal
switches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from match3_simulator.retention import ChurnSchedule, MasteryConfig
from match3_simulator.experiments.arm_base import CalaArm, CalaArmConfig
from match3_simulator.experiments.contexts import Arm, ContextModule, DeterministicZContext, VariationalZContext, build_context
from match3_simulator.experiments.encoder_base import PrefixEncoderConfig
from match3_simulator.experiments.retention import ActionSummaryHead, AssignmentHead, CompletionHead, EvidenceAdapter, MarginHead, RetentionHead
from match3_simulator.experiments.encoder import ENCODERS, SpatialEncoderConfig, make_encoder
from match3_simulator.experiments.policy import SpecialAwareActionPolicy, SpecialAwarePolicyConfig


@dataclass(frozen=True)
class CalaArm4Config(CalaArmConfig):
    encoder: str = "current"
    policy_specials: bool = False
    policy_special_features: bool = False
    cnn_channels: int = 32
    cnn_layers: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def variant(self) -> str:
        enc = self.encoder if Arm(self.arm).uses_encoder else "na"
        pol = "special" if (self.policy_specials or self.policy_special_features) else "current"
        if self.policy_specials and not self.policy_special_features:
            pol = "specialemb"
        if self.policy_special_features and not self.policy_specials:
            pol = "specialfeat"
        return f"enc-{enc}__pol-{pol}"


class DeterministicZContext4(DeterministicZContext):
    def __init__(self, encoder_config: PrefixEncoderConfig, *, l2_weight: float, encoder: str, spatial: SpatialEncoderConfig):
        super().__init__(encoder_config, l2_weight=l2_weight)
        self.encoder = make_encoder(encoder, encoder_config, spatial=spatial)


class VariationalZContext4(VariationalZContext):
    def __init__(self, encoder_config: PrefixEncoderConfig, *, kl_weight: float, encoder: str, spatial: SpatialEncoderConfig):
        super().__init__(encoder_config, kl_weight=kl_weight)
        self.encoder = make_encoder(encoder, encoder_config, spatial=spatial)


def build_context4(config: CalaArm4Config) -> ContextModule:
    arm = Arm(config.arm)
    enc = PrefixEncoderConfig(latent_dimensions=config.latent_dimensions, hidden_size=config.encoder_hidden)
    spatial = SpatialEncoderConfig(channels=config.cnn_channels, layers=config.cnn_layers)
    if arm is Arm.DET_Z:
        return DeterministicZContext4(enc, l2_weight=config.l2_weight, encoder=config.encoder, spatial=spatial)
    if arm is Arm.VAR_Z:
        return VariationalZContext4(enc, kl_weight=config.kl_weight, encoder=config.encoder, spatial=spatial)
    return build_context(arm, feature_width=config.feature_width, encoder_config=enc, kl_weight=config.kl_weight, l2_weight=config.l2_weight)


class CalaArm4(CalaArm):
    """See module docstring. Construction mirrors ``CalaArm.__init__`` with the two switches."""

    def __init__(self, config: CalaArm4Config, *, churn: ChurnSchedule, mastery: MasteryConfig):
        nn.Module.__init__(self)
        self.config = config
        self.arm = Arm(config.arm)
        self.context = build_context4(config)
        self.width = int(self.context.width)
        self.policy = SpecialAwareActionPolicy(SpecialAwarePolicyConfig(skill_dimensions=self.width, d_model=config.policy_d_model, n_layers=config.policy_layers, n_heads=config.policy_heads,
                                                                        use_specials=config.policy_specials, special_features=config.policy_special_features))
        self.retention = RetentionHead(self.width, churn=churn, mastery=mastery, warmup_scale=config.warmup_scale)
        self.assignment = AssignmentHead(self.width)
        self.completion = CompletionHead(self.width)
        self.margin = MarginHead(self.width)
        self.action_summary = ActionSummaryHead(self.width)
        self.evidence = EvidenceAdapter(self.width)

    @property
    def is_frozen_equivalent(self) -> bool:
        return self.config.encoder == "current" and self.policy.is_frozen_equivalent

    def objective(self, batch: dict[str, torch.Tensor], *, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """As ``CalaArm.objective`` with the target transitions' specials (``t_specials``) handed to the policy."""
        ctx, penalty, _ = self.contexts(batch, S=1, generator=generator)
        c = ctx[:, 0]
        level, tier = batch["level"], batch["tier"]
        logp = self.policy(board=batch["t_boards"], goal_colour=batch["t_colour"], moves_left=batch["t_moves"], goals_left=batch["t_goals"], skill=c[batch["t_row"]], legal_actions=batch["t_legal"], specials=batch.get("t_specials"))
        action = -logp.gather(1, batch["t_actions"].long().unsqueeze(1)).mean()
        retention = self.retention.nll(churn=batch["C"], completion=batch["R"], margin=batch["Q"], mastery_after=batch["mastery_after"], level=level, context=c, attempt=batch["attempt"]).mean()
        assignment = -self.assignment.log_prob(batch["E"], c, level, tier).mean()
        completion = -self.completion.log_prob(batch["R"], batch["E"], c, level, tier).mean()
        margin = -self.margin.log_prob(batch["Q"], batch["E"], c, level, tier).mean()
        summary = -self.action_summary.log_prob(batch["summaries"], c, level).mean()
        evidence = -self.evidence.log_prob(batch["evidence"], c, level).mean() / 12.0
        aux = assignment + completion + margin + summary + evidence
        total = action + self.config.retention_weight * retention + self.config.aux_weight * aux + penalty.mean()
        return {"total": total, "action": action, "retention": retention, "assignment": assignment, "completion": completion, "margin": margin, "action_summary": summary, "evidence": evidence,
                "penalty": penalty.mean(), "n_rows": torch.tensor(float(c.shape[0])), "n_transitions": torch.tensor(float(logp.shape[0]))}

    def policy_fn(self, context: torch.Tensor):
        """Inputs: fixed context per row (N, width). Outputs: callable (boards, goal_colour, moves_left, goals_left, legal, idx, specials) -> log-probabilities."""
        if context.ndim != 2 or context.shape[1] != self.width:
            raise ValueError(f"context must have shape (N, {self.width})")

        def call(boards, goal_colour, moves_left, goals_left, legal, idx, specials=None):
            return self.policy(board=boards, goal_colour=goal_colour, moves_left=moves_left, goals_left=goals_left, skill=context[idx], legal_actions=legal, specials=specials)

        return call

    def parameter_counts(self) -> dict[str, int]:
        counts = super().parameter_counts()
        enc = getattr(self.context, "encoder", None)
        counts["encoder"] = 0 if enc is None else int(sum(p.numel() for p in enc.parameters()))
        counts["policy_special_embedding"] = 0 if self.policy.special is None else int(sum(p.numel() for p in self.policy.special.parameters()))
        return counts


def save_arm4(model: CalaArm4, path, *, extra: dict[str, object] | None = None) -> None:
    torch.save({"arm_config": model.config.to_dict(), "state_dict": model.state_dict(), **(extra or {})}, path)


def load_arm4(path, *, churn: ChurnSchedule, mastery: MasteryConfig, device: torch.device = torch.device("cpu")) -> tuple[CalaArm4, dict[str, object]]:
    """Load a CaLA-4 checkpoint **or** a frozen CaLA checkpoint (its ``arm_config`` lacks the switches -> defaults = frozen-equivalent module, strict load)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CalaArm4(CalaArm4Config(**checkpoint["arm_config"]), churn=churn, mastery=mastery).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), {k: v for k, v in checkpoint.items() if k != "state_dict"}


__all__ = ["CalaArm4", "CalaArm4Config", "DeterministicZContext4", "VariationalZContext4", "build_context4", "load_arm4", "save_arm4"]
