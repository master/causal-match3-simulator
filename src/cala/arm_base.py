"""
Implementation of the CaLA arm: the adjustment context, the shared legal-action policy (``ContinuousActionPolicy``, 4 layers, 4 heads, 128 swap slots,
and generic context width), the learned retention head, and the auxiliary anchoring heads.
The training objective is action NLL + retention NLL + auxiliary NLLs + context penalty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from match3_simulator.learned_model.action_policy import ActionPolicyConfig, ContinuousActionPolicy
from match3_simulator.retention import ChurnSchedule, MasteryConfig
from match3_simulator.experiments.contexts import Arm, ContextModule, build_context
from match3_simulator.experiments.encoder_base import PrefixEncoderConfig
from match3_simulator.experiments.retention import ActionSummaryHead, AssignmentHead, CompletionHead, EvidenceAdapter, MarginHead, RetentionHead


@dataclass(frozen=True)
class CalaArmConfig:
    arm: str = "var_z"
    feature_width: int = 0
    latent_dimensions: int = 4
    encoder_hidden: int = 64
    policy_d_model: int = 128
    policy_layers: int = 4
    policy_heads: int = 4
    kl_weight: float = 1.0
    l2_weight: float = 1e-3
    retention_weight: float = 1.0
    aux_weight: float = 0.1
    warmup_scale: float = 0.005

    def __post_init__(self) -> None:
        Arm(self.arm)
        if self.arm == "handcrafted" and self.feature_width < 1:
            raise ValueError("the handcrafted arm needs feature_width")
        if min(self.retention_weight, self.aux_weight, self.kl_weight, self.l2_weight) < 0:
            raise ValueError("weights must be non-negative")
        if self.policy_d_model % self.policy_heads:
            raise ValueError("policy_d_model must be divisible by policy_heads")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CalaArm(nn.Module):
    """Context + policy + heads for one arm (see module docstring)."""

    def __init__(self, config: CalaArmConfig, *, churn: ChurnSchedule, mastery: MasteryConfig):
        super().__init__()
        self.config = config
        self.arm = Arm(config.arm)
        enc = PrefixEncoderConfig(latent_dimensions=config.latent_dimensions, hidden_size=config.encoder_hidden)
        self.context: ContextModule = build_context(self.arm, feature_width=config.feature_width, encoder_config=enc, kl_weight=config.kl_weight, l2_weight=config.l2_weight)
        self.width = int(self.context.width)
        self.policy = ContinuousActionPolicy(ActionPolicyConfig(skill_dimensions=self.width, d_model=config.policy_d_model, n_layers=config.policy_layers, n_heads=config.policy_heads))
        self.retention = RetentionHead(self.width, churn=churn, mastery=mastery, warmup_scale=config.warmup_scale)
        self.assignment = AssignmentHead(self.width)
        self.completion = CompletionHead(self.width)
        self.margin = MarginHead(self.width)
        self.action_summary = ActionSummaryHead(self.width)
        self.evidence = EvidenceAdapter(self.width)

    # --- context ---------------------------------------------------------------------------------------------------------
    def contexts(self, batch: dict[str, torch.Tensor], *, S: int = 1, generator: torch.Generator | None = None, eps: torch.Tensor | None = None):
        """Inputs: batch with prefix / base / extra; S draws; generator or fixed eps. Outputs: (context (N, S, width), penalty (N,), info)."""
        return self.context(prefix=batch.get("prefix"), base=batch["base"], extra=batch.get("extra"), S=S, generator=generator, eps=eps)

    # --- objective -------------------------------------------------------------------------------------------------------
    def objective(self, batch: dict[str, torch.Tensor], *, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """
        Joint training objective on one batch of (player, target attempt) rows plus the target attempt's logged transitions.
        """
        ctx, penalty, _ = self.contexts(batch, S=1, generator=generator)
        c = ctx[:, 0]
        level, tier = batch["level"], batch["tier"]
        logp = self.policy(board=batch["t_boards"], goal_colour=batch["t_colour"], moves_left=batch["t_moves"], goals_left=batch["t_goals"], skill=c[batch["t_row"]], legal_actions=batch["t_legal"])
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

    # --- query-time callables ----------------------------------------------------------------------------------------------
    def policy_fn(self, context: torch.Tensor):
        """Inputs: fixed context per row (N, width). Outputs: callable (boards, goal_colour, moves_left, goals_left, legal) -> log-probabilities (N, 128)."""
        if context.ndim != 2 or context.shape[1] != self.width:
            raise ValueError(f"context must have shape (N, {self.width})")

        def call(boards, goal_colour, moves_left, goals_left, legal):
            return self.policy(board=boards, goal_colour=goal_colour, moves_left=moves_left, goals_left=goals_left, skill=context[: boards.shape[0]] if context.shape[0] != boards.shape[0] else context, legal_actions=legal)

        return call

    def retention_fn(self, context: torch.Tensor):
        """Inputs: context (N, width). Outputs: callable (completion, margin, mastery_after, level, attempt) numpy -> churn probability numpy (N,)."""
        device = context.device

        @torch.no_grad()
        def call(completion, margin, mastery_after, level, attempt):
            t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
            p = self.retention.probabilities(completion=t(completion, torch.float32), margin=t(margin, torch.float32), mastery_after=t(mastery_after, torch.float32), level=t(level, torch.long), context=context,
                                             attempt=t(attempt, torch.long))
            return p.detach().cpu().numpy().astype(np.float64)

        return call

    def parameter_counts(self) -> dict[str, int]:
        count = lambda m: int(sum(p.numel() for p in m.parameters()))
        return {"context": count(self.context), "policy": count(self.policy), "retention": count(self.retention), "aux_heads": count(self.assignment) + count(self.completion) + count(self.margin) + count(self.action_summary) + count(self.evidence),
                "total": count(self)}


def save_arm(model: CalaArm, path, *, extra: dict[str, object] | None = None) -> None:
    torch.save({"arm_config": model.config.to_dict(), "state_dict": model.state_dict(), **(extra or {})}, path)


def load_arm(path, *, churn: ChurnSchedule, mastery: MasteryConfig, device: torch.device = torch.device("cpu")) -> tuple[CalaArm, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CalaArm(CalaArmConfig(**checkpoint["arm_config"]), churn=churn, mastery=mastery).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), {k: v for k, v in checkpoint.items() if k not in ("state_dict",)}


__all__ = ["CalaArm", "CalaArmConfig", "load_arm", "save_arm"]
