"""Context-conditioned recurrent state-space model for gameplay dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tokens import ACTION_SLOTS, N_CELLS


@dataclass(frozen=True)
class RSSMConfig:
    n_colours: int = 6
    hidden_size: int = 128
    stochastic_size: int = 32
    observation_size: int = 128
    context_size: int = 4
    action_size: int = ACTION_SLOTS

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")


class FastRSSM(nn.Module):
    """Gaussian RSSM with deterministic recurrence and typed state decoders."""

    def __init__(self, config: RSSMConfig = RSSMConfig()):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        stochastic = config.stochastic_size
        context = config.context_size
        self.action = nn.Embedding(config.action_size, hidden)
        self.recurrent = nn.GRUCell(hidden + stochastic + context, hidden)
        self.prior = nn.Sequential(
            nn.Linear(hidden + context, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * stochastic),
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden + context + config.observation_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * stochastic),
        )
        decoder_input = hidden + stochastic + context
        self.board_decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, N_CELLS * config.n_colours),
        )
        self.counter_decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    @staticmethod
    def _distribution_parameters(
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_scale = values.chunk(2, dim=-1)
        log_scale = raw_scale.clamp(-6.0, 3.0)
        return mean, log_scale

    @staticmethod
    def _sample(
        mean: torch.Tensor,
        log_scale: torch.Tensor,
        sample: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not sample:
            return mean
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return mean + noise * log_scale.exp()

    @staticmethod
    def _kl(
        posterior_mean: torch.Tensor,
        posterior_log_scale: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_log_scale: torch.Tensor,
    ) -> torch.Tensor:
        posterior_variance = torch.exp(2.0 * posterior_log_scale)
        prior_variance = torch.exp(2.0 * prior_log_scale)
        return 0.5 * torch.sum(
            (
                posterior_variance
                + (posterior_mean - prior_mean).square()
            )
            / prior_variance
            - 1.0
            + 2.0 * (prior_log_scale - posterior_log_scale),
            dim=-1,
        )

    def _decode(
        self,
        hidden: torch.Tensor,
        stochastic: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((hidden, stochastic, context), dim=-1)
        board_logits = self.board_decoder(features).view(
            *features.shape[:-1], N_CELLS, self.config.n_colours
        )
        counter_mean = self.counter_decoder(features)
        return board_logits, counter_mean

    def _initial_state(
        self, batch_size: int, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = reference.new_zeros(batch_size, self.config.hidden_size)
        stochastic = reference.new_zeros(batch_size, self.config.stochastic_size)
        return hidden, stochastic

    def initial_state(
        self, batch_size: int, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero deterministic and stochastic states for imagination."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return self._initial_state(batch_size, reference)

    def imagine_step(
        self,
        *,
        hidden: torch.Tensor,
        stochastic: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        sample_prior: bool = True,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance one prior transition so actions can depend on generated state."""
        batch_size = hidden.shape[0]
        if hidden.shape != (batch_size, self.config.hidden_size):
            raise ValueError("hidden has the wrong shape")
        if stochastic.shape != (batch_size, self.config.stochastic_size):
            raise ValueError("stochastic has the wrong shape")
        if action.shape != (batch_size,):
            raise ValueError("action must have shape (batch,)")
        if context.shape != (batch_size, self.config.context_size):
            raise ValueError("context has the wrong shape")
        return self._prior_step(
            hidden=hidden,
            stochastic=stochastic,
            action=action,
            context=context,
            sample_prior=sample_prior,
            generator=generator,
        )

    def _prior_step(
        self,
        *,
        hidden: torch.Tensor,
        stochastic: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        sample_prior: bool,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        next_hidden = self.recurrent(
            torch.cat(
                (self.action(action.long()), stochastic, context), dim=-1
            ),
            hidden,
        )
        prior_mean, prior_log_scale = self._distribution_parameters(
            self.prior(torch.cat((next_hidden, context), dim=-1))
        )
        next_stochastic = self._sample(
            prior_mean, prior_log_scale, sample_prior, generator
        )
        board_logits, counter_mean = self._decode(
            next_hidden, next_stochastic, context
        )
        return {
            "hidden": next_hidden,
            "prior_mean": prior_mean,
            "prior_log_scale": prior_log_scale,
            "stochastic": next_stochastic,
            "board_logits": board_logits,
            "counter_mean": counter_mean,
        }

    def observe_step(
        self,
        *,
        hidden: torch.Tensor,
        stochastic: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        observation: torch.Tensor,
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance one transition and condition its latent on an observation."""
        batch_size = hidden.shape[0]
        if observation.shape != (batch_size, self.config.observation_size):
            raise ValueError("observation has the wrong shape")
        prior = self._prior_step(
            hidden=hidden,
            stochastic=stochastic,
            action=action,
            context=context,
            sample_prior=False,
            generator=generator,
        )
        posterior_mean, posterior_log_scale = self._distribution_parameters(
            self.posterior(
                torch.cat(
                    (prior["hidden"], context, observation), dim=-1
                )
            )
        )
        next_stochastic = self._sample(
            posterior_mean, posterior_log_scale, sample_posterior, generator
        )
        board_logits, counter_mean = self._decode(
            prior["hidden"], next_stochastic, context
        )
        return {
            "hidden": prior["hidden"],
            "prior_mean": prior["prior_mean"],
            "prior_log_scale": prior["prior_log_scale"],
            "posterior_mean": posterior_mean,
            "posterior_log_scale": posterior_log_scale,
            "stochastic": next_stochastic,
            "board_logits": board_logits,
            "counter_mean": counter_mean,
            "kl": self._kl(
                posterior_mean,
                posterior_log_scale,
                prior["prior_mean"],
                prior["prior_log_scale"],
            ),
        }

    def observe(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        context: torch.Tensor,
        sample_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Filter an observed sequence and return aligned latent predictions."""
        if observations.ndim != 3:
            raise ValueError("observations must have shape (batch, steps, features)")
        batch_size, steps, observation_size = observations.shape
        if observation_size != self.config.observation_size:
            raise ValueError("observation feature size does not match config")
        if actions.shape != (batch_size, steps):
            raise ValueError("actions must align with observations")
        if context.shape != (batch_size, self.config.context_size):
            raise ValueError("context has the wrong shape")
        hidden, stochastic = self.initial_state(batch_size, observations)
        outputs: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "hidden",
                "prior_mean",
                "prior_log_scale",
                "posterior_mean",
                "posterior_log_scale",
                "stochastic",
                "board_logits",
                "counter_mean",
                "kl",
            )
        }
        for step in range(steps):
            values = self.observe_step(
                hidden=hidden,
                stochastic=stochastic,
                action=actions[:, step],
                context=context,
                observation=observations[:, step],
                sample_posterior=sample_posterior,
            )
            hidden = values["hidden"]
            stochastic = values["stochastic"]
            for name, value in values.items():
                outputs[name].append(value)
        return {
            name: torch.stack(values, dim=1) for name, values in outputs.items()
        }

    def imagine(
        self,
        *,
        actions: torch.Tensor,
        context: torch.Tensor,
        sample_prior: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Generate latent states and decoded observations without inputs."""
        if actions.ndim != 2:
            raise ValueError("actions must have shape (batch, steps)")
        batch_size, steps = actions.shape
        if context.shape != (batch_size, self.config.context_size):
            raise ValueError("context has the wrong shape")
        hidden, stochastic = self.initial_state(batch_size, context)
        outputs: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "hidden",
                "prior_mean",
                "prior_log_scale",
                "stochastic",
                "board_logits",
                "counter_mean",
            )
        }
        for step in range(steps):
            result = self.imagine_step(
                hidden=hidden,
                stochastic=stochastic,
                action=actions[:, step],
                context=context,
                sample_prior=sample_prior,
            )
            hidden = result["hidden"]
            stochastic = result["stochastic"]
            for name, value in result.items():
                outputs[name].append(value)
        return {
            name: torch.stack(values, dim=1)
            for name, values in outputs.items()
        }

    def parameter_counts(self) -> dict[str, int]:
        """Return trainable parameters by component and in total."""
        components = {
            "action_embedding": self.action,
            "recurrent": self.recurrent,
            "prior": self.prior,
            "posterior": self.posterior,
            "board_decoder": self.board_decoder,
            "counter_decoder": self.counter_decoder,
        }
        counts = {
            name: sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            for name, module in components.items()
        }
        counts["total"] = sum(counts.values())
        return counts


__all__ = ["FastRSSM", "RSSMConfig"]