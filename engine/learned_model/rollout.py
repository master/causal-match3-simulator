"""Closed-loop adapters from learned action logits to simulator actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..scm import DEFAULT_GOAL_COLOUR
from ..spec import Action, PlayerSkill, State
from .action_policy import ContinuousActionPolicy
from .arms import GenerativeWorldModel
from .tokens import action_to_index, index_to_action, legal_mask


@dataclass
class NetworkActionPolicy:
    """Feed each evolving state into a learned legal-action distribution."""

    model: ContinuousActionPolicy
    device: torch.device = torch.device("cpu")
    stochastic: bool = False
    seed: int = 0
    skill_override: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.model.to(self.device).eval()
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        if self.skill_override is not None:
            values = np.asarray(self.skill_override, dtype=np.float32)
            if values.shape != (self.model.config.skill_dimensions,):
                raise ValueError("skill_override has the wrong shape")
            self.skill_override = values

    @torch.no_grad()
    def __call__(self, state: State, player: PlayerSkill) -> Action | None:
        mask = legal_mask(state.board)
        if not np.any(mask):
            return None
        skill = (
            self.skill_override
            if self.skill_override is not None
            else player.as_array().astype(np.float32)
        )
        log_probabilities = self.model(
            board=torch.as_tensor(
                state.board.reshape(1, -1), dtype=torch.long, device=self.device
            ),
            goal_colour=torch.tensor(
                [state.goal_colour], dtype=torch.long, device=self.device
            ),
            moves_left=torch.tensor(
                [state.moves_left], dtype=torch.long, device=self.device
            ),
            goals_left=torch.tensor(
                [state.goals_left], dtype=torch.long, device=self.device
            ),
            skill=torch.as_tensor(
                skill[None, :], dtype=torch.float32, device=self.device
            ),
            legal_actions=torch.as_tensor(
                mask[None, :], dtype=torch.bool, device=self.device
            ),
        )[0]
        if self.stochastic:
            index = int(
                torch.multinomial(
                    log_probabilities.exp(),
                    num_samples=1,
                    generator=self.generator,
                )
            )
        else:
            index = int(torch.argmax(log_probabilities))
        return index_to_action(index)


@dataclass(frozen=True)
class LearnedDynamicsRollout:
    """States and actions generated after conditioning on a supplied opening."""

    states: tuple[State, ...]
    actions: tuple[Action, ...]

    @property
    def outcome(self) -> int:
        return int(self.states[-1].won)

    @property
    def completion_margin(self) -> float:
        initial = self.states[0]
        terminal = self.states[-1]
        if terminal.won:
            return terminal.moves_left / initial.moves_left
        initial_goals = max(1, initial.goals_left)
        return -terminal.goals_left / initial_goals


def _decoded_board(
    logits: torch.Tensor,
    *,
    stochastic: bool,
    generator: torch.Generator,
) -> np.ndarray:
    if stochastic:
        colours = torch.multinomial(
            logits.softmax(dim=-1),
            num_samples=1,
            generator=generator,
        ).squeeze(-1)
    else:
        colours = logits.argmax(dim=-1)
    return colours.reshape(8, 8).detach().cpu().numpy().astype(np.int8)


@torch.no_grad()
def sample_learned_initial_state(
    model: GenerativeWorldModel,
    *,
    level_index: int,
    tier_index: int,
    served_difficulty: float,
    goal_colour: int = DEFAULT_GOAL_COLOUR,
    stochastic: bool = True,
    seed: int = 0,
) -> State:
    """Sample a task-conditioned opening without invoking simulator mechanics."""
    if not 0 <= level_index < model.config.dynamics.n_levels:
        raise ValueError("level_index is outside the model vocabulary")
    if not 0 <= tier_index < model.config.dynamics.n_tiers:
        raise ValueError("tier_index is outside the model vocabulary")
    if not 0 <= goal_colour < model.config.dynamics.n_colours:
        raise ValueError("goal_colour is outside the model vocabulary")
    device = next(model.parameters()).device
    model.eval()
    context = model.dynamics.encode_task(
        torch.tensor([level_index], dtype=torch.long, device=device),
        torch.tensor([tier_index], dtype=torch.long, device=device),
        torch.tensor(
            [served_difficulty], dtype=torch.float32, device=device
        ),
    )
    board_logits, counter_mean = model.dynamics.initial_state_predictions(
        context
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    moves_left = int(
        torch.round(
            counter_mean[0, 0] * model.config.dynamics.max_moves_left
        ).item()
    )
    goals_left = int(
        torch.round(
            counter_mean[0, 1] * model.config.dynamics.goals_scale
        ).item()
    )
    return State(
        board=_decoded_board(
            board_logits[0], stochastic=stochastic, generator=generator
        ),
        moves_left=min(
            model.config.dynamics.max_moves_left, max(1, moves_left)
        ),
        goals_left=max(1, goals_left),
        goal_colour=goal_colour,
        t=0,
    )


@torch.no_grad()
def rollout_learned_dynamics(
    model: GenerativeWorldModel,
    initial_state: State,
    *,
    level_index: int,
    tier_index: int,
    served_difficulty: float,
    player_context: torch.Tensor | np.ndarray,
    max_steps: int | None = None,
    stochastic: bool = False,
    seed: int = 0,
) -> LearnedDynamicsRollout:
    """Run policy and RSSM dynamics without applying the engine transition."""
    if not 0 <= level_index < model.config.dynamics.n_levels:
        raise ValueError("level_index is outside the model vocabulary")
    if not 0 <= tier_index < model.config.dynamics.n_tiers:
        raise ValueError("tier_index is outside the model vocabulary")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    device = next(model.parameters()).device
    model.eval()
    skill = torch.as_tensor(
        player_context,
        dtype=model.behavior.skill.weight.dtype,
        device=device,
    )
    if skill.shape != (model.config.prefix.skill_dimensions,):
        raise ValueError("player_context has the wrong shape")
    levels = torch.tensor([level_index], dtype=torch.long, device=device)
    tiers = torch.tensor([tier_index], dtype=torch.long, device=device)
    served = torch.tensor(
        [served_difficulty], dtype=torch.float32, device=device
    )
    task_context = model.dynamics.encode_task(levels, tiers, served)
    hidden, stochastic_state = model.dynamics.rssm.initial_state(1, task_context)
    generator = torch.Generator(device=device).manual_seed(seed)
    step_limit = initial_state.moves_left if max_steps is None else min(
        max_steps, initial_state.moves_left
    )
    states = [initial_state.copy()]
    actions: list[Action] = []

    for step_index in range(step_limit):
        state = states[-1]
        if state.terminal:
            break
        mask = legal_mask(state.board)
        if not np.any(mask):
            break
        board = torch.as_tensor(
            state.board.reshape(1, -1), dtype=torch.long, device=device
        )
        legal_actions = torch.as_tensor(
            mask[None, :], dtype=torch.bool, device=device
        )
        log_probabilities = model.behavior(
            board=board,
            goal_colour=torch.tensor(
                [state.goal_colour], dtype=torch.long, device=device
            ),
            moves_left=torch.tensor(
                [state.moves_left], dtype=torch.long, device=device
            ),
            goals_left=torch.tensor(
                [state.goals_left], dtype=torch.long, device=device
            ),
            skill=skill.unsqueeze(0),
            legal_actions=legal_actions,
        )[0]
        if stochastic:
            action_index = int(
                torch.multinomial(
                    log_probabilities.exp(),
                    num_samples=1,
                    generator=generator,
                )
            )
        else:
            action_index = int(torch.argmax(log_probabilities))
        action = index_to_action(action_index)
        action_tensor = torch.tensor(
            [action_to_index(action)], dtype=torch.long, device=device
        )
        if step_index == 0:
            observation = model.dynamics.encode_observation(
                board.unsqueeze(1),
                torch.tensor(
                    [[state.goal_colour]], dtype=torch.long, device=device
                ),
                torch.tensor(
                    [[state.moves_left]], dtype=torch.long, device=device
                ),
                torch.tensor(
                    [[state.goals_left]], dtype=torch.long, device=device
                ),
            )[:, 0]
            result = model.dynamics.rssm.observe_step(
                hidden=hidden,
                stochastic=stochastic_state,
                action=action_tensor,
                context=task_context,
                observation=observation,
                sample_posterior=stochastic,
                generator=generator,
            )
        else:
            result = model.dynamics.rssm.imagine_step(
                hidden=hidden,
                stochastic=stochastic_state,
                action=action_tensor,
                context=task_context,
                sample_prior=stochastic,
                generator=generator,
            )
        hidden = result["hidden"]
        stochastic_state = result["stochastic"]
        decoded_goals = int(
            torch.round(
                result["counter_mean"][0, 1]
                * model.config.dynamics.goals_scale
            ).item()
        )
        next_state = State(
            board=_decoded_board(
                result["board_logits"][0],
                stochastic=stochastic,
                generator=generator,
            ),
            moves_left=max(0, state.moves_left - 1),
            goals_left=min(state.goals_left, max(0, decoded_goals)),
            goal_colour=state.goal_colour,
            t=state.t + 1,
        )
        actions.append(action)
        states.append(next_state)

    return LearnedDynamicsRollout(tuple(states), tuple(actions))


__all__ = [
    "LearnedDynamicsRollout",
    "NetworkActionPolicy",
    "rollout_learned_dynamics",
    "sample_learned_initial_state",
]