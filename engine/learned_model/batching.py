"""Build strict-prefix encoder inputs and target-episode decoder tensors."""

from __future__ import annotations

import numpy as np
import torch

from ..evidence import EVIDENCE_NAMES
from ..retention import (
    MasteryConfig,
    PlayerTrajectory,
    completion_margin,
    update_mastery,
)
from ..scm import LEVELS, TIER_NAMES
from ..spec import BENCHMARK_CONFIG, BenchmarkConfig
from .data import gameplay_transition_dataset_from_episodes
from .model import PredictiveTarget
from .tokens import action_to_index, legal_mask


def build_prefix_target_batch(
    trajectories: list[PlayerTrajectory],
    *,
    target_attempt: int,
    device: torch.device = torch.device("cpu"),
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
) -> tuple[dict[str, torch.Tensor], PredictiveTarget]:
    """Encode attempts before ``target_attempt`` and score that attempt only."""
    if target_attempt < 1:
        raise ValueError("target_attempt must be positive")
    selected: list[tuple[PlayerTrajectory, object, tuple[object, ...]]] = []
    for trajectory in trajectories:
        targets = [
            record
            for record in trajectory.attempts
            if record.attempt_id == target_attempt
        ]
        if len(targets) != 1:
            raise ValueError(
                f"player {trajectory.player_id} lacks exactly one target attempt"
            )
        prefix = tuple(
            record
            for record in trajectory.attempts
            if record.attempt_id < target_attempt
        )
        selected.append((trajectory, targets[0], prefix))
    if not selected:
        raise ValueError("at least one trajectory is required")

    batch_size = len(selected)
    max_episodes = max(1, max(len(prefix) for _, _, prefix in selected))
    max_steps = max(
        1,
        max(
            (len(record.episode.actions) for _, _, prefix in selected for record in prefix),
            default=0,
        ),
    )
    proxy_dimensions = len(EVIDENCE_NAMES)
    boards = np.zeros((batch_size, max_episodes, max_steps, 64), dtype=np.int64)
    actions = np.zeros((batch_size, max_episodes, max_steps), dtype=np.int64)
    moves_left = np.zeros((batch_size, max_episodes, max_steps), dtype=np.int64)
    goals_left = np.zeros((batch_size, max_episodes, max_steps), dtype=np.int64)
    step_mask = np.zeros((batch_size, max_episodes, max_steps), dtype=bool)
    levels = np.zeros((batch_size, max_episodes), dtype=np.int64)
    tiers = np.zeros((batch_size, max_episodes), dtype=np.int64)
    served = np.zeros((batch_size, max_episodes), dtype=np.float32)
    outcomes = np.zeros((batch_size, max_episodes), dtype=np.float32)
    proxies = np.zeros((batch_size, max_episodes, proxy_dimensions), dtype=np.float32)
    episode_mask = np.zeros((batch_size, max_episodes), dtype=bool)
    level_ids = {level.name: index for index, level in enumerate(LEVELS)}
    tier_ids = {tier: index for index, tier in enumerate(TIER_NAMES)}

    target_served = []
    target_levels = []
    target_tiers = []
    target_evidence = []
    target_outcomes = []
    target_mastery_before = []
    target_completion_margin = []
    target_churn = []
    target_action_player = []
    target_boards = []
    target_goal_colours = []
    target_moves_left = []
    target_goals_left = []
    target_legal_actions = []
    target_actions = []

    for player_index, (_, target, prefix) in enumerate(selected):
        for episode_index, record in enumerate(prefix):
            episode = record.episode
            episode_mask[player_index, episode_index] = True
            levels[player_index, episode_index] = level_ids[episode.level.name]
            tiers[player_index, episode_index] = tier_ids[episode.tier]
            served[player_index, episode_index] = episode.E
            outcomes[player_index, episode_index] = episode.R
            proxies[player_index, episode_index] = episode.proxy
            for step_index, action in enumerate(episode.actions):
                state = episode.states[step_index]
                boards[player_index, episode_index, step_index] = state.board.ravel()
                actions[player_index, episode_index, step_index] = action_to_index(action)
                moves_left[player_index, episode_index, step_index] = state.moves_left
                goals_left[player_index, episode_index, step_index] = state.goals_left
                step_mask[player_index, episode_index, step_index] = True

        episode = target.episode
        target_served.append(episode.E)
        target_levels.append(level_ids[episode.level.name])
        target_tiers.append(tier_ids[episode.tier])
        target_evidence.append(episode.proxy)
        target_outcomes.append(episode.R)
        mastery_before = mastery_config.initial
        for record in prefix:
            mastery_before = update_mastery(
                mastery_before, record.episode.R, mastery_config
            )
        target_mastery_before.append(mastery_before)
        target_completion_margin.append(completion_margin(episode))
        target_churn.append(target.churn_after)
        for step_index, action in enumerate(episode.actions):
            state = episode.states[step_index]
            target_action_player.append(player_index)
            target_boards.append(state.board.ravel())
            target_goal_colours.append(state.goal_colour)
            target_moves_left.append(state.moves_left)
            target_goals_left.append(state.goals_left)
            target_legal_actions.append(legal_mask(state.board))
            target_actions.append(action_to_index(action))

    tensor = lambda values, dtype: torch.as_tensor(values, dtype=dtype, device=device)
    prefix_batch = {
        "boards": tensor(boards, torch.long),
        "actions": tensor(actions, torch.long),
        "moves_left": tensor(moves_left, torch.long),
        "goals_left": tensor(goals_left, torch.long),
        "step_mask": tensor(step_mask, torch.bool),
        "levels": tensor(levels, torch.long),
        "tiers": tensor(tiers, torch.long),
        "served_difficulty": tensor(served, torch.float32),
        "outcomes": tensor(outcomes, torch.float32),
        "proxies": tensor(proxies, torch.float32),
        "episode_mask": tensor(episode_mask, torch.bool),
        "baseline_evidence": torch.zeros(
            (batch_size, proxy_dimensions), dtype=torch.float32, device=device
        ),
    }
    predictive_target = PredictiveTarget(
        episode_player=torch.arange(batch_size, device=device),
        served_difficulty=tensor(target_served, torch.float32),
        levels=tensor(target_levels, torch.long),
        tiers=tensor(target_tiers, torch.long),
        evidence=tensor(np.asarray(target_evidence), torch.float32),
        outcomes=tensor(target_outcomes, torch.float32),
        mastery_before=tensor(target_mastery_before, torch.float32),
        completion_margin=tensor(target_completion_margin, torch.float32),
        churn=tensor(target_churn, torch.float32),
        churn_mask=torch.ones((batch_size,), dtype=torch.bool, device=device),
        churn_scale=torch.full(
            (batch_size,),
            benchmark.warmup_churn_scale
            if target_attempt < benchmark.landmark_attempt
            else 1.0,
            dtype=torch.float32,
            device=device,
        ),
        action_player=tensor(target_action_player, torch.long),
        boards=tensor(np.asarray(target_boards).reshape(-1, 64), torch.long),
        goal_colours=tensor(target_goal_colours, torch.long),
        moves_left=tensor(target_moves_left, torch.long),
        goals_left=tensor(target_goals_left, torch.long),
        legal_actions=tensor(
            np.asarray(target_legal_actions).reshape(-1, 128), torch.bool
        ),
        actions=tensor(target_actions, torch.long),
    )
    return prefix_batch, predictive_target


def build_generative_training_batch(
    trajectories: list[PlayerTrajectory],
    *,
    target_attempt: int,
    device: torch.device = torch.device("cpu"),
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
) -> tuple[
    dict[str, torch.Tensor],
    PredictiveTarget,
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    """Build aligned structural, dynamics, and isolated oracle tensors."""
    prefix, target = build_prefix_target_batch(
        trajectories,
        target_attempt=target_attempt,
        device=device,
        mastery_config=mastery_config,
        benchmark=benchmark,
    )
    target_records = [
        next(
            record
            for record in trajectory.attempts
            if record.attempt_id == target_attempt
        )
        for trajectory in trajectories
    ]
    transition_data = gameplay_transition_dataset_from_episodes(
        [record.episode for record in target_records],
        player_ids=[trajectory.player_id for trajectory in trajectories],
        attempt_ids=[target_attempt] * len(trajectories),
    )
    transitions = transition_data.batch(transition_data.episodes, device=device)
    oracle_skill = torch.as_tensor(
        np.asarray(
            [trajectory.player.as_array() for trajectory in trajectories]
        ),
        dtype=torch.float32,
        device=device,
    )
    return prefix, target, transitions, oracle_skill


__all__ = ["build_generative_training_batch", "build_prefix_target_batch"]