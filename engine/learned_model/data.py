"""Datasets for learned action policies and future prefix inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..retention import PlayerTrajectory
from ..scm import LEVELS, TIER_NAMES, Episode
from .tokens import ACTION_SLOTS, N_CELLS, action_to_index, legal_mask


@dataclass(frozen=True)
class ActionDataset:
    """Flat state-action rows retaining episode groups and oracle skill."""

    episode_indices: np.ndarray
    player_indices: np.ndarray
    boards: np.ndarray
    goal_colours: np.ndarray
    moves_left: np.ndarray
    goals_left: np.ndarray
    skills: np.ndarray
    actions: np.ndarray
    legal_actions: np.ndarray

    def __post_init__(self) -> None:
        n_rows = len(self.actions)
        expected_vectors = {
            "episode_indices": self.episode_indices,
            "player_indices": self.player_indices,
            "goal_colours": self.goal_colours,
            "moves_left": self.moves_left,
            "goals_left": self.goals_left,
        }
        for name, values in expected_vectors.items():
            if np.shape(values) != (n_rows,):
                raise ValueError(f"{name} must align with action rows")
        if np.shape(self.boards) != (n_rows, 64):
            raise ValueError("boards must have shape (rows, 64)")
        if np.shape(self.skills) != (n_rows, 4):
            raise ValueError("skills must have shape (rows, 4)")
        if np.shape(self.legal_actions) != (n_rows, ACTION_SLOTS):
            raise ValueError("legal_actions must have shape (rows, 128)")
        if n_rows and not np.all(
            np.asarray(self.legal_actions)[np.arange(n_rows), self.actions]
        ):
            raise ValueError("dataset contains an action illegal in its current state")

    def __len__(self) -> int:
        return len(self.actions)

    def subset(self, rows: np.ndarray) -> "ActionDataset":
        index = np.asarray(rows, dtype=np.int64)
        return ActionDataset(
            episode_indices=np.asarray(self.episode_indices)[index],
            player_indices=np.asarray(self.player_indices)[index],
            boards=np.asarray(self.boards)[index],
            goal_colours=np.asarray(self.goal_colours)[index],
            moves_left=np.asarray(self.moves_left)[index],
            goals_left=np.asarray(self.goals_left)[index],
            skills=np.asarray(self.skills)[index],
            actions=np.asarray(self.actions)[index],
            legal_actions=np.asarray(self.legal_actions)[index],
        )

    def batch(
        self,
        rows: np.ndarray,
        *,
        device: torch.device,
        include_skill: bool,
    ) -> dict[str, torch.Tensor]:
        index = np.asarray(rows, dtype=np.int64)
        skills = np.asarray(self.skills)[index]
        if not include_skill:
            skills = np.zeros_like(skills)
        return {
            "board": torch.as_tensor(
                np.asarray(self.boards)[index], dtype=torch.long, device=device
            ),
            "goal_colour": torch.as_tensor(
                np.asarray(self.goal_colours)[index], dtype=torch.long, device=device
            ),
            "moves_left": torch.as_tensor(
                np.asarray(self.moves_left)[index], dtype=torch.long, device=device
            ),
            "goals_left": torch.as_tensor(
                np.asarray(self.goals_left)[index], dtype=torch.long, device=device
            ),
            "skill": torch.as_tensor(skills, dtype=torch.float32, device=device),
            "legal_actions": torch.as_tensor(
                np.asarray(self.legal_actions)[index], dtype=torch.bool, device=device
            ),
            "action": torch.as_tensor(
                np.asarray(self.actions)[index], dtype=torch.long, device=device
            ),
        }


@dataclass(frozen=True)
class GameplayTransitionDataset:
    """Logged transition rows grouped into complete gameplay episodes."""

    episode_ids: np.ndarray
    player_ids: np.ndarray
    attempt_ids: np.ndarray
    step_ids: np.ndarray
    boards: np.ndarray
    next_boards: np.ndarray
    actions: np.ndarray
    goal_colours: np.ndarray
    moves_left: np.ndarray
    goals_left: np.ndarray
    next_moves_left: np.ndarray
    next_goals_left: np.ndarray
    levels: np.ndarray
    tiers: np.ndarray
    served_difficulty: np.ndarray
    specials: np.ndarray | None = None
    next_specials: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_rows = len(self.episode_ids)
        if n_rows == 0:
            raise ValueError("transition dataset cannot be empty")
        vectors = {
            "player_ids": self.player_ids,
            "attempt_ids": self.attempt_ids,
            "step_ids": self.step_ids,
            "actions": self.actions,
            "goal_colours": self.goal_colours,
            "moves_left": self.moves_left,
            "goals_left": self.goals_left,
            "next_moves_left": self.next_moves_left,
            "next_goals_left": self.next_goals_left,
            "levels": self.levels,
            "tiers": self.tiers,
            "served_difficulty": self.served_difficulty,
        }
        for name, values in vectors.items():
            if np.shape(values) != (n_rows,):
                raise ValueError(f"{name} must align with transition rows")
        if np.shape(self.boards) != (n_rows, N_CELLS):
            raise ValueError("boards must have shape (rows, 64)")
        if np.shape(self.next_boards) != (n_rows, N_CELLS):
            raise ValueError("next_boards must have shape (rows, 64)")
        for name, values in {
            "specials": self.specials,
            "next_specials": self.next_specials,
        }.items():
            if values is None:
                object.__setattr__(
                    self, name, np.zeros((n_rows, N_CELLS), dtype=np.int64)
                )
            elif np.shape(values) != (n_rows, N_CELLS):
                raise ValueError(f"{name} must have shape (rows, 64)")
        if np.any((self.actions < 0) | (self.actions >= ACTION_SLOTS)):
            raise ValueError("actions contain an index outside the vocabulary")
        if np.any((self.levels < 0) | (self.levels >= len(LEVELS))):
            raise ValueError("levels contain an unknown index")
        if np.any((self.tiers < 0) | (self.tiers >= len(TIER_NAMES))):
            raise ValueError("tiers contain an unknown index")

        for episode_id in self.episodes:
            rows = np.flatnonzero(self.episode_ids == episode_id)
            ordered = rows[np.argsort(self.step_ids[rows])]
            if not np.array_equal(
                self.step_ids[ordered], np.arange(len(ordered))
            ):
                raise ValueError("episode steps must be contiguous and zero-based")
            for name, values in {
                "player_ids": self.player_ids,
                "attempt_ids": self.attempt_ids,
                "levels": self.levels,
                "tiers": self.tiers,
                "served_difficulty": self.served_difficulty,
            }.items():
                if not np.all(values[rows] == values[rows[0]]):
                    raise ValueError(f"{name} must be constant within an episode")

    @property
    def episodes(self) -> np.ndarray:
        return np.unique(self.episode_ids)

    def batch(
        self,
        episode_ids: np.ndarray,
        *,
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, torch.Tensor]:
        """Pad complete selected episodes into the ``GameplayRSSM`` contract."""
        selected = np.asarray(episode_ids, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("episode_ids must be a non-empty vector")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("episode_ids must be unique")
        if not np.all(np.isin(selected, self.episodes)):
            raise ValueError("episode_ids contain an unknown episode")

        episode_rows = []
        for episode_id in selected:
            rows = np.flatnonzero(self.episode_ids == episode_id)
            episode_rows.append(rows[np.argsort(self.step_ids[rows])])
        batch_size = len(episode_rows)
        max_steps = max(map(len, episode_rows))
        boards = np.zeros((batch_size, max_steps, N_CELLS), dtype=np.int64)
        next_boards = np.zeros_like(boards)
        actions = np.zeros((batch_size, max_steps), dtype=np.int64)
        goal_colours = np.zeros_like(actions)
        moves_left = np.zeros_like(actions)
        goals_left = np.zeros_like(actions)
        next_moves_left = np.zeros_like(actions)
        next_goals_left = np.zeros_like(actions)
        step_mask = np.zeros((batch_size, max_steps), dtype=bool)
        levels = np.zeros(batch_size, dtype=np.int64)
        tiers = np.zeros(batch_size, dtype=np.int64)
        served_difficulty = np.zeros(batch_size, dtype=np.float32)

        for batch_index, rows in enumerate(episode_rows):
            steps = len(rows)
            boards[batch_index, :steps] = self.boards[rows]
            next_boards[batch_index, :steps] = self.next_boards[rows]
            actions[batch_index, :steps] = self.actions[rows]
            goal_colours[batch_index, :steps] = self.goal_colours[rows]
            moves_left[batch_index, :steps] = self.moves_left[rows]
            goals_left[batch_index, :steps] = self.goals_left[rows]
            next_moves_left[batch_index, :steps] = self.next_moves_left[rows]
            next_goals_left[batch_index, :steps] = self.next_goals_left[rows]
            step_mask[batch_index, :steps] = True
            levels[batch_index] = self.levels[rows[0]]
            tiers[batch_index] = self.tiers[rows[0]]
            served_difficulty[batch_index] = self.served_difficulty[rows[0]]

        tensor = lambda values, dtype: torch.as_tensor(
            values, dtype=dtype, device=device
        )
        return {
            "boards": tensor(boards, torch.long),
            "next_boards": tensor(next_boards, torch.long),
            "actions": tensor(actions, torch.long),
            "goal_colours": tensor(goal_colours, torch.long),
            "moves_left": tensor(moves_left, torch.long),
            "goals_left": tensor(goals_left, torch.long),
            "next_moves_left": tensor(next_moves_left, torch.long),
            "next_goals_left": tensor(next_goals_left, torch.long),
            "levels": tensor(levels, torch.long),
            "tiers": tensor(tiers, torch.long),
            "served_difficulty": tensor(served_difficulty, torch.float32),
            "step_mask": tensor(step_mask, torch.bool),
        }


def load_gameplay_transition_dataset(
    path: str | Path,
) -> GameplayTransitionDataset:
    """Load the strict logged schema used by the generative world model."""
    required = {
        "schema_version",
        "board_before",
        "board_after",
        "action_index",
        "moves_left",
        "moves_left_next",
        "goals_left",
        "goals_left_next",
        "goal_colour",
        "level",
        "tier",
        "served_difficulty",
        "episode_id",
        "player_id",
        "attempt_id",
        "step_id",
    }
    with np.load(Path(path), allow_pickle=False) as values:
        missing = required - set(values.files)
        if missing:
            raise ValueError(
                "transition artifact lacks required logged fields: "
                + ", ".join(sorted(missing))
            )
        version = int(values["schema_version"][0])
        if version not in (2, 3):
            raise ValueError(f"unsupported transition schema {version}")
        specials = (
            values["specials_before"].reshape(-1, N_CELLS).astype(np.int64)
            if version >= 3
            else np.zeros_like(
                values["board_before"].reshape(-1, N_CELLS), dtype=np.int64
            )
        )
        next_specials = (
            values["specials_after"].reshape(-1, N_CELLS).astype(np.int64)
            if version >= 3
            else np.zeros_like(
                values["board_after"].reshape(-1, N_CELLS), dtype=np.int64
            )
        )
        return GameplayTransitionDataset(
            episode_ids=values["episode_id"].astype(np.int64),
            player_ids=values["player_id"].astype(np.int64),
            attempt_ids=values["attempt_id"].astype(np.int64),
            step_ids=values["step_id"].astype(np.int64),
            boards=values["board_before"].reshape(-1, N_CELLS).astype(np.int64),
            next_boards=values["board_after"].reshape(-1, N_CELLS).astype(
                np.int64
            ),
            actions=values["action_index"].astype(np.int64),
            goal_colours=values["goal_colour"].astype(np.int64),
            moves_left=values["moves_left"].astype(np.int64),
            goals_left=values["goals_left"].astype(np.int64),
            next_moves_left=values["moves_left_next"].astype(np.int64),
            next_goals_left=values["goals_left_next"].astype(np.int64),
            levels=values["level"].astype(np.int64),
            tiers=values["tier"].astype(np.int64),
            served_difficulty=values["served_difficulty"].astype(np.float32),
            specials=specials,
            next_specials=next_specials,
        )


def gameplay_transition_dataset_from_episodes(
    episodes: list[Episode],
    *,
    player_ids: list[int] | None = None,
    attempt_ids: list[int] | None = None,
) -> GameplayTransitionDataset:
    """Convert in-memory episodes to the logged generative sequence schema."""
    if not episodes:
        raise ValueError("at least one episode is required")
    if (player_ids is None) != (attempt_ids is None):
        raise ValueError("player_ids and attempt_ids must be supplied together")
    if player_ids is None:
        player_ids = list(range(len(episodes)))
        attempt_ids = [1] * len(episodes)
    assert attempt_ids is not None
    if len(player_ids) != len(episodes) or len(attempt_ids) != len(episodes):
        raise ValueError("trajectory identifiers must align with episodes")

    level_ids = {level.name: index for index, level in enumerate(LEVELS)}
    tier_ids = {tier: index for index, tier in enumerate(TIER_NAMES)}
    rows: dict[str, list] = {
        name: []
        for name in (
            "episode_ids",
            "player_ids",
            "attempt_ids",
            "step_ids",
            "boards",
            "next_boards",
            "specials",
            "next_specials",
            "actions",
            "goal_colours",
            "moves_left",
            "goals_left",
            "next_moves_left",
            "next_goals_left",
            "levels",
            "tiers",
            "served_difficulty",
        )
    }
    for episode_index, episode in enumerate(episodes):
        for step_index, action in enumerate(episode.actions):
            state = episode.states[step_index]
            next_state = episode.states[step_index + 1]
            rows["episode_ids"].append(episode_index)
            rows["player_ids"].append(player_ids[episode_index])
            rows["attempt_ids"].append(attempt_ids[episode_index])
            rows["step_ids"].append(step_index)
            rows["boards"].append(state.board.ravel())
            rows["next_boards"].append(next_state.board.ravel())
            rows["specials"].append(state.specials.ravel())
            rows["next_specials"].append(next_state.specials.ravel())
            rows["actions"].append(action_to_index(action))
            rows["goal_colours"].append(state.goal_colour)
            rows["moves_left"].append(state.moves_left)
            rows["goals_left"].append(state.goals_left)
            rows["next_moves_left"].append(next_state.moves_left)
            rows["next_goals_left"].append(next_state.goals_left)
            rows["levels"].append(level_ids[episode.level.name])
            rows["tiers"].append(tier_ids[episode.tier])
            rows["served_difficulty"].append(episode.E)
    if not rows["actions"]:
        raise ValueError("episodes contain no observed actions")
    return GameplayTransitionDataset(
        episode_ids=np.asarray(rows["episode_ids"], dtype=np.int64),
        player_ids=np.asarray(rows["player_ids"], dtype=np.int64),
        attempt_ids=np.asarray(rows["attempt_ids"], dtype=np.int64),
        step_ids=np.asarray(rows["step_ids"], dtype=np.int64),
        boards=np.asarray(rows["boards"], dtype=np.int8),
        next_boards=np.asarray(rows["next_boards"], dtype=np.int8),
        specials=np.asarray(rows["specials"], dtype=np.int8),
        next_specials=np.asarray(rows["next_specials"], dtype=np.int8),
        actions=np.asarray(rows["actions"], dtype=np.int64),
        goal_colours=np.asarray(rows["goal_colours"], dtype=np.int64),
        moves_left=np.asarray(rows["moves_left"], dtype=np.int64),
        goals_left=np.asarray(rows["goals_left"], dtype=np.int64),
        next_moves_left=np.asarray(rows["next_moves_left"], dtype=np.int64),
        next_goals_left=np.asarray(rows["next_goals_left"], dtype=np.int64),
        levels=np.asarray(rows["levels"], dtype=np.int64),
        tiers=np.asarray(rows["tiers"], dtype=np.int64),
        served_difficulty=np.asarray(
            rows["served_difficulty"], dtype=np.float32
        ),
    )


def action_dataset_from_episodes(
    episodes: list[Episode],
    *,
    player_indices: list[int] | None = None,
) -> ActionDataset:
    """Flatten observed current states and actions without using next states."""
    if player_indices is None:
        player_indices = list(range(len(episodes)))
    if len(player_indices) != len(episodes):
        raise ValueError("player_indices must align with episodes")
    episode_indices: list[int] = []
    action_player_indices: list[int] = []
    boards: list[np.ndarray] = []
    goal_colours: list[int] = []
    moves_left: list[int] = []
    goals_left: list[int] = []
    skills: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    for episode_index, episode in enumerate(episodes):
        for step, action in enumerate(episode.actions):
            state = episode.states[step]
            episode_indices.append(episode_index)
            action_player_indices.append(player_indices[episode_index])
            boards.append(state.board.reshape(-1))
            goal_colours.append(state.goal_colour)
            moves_left.append(state.moves_left)
            goals_left.append(state.goals_left)
            skills.append(episode.player.as_array())
            actions.append(action_to_index(action))
            masks.append(legal_mask(state.board))
    if not actions:
        raise ValueError("episodes contain no observed actions")
    return ActionDataset(
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        player_indices=np.asarray(action_player_indices, dtype=np.int64),
        boards=np.asarray(boards, dtype=np.int8),
        goal_colours=np.asarray(goal_colours, dtype=np.int64),
        moves_left=np.asarray(moves_left, dtype=np.int64),
        goals_left=np.asarray(goals_left, dtype=np.int64),
        skills=np.asarray(skills, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        legal_actions=np.asarray(masks, dtype=bool),
    )


def action_dataset_from_trajectories(
    trajectories: list[PlayerTrajectory],
) -> ActionDataset:
    """Flatten longitudinal episodes while retaining stable player groups."""
    episodes = [
        record.episode
        for trajectory in trajectories
        for record in trajectory.attempts
    ]
    player_indices = [
        trajectory.player_id
        for trajectory in trajectories
        for _ in trajectory.attempts
    ]
    return action_dataset_from_episodes(
        episodes, player_indices=player_indices
    )


def split_action_dataset_by_episode(
    dataset: ActionDataset,
    *,
    validation_fraction: float = 0.20,
    seed: int = 0,
) -> tuple[ActionDataset, ActionDataset]:
    """Split complete episodes so no trajectory contributes to both sets."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    episodes = np.unique(dataset.episode_indices)
    if len(episodes) < 2:
        raise ValueError("at least two episodes are required for a split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    n_validation = min(
        len(episodes) - 1,
        max(1, int(round(validation_fraction * len(episodes)))),
    )
    validation_episodes = shuffled[:n_validation]
    validation_mask = np.isin(dataset.episode_indices, validation_episodes)
    return (
        dataset.subset(np.flatnonzero(~validation_mask)),
        dataset.subset(np.flatnonzero(validation_mask)),
    )


def split_action_dataset_by_player(
    dataset: ActionDataset,
    *,
    validation_fraction: float = 0.20,
    seed: int = 0,
) -> tuple[ActionDataset, ActionDataset]:
    """Split all episodes from a player into exactly one partition."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    players = np.unique(dataset.player_indices)
    if len(players) < 2:
        raise ValueError("at least two players are required for a split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(players)
    n_validation = min(
        len(players) - 1,
        max(1, int(round(validation_fraction * len(players)))),
    )
    validation_players = shuffled[:n_validation]
    validation_mask = np.isin(dataset.player_indices, validation_players)
    return (
        dataset.subset(np.flatnonzero(~validation_mask)),
        dataset.subset(np.flatnonzero(validation_mask)),
    )


def split_player_trajectories(
    trajectories: list[PlayerTrajectory],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[
    list[PlayerTrajectory],
    list[PlayerTrajectory],
    list[PlayerTrajectory],
]:
    """Create deterministic player-disjoint train, validation, and test sets."""
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    if len(trajectories) < 3:
        raise ValueError("at least three player trajectories are required")
    player_ids = [trajectory.player_id for trajectory in trajectories]
    if len(set(player_ids)) != len(player_ids):
        raise ValueError("player trajectory IDs must be unique")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(player_ids)
    n_validation = max(1, int(round(validation_fraction * len(shuffled))))
    n_test = max(1, int(round(test_fraction * len(shuffled))))
    while n_validation + n_test >= len(shuffled):
        if n_validation >= n_test and n_validation > 1:
            n_validation -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError("fractions leave no training trajectories")
    validation_ids = set(shuffled[:n_validation].tolist())
    test_ids = set(shuffled[n_validation : n_validation + n_test].tolist())
    train = [
        trajectory
        for trajectory in trajectories
        if trajectory.player_id not in validation_ids | test_ids
    ]
    validation = [
        trajectory
        for trajectory in trajectories
        if trajectory.player_id in validation_ids
    ]
    test = [
        trajectory
        for trajectory in trajectories
        if trajectory.player_id in test_ids
    ]
    return train, validation, test


__all__ = [
    "ActionDataset",
    "GameplayTransitionDataset",
    "action_dataset_from_episodes",
    "action_dataset_from_trajectories",
    "gameplay_transition_dataset_from_episodes",
    "load_gameplay_transition_dataset",
    "split_action_dataset_by_episode",
    "split_action_dataset_by_player",
    "split_player_trajectories",
]