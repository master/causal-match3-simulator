"""Experienced-mastery and churn mechanisms for longitudinal trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from .spec import BENCHMARK_CONFIG, BenchmarkConfig, SKILL_NAMES, PlayerSkill


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -array))


@dataclass(frozen=True)
class MasteryConfig:
    """Experienced-mastery state carried between player attempts."""

    initial: float = 0.35
    update_rate: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 <= self.initial <= 1.0:
            raise ValueError("initial mastery must lie in [0, 1]")
        if not 0.0 < self.update_rate <= 1.0:
            raise ValueError("update_rate must lie in (0, 1]")


@dataclass(frozen=True)
class ChurnConfig:
    """Mastery and current-challenge mismatch hazard configuration."""

    intercept: float = -14.0
    deviation_coefficient: float = 512.0
    overchallenge_deviation_coefficient: float | None = None
    mastery_target: float = 0.35
    margin_deviation_coefficient: float = 0.0
    margin_overchallenge_deviation_coefficient: float | None = None
    margin_target: float = 0.0

    def __post_init__(self) -> None:
        if self.deviation_coefficient <= 0:
            raise ValueError("deviation_coefficient must be positive")
        if (
            self.overchallenge_deviation_coefficient is not None
            and self.overchallenge_deviation_coefficient <= 0
        ):
            raise ValueError(
                "overchallenge_deviation_coefficient must be positive"
            )
        if not 0.0 < self.mastery_target < 1.0:
            raise ValueError("mastery_target must lie in (0, 1)")
        if self.margin_deviation_coefficient < 0:
            raise ValueError("margin_deviation_coefficient must be non-negative")
        if (
            self.margin_overchallenge_deviation_coefficient is not None
            and self.margin_overchallenge_deviation_coefficient < 0
        ):
            raise ValueError(
                "margin_overchallenge_deviation_coefficient must be non-negative"
            )
        if not -1.0 <= self.margin_target <= 1.0:
            raise ValueError("margin_target must lie in [-1, 1]")


@dataclass(frozen=True)
class ChurnSchedule:
    """Level-specific mismatch sensitivity with shared mastery target."""

    level_names: tuple[str, ...] = ("orchard", "harbour", "foundry")
    intercepts: tuple[float, ...] = (-14.0, -10.0, -16.0)
    deviation_coefficients: tuple[float, ...] = (512.0, 128.0, 192.0)
    overchallenge_deviation_coefficients: tuple[float, ...] | None = None
    mastery_target: float = 0.35
    margin_deviation_coefficients: tuple[float, ...] = (0.0, 0.0, 0.0)
    margin_overchallenge_deviation_coefficients: tuple[float, ...] | None = None
    margin_targets: tuple[float, ...] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        n_levels = len(self.level_names)
        if len(set(self.level_names)) != n_levels:
            raise ValueError("level_names must be unique")
        if len(self.intercepts) != n_levels:
            raise ValueError("intercepts must align with levels")
        if len(self.deviation_coefficients) != n_levels:
            raise ValueError("deviation coefficients must align with levels")
        if any(value <= 0 for value in self.deviation_coefficients):
            raise ValueError("deviation coefficients must be positive")
        if (
            self.overchallenge_deviation_coefficients is not None
            and len(self.overchallenge_deviation_coefficients) != n_levels
        ):
            raise ValueError("overchallenge deviation coefficients must align with levels")
        if self.overchallenge_deviation_coefficients is not None and any(
            value <= 0 for value in self.overchallenge_deviation_coefficients
        ):
            raise ValueError("overchallenge deviation coefficients must be positive")
        if not 0.0 < self.mastery_target < 1.0:
            raise ValueError("mastery_target must lie in (0, 1)")
        if len(self.margin_deviation_coefficients) != n_levels:
            raise ValueError("margin deviation coefficients must align with levels")
        if any(value < 0 for value in self.margin_deviation_coefficients):
            raise ValueError("margin deviation coefficients must be non-negative")
        if (
            self.margin_overchallenge_deviation_coefficients is not None
            and len(self.margin_overchallenge_deviation_coefficients) != n_levels
        ):
            raise ValueError(
                "margin overchallenge deviation coefficients must align with levels"
            )
        if self.margin_overchallenge_deviation_coefficients is not None and any(
            value < 0
            for value in self.margin_overchallenge_deviation_coefficients
        ):
            raise ValueError(
                "margin overchallenge deviation coefficients must be non-negative"
            )
        if len(self.margin_targets) != n_levels:
            raise ValueError("margin targets must align with levels")
        if any(not -1.0 <= value <= 1.0 for value in self.margin_targets):
            raise ValueError("margin targets must lie in [-1, 1]")

    def for_level(self, level_name: str) -> ChurnConfig:
        index = self.level_names.index(level_name)
        return ChurnConfig(
            intercept=self.intercepts[index],
            deviation_coefficient=self.deviation_coefficients[index],
            overchallenge_deviation_coefficient=(
                None
                if self.overchallenge_deviation_coefficients is None
                else self.overchallenge_deviation_coefficients[index]
            ),
            mastery_target=self.mastery_target,
            margin_deviation_coefficient=self.margin_deviation_coefficients[index],
            margin_overchallenge_deviation_coefficient=(
                None
                if self.margin_overchallenge_deviation_coefficients is None
                else self.margin_overchallenge_deviation_coefficients[index]
            ),
            margin_target=self.margin_targets[index],
        )


CHURN_SCHEDULE = ChurnSchedule()


@dataclass(frozen=True)
class WinPropensityModel:
    """Frozen logistic response surface calibrated against engine rollouts."""

    level_names: tuple[str, ...]
    tier_names: tuple[str, ...]
    intercepts: tuple[tuple[float, ...], ...]
    skill_coefficients: tuple[tuple[float, ...], ...]
    difficulty_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        n_levels = len(self.level_names)
        n_tiers = len(self.tier_names)
        if len(set(self.level_names)) != n_levels:
            raise ValueError("level_names must be unique")
        if len(set(self.tier_names)) != n_tiers:
            raise ValueError("tier_names must be unique")
        if np.shape(self.intercepts) != (n_levels, n_tiers):
            raise ValueError("intercepts must have shape (levels, tiers)")
        if np.shape(self.skill_coefficients) != (n_levels, len(SKILL_NAMES)):
            raise ValueError("skill_coefficients must have shape (levels, skills)")
        if np.shape(self.difficulty_coefficients) != (n_levels,):
            raise ValueError("difficulty_coefficients must have shape (levels,)")
        if np.any(np.asarray(self.skill_coefficients) < 0):
            raise ValueError("skill coefficients must be non-negative")
        if np.any(np.asarray(self.difficulty_coefficients) <= 0):
            raise ValueError("difficulty coefficients must be positive")

    def logits(
        self,
        level_name: str,
        tier_indices: np.ndarray,
        skills: np.ndarray,
        served_difficulty: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate response-surface logits for one level."""
        level_index = self.level_names.index(level_name)
        tiers = np.asarray(tier_indices, dtype=np.int64)
        skill_array = np.asarray(skills, dtype=np.float64)
        if skill_array.ndim != 2 or skill_array.shape[1] != len(SKILL_NAMES):
            raise ValueError("skills must have shape (players, skills)")
        if tiers.shape != (skill_array.shape[0],):
            raise ValueError("tier_indices must align with skills")
        if np.any((tiers < 0) | (tiers >= len(self.tier_names))):
            raise ValueError("tier index outside model support")
        served = np.asarray(served_difficulty, dtype=np.float64)
        if served.ndim > 1 or (served.ndim == 1 and served.shape != tiers.shape):
            raise ValueError("served_difficulty must be scalar or align with skills")

        intercept = np.asarray(self.intercepts[level_index])[tiers]
        coefficient = np.asarray(self.skill_coefficients[level_index])
        return (
            intercept
            + skill_array @ coefficient
            - self.difficulty_coefficients[level_index] * served
        )

    def probabilities(
        self,
        level_name: str,
        tier_indices: np.ndarray,
        skills: np.ndarray,
        served_difficulty: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate oracle win propensity for one level."""
        return _sigmoid(
            self.logits(level_name, tier_indices, skills, served_difficulty)
        )

    def probability(
        self,
        level_name: str,
        tier_index: int,
        skill: PlayerSkill,
        served_difficulty: float,
    ) -> float:
        values = self.probabilities(
            level_name,
            np.asarray([tier_index]),
            skill.as_array()[None, :],
            served_difficulty,
        )
        return float(values[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "level_names": list(self.level_names),
            "tier_names": list(self.tier_names),
            "intercepts": [list(row) for row in self.intercepts],
            "skill_coefficients": [list(row) for row in self.skill_coefficients],
            "difficulty_coefficients": list(self.difficulty_coefficients),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "WinPropensityModel":
        return cls(
            level_names=tuple(str(value) for value in values["level_names"]),
            tier_names=tuple(str(value) for value in values["tier_names"]),
            intercepts=tuple(
                tuple(float(value) for value in row)
                for row in values["intercepts"]
            ),
            skill_coefficients=tuple(
                tuple(float(value) for value in row)
                for row in values["skill_coefficients"]
            ),
            difficulty_coefficients=tuple(
                float(value) for value in values["difficulty_coefficients"]
            ),
        )


@dataclass(frozen=True)
class AttemptRecord:
    """One ordered episode plus its post-episode churn decision."""

    player_id: int
    attempt_id: int
    episode: object
    mastery_before: float
    mastery_after: float
    completion_margin: float
    win_probability: float
    churn_probability: float
    churn_after: int


@dataclass(frozen=True)
class PlayerTrajectory:
    """Ordered attempts sharing one player-level skill draw."""

    player_id: int
    player: PlayerSkill
    attempts: tuple[AttemptRecord, ...]

    @property
    def churned(self) -> bool:
        return bool(self.attempts and self.attempts[-1].churn_after)

    @property
    def churn_attempt(self) -> int | None:
        return self.attempts[-1].attempt_id if self.churned else None


def update_mastery(
    mastery_before: np.ndarray | float,
    outcome: np.ndarray | int,
    config: MasteryConfig = MasteryConfig(),
) -> np.ndarray | float:
    """Update experienced mastery from a realized binary completion outcome."""
    mastery = np.asarray(mastery_before, dtype=np.float64)
    realized = np.asarray(outcome, dtype=np.float64)
    if np.any((mastery < 0.0) | (mastery > 1.0)):
        raise ValueError("mastery_before must lie in [0, 1]")
    if np.any((realized != 0.0) & (realized != 1.0)):
        raise ValueError("outcome must be binary")
    updated = mastery + config.update_rate * (realized - mastery)
    updated = np.clip(updated, 0.0, 1.0)
    return float(updated) if updated.ndim == 0 else updated


def completion_margin(episode: object) -> float:
    """Return signed distance from failure using unused moves or unmet quota."""
    terminal_state = episode.states[-1]
    if episode.R:
        margin = terminal_state.moves_left / episode.difficulty.move_budget
    else:
        margin = -terminal_state.goals_left / episode.served_goal_count
    if not -1.0 <= margin <= 1.0:
        raise ValueError("completion margin must lie in [-1, 1]")
    return float(margin)


def mastery_mismatch_hazard(
    mastery_after: np.ndarray | float,
    config: ChurnConfig = ChurnConfig(),
    *,
    completion_margin: np.ndarray | float | None = None,
) -> np.ndarray:
    """Return churn probability from longitudinal and current challenge."""
    mastery = np.asarray(mastery_after, dtype=np.float64)
    if np.any((mastery < 0.0) | (mastery > 1.0)):
        raise ValueError("mastery_after must lie in [0, 1]")
    mastery_deviation = mastery - config.mastery_target
    mastery_coefficient = np.where(
        mastery_deviation < 0.0,
        (
            config.deviation_coefficient
            if config.overchallenge_deviation_coefficient is None
            else config.overchallenge_deviation_coefficient
        ),
        config.deviation_coefficient,
    )
    logit = config.intercept + mastery_coefficient * mastery_deviation**2
    if completion_margin is None:
        if config.margin_deviation_coefficient:
            raise ValueError("completion_margin is required by churn config")
    else:
        margin = np.asarray(completion_margin, dtype=np.float64)
        if np.any((margin < -1.0) | (margin > 1.0)):
            raise ValueError("completion_margin must lie in [-1, 1]")
        margin_deviation = margin - config.margin_target
        margin_coefficient = np.where(
            margin_deviation < 0.0,
            (
                config.margin_deviation_coefficient
                if config.margin_overchallenge_deviation_coefficient is None
                else config.margin_overchallenge_deviation_coefficient
            ),
            config.margin_deviation_coefficient,
        )
        logit = logit + margin_coefficient * margin_deviation**2
    return _sigmoid(logit)


def expected_mastery_churn(
    win_probability: np.ndarray | float,
    mastery_before: np.ndarray | float,
    *,
    mastery_config: MasteryConfig = MasteryConfig(),
    churn_config: ChurnConfig = ChurnConfig(),
) -> np.ndarray:
    """Integrate post-attempt churn over a realized binary outcome."""
    probability = np.asarray(win_probability, dtype=np.float64)
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("win_probability must lie in [0, 1]")
    mastery = np.asarray(mastery_before, dtype=np.float64)
    after_win = update_mastery(mastery, np.ones_like(mastery), mastery_config)
    after_loss = update_mastery(mastery, np.zeros_like(mastery), mastery_config)
    return (
        probability * mastery_mismatch_hazard(after_win, churn_config)
        + (1.0 - probability) * mastery_mismatch_hazard(after_loss, churn_config)
    )


def sample_C(
    mastery_after: float,
    *,
    completion_margin: float | None = None,
    active: bool = True,
    hazard_scale: float = 1.0,
    config: ChurnConfig = ChurnConfig(),
    name: str = "C",
    value: int | None = None,
) -> int:
    """Sample absorbing churn status after an attempt."""
    if not 0.0 <= hazard_scale <= 1.0:
        raise ValueError("hazard_scale must lie in [0, 1]")
    if not active:
        return int(pyro.deterministic(name, torch.tensor(1)))
    hazard = hazard_scale * float(
        mastery_mismatch_hazard(
            mastery_after,
            config,
            completion_margin=completion_margin,
        )
    )
    if value is not None:
        outcome = pyro.deterministic(name, torch.tensor(int(value)))
    else:
        outcome = pyro.sample(name, dist.Bernoulli(torch.tensor(hazard)))
    return int(outcome)


def simulate_player_trajectory(
    propensity_model: WinPropensityModel,
    *,
    player_id: int,
    seed: int,
    max_attempts: int = 30,
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
    churn_config: ChurnConfig | ChurnSchedule = CHURN_SCHEDULE,
    mastery_config: MasteryConfig = MasteryConfig(),
    player: PlayerSkill | None = None,
    dda_gains: tuple[float, ...] | None = None,
    e_sigmas: tuple[float, ...] | None = None,
) -> PlayerTrajectory:
    """Simulate ordered attempts with mastery updates and absorbing churn."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    from .scm import TIER_NAMES, ground_truth_model, sample_K

    skill_seed = int(
        np.random.SeedSequence([seed, player_id, 0]).generate_state(1)[0]
    )
    pyro.set_rng_seed(skill_seed)
    skill = sample_K(value=player)
    attempts: list[AttemptRecord] = []
    mastery = mastery_config.initial
    for attempt_id in range(1, max_attempts + 1):
        episode_seed = int(
            np.random.SeedSequence([seed, player_id, attempt_id]).generate_state(1)[0]
        )
        pyro.set_rng_seed(episode_seed)
        episode = ground_truth_model(
            player=skill,
            dda_gains=dda_gains,
            e_sigmas=e_sigmas,
        )
        tier_index = TIER_NAMES.index(episode.tier)
        win_probability = propensity_model.probability(
            episode.level.name,
            tier_index,
            skill,
            episode.E,
        )
        mastery_before = mastery
        mastery_after = update_mastery(mastery_before, episode.R, mastery_config)
        margin = completion_margin(episode)
        current_churn_config = (
            churn_config.for_level(episode.level.name)
            if hasattr(churn_config, "for_level")
            else churn_config
        )
        hazard_scale = (
            benchmark.warmup_churn_scale
            if attempt_id < benchmark.landmark_attempt
            else 1.0
        )
        churn_probability = hazard_scale * float(
            mastery_mismatch_hazard(
                mastery_after,
                current_churn_config,
                completion_margin=margin,
            )
        )
        churn_after = sample_C(
            mastery_after,
            completion_margin=margin,
            hazard_scale=hazard_scale,
            config=current_churn_config,
            name=f"C/{attempt_id}",
        )
        attempts.append(
            AttemptRecord(
                player_id=player_id,
                attempt_id=attempt_id,
                episode=episode,
                mastery_before=mastery_before,
                mastery_after=mastery_after,
                completion_margin=margin,
                win_probability=win_probability,
                churn_probability=churn_probability,
                churn_after=churn_after,
            )
        )
        mastery = mastery_after
        if churn_after:
            break
    return PlayerTrajectory(player_id, skill, tuple(attempts))


__all__ = [
    "AttemptRecord",
    "CHURN_SCHEDULE",
    "ChurnConfig",
    "ChurnSchedule",
    "completion_margin",
    "expected_mastery_churn",
    "MasteryConfig",
    "PlayerTrajectory",
    "WinPropensityModel",
    "mastery_mismatch_hazard",
    "sample_C",
    "simulate_player_trajectory",
    "update_mastery",
]