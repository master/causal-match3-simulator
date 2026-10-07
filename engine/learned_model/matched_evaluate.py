"""Fixed-context causal response curves for matched generative model arms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import numpy as np
import torch

from ..calibrate import load_win_propensity_model
from ..retention import MasteryConfig, simulate_player_trajectory
from ..scm import LEVELS, TIER_NAMES, TIER_PROBS
from ..spec import BENCHMARK_CONFIG
from .action_policy import ActionPolicyConfig
from .arms import GenerativeModelConfig, GenerativeWorldModel, ModelArm
from .batching import build_prefix_target_batch
from .data import split_player_trajectories
from .encoder import PrefixEncoderConfig
from .generative import GameplayRSSMConfig
from .matched_experiment import MatchedGenerativeTrainConfig
from .rollout import rollout_learned_dynamics, sample_learned_initial_state


def load_generative_world_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device = torch.device("cpu"),
    expected_arm: ModelArm | None = None,
) -> GenerativeWorldModel:
    """Reconstruct a matched arm from a smoke or production checkpoint."""
    payload = torch.load(
        Path(path), map_location=device, weights_only=False
    )
    if payload.get("schema_version") not in (1, 2):
        raise ValueError("unsupported matched checkpoint schema")
    checkpoint_arm = ModelArm(payload["arm"])
    if expected_arm is not None and checkpoint_arm is not ModelArm(expected_arm):
        raise ValueError(
            f"checkpoint arm {checkpoint_arm.value} does not match "
            f"expected arm {ModelArm(expected_arm).value}"
        )
    raw_config = payload.get("model_config", payload.get("config"))
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint lacks a matched model configuration")
    config = GenerativeModelConfig(
        dynamics=GameplayRSSMConfig(**raw_config["dynamics"]),
        prefix=PrefixEncoderConfig(**raw_config["prefix"]),
        behavior=ActionPolicyConfig(**raw_config["behavior"]),
        mastery=MasteryConfig(**raw_config.get("mastery", {})),
    )
    state_dict = payload.get("model_state_dict", payload.get("state_dict"))
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint lacks a model state dictionary")
    model = GenerativeWorldModel(checkpoint_arm, config)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "churn_head.raw_overchallenge_deviation",
        "churn_head.raw_margin_deviation",
        "churn_head.raw_margin_overchallenge_deviation",
        "churn_head.raw_margin_target",
    }
    if set(incompatible.missing_keys) - allowed_missing:
        raise ValueError(
            "checkpoint lacks required model parameters: "
            + ", ".join(sorted(set(incompatible.missing_keys) - allowed_missing))
        )
    if incompatible.unexpected_keys:
        raise ValueError(
            "checkpoint contains unexpected model parameters: "
            + ", ".join(sorted(incompatible.unexpected_keys))
        )
    if "churn_head.raw_margin_deviation" in incompatible.missing_keys:
        with torch.no_grad():
            model.churn_head.raw_margin_deviation.fill_(-20.0)
    with torch.no_grad():
        if "churn_head.raw_overchallenge_deviation" in incompatible.missing_keys:
            model.churn_head.raw_overchallenge_deviation.copy_(
                model.churn_head.raw_deviation
            )
        if (
            "churn_head.raw_margin_overchallenge_deviation"
            in incompatible.missing_keys
        ):
            model.churn_head.raw_margin_overchallenge_deviation.copy_(
                model.churn_head.raw_margin_deviation
            )
    return model.to(device=device, dtype=torch.float32).eval()


def _model_device(model: GenerativeWorldModel) -> torch.device:
    return next(model.parameters()).device


def _prefix_to_device(
    prefix: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in prefix.items()}


@torch.no_grad()
def induced_response_curves(
    model: GenerativeWorldModel,
    prefix: Mapping[str, torch.Tensor],
    *,
    mastery_before: torch.Tensor | np.ndarray,
    oracle_skill: torch.Tensor | np.ndarray | None = None,
    e_grid: Sequence[float] = BENCHMARK_CONFIG.e_grid,
    tier_probabilities: Sequence[float] = TIER_PROBS,
) -> dict[str, object]:
    """Infer player context once and sweep a next-attempt intervention on E."""
    grid = np.asarray(e_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("e_grid must be a non-empty finite vector")
    if len(np.unique(grid)) != len(grid):
        raise ValueError("e_grid values must be unique")
    tier_weights = np.asarray(tier_probabilities, dtype=np.float64)
    if tier_weights.shape != (len(TIER_NAMES),) or np.any(tier_weights < 0):
        raise ValueError("tier_probabilities must align with simulator tiers")
    if not np.isclose(tier_weights.sum(), 1.0):
        raise ValueError("tier_probabilities must sum to one")

    device = _model_device(model)
    model.eval()
    device_prefix = _prefix_to_device(prefix, device)
    batch_size = device_prefix["episode_mask"].shape[0]
    mastery = torch.as_tensor(
        mastery_before,
        dtype=model.behavior.skill.weight.dtype,
        device=device,
    )
    if mastery.shape != (batch_size,):
        raise ValueError("mastery_before must align with the prefix player batch")
    oracle = None
    if oracle_skill is not None:
        oracle = torch.as_tensor(
            oracle_skill,
            dtype=model.behavior.skill.weight.dtype,
            device=device,
        )
        if oracle.shape != (batch_size, model.config.prefix.skill_dimensions):
            raise ValueError("oracle_skill must align with the prefix player batch")

    context = model.player_context(
        prefix=device_prefix,
        oracle_skill=oracle,
        sample=False,
    )
    n_tiers = len(TIER_NAMES)
    expanded_skill = context.skill.repeat_interleave(n_tiers, dim=0)
    expanded_mastery = mastery.repeat_interleave(n_tiers)
    tiers = torch.arange(n_tiers, device=device).repeat(batch_size)
    weights = torch.as_tensor(
        tier_weights,
        dtype=expanded_skill.dtype,
        device=device,
    )
    update_rate = model.config.mastery.update_rate
    mastery_after_win = expanded_mastery + update_rate * (
        1.0 - expanded_mastery
    )
    mastery_after_loss = expanded_mastery - update_rate * expanded_mastery

    levels: dict[str, object] = {}
    for level_index, level in enumerate(LEVELS):
        level_indices = torch.full_like(tiers, level_index)
        churn_after_win = model.churn_head.probabilities(
            mastery_after_win, level_indices
        )
        churn_after_loss = model.churn_head.probabilities(
            mastery_after_loss, level_indices
        )
        rows = []
        for served_difficulty in grid:
            served = torch.full(
                (batch_size * n_tiers,),
                float(served_difficulty),
                dtype=expanded_skill.dtype,
                device=device,
            )
            win = model.win_head.probabilities(
                served,
                expanded_skill,
                level_indices,
                tiers,
            ).reshape(batch_size, n_tiers)
            churn = (
                win.reshape(-1) * churn_after_win
                + (1.0 - win.reshape(-1)) * churn_after_loss
            ).reshape(batch_size, n_tiers)
            rows.append(
                {
                    "e": float(served_difficulty),
                    "win_probability": float((win * weights).sum(-1).mean()),
                    "churn_probability": float(
                        (churn * weights).sum(-1).mean()
                    ),
                }
            )
        recommendation_index = int(
            np.argmin([row["churn_probability"] for row in rows])
        )
        levels[level.name] = {
            "rows": rows,
            "recommended_e": rows[recommendation_index]["e"],
        }

    return {
        "schema_version": 1,
        "query": "structural_g_computation",
        "fixed_player_context": True,
        "completion_margin": "held_at_learned_target",
        "arm": model.arm.value,
        "n_players": batch_size,
        "e_grid": grid.tolist(),
        "tier_probabilities": tier_weights.tolist(),
        "levels": levels,
    }


def evaluate_matched_response_curves(
    models: Mapping[ModelArm, GenerativeWorldModel],
    prefix: Mapping[str, torch.Tensor],
    *,
    mastery_before: torch.Tensor | np.ndarray,
    oracle_skill: torch.Tensor | np.ndarray,
    e_grid: Sequence[float] = BENCHMARK_CONFIG.e_grid,
    tier_probabilities: Sequence[float] = TIER_PROBS,
) -> dict[str, object]:
    """Evaluate the same fixed-player intervention sweep for all four arms."""
    missing = set(ModelArm) - set(models)
    if missing:
        raise ValueError(
            "models must contain every arm: "
            + ", ".join(sorted(arm.value for arm in missing))
        )
    reports = {
        arm.value: induced_response_curves(
            models[arm],
            prefix,
            mastery_before=mastery_before,
            oracle_skill=oracle_skill if arm is ModelArm.ORACLE else None,
            e_grid=e_grid,
            tier_probabilities=tier_probabilities,
        )
        for arm in ModelArm
    }
    return {
        "schema_version": 1,
        "query": "matched_structural_g_computation",
        "e_grid": list(map(float, e_grid)),
        "arms": reports,
    }


@torch.no_grad()
def imagine_response_curves(
    model: GenerativeWorldModel,
    prefix: Mapping[str, torch.Tensor],
    *,
    mastery_before: torch.Tensor | np.ndarray,
    oracle_skill: torch.Tensor | np.ndarray | None = None,
    e_grid: Sequence[float] = BENCHMARK_CONFIG.e_grid,
    tier_probabilities: Sequence[float] = TIER_PROBS,
    rollouts_per_player: int = 1,
    max_steps: int | None = None,
    stochastic: bool = True,
    seed: int = 0,
) -> dict[str, object]:
    """Estimate churn curves through learned S0, policy, and RSSM dynamics."""
    grid = np.asarray(e_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("e_grid must be a non-empty finite vector")
    if len(np.unique(grid)) != len(grid):
        raise ValueError("e_grid values must be unique")
    if rollouts_per_player < 1:
        raise ValueError("rollouts_per_player must be positive")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    tier_weights = np.asarray(tier_probabilities, dtype=np.float64)
    if tier_weights.shape != (len(TIER_NAMES),) or np.any(tier_weights < 0):
        raise ValueError("tier_probabilities must align with simulator tiers")
    if not np.isclose(tier_weights.sum(), 1.0):
        raise ValueError("tier_probabilities must sum to one")

    device = _model_device(model)
    model.eval()
    device_prefix = _prefix_to_device(prefix, device)
    batch_size = device_prefix["episode_mask"].shape[0]
    mastery = torch.as_tensor(
        mastery_before,
        dtype=model.behavior.skill.weight.dtype,
        device=device,
    )
    if mastery.shape != (batch_size,):
        raise ValueError("mastery_before must align with the prefix player batch")
    oracle = None
    if oracle_skill is not None:
        oracle = torch.as_tensor(
            oracle_skill,
            dtype=model.behavior.skill.weight.dtype,
            device=device,
        )
        if oracle.shape != (batch_size, model.config.prefix.skill_dimensions):
            raise ValueError("oracle_skill must align with the prefix player batch")
    context = model.player_context(
        prefix=device_prefix,
        oracle_skill=oracle,
        sample=False,
    )

    levels: dict[str, object] = {}
    for level_index, level in enumerate(LEVELS):
        rows = []
        for served_difficulty in grid:
            tier_win = np.zeros((batch_size, len(TIER_NAMES)), dtype=np.float64)
            tier_churn = np.zeros_like(tier_win)
            for tier_index in range(len(TIER_NAMES)):
                for player_index in range(batch_size):
                    outcomes = []
                    churn_probabilities = []
                    for replicate in range(rollouts_per_player):
                        stream_seeds = np.random.SeedSequence(
                            [seed, level_index, tier_index, player_index, replicate]
                        ).generate_state(2)
                        initial_seed = int(stream_seeds[0])
                        dynamics_seed = int(stream_seeds[1])
                        initial_state = sample_learned_initial_state(
                            model,
                            level_index=level_index,
                            tier_index=tier_index,
                            served_difficulty=float(served_difficulty),
                            stochastic=stochastic,
                            seed=initial_seed,
                        )
                        rollout = rollout_learned_dynamics(
                            model,
                            initial_state,
                            level_index=level_index,
                            tier_index=tier_index,
                            served_difficulty=float(served_difficulty),
                            player_context=context.skill[player_index],
                            max_steps=max_steps,
                            stochastic=stochastic,
                            seed=dynamics_seed,
                        )
                        outcome = rollout.outcome
                        mastery_after = mastery[player_index] + (
                            model.config.mastery.update_rate
                            * (outcome - mastery[player_index])
                        )
                        churn_probability = model.churn_head.probabilities(
                            mastery_after.reshape(1),
                            torch.tensor(
                                [level_index], dtype=torch.long, device=device
                            ),
                            completion_margin=torch.tensor(
                                [rollout.completion_margin],
                                dtype=mastery_after.dtype,
                                device=device,
                            ),
                        )[0]
                        outcomes.append(outcome)
                        churn_probabilities.append(float(churn_probability))
                    tier_win[player_index, tier_index] = np.mean(outcomes)
                    tier_churn[player_index, tier_index] = np.mean(
                        churn_probabilities
                    )
            rows.append(
                {
                    "e": float(served_difficulty),
                    "win_probability": float(
                        np.mean(tier_win @ tier_weights)
                    ),
                    "churn_probability": float(
                        np.mean(tier_churn @ tier_weights)
                    ),
                    "n_rollouts": int(
                        batch_size
                        * len(TIER_NAMES)
                        * rollouts_per_player
                    ),
                }
            )
        recommendation_index = int(
            np.argmin([row["churn_probability"] for row in rows])
        )
        levels[level.name] = {
            "rows": rows,
            "recommended_e": rows[recommendation_index]["e"],
        }
    return {
        "schema_version": 1,
        "query": "free_running_rssm_g_computation",
        "initial_state": "learned_task_decoder",
        "dynamics": "s0_posterior_then_rssm_prior",
        "fixed_player_context": True,
        "arm": model.arm.value,
        "n_players": batch_size,
        "e_grid": grid.tolist(),
        "tier_probabilities": tier_weights.tolist(),
        "rollouts_per_player": rollouts_per_player,
        "stochastic": stochastic,
        "common_random_numbers_across_e": True,
        "seed": seed,
        "levels": levels,
    }


def evaluate_matched_imagination_curves(
    models: Mapping[ModelArm, GenerativeWorldModel],
    prefix: Mapping[str, torch.Tensor],
    *,
    mastery_before: torch.Tensor | np.ndarray,
    oracle_skill: torch.Tensor | np.ndarray,
    e_grid: Sequence[float] = BENCHMARK_CONFIG.e_grid,
    tier_probabilities: Sequence[float] = TIER_PROBS,
    rollouts_per_player: int = 1,
    max_steps: int | None = None,
    stochastic: bool = True,
    seed: int = 0,
) -> dict[str, object]:
    """Run the same learned-dynamics intervention query for all four arms."""
    missing = set(ModelArm) - set(models)
    if missing:
        raise ValueError(
            "models must contain every arm: "
            + ", ".join(sorted(arm.value for arm in missing))
        )
    reports = {
        arm.value: imagine_response_curves(
            models[arm],
            prefix,
            mastery_before=mastery_before,
            oracle_skill=oracle_skill if arm is ModelArm.ORACLE else None,
            e_grid=e_grid,
            tier_probabilities=tier_probabilities,
            rollouts_per_player=rollouts_per_player,
            max_steps=max_steps,
            stochastic=stochastic,
            seed=seed,
        )
        for arm in ModelArm
    }
    return {
        "schema_version": 1,
        "query": "matched_free_running_rssm_g_computation",
        "e_grid": list(map(float, e_grid)),
        "rollouts_per_player": rollouts_per_player,
        "stochastic": stochastic,
        "common_random_numbers_across_e": True,
        "seed": seed,
        "arms": reports,
    }


def evaluate_matched_experiment_directory(
    experiment_dir: str | Path,
    *,
    target_attempt: int | None = None,
    e_grid: Sequence[float] = BENCHMARK_CONFIG.e_grid,
    rollouts_per_player: int = 1,
    max_steps: int | None = None,
    max_players: int | None = None,
    stochastic: bool = True,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, object]:
    """Reload best arms and evaluate their reconstructed held-out test prefix."""
    directory = Path(experiment_dir)
    training_report = json.loads((directory / "metrics.json").read_text())
    raw_config = dict(training_report["config"])
    for name in ("target_attempts", "dda_gains", "e_sigmas"):
        if name in raw_config:
            raw_config[name] = tuple(raw_config[name])
    config = MatchedGenerativeTrainConfig(**raw_config)
    query_attempt = (
        max(config.target_attempts)
        if target_attempt is None
        else target_attempt
    )
    if not 1 <= query_attempt <= config.max_attempts:
        raise ValueError("target_attempt must lie within the simulated history")
    if max_players is not None and max_players < 1:
        raise ValueError("max_players must be positive")

    propensity_model = (
        load_win_propensity_model()
        if config.propensity_path is None
        else load_win_propensity_model(config.propensity_path)
    )
    trajectories = [
        simulate_player_trajectory(
            propensity_model,
            player_id=player_id,
            seed=config.seed,
            max_attempts=config.max_attempts,
            dda_gains=config.dda_gains,
            e_sigmas=config.e_sigmas,
        )
        for player_id in range(config.n_players)
    ]
    _, _, test = split_player_trajectories(
        trajectories,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed + 1,
    )
    recorded_test = training_report.get("splits", {}).get("test", {}).get(
        "player_ids"
    )
    reconstructed_ids = [trajectory.player_id for trajectory in test]
    if recorded_test is not None and reconstructed_ids != recorded_test:
        raise RuntimeError("reconstructed test split does not match the manifest")
    eligible = [
        trajectory
        for trajectory in test
        if any(
            record.attempt_id == query_attempt
            for record in trajectory.attempts
        )
    ]
    if max_players is not None:
        eligible = eligible[:max_players]
    if not eligible:
        raise ValueError("no held-out players survived to the query attempt")
    prefix, target = build_prefix_target_batch(
        eligible, target_attempt=query_attempt, device=device
    )
    oracle_skill = torch.as_tensor(
        np.asarray([trajectory.player.as_array() for trajectory in eligible]),
        dtype=torch.float32,
        device=device,
    )
    models = {
        arm: load_generative_world_model_checkpoint(
            directory / training_report["arms"][arm.value]["checkpoint"],
            device=device,
            expected_arm=arm,
        )
        for arm in ModelArm
    }
    structural = evaluate_matched_response_curves(
        models,
        prefix,
        mastery_before=target.mastery_before,
        oracle_skill=oracle_skill,
        e_grid=e_grid,
    )
    imagination = evaluate_matched_imagination_curves(
        models,
        prefix,
        mastery_before=target.mastery_before,
        oracle_skill=oracle_skill,
        e_grid=e_grid,
        rollouts_per_player=rollouts_per_player,
        max_steps=max_steps,
        stochastic=stochastic,
        seed=seed,
    )
    report = {
        "schema_version": 1,
        "training_manifest": "metrics.json",
        "cohort": {
            "split": "test",
            "target_attempt": query_attempt,
            "n_players": len(eligible),
            "player_ids": [trajectory.player_id for trajectory in eligible],
            "strict_prefix": True,
        },
        "structural": structural,
        "imagination": imagination,
    }
    (directory / "response-curves.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


__all__ = [
    "evaluate_matched_experiment_directory",
    "evaluate_matched_imagination_curves",
    "evaluate_matched_response_curves",
    "imagine_response_curves",
    "induced_response_curves",
    "load_generative_world_model_checkpoint",
]