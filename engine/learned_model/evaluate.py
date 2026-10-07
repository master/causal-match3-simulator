"""Evaluate a learned prefix model with real-engine closed-loop rollouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pyro
import torch

from ..calibrate import load_win_propensity_model
from ..causal_queries import select_grid_optimum
from ..retention import (
    CHURN_SCHEDULE,
    MasteryConfig,
    PlayerTrajectory,
    WinPropensityModel,
    expected_mastery_churn,
)
from ..scm import LEVELS, TIER_NAMES, TIER_PROBS, ground_truth_model
from ..simulate import simulate_players
from ..spec import BENCHMARK_CONFIG, PlayerSkill
from .action_policy import ActionPolicyConfig, ContinuousActionPolicy
from .data import split_player_trajectories
from .encoder import PrefixEncoderConfig
from .experiment import (
    LearnedExperimentConfig,
    _posterior_skill_evaluation,
    _truncate_at_target,
)
from .model import ContinuousCausalVAE
from .rollout import NetworkActionPolicy
from .train import resolve_device


def load_continuous_vae_checkpoint(
    path: str | Path, *, device: torch.device = torch.device("cpu")
) -> ContinuousCausalVAE:
    payload = torch.load(Path(path), map_location="cpu")
    action_config = dict(payload["action_config"])
    if "use_immediate_features" not in action_config:
        action_config["use_immediate_features"] = False
    model = ContinuousCausalVAE(
        PrefixEncoderConfig(**payload["encoder_config"]),
        ActionPolicyConfig(**action_config),
    )
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {
        "churn_head.raw_overchallenge_deviation",
        "churn_head.raw_margin_deviation",
        "churn_head.raw_margin_overchallenge_deviation",
        "churn_head.raw_margin_target",
    }
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise ValueError("checkpoint parameters do not match the continuous VAE")
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


def load_action_policy_checkpoint(
    path: str | Path, *, device: torch.device = torch.device("cpu")
) -> ContinuousActionPolicy:
    payload = torch.load(Path(path), map_location="cpu")
    config = dict(payload["config"])
    if "use_immediate_features" not in config:
        config["use_immediate_features"] = False
    model = ContinuousActionPolicy(ActionPolicyConfig(**config))
    model.load_state_dict(payload["state_dict"])
    return model.to(device=device, dtype=torch.float32).eval()


@torch.no_grad()
def _head_curves(
    model: ContinuousCausalVAE,
    skills: np.ndarray,
    e_grid: tuple[float, ...],
    *,
    mastery_before: np.ndarray | None = None,
    device: torch.device,
) -> dict[str, list[dict[str, float]]]:
    skill_array = np.asarray(skills, dtype=np.float32)
    mastery_array = (
        np.full(len(skill_array), MasteryConfig().initial, dtype=np.float32)
        if mastery_before is None
        else np.asarray(mastery_before, dtype=np.float32)
    )
    if mastery_array.shape != (len(skill_array),):
        raise ValueError("mastery_before must align with skills")
    tier_probabilities = torch.as_tensor(
        TIER_PROBS, dtype=torch.float32, device=device
    )
    curves: dict[str, list[dict[str, float]]] = {}
    for level_index, level in enumerate(LEVELS):
        rows = []
        for served_difficulty in e_grid:
            expanded_skill = torch.as_tensor(
                np.repeat(skill_array, len(TIER_NAMES), axis=0),
                dtype=torch.float32,
                device=device,
            )
            tiers = torch.arange(len(TIER_NAMES), device=device).repeat(
                len(skill_array)
            )
            levels = torch.full_like(tiers, level_index)
            served = torch.full(
                (len(expanded_skill),),
                served_difficulty,
                dtype=torch.float32,
                device=device,
            )
            win = model.win_head.probabilities(
                served, expanded_skill, levels, tiers
            ).reshape(len(skill_array), len(TIER_NAMES))
            expanded_mastery = torch.as_tensor(
                np.repeat(mastery_array, len(TIER_NAMES)),
                dtype=torch.float32,
                device=device,
            )
            churn = model.expected_churn_probability(
                win.reshape(-1), expanded_mastery, levels
            ).reshape_as(win)
            rows.append(
                {
                    "e": served_difficulty,
                    "head_win_probability": float(
                        (win * tier_probabilities).sum(dim=1).mean()
                    ),
                    "head_churn_probability": float(
                        (churn * tier_probabilities).sum(dim=1).mean()
                    ),
                }
            )
        curves[level.name] = rows
    return curves


def oracle_propensity_curves(
    model: WinPropensityModel,
    skills: np.ndarray,
    *,
    e_grid: tuple[float, ...] = (-1.0, 0.0, 1.0),
) -> dict[str, list[dict[str, float]]]:
    """Evaluate the frozen engine-derived response surface over tier mixtures."""
    skill_array = np.asarray(skills, dtype=np.float64)
    tier_indices = np.tile(np.arange(len(TIER_NAMES)), len(skill_array))
    expanded_skills = np.repeat(skill_array, len(TIER_NAMES), axis=0)
    curves: dict[str, list[dict[str, float]]] = {}
    for level in LEVELS:
        rows = []
        for served_difficulty in e_grid:
            probabilities = model.probabilities(
                level.name,
                tier_indices,
                expanded_skills,
                served_difficulty,
            ).reshape(len(skill_array), len(TIER_NAMES))
            rows.append(
                {
                    "e": served_difficulty,
                    "win_probability": float(
                        np.mean(probabilities @ np.asarray(TIER_PROBS))
                    ),
                }
            )
        curves[level.name] = rows
    return curves


def _curve_mae(
    left: dict[str, list[dict[str, float]]],
    left_key: str,
    right: dict[str, list[dict[str, float]]],
    right_key: str,
) -> float:
    differences = [
        abs(left_row[left_key] - right_row[right_key])
        for level in left
        for left_row, right_row in zip(left[level], right[level])
    ]
    return float(np.mean(differences))


def evaluate_causal_query(
    model: ContinuousCausalVAE,
    inferred_skills: np.ndarray,
    true_skills: np.ndarray,
    propensity_model: WinPropensityModel,
    *,
    e_grid: tuple[float, ...] = BENCHMARK_CONFIG.e_grid,
    mastery_before: np.ndarray | None = None,
    mastery_config: MasteryConfig = MasteryConfig(),
    device: torch.device = torch.device("cpu"),
) -> dict[str, object]:
    """Compare learned and oracle do-curves on held-out player populations."""
    inferred = np.asarray(inferred_skills, dtype=np.float32)
    truth = np.asarray(true_skills, dtype=np.float64)
    if inferred.shape != truth.shape or inferred.ndim != 2 or inferred.shape[1] != 4:
        raise ValueError("inferred and true skills must have matching (players, 4) shapes")
    mastery = (
        np.full(len(truth), mastery_config.initial, dtype=np.float64)
        if mastery_before is None
        else np.asarray(mastery_before, dtype=np.float64)
    )
    if mastery.shape != (len(truth),):
        raise ValueError("mastery_before must align with skills")
    learned = _head_curves(
        model,
        inferred,
        e_grid,
        mastery_before=mastery,
        device=device,
    )
    tier_indices = np.tile(np.arange(len(TIER_NAMES)), len(truth))
    expanded_truth = np.repeat(truth, len(TIER_NAMES), axis=0)
    tier_probabilities = np.asarray(TIER_PROBS)
    levels: dict[str, object] = {}
    for level in LEVELS:
        oracle_values = []
        for served_difficulty in e_grid:
            win_probability = propensity_model.probabilities(
                level.name,
                tier_indices,
                expanded_truth,
                served_difficulty,
            ).reshape(len(truth), len(TIER_NAMES))
            hazard = expected_mastery_churn(
                win_probability,
                np.repeat(mastery, len(TIER_NAMES)).reshape(
                    len(truth), len(TIER_NAMES)
                ),
                mastery_config=mastery_config,
                churn_config=CHURN_SCHEDULE.for_level(level.name),
            )
            oracle_values.append(float(np.mean(hazard @ tier_probabilities)))
        learned_values = [
            float(row["head_churn_probability"])
            for row in learned[level.name]
        ]
        grid = np.asarray(e_grid, dtype=np.float64)
        oracle_array = np.asarray(oracle_values)
        learned_array = np.asarray(learned_values)
        oracle_optimum = select_grid_optimum(grid, oracle_array)
        learned_optimum = select_grid_optimum(grid, learned_array)
        learned_index = int(np.flatnonzero(grid == learned_optimum)[0])
        levels[level.name] = {
            "grid": list(e_grid),
            "oracle": oracle_values,
            "learned": learned_values,
            "oracle_optimum": oracle_optimum,
            "learned_optimum": learned_optimum,
            "recommendation_gap": abs(learned_optimum - oracle_optimum),
            "oracle_regret": float(
                oracle_array[learned_index] - np.min(oracle_array)
            ),
            "curve_mae": float(np.mean(np.abs(learned_array - oracle_array))),
        }
    return {"n_players": len(truth), "levels": levels}


def evaluate_closed_loop(
    model: ContinuousCausalVAE,
    skills: np.ndarray,
    *,
    e_grid: tuple[float, ...] = (-1.0, 0.0, 1.0),
    rollouts_per_player: int = 1,
    seed: int = 0,
    stochastic: bool = True,
    device: torch.device = torch.device("cpu"),
) -> dict[str, object]:
    """Compare learned-head curves with learned-policy engine outcomes."""
    skill_array = np.asarray(skills, dtype=np.float32)
    if skill_array.ndim != 2 or skill_array.shape[1] != 4 or len(skill_array) == 0:
        raise ValueError("skills must have shape (players, 4)")
    if not e_grid or rollouts_per_player < 1:
        raise ValueError("e_grid and rollouts_per_player must be non-empty")
    model.to(device).eval()
    head_curves = _head_curves(model, skill_array, e_grid, device=device)
    rollout_curves = rollout_win_curves(
        model.action_policy,
        skill_array,
        e_grid=e_grid,
        rollouts_per_player=rollouts_per_player,
        seed=seed,
        stochastic=stochastic,
        device=device,
    )
    absolute_gaps = []
    levels: dict[str, list[dict[str, float]]] = {}
    for level in LEVELS:
        rows = []
        for e_index in range(len(e_grid)):
            head_row = head_curves[level.name][e_index]
            rollout_row = rollout_curves[level.name][e_index]
            rollout_win_probability = rollout_row["rollout_win_probability"]
            gap = abs(
                rollout_win_probability - head_row["head_win_probability"]
            )
            absolute_gaps.append(gap)
            rows.append(
                {
                    **head_row,
                    "rollout_win_probability": rollout_win_probability,
                    "absolute_win_gap": gap,
                    "n_rollouts": rollout_row["n_rollouts"],
                }
            )
        levels[level.name] = rows
    return {
        "schema_version": 1,
        "e_grid": list(e_grid),
        "n_players": len(skill_array),
        "rollouts_per_player": rollouts_per_player,
        "stochastic_policy": stochastic,
        "seed": seed,
        "mean_absolute_win_gap": float(np.mean(absolute_gaps)),
        "maximum_absolute_win_gap": float(np.max(absolute_gaps)),
        "levels": levels,
    }


def rollout_win_curves(
    policy_model: ContinuousActionPolicy | None,
    skills: np.ndarray,
    *,
    e_grid: tuple[float, ...] = (-1.0, 0.0, 1.0),
    rollouts_per_player: int = 1,
    seed: int = 0,
    stochastic: bool = True,
    device: torch.device = torch.device("cpu"),
) -> dict[str, list[dict[str, float]]]:
    """Roll a learned action policy through the engine under common seeds."""
    skill_array = np.asarray(skills, dtype=np.float32)
    if skill_array.ndim != 2 or skill_array.shape[1] != 4 or len(skill_array) == 0:
        raise ValueError("skills must have shape (players, 4)")
    if policy_model is not None:
        policy_model.to(device).eval()
    levels: dict[str, list[dict[str, float]]] = {}
    for level_index, level in enumerate(LEVELS):
        rows = []
        for served_difficulty in e_grid:
            outcomes = []
            for player_index, values in enumerate(skill_array):
                player = PlayerSkill(tuple(float(value) for value in values))
                for replicate in range(rollouts_per_player):
                    rollout_seed = int(
                        np.random.SeedSequence(
                            [seed, level_index, player_index, replicate]
                        ).generate_state(1)[0]
                    )
                    pyro.set_rng_seed(rollout_seed)
                    policy = (
                        NetworkActionPolicy(
                            policy_model,
                            device=device,
                            stochastic=stochastic,
                            seed=rollout_seed,
                        )
                        if policy_model is not None
                        else None
                    )
                    episode = ground_truth_model(
                        level=level,
                        player=player,
                        E=served_difficulty,
                        action_policy=policy,
                    )
                    outcomes.append(episode.R)
            rows.append(
                {
                    "e": served_difficulty,
                    "rollout_win_probability": float(np.mean(outcomes)),
                    "n_rollouts": len(outcomes),
                }
            )
        levels[level.name] = rows
    return levels


def evaluate_experiment_directory(
    experiment_dir: str | Path,
    *,
    e_grid: tuple[float, ...] = (-1.0, 0.0, 1.0),
    max_players: int = 5,
    rollouts_per_player: int = 1,
    seed: int = 0,
) -> dict[str, object]:
    experiment_dir = Path(experiment_dir)
    metrics = json.loads((experiment_dir / "metrics.json").read_text())
    allowed_config = {field.name for field in fields(LearnedExperimentConfig)}
    experiment_config = LearnedExperimentConfig(
        **{
            name: value
            for name, value in metrics["config"].items()
            if name in allowed_config
        }
    )
    device = resolve_device(experiment_config.device)
    model = load_continuous_vae_checkpoint(
        experiment_dir / "continuous_vae.pt", device=device
    )
    state_only = load_action_policy_checkpoint(
        experiment_dir / "state_only_action.pt", device=device
    )
    oracle_skill = load_action_policy_checkpoint(
        experiment_dir / "oracle_skill_action.pt", device=device
    )
    propensity_path = Path(metrics["propensity_artifact"]["path"])
    if not propensity_path.is_absolute():
        propensity_path = Path(__file__).resolve().parents[1] / propensity_path
    propensity_model = load_win_propensity_model(propensity_path)
    evaluation_path = experiment_dir / "evaluation_players.npz"
    if evaluation_path.exists():
        with np.load(evaluation_path) as values:
            all_player_ids = values["player_id"].astype(int).tolist()
            all_true_skills = values["true_skill"]
            all_inferred_skills = values["inferred_skill"]
    else:
        trajectories = simulate_players(
            experiment_config.n_players,
            propensity_model,
            seed=experiment_config.seed,
            max_attempts=experiment_config.max_attempts,
            progress_every=max(1, experiment_config.n_players // 10),
        )
        eligible = [
            trajectory
            for trajectory in trajectories
            if any(
                record.attempt_id == experiment_config.target_attempt
                for record in trajectory.attempts
            )
        ]
        eligible = _truncate_at_target(eligible, experiment_config.target_attempt)
        _, _, test_trajectories = split_player_trajectories(
            eligible,
            validation_fraction=experiment_config.validation_fraction,
            test_fraction=experiment_config.test_fraction,
            seed=experiment_config.seed,
        )
        selected: list[PlayerTrajectory] = test_trajectories
        _, inferred_by_player = _posterior_skill_evaluation(
            model,
            selected,
            target_attempt=experiment_config.target_attempt,
            players_per_batch=experiment_config.players_per_batch,
            device=device,
        )
        all_player_ids = [trajectory.player_id for trajectory in selected]
        all_inferred_skills = np.asarray(
            [inferred_by_player[player_id] for player_id in all_player_ids]
        )
        all_true_skills = np.asarray(
            [trajectory.player.as_array() for trajectory in selected]
        )
    player_ids = all_player_ids[:max_players]
    true_skills = all_true_skills[:max_players]
    inferred_skills = all_inferred_skills[:max_players]
    report = evaluate_closed_loop(
        model,
        inferred_skills,
        e_grid=e_grid,
        rollouts_per_player=rollouts_per_player,
        seed=seed,
        device=device,
    )
    report["experiment_dir"] = str(experiment_dir)
    report["source_metrics"] = "metrics.json"
    report["model_config"] = metrics.get("model_config")
    report["propensity_artifact"] = metrics["propensity_artifact"]
    report["player_ids"] = player_ids
    report["causal_query"] = evaluate_causal_query(
        model,
        all_inferred_skills,
        all_true_skills,
        propensity_model,
        device=device,
    )
    baseline_curves = {
        "state_only": rollout_win_curves(
            state_only,
            np.zeros_like(inferred_skills),
            e_grid=e_grid,
            rollouts_per_player=rollouts_per_player,
            seed=seed,
            device=device,
        ),
        "oracle_skill": rollout_win_curves(
            oracle_skill,
            true_skills,
            e_grid=e_grid,
            rollouts_per_player=rollouts_per_player,
            seed=seed,
            device=device,
        ),
        "ground_truth_inferred_skill": rollout_win_curves(
            None,
            inferred_skills,
            e_grid=e_grid,
            rollouts_per_player=rollouts_per_player,
            seed=seed,
            device=device,
        ),
        "ground_truth_true_skill": rollout_win_curves(
            None,
            true_skills,
            e_grid=e_grid,
            rollouts_per_player=rollouts_per_player,
            seed=seed,
            device=device,
        ),
    }
    report["baseline_rollout_win_curves"] = baseline_curves
    oracle_inferred = oracle_propensity_curves(
        propensity_model, inferred_skills, e_grid=e_grid
    )
    oracle_true = oracle_propensity_curves(
        propensity_model, true_skills, e_grid=e_grid
    )
    report["oracle_propensity_curves"] = {
        "inferred_skill": oracle_inferred,
        "true_skill": oracle_true,
    }
    report["reference_gaps"] = {
        "learned_head_vs_oracle_propensity_mae": _curve_mae(
            report["levels"],
            "head_win_probability",
            oracle_inferred,
            "win_probability",
        ),
        "inferred_policy_vs_oracle_propensity_mae": _curve_mae(
            report["levels"],
            "rollout_win_probability",
            oracle_inferred,
            "win_probability",
        ),
        "oracle_skill_policy_vs_oracle_propensity_mae": _curve_mae(
            baseline_curves["oracle_skill"],
            "rollout_win_probability",
            oracle_true,
            "win_probability",
        ),
        "ground_truth_policy_vs_oracle_propensity_mae": _curve_mae(
            baseline_curves["ground_truth_inferred_skill"],
            "rollout_win_probability",
            oracle_inferred,
            "win_probability",
        ),
        "inferred_policy_vs_ground_truth_policy_mae": _curve_mae(
            report["levels"],
            "rollout_win_probability",
            baseline_curves["ground_truth_inferred_skill"],
            "rollout_win_probability",
        ),
    }
    path = experiment_dir / "closed_loop.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path}")
    return report


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--e-grid", type=float, nargs="+", default=(-1.0, 0.0, 1.0))
    parser.add_argument("--max-players", type=int, default=5)
    parser.add_argument("--rollouts-per-player", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = evaluate_experiment_directory(
        args.experiment_dir,
        e_grid=tuple(args.e_grid),
        max_players=args.max_players,
        rollouts_per_player=args.rollouts_per_player,
        seed=args.seed,
    )
    print(
        f"closed-loop/head win MAE={report['mean_absolute_win_gap']:.3f} "
        f"max={report['maximum_absolute_win_gap']:.3f}"
    )
    print(
        "reference MAE: "
        f"head={report['reference_gaps']['learned_head_vs_oracle_propensity_mae']:.3f} "
        f"latent-policy={report['reference_gaps']['inferred_policy_vs_oracle_propensity_mae']:.3f} "
        f"oracle-policy={report['reference_gaps']['oracle_skill_policy_vs_oracle_propensity_mae']:.3f} "
        f"ground-truth-policy={report['reference_gaps']['ground_truth_policy_vs_oracle_propensity_mae']:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "evaluate_closed_loop",
    "evaluate_causal_query",
    "evaluate_experiment_directory",
    "load_action_policy_checkpoint",
    "load_continuous_vae_checkpoint",
    "oracle_propensity_curves",
    "rollout_win_curves",
]