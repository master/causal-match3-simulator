"""Small matched-arm smoke experiment for generative model integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from ..calibrate import load_win_propensity_model
from ..retention import PlayerTrajectory, simulate_player_trajectory
from ..scm import DDA_GAINS, E_SIGMAS
from ..spec import BENCHMARK_CONFIG
from .action_policy import ActionPolicyConfig
from .arms import GenerativeModelConfig, GenerativeWorldModel, ModelArm
from .batching import build_generative_training_batch
from .data import split_player_trajectories
from .encoder import PrefixEncoderConfig
from .generative import GameplayRSSMConfig
from .train import train_generative_world_model_step


@dataclass(frozen=True)
class MatchedGenerativeSmokeConfig:
    n_players: int = 3
    target_attempt: int = 2
    embedding_size: int = 8
    observation_size: int = 16
    task_context_size: int = 8
    hidden_size: int = 16
    stochastic_size: int = 8
    behavior_size: int = 16
    prefix_hidden_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    context_kl_weight: float = 0.2
    dynamics_kl_weight: float = 0.2
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        integer_fields = (
            "n_players",
            "target_attempt",
            "embedding_size",
            "observation_size",
            "task_context_size",
            "hidden_size",
            "stochastic_size",
            "behavior_size",
            "prefix_hidden_size",
        )
        if any(getattr(self, name) < 1 for name in integer_fields):
            raise ValueError("smoke dimensions and counts must be positive")
        if self.behavior_size % 2:
            raise ValueError("behavior_size must be divisible by two heads")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")


@dataclass(frozen=True)
class MatchedGenerativeTrainConfig:
    """Configuration for a player-disjoint four-arm experiment."""

    n_players: int = 2_400
    max_attempts: int = BENCHMARK_CONFIG.landmark_attempt
    target_attempts: tuple[int, ...] = tuple(
        range(1, BENCHMARK_CONFIG.landmark_attempt + 1)
    )
    batch_size: int = 32
    epochs: int = 50
    patience: int = 8
    min_delta: float = 1e-4
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    max_updates_per_epoch: int | None = None
    embedding_size: int = 32
    observation_size: int = 128
    task_context_size: int = 32
    hidden_size: int = 128
    stochastic_size: int = 32
    behavior_size: int = 128
    prefix_hidden_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    context_kl_weight: float = 0.2
    dynamics_kl_weight: float = 0.2
    seed: int = 4201
    device: str = "cpu"
    dda_gains: tuple[float, ...] = DDA_GAINS
    e_sigmas: tuple[float, ...] = E_SIGMAS
    propensity_path: str | None = None

    def __post_init__(self) -> None:
        if self.n_players < 3:
            raise ValueError("n_players must permit three player splits")
        attempts = tuple(self.target_attempts)
        if not attempts or attempts != tuple(sorted(set(attempts))):
            raise ValueError("target_attempts must be non-empty, unique, and sorted")
        if attempts[0] < 1 or attempts[-1] > self.max_attempts:
            raise ValueError("target_attempts must lie between 1 and max_attempts")
        positive_integers = (
            "batch_size",
            "epochs",
            "patience",
            "embedding_size",
            "observation_size",
            "task_context_size",
            "hidden_size",
            "stochastic_size",
            "behavior_size",
            "prefix_hidden_size",
        )
        if any(getattr(self, name) < 1 for name in positive_integers):
            raise ValueError("training dimensions and counts must be positive")
        if self.behavior_size % 2:
            raise ValueError("behavior_size must be divisible by two heads")
        if self.max_updates_per_epoch is not None and self.max_updates_per_epoch < 1:
            raise ValueError("max_updates_per_epoch must be positive")
        if self.validation_fraction <= 0 or self.test_fraction <= 0:
            raise ValueError("validation and test fractions must be positive")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation and test fractions must sum below one")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.context_kl_weight < 0 or self.dynamics_kl_weight < 0:
            raise ValueError("KL weights must be non-negative")


def _model_config(
    config: MatchedGenerativeSmokeConfig | MatchedGenerativeTrainConfig,
) -> GenerativeModelConfig:
    return GenerativeModelConfig(
        dynamics=GameplayRSSMConfig(
            embedding_size=config.embedding_size,
            observation_size=config.observation_size,
            task_context_size=config.task_context_size,
            hidden_size=config.hidden_size,
            stochastic_size=config.stochastic_size,
        ),
        prefix=PrefixEncoderConfig(hidden_size=config.prefix_hidden_size),
        behavior=ActionPolicyConfig(
            d_model=config.behavior_size,
            n_layers=1,
            n_heads=2,
        ),
    )


def run_matched_generative_smoke(
    config: MatchedGenerativeSmokeConfig,
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    """Simulate one shared batch and update every generative arm once."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    propensity_model = load_win_propensity_model()
    benchmark = replace(BENCHMARK_CONFIG, warmup_churn_scale=0.0)
    trajectories = [
        simulate_player_trajectory(
            propensity_model,
            player_id=player_id,
            seed=config.seed,
            max_attempts=config.target_attempt,
            benchmark=benchmark,
        )
        for player_id in range(config.n_players)
    ]
    prefix, target, transitions, oracle_skill = build_generative_training_batch(
        trajectories,
        target_attempt=config.target_attempt,
        device=device,
        benchmark=benchmark,
    )
    model_config = _model_config(config)
    arm_reports: dict[str, object] = {}
    for arm in ModelArm:
        torch.manual_seed(config.seed)
        model = GenerativeWorldModel(arm, model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        started = time.perf_counter()
        metrics = train_generative_world_model_step(
            model,
            prefix=prefix,
            target=target,
            transitions=transitions,
            optimizer=optimizer,
            oracle_skill=oracle_skill if arm is ModelArm.ORACLE else None,
            context_kl_weight=config.context_kl_weight,
            dynamics_kl_weight=config.dynamics_kl_weight,
            gradient_clip=config.gradient_clip,
        )
        runtime = time.perf_counter() - started
        checkpoint_name = f"{arm.value}.pt"
        torch.save(
            {
                "schema_version": 1,
                "arm": arm.value,
                "config": asdict(model_config),
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
            },
            output / checkpoint_name,
        )
        arm_reports[arm.value] = {
            **metrics,
            "parameters": model.parameter_counts(),
            "runtime_seconds": runtime,
            "checkpoint": checkpoint_name,
            "seed": config.seed,
        }
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "smoke",
        "config": json.loads(json.dumps(asdict(config))),
        "n_players": len(trajectories),
        "target_attempt": config.target_attempt,
        "n_target_transitions": int(transitions["step_mask"].sum()),
        "arms": arm_reports,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


_OBJECTIVE_CHANNELS = (
    "assignment_nll",
    "evidence_nll",
    "win_nll",
    "churn_nll",
    "action_nll",
    "context_kl",
    "board_nll",
    "counter_mse",
    "dynamics_kl",
)


def _batch_specs(
    trajectories: list[PlayerTrajectory],
    target_attempts: tuple[int, ...],
    batch_size: int,
) -> list[tuple[list[PlayerTrajectory], int]]:
    specs: list[tuple[list[PlayerTrajectory], int]] = []
    for target_attempt in target_attempts:
        eligible = [
            trajectory
            for trajectory in trajectories
            if any(
                record.attempt_id == target_attempt
                for record in trajectory.attempts
            )
        ]
        for start in range(0, len(eligible), batch_size):
            specs.append((eligible[start : start + batch_size], target_attempt))
    if not specs:
        raise ValueError("split has no eligible target attempts")
    return specs


def _new_accumulator() -> dict[str, dict[str, float]]:
    return {
        "sums": {name: 0.0 for name in _OBJECTIVE_CHANNELS},
        "counts": {name: 0.0 for name in _OBJECTIVE_CHANNELS},
    }


def _accumulate_metrics(
    accumulator: dict[str, dict[str, float]],
    metrics: dict[str, float | torch.Tensor | object],
    target: object,
    transitions: dict[str, torch.Tensor],
) -> None:
    episode_count = float(target.outcomes.numel())
    transition_count = float(transitions["step_mask"].sum().item())
    state_count = episode_count + transition_count
    channel_counts = {
        "assignment_nll": episode_count,
        "evidence_nll": episode_count,
        "win_nll": episode_count,
        "churn_nll": float(target.churn_mask.sum().item()),
        "action_nll": float(target.actions.numel()),
        "context_kl": episode_count,
        "board_nll": state_count,
        "counter_mse": state_count,
        "dynamics_kl": transition_count,
    }
    for name, count in channel_counts.items():
        value = metrics[name]
        scalar = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
        accumulator["sums"][name] += scalar * count
        accumulator["counts"][name] += count


def _finalize_metrics(
    accumulator: dict[str, dict[str, float]],
    config: MatchedGenerativeTrainConfig,
) -> dict[str, float]:
    metrics = {
        name: (
            accumulator["sums"][name] / accumulator["counts"][name]
            if accumulator["counts"][name]
            else 0.0
        )
        for name in _OBJECTIVE_CHANNELS
    }
    metrics["loss"] = (
        metrics["assignment_nll"]
        + metrics["evidence_nll"]
        + metrics["win_nll"]
        + metrics["churn_nll"]
        + metrics["action_nll"]
        + config.context_kl_weight * metrics["context_kl"]
        + metrics["board_nll"]
        + metrics["counter_mse"]
        + config.dynamics_kl_weight * metrics["dynamics_kl"]
    )
    return metrics


def _evaluate_arms(
    models: dict[ModelArm, GenerativeWorldModel],
    selected_arms: list[ModelArm],
    specs: list[tuple[list[PlayerTrajectory], int]],
    config: MatchedGenerativeTrainConfig,
    device: torch.device,
) -> tuple[dict[ModelArm, dict[str, float]], dict[ModelArm, float]]:
    accumulators = {arm: _new_accumulator() for arm in selected_arms}
    runtimes = {arm: 0.0 for arm in selected_arms}
    for trajectories, target_attempt in specs:
        prefix, target, transitions, oracle_skill = build_generative_training_batch(
            trajectories,
            target_attempt=target_attempt,
            device=device,
        )
        for arm in selected_arms:
            started = time.perf_counter()
            models[arm].eval()
            with torch.no_grad():
                result = models[arm].objective(
                    prefix=prefix,
                    target=target,
                    transitions=transitions,
                    oracle_skill=(
                        oracle_skill if arm is ModelArm.ORACLE else None
                    ),
                    context_kl_weight=config.context_kl_weight,
                    dynamics_kl_weight=config.dynamics_kl_weight,
                    sample_context=False,
                    sample_dynamics=False,
                )
            runtimes[arm] += time.perf_counter() - started
            _accumulate_metrics(accumulators[arm], result, target, transitions)
    return (
        {
            arm: _finalize_metrics(accumulators[arm], config)
            for arm in selected_arms
        },
        runtimes,
    )


def _split_report(
    trajectories: list[PlayerTrajectory],
    specs: list[tuple[list[PlayerTrajectory], int]],
) -> dict[str, object]:
    examples_by_attempt: dict[str, int] = {}
    for batch, target_attempt in specs:
        key = str(target_attempt)
        examples_by_attempt[key] = examples_by_attempt.get(key, 0) + len(batch)
    return {
        "n_players": len(trajectories),
        "player_ids": [trajectory.player_id for trajectory in trajectories],
        "n_examples": sum(examples_by_attempt.values()),
        "n_batches": len(specs),
        "examples_by_target_attempt": examples_by_attempt,
    }


def run_matched_generative_experiment(
    config: MatchedGenerativeTrainConfig,
    *,
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Train and select four matched arms, then score an untouched test split."""
    experiment_started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    emit = progress or (lambda _: None)

    propensity_model = (
        load_win_propensity_model()
        if config.propensity_path is None
        else load_win_propensity_model(config.propensity_path)
    )
    emit(
        f"simulating {config.n_players} players through attempt {config.max_attempts}"
    )
    simulation_started = time.perf_counter()
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
    simulation_seconds = time.perf_counter() - simulation_started
    train, validation, test = split_player_trajectories(
        trajectories,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed + 1,
    )
    train_specs = _batch_specs(train, config.target_attempts, config.batch_size)
    validation_specs = _batch_specs(
        validation, config.target_attempts, config.batch_size
    )
    test_specs = _batch_specs(test, config.target_attempts, config.batch_size)
    emit(
        "split players into "
        f"{len(train)} train / {len(validation)} validation / {len(test)} test"
    )

    model_config = _model_config(config)
    models: dict[ModelArm, GenerativeWorldModel] = {}
    optimizers: dict[ModelArm, torch.optim.Optimizer] = {}
    for arm in ModelArm:
        torch.manual_seed(config.seed)
        model = GenerativeWorldModel(arm, model_config).to(device)
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    states: dict[ModelArm, dict[str, object]] = {
        arm: {
            "best_loss": float("inf"),
            "best_epoch": 0,
            "stale_epochs": 0,
            "stopped": False,
            "updates": 0,
            "runtime_seconds": 0.0,
            "history": [],
        }
        for arm in ModelArm
    }
    checkpoints = {
        arm: output / f"{arm.value}-best.pt" for arm in ModelArm
    }

    for epoch in range(1, config.epochs + 1):
        active_arms = [arm for arm in ModelArm if not states[arm]["stopped"]]
        if not active_arms:
            break
        order = np.random.default_rng(config.seed + epoch).permutation(
            len(train_specs)
        )
        if config.max_updates_per_epoch is not None:
            order = order[: config.max_updates_per_epoch]
        emit(
            f"epoch {epoch}: {len(order)} shared batches across "
            f"{len(active_arms)} active arms"
        )
        train_accumulators = {
            arm: _new_accumulator() for arm in active_arms
        }
        for update_index, spec_index in enumerate(order):
            batch, target_attempt = train_specs[int(spec_index)]
            prefix, target, transitions, oracle_skill = (
                build_generative_training_batch(
                    batch,
                    target_attempt=target_attempt,
                    device=device,
                )
            )
            for arm_index, arm in enumerate(active_arms):
                step_seed = int(
                    np.random.SeedSequence(
                        [config.seed, epoch, update_index, arm_index]
                    ).generate_state(1)[0]
                )
                torch.manual_seed(step_seed)
                step_started = time.perf_counter()
                metrics = train_generative_world_model_step(
                    models[arm],
                    prefix=prefix,
                    target=target,
                    transitions=transitions,
                    optimizer=optimizers[arm],
                    oracle_skill=(
                        oracle_skill if arm is ModelArm.ORACLE else None
                    ),
                    context_kl_weight=config.context_kl_weight,
                    dynamics_kl_weight=config.dynamics_kl_weight,
                    gradient_clip=config.gradient_clip,
                )
                states[arm]["runtime_seconds"] += (
                    time.perf_counter() - step_started
                )
                states[arm]["updates"] += 1
                _accumulate_metrics(
                    train_accumulators[arm], metrics, target, transitions
                )

        validation_metrics, validation_runtimes = _evaluate_arms(
            models, active_arms, validation_specs, config, device
        )
        for arm in active_arms:
            states[arm]["runtime_seconds"] += validation_runtimes[arm]
            train_metrics = _finalize_metrics(
                train_accumulators[arm], config
            )
            current_loss = validation_metrics[arm]["loss"]
            if not np.isfinite(current_loss):
                raise RuntimeError(
                    f"{arm.value} produced non-finite validation loss"
                )
            improved = current_loss < states[arm]["best_loss"] - config.min_delta
            if improved:
                states[arm]["best_loss"] = current_loss
                states[arm]["best_epoch"] = epoch
                states[arm]["stale_epochs"] = 0
                torch.save(
                    {
                        "schema_version": 2,
                        "arm": arm.value,
                        "epoch": epoch,
                        "validation_loss": current_loss,
                        "model_config": asdict(model_config),
                        "train_config": asdict(config),
                        "model_state_dict": models[arm].state_dict(),
                        "optimizer_state_dict": optimizers[arm].state_dict(),
                    },
                    checkpoints[arm],
                )
            else:
                states[arm]["stale_epochs"] += 1
                if states[arm]["stale_epochs"] >= config.patience:
                    states[arm]["stopped"] = True
            states[arm]["history"].append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics[arm],
                    "improved": improved,
                }
            )

    for arm in ModelArm:
        checkpoint = torch.load(
            checkpoints[arm], map_location=device, weights_only=False
        )
        models[arm].load_state_dict(checkpoint["model_state_dict"])
    all_arms = list(ModelArm)
    selected_validation, validation_runtimes = _evaluate_arms(
        models, all_arms, validation_specs, config, device
    )
    test_metrics, test_runtimes = _evaluate_arms(
        models, all_arms, test_specs, config, device
    )
    for arm in ModelArm:
        states[arm]["runtime_seconds"] += (
            validation_runtimes[arm] + test_runtimes[arm]
        )

    report: dict[str, object] = {
        "schema_version": 2,
        "status": "complete",
        "config": json.loads(json.dumps(asdict(config))),
        "simulation_seconds": simulation_seconds,
        "runtime_seconds": time.perf_counter() - experiment_started,
        "splits": {
            "train": _split_report(train, train_specs),
            "validation": _split_report(validation, validation_specs),
            "test": _split_report(test, test_specs),
        },
        "arms": {
            arm.value: {
                "parameters": models[arm].parameter_counts(),
                "checkpoint": checkpoints[arm].name,
                "best_epoch": states[arm]["best_epoch"],
                "epochs_ran": len(states[arm]["history"]),
                "train_updates": states[arm]["updates"],
                "stopped_early": states[arm]["stopped"],
                "runtime_seconds": states[arm]["runtime_seconds"],
                "history": states[arm]["history"],
                "validation": selected_validation[arm],
                "test": test_metrics[arm],
            }
            for arm in ModelArm
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    emit(f"finished matched experiment in {report['runtime_seconds']:.1f}s")
    return report


__all__ = [
    "MatchedGenerativeSmokeConfig",
    "MatchedGenerativeTrainConfig",
    "run_matched_generative_experiment",
    "run_matched_generative_smoke",
]
