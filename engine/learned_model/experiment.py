"""Run player-disjoint action and continuous-prefix learning experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ..calibrate import WIN_PROPENSITY_PATH, load_win_propensity_model
from ..retention import PlayerTrajectory
from ..simulate import simulate_players
from ..spec import SKILL_NAMES
from .action_policy import ActionPolicyConfig, ContinuousActionPolicy
from .batching import build_prefix_target_batch
from .data import (
    ActionDataset,
    action_dataset_from_episodes,
    action_dataset_from_trajectories,
    split_player_trajectories,
)
from .encoder import PrefixEncoderConfig
from .model import ContinuousCausalVAE, PredictiveTarget
from .train import (
    ActionTrainConfig,
    VAETrainConfig,
    WinHeadTrainConfig,
    evaluate_action_policy,
    evaluate_continuous_vae,
    fit_win_head,
    resolve_device,
    train_action_policy,
    train_continuous_vae,
)


@dataclass(frozen=True)
class LearnedExperimentConfig:
    n_players: int = 32
    max_attempts: int = 20
    target_attempt: int = 20
    first_training_attempt: int = 2
    win_head_first_attempt: int = 8
    action_epochs: int = 2
    vae_epochs: int = 5
    kl_warmup_epochs: int = 5
    win_head_epochs: int = 400
    win_head_patience: int = 50
    action_batch_size: int = 128
    players_per_batch: int = 8
    d_model: int = 32
    n_layers: int = 1
    n_heads: int = 4
    encoder_hidden_size: int = 32
    learning_rate: float = 3e-4
    win_head_learning_rate: float = 3e-3
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.n_players < 3:
            raise ValueError("n_players must be at least three")
        if not 1 <= self.target_attempt <= self.max_attempts:
            raise ValueError("target_attempt must lie within max_attempts")
        if not 1 <= self.first_training_attempt <= self.target_attempt:
            raise ValueError("first_training_attempt must not exceed target_attempt")
        if not self.first_training_attempt <= self.win_head_first_attempt <= self.target_attempt:
            raise ValueError("win_head_first_attempt must lie in the training range")
        if self.action_epochs < 1 or self.vae_epochs < 1:
            raise ValueError("training epochs must be positive")
        if self.win_head_epochs < 1 or self.win_head_patience < 1:
            raise ValueError("win-head epochs and patience must be positive")
        if self.players_per_batch < 1 or self.action_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunks(values: list, size: int) -> list[list]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _truncate_at_target(
    trajectories: list[PlayerTrajectory], target_attempt: int
) -> list[PlayerTrajectory]:
    return [
        PlayerTrajectory(
            trajectory.player_id,
            trajectory.player,
            tuple(
                record
                for record in trajectory.attempts
                if record.attempt_id <= target_attempt
            ),
        )
        for trajectory in trajectories
    ]


def _target_action_dataset(
    trajectories: list[PlayerTrajectory], target_attempt: int
) -> ActionDataset:
    records = [
        record
        for trajectory in trajectories
        for record in trajectory.attempts
        if record.attempt_id == target_attempt
    ]
    if len(records) != len(trajectories):
        raise ValueError("every trajectory must contain the target attempt")
    return action_dataset_from_episodes(
        [record.episode for record in records],
        player_indices=[record.player_id for record in records],
    )


def _prefix_batches(
    trajectories: list[PlayerTrajectory],
    *,
    target_attempts: tuple[int, ...],
    players_per_batch: int,
) -> list[tuple[dict[str, torch.Tensor], PredictiveTarget]]:
    return [
        build_prefix_target_batch(chunk, target_attempt=target_attempt)
        for target_attempt in target_attempts
        for chunk in _chunks(trajectories, players_per_batch)
    ]


@torch.no_grad()
def _posterior_skill_evaluation(
    model: ContinuousCausalVAE,
    trajectories: list[PlayerTrajectory],
    *,
    target_attempt: int,
    players_per_batch: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[int, np.ndarray]]:
    model.to(device).eval()
    inferred_parts = []
    true_parts = []
    scale_parts = []
    player_ids: list[int] = []
    for chunk in _chunks(trajectories, players_per_batch):
        prefix, _ = build_prefix_target_batch(
            chunk, target_attempt=target_attempt, device=device
        )
        mean, log_scale = model.posterior(prefix)
        inferred_parts.append(model.skill_transform(mean).cpu().numpy())
        scale_parts.append(log_scale.exp().cpu().numpy())
        true_parts.append(
            np.asarray([trajectory.player.as_array() for trajectory in chunk])
        )
        player_ids.extend(trajectory.player_id for trajectory in chunk)

    inferred = np.concatenate(inferred_parts)
    true = np.concatenate(true_parts)
    posterior_scale = np.concatenate(scale_parts)
    dimensions = []
    for index, name in enumerate(SKILL_NAMES):
        if len(true) < 2 or np.std(true[:, index]) == 0 or np.std(inferred[:, index]) == 0:
            correlation = None
        else:
            correlation = float(np.corrcoef(true[:, index], inferred[:, index])[0, 1])
        dimensions.append(
            {
                "name": name,
                "correlation": correlation,
                "rmse": float(
                    np.sqrt(np.mean((true[:, index] - inferred[:, index]) ** 2))
                ),
                "posterior_mean_std": float(np.std(inferred[:, index])),
                "true_std": float(np.std(true[:, index])),
            }
        )
    metrics: dict[str, object] = {
        "dimensions": dimensions,
        "mean_whitened_posterior_scale": posterior_scale.mean(axis=0).tolist(),
    }
    inferred_by_player = {
        player_id: inferred[index].astype(np.float32)
        for index, player_id in enumerate(player_ids)
    }
    return metrics, inferred_by_player


def _replace_oracle_skills(
    dataset: ActionDataset, inferred_by_player: dict[int, np.ndarray]
) -> ActionDataset:
    inferred = np.asarray(
        [inferred_by_player[int(player_id)] for player_id in dataset.player_indices],
        dtype=np.float32,
    )
    return replace(dataset, skills=inferred)


def _state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def run_learned_experiment(
    config: LearnedExperimentConfig,
    *,
    output_dir: str | Path,
    propensity_path: str | Path = WIN_PROPENSITY_PATH,
) -> dict[str, object]:
    """Run one reproducible development experiment and persist its artifacts."""
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    propensity_path = Path(propensity_path)
    propensity_model = load_win_propensity_model(propensity_path)
    package_root = Path(__file__).resolve().parents[1]
    try:
        propensity_label = str(propensity_path.resolve().relative_to(package_root))
    except ValueError:
        propensity_label = str(propensity_path.resolve())

    print(f"simulating {config.n_players} player histories", flush=True)
    trajectories = simulate_players(
        config.n_players,
        propensity_model,
        seed=config.seed,
        max_attempts=config.max_attempts,
        progress_every=max(1, config.n_players // 10),
    )
    eligible = [
        trajectory
        for trajectory in trajectories
        if any(
            record.attempt_id == config.target_attempt
            for record in trajectory.attempts
        )
    ]
    if len(eligible) < 3:
        raise ValueError("fewer than three players reached the target attempt")
    eligible = _truncate_at_target(eligible, config.target_attempt)
    train_trajectories, validation_trajectories, test_trajectories = (
        split_player_trajectories(
            eligible,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=config.seed,
        )
    )

    train_actions = action_dataset_from_trajectories(train_trajectories)
    validation_actions = action_dataset_from_trajectories(validation_trajectories)
    test_target_actions = _target_action_dataset(
        test_trajectories, config.target_attempt
    )
    action_config = ActionPolicyConfig(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    )
    action_train_config = ActionTrainConfig(
        epochs=config.action_epochs,
        batch_size=config.action_batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed + 1,
        device=config.device,
    )
    device = resolve_device(config.device)

    print("training state-only action baseline", flush=True)
    torch.manual_seed(config.seed + 101)
    state_only = ContinuousActionPolicy(action_config)
    state_history = train_action_policy(
        state_only,
        train_actions,
        validation_actions,
        include_skill=False,
        config=action_train_config,
    )
    state_metrics = evaluate_action_policy(
        state_only,
        test_target_actions,
        include_skill=False,
        device=device,
    )

    print("training oracle-skill action ceiling", flush=True)
    torch.manual_seed(config.seed + 101)
    oracle_skill = ContinuousActionPolicy(action_config)
    oracle_history = train_action_policy(
        oracle_skill,
        train_actions,
        validation_actions,
        include_skill=True,
        config=action_train_config,
    )
    oracle_metrics = evaluate_action_policy(
        oracle_skill,
        test_target_actions,
        include_skill=True,
        device=device,
    )

    training_target_attempts = tuple(
        range(config.first_training_attempt, config.target_attempt + 1)
    )
    train_batches = _prefix_batches(
        train_trajectories,
        target_attempts=training_target_attempts,
        players_per_batch=config.players_per_batch,
    )
    validation_batches = _prefix_batches(
        validation_trajectories,
        target_attempts=(config.target_attempt,),
        players_per_batch=config.players_per_batch,
    )
    test_batches = _prefix_batches(
        test_trajectories,
        target_attempts=(config.target_attempt,),
        players_per_batch=config.players_per_batch,
    )
    torch.manual_seed(config.seed + 202)
    vae = ContinuousCausalVAE(
        PrefixEncoderConfig(hidden_size=config.encoder_hidden_size),
        action_config,
    )
    print("training continuous strict-prefix VAE", flush=True)
    vae_history = train_continuous_vae(
        vae,
        train_batches,
        validation_batches,
        config=VAETrainConfig(
            epochs=config.vae_epochs,
            learning_rate=config.learning_rate,
            kl_warmup_epochs=config.kl_warmup_epochs,
            seed=config.seed + 2,
            device=config.device,
        ),
    )
    batches_per_attempt = (
        len(train_trajectories) + config.players_per_batch - 1
    ) // config.players_per_batch
    win_head_start = (
        config.win_head_first_attempt - config.first_training_attempt
    ) * batches_per_attempt
    win_head_train_batches = train_batches[win_head_start:]
    win_head_validation_batches = _prefix_batches(
        validation_trajectories,
        target_attempts=tuple(
            range(config.win_head_first_attempt, config.target_attempt + 1)
        ),
        players_per_batch=config.players_per_batch,
    )
    print("refitting constrained win head on frozen posteriors", flush=True)
    win_head_fit = fit_win_head(
        vae,
        win_head_train_batches,
        win_head_validation_batches,
        config=WinHeadTrainConfig(
            epochs=config.win_head_epochs,
            learning_rate=config.win_head_learning_rate,
            patience=config.win_head_patience,
            seed=config.seed + 3,
            device=config.device,
        ),
    )
    vae_test_metrics = evaluate_continuous_vae(
        vae, test_batches, device=device
    )
    posterior_metrics, inferred_by_player = _posterior_skill_evaluation(
        vae,
        test_trajectories,
        target_attempt=config.target_attempt,
        players_per_batch=config.players_per_batch,
        device=device,
    )
    inferred_test_actions = _replace_oracle_skills(
        test_target_actions, inferred_by_player
    )
    latent_action_metrics = evaluate_action_policy(
        vae.action_policy,
        inferred_test_actions,
        include_skill=True,
        device=device,
    )

    checkpoints = {
        "state_only": output / "state_only_action.pt",
        "oracle_skill": output / "oracle_skill_action.pt",
        "continuous_vae": output / "continuous_vae.pt",
    }
    evaluation_players_path = output / "evaluation_players.npz"
    np.savez_compressed(
        evaluation_players_path,
        player_id=np.asarray(
            [trajectory.player_id for trajectory in test_trajectories],
            dtype=np.int64,
        ),
        true_skill=np.asarray(
            [trajectory.player.as_array() for trajectory in test_trajectories],
            dtype=np.float32,
        ),
        inferred_skill=np.asarray(
            [
                inferred_by_player[trajectory.player_id]
                for trajectory in test_trajectories
            ],
            dtype=np.float32,
        ),
    )
    torch.save(
        {"config": asdict(action_config), "state_dict": _state_dict_cpu(state_only)},
        checkpoints["state_only"],
    )
    torch.save(
        {"config": asdict(action_config), "state_dict": _state_dict_cpu(oracle_skill)},
        checkpoints["oracle_skill"],
    )
    torch.save(
        {
            "encoder_config": asdict(vae.encoder.config),
            "action_config": asdict(vae.action_policy.config),
            "state_dict": _state_dict_cpu(vae),
        },
        checkpoints["continuous_vae"],
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "model_config": {
            "action": asdict(action_config),
            "encoder": asdict(vae.encoder.config),
        },
        "propensity_artifact": {
            "path": propensity_label,
            "sha256": _sha256(propensity_path),
        },
        "partitions": {
            "simulated_players": len(trajectories),
            "eligible_players": len(eligible),
            "excluded_before_target": len(trajectories) - len(eligible),
            "train_players": len(train_trajectories),
            "validation_players": len(validation_trajectories),
            "test_players": len(test_trajectories),
            "train_actions": len(train_actions),
            "validation_actions": len(validation_actions),
            "test_target_actions": len(test_target_actions),
        },
        "action_models": {
            "state_only": {
                "history": state_history,
                "target_test": state_metrics,
            },
            "oracle_skill": {
                "history": oracle_history,
                "target_test": oracle_metrics,
            },
            "inferred_skill": {
                "target_test": latent_action_metrics,
                "nll_gain_over_state_only": (
                    state_metrics["nll"] - latent_action_metrics["nll"]
                ),
            },
        },
        "continuous_vae": {
            "training_target_attempts": list(training_target_attempts),
            "training_target_actions": sum(
                int(target.actions.numel()) for _, target in train_batches
            ),
            "win_head_fit": win_head_fit,
            "history": vae_history,
            "target_test": vae_test_metrics,
            "posterior": posterior_metrics,
        },
        "checkpoints": {
            name: path.name for name, path in checkpoints.items()
        },
        "oracle_evaluation_artifact": evaluation_players_path.name,
        "runtime_seconds": time.perf_counter() - started,
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
    }
    report_path = output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path}", flush=True)
    return report


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--target-attempt", type=int, default=20)
    parser.add_argument("--first-training-attempt", type=int, default=2)
    parser.add_argument("--win-head-first-attempt", type=int, default=8)
    parser.add_argument("--action-epochs", type=int, default=2)
    parser.add_argument("--vae-epochs", type=int, default=5)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--win-head-epochs", type=int, default=400)
    parser.add_argument("--win-head-patience", type=int, default=50)
    parser.add_argument("--action-batch-size", type=int, default=128)
    parser.add_argument("--players-per-batch", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--encoder-hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--win-head-learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--propensity", type=Path, default=WIN_PROPENSITY_PATH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = LearnedExperimentConfig(
        n_players=args.players,
        max_attempts=args.max_attempts,
        target_attempt=args.target_attempt,
        first_training_attempt=args.first_training_attempt,
        win_head_first_attempt=args.win_head_first_attempt,
        action_epochs=args.action_epochs,
        vae_epochs=args.vae_epochs,
        kl_warmup_epochs=args.kl_warmup_epochs,
        win_head_epochs=args.win_head_epochs,
        win_head_patience=args.win_head_patience,
        action_batch_size=args.action_batch_size,
        players_per_batch=args.players_per_batch,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        encoder_hidden_size=args.encoder_hidden_size,
        learning_rate=args.learning_rate,
        win_head_learning_rate=args.win_head_learning_rate,
        seed=args.seed,
        device=args.device,
    )
    report = run_learned_experiment(
        config,
        output_dir=args.out,
        propensity_path=args.propensity,
    )
    print(
        "target action NLL: "
        f"state={report['action_models']['state_only']['target_test']['nll']:.3f} "
        f"oracle={report['action_models']['oracle_skill']['target_test']['nll']:.3f} "
        f"latent={report['action_models']['inferred_skill']['target_test']['nll']:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["LearnedExperimentConfig", "run_learned_experiment"]