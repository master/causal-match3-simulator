"""Deterministic trainers for action-policy baselines."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from .action_policy import ContinuousActionPolicy
from .arms import GenerativeWorldModel
from .data import ActionDataset
from .generative import GameplayRSSM
from .model import ContinuousCausalVAE, PredictiveTarget


@dataclass(frozen=True)
class ActionTrainConfig:
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class VAETrainConfig:
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    kl_warmup_epochs: int = 5
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.kl_warmup_epochs < 1:
            raise ValueError("epochs and kl_warmup_epochs must be positive")
        if self.kl_warmup_epochs > self.epochs:
            raise ValueError("kl_warmup_epochs must not exceed epochs")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class WinHeadTrainConfig:
    epochs: int = 400
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 50
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_gameplay_rssm_step(
    model: GameplayRSSM,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    kl_weight: float = 1.0,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    """Run one optimizer step for a logged gameplay transition batch."""
    if gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = model.objective(
        **batch,
        kl_weight=kl_weight,
        sample_posterior=True,
    )
    result["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip
    )
    optimizer.step()
    return {
        name: float(result[name].detach())
        for name in ("loss", "board_nll", "counter_mse", "kl")
    } | {"gradient_norm": float(gradient_norm)}


def train_generative_world_model_step(
    model: GenerativeWorldModel,
    *,
    prefix: dict[str, torch.Tensor],
    target: PredictiveTarget,
    transitions: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    oracle_skill: torch.Tensor | None = None,
    context_kl_weight: float = 1.0,
    dynamics_kl_weight: float = 1.0,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    """Run one matched structural and generative optimizer step."""
    if gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = model.objective(
        prefix=prefix,
        target=target,
        transitions=transitions,
        oracle_skill=oracle_skill,
        context_kl_weight=context_kl_weight,
        dynamics_kl_weight=dynamics_kl_weight,
        sample_context=True,
        sample_dynamics=True,
    )
    loss = result["loss"]
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip
    )
    optimizer.step()
    names = (
        "loss",
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
    metrics = {}
    for name in names:
        value = result[name]
        assert isinstance(value, torch.Tensor)
        metrics[name] = float(value.detach())
    metrics["gradient_norm"] = float(gradient_norm)
    return metrics


@torch.no_grad()
def evaluate_action_policy(
    policy: ContinuousActionPolicy,
    dataset: ActionDataset,
    *,
    include_skill: bool,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, float]:
    policy.eval()
    total_nll = 0.0
    total_correct = 0
    total_uniform_nll = 0.0
    n_rows = 0
    for start in range(0, len(dataset), batch_size):
        rows = np.arange(start, min(start + batch_size, len(dataset)))
        batch = dataset.batch(rows, device=device, include_skill=include_skill)
        action = batch.pop("action")
        log_probabilities = policy(**batch)
        row_index = torch.arange(len(rows), device=device)
        total_nll += float(-log_probabilities[row_index, action].sum())
        total_correct += int((log_probabilities.argmax(dim=1) == action).sum())
        total_uniform_nll += float(
            batch["legal_actions"].sum(dim=1).to(torch.float32).log().sum()
        )
        n_rows += len(rows)
    return {
        "nll": total_nll / n_rows,
        "top1_accuracy": total_correct / n_rows,
        "uniform_legal_nll": total_uniform_nll / n_rows,
        "n_actions": float(n_rows),
    }


def action_nll(
    policy: ContinuousActionPolicy,
    dataset: ActionDataset,
    *,
    include_skill: bool,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    return evaluate_action_policy(
        policy,
        dataset,
        include_skill=include_skill,
        device=device,
        batch_size=batch_size,
    )["nll"]


def train_action_policy(
    policy: ContinuousActionPolicy,
    train_data: ActionDataset,
    validation_data: ActionDataset,
    *,
    include_skill: bool,
    config: ActionTrainConfig = ActionTrainConfig(),
) -> list[dict[str, float]]:
    """Fit masked categorical likelihood on observed state-action pairs."""
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    policy.to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        policy.train()
        rows = rng.permutation(len(train_data))
        total = 0.0
        for start in range(0, len(rows), config.batch_size):
            index = rows[start : start + config.batch_size]
            batch = train_data.batch(
                index, device=device, include_skill=include_skill
            )
            action = batch.pop("action")
            loss = policy.negative_log_likelihood(action=action, **batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_nll": total / len(train_data),
                "validation_nll": action_nll(
                    policy,
                    validation_data,
                    include_skill=include_skill,
                    device=device,
                ),
            }
        )
    return history


def _target_to_device(
    target: PredictiveTarget, device: torch.device
) -> PredictiveTarget:
    return PredictiveTarget(
        **{
            name: value.to(device)
            for name, value in target.__dict__.items()
        }
    )


def _batch_to_device(
    batch: tuple[dict[str, torch.Tensor], PredictiveTarget],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], PredictiveTarget]:
    prefix, target = batch
    return (
        {name: value.to(device) for name, value in prefix.items()},
        _target_to_device(target, device),
    )


@torch.no_grad()
def evaluate_continuous_vae(
    model: ContinuousCausalVAE,
    batches: list[tuple[dict[str, torch.Tensor], PredictiveTarget]],
    *,
    device: torch.device,
) -> dict[str, float]:
    if not batches:
        raise ValueError("at least one validation batch is required")
    model.eval()
    names = (
        "assignment_nll",
        "evidence_nll",
        "win_nll",
        "churn_nll",
        "action_nll",
        "kl",
    )
    totals = {name: 0.0 for name in names}
    weights = {name: 0 for name in names}
    for batch in batches:
        prefix, target = _batch_to_device(batch, device)
        result = model.predictive_objective(
            prefix,
            target,
            kl_weight=1.0,
            sample_posterior=False,
        )
        episode_count = int(target.episode_player.numel())
        action_count = int(target.actions.numel())
        churn_count = int(target.churn_mask.sum())
        channel_weights = {
            "assignment_nll": episode_count,
            "evidence_nll": episode_count,
            "win_nll": episode_count,
            "churn_nll": churn_count,
            "action_nll": action_count,
            "kl": prefix["episode_mask"].shape[0],
        }
        for name, weight in channel_weights.items():
            if weight:
                totals[name] += float(result[name]) * weight
                weights[name] += weight
    metrics = {
        name: totals[name] / weights[name] if weights[name] else 0.0
        for name in names
    }
    metrics["loss"] = sum(metrics.values())
    return {"loss": metrics.pop("loss"), **metrics}


def train_continuous_vae(
    model: ContinuousCausalVAE,
    train_batches: list[tuple[dict[str, torch.Tensor], PredictiveTarget]],
    validation_batches: list[tuple[dict[str, torch.Tensor], PredictiveTarget]],
    *,
    config: VAETrainConfig = VAETrainConfig(),
) -> list[dict[str, float]]:
    """Optimize predictive-prefix likelihood with a linear KL warm-up."""
    if not train_batches or not validation_batches:
        raise ValueError("training and validation batches cannot be empty")
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []
    channel_names = (
        "loss",
        "assignment_nll",
        "evidence_nll",
        "win_nll",
        "churn_nll",
        "action_nll",
        "kl",
    )
    for epoch in range(config.epochs):
        model.train()
        kl_weight = min(1.0, (epoch + 1) / config.kl_warmup_epochs)
        totals = {name: 0.0 for name in channel_names}
        for batch_index in rng.permutation(len(train_batches)):
            prefix, target = _batch_to_device(
                train_batches[int(batch_index)], device
            )
            result = model.predictive_objective(
                prefix, target, kl_weight=kl_weight, sample_posterior=True
            )
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            for name in channel_names:
                totals[name] += float(result[name].detach())

        validation = evaluate_continuous_vae(
            model, validation_batches, device=device
        )
        row = {
            "epoch": float(epoch + 1),
            "kl_weight": kl_weight,
        }
        row.update(
            {
                f"train_{name}": total / len(train_batches)
                for name, total in totals.items()
            }
        )
        row.update(
            {f"validation_{name}": value for name, value in validation.items()}
        )
        history.append(row)
    return history


@torch.no_grad()
def _win_head_data(
    model: ContinuousCausalVAE,
    batches: list[tuple[dict[str, torch.Tensor], PredictiveTarget]],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    model.eval()
    rows: list[tuple[torch.Tensor, ...]] = []
    for batch in batches:
        prefix, target = _batch_to_device(batch, device)
        mean, _ = model.posterior(prefix)
        skill = model.skill_transform(mean)[target.episode_player.long()]
        rows.append(
            (
                skill,
                target.served_difficulty,
                target.levels.long(),
                target.tiers.long(),
                target.outcomes.to(skill.dtype),
            )
        )
    return tuple(torch.cat([row[index] for row in rows]) for index in range(5))


def fit_win_head(
    model: ContinuousCausalVAE,
    train_batches: list[tuple[dict[str, torch.Tensor], PredictiveTarget]],
    validation_batches: list[tuple[dict[str, torch.Tensor], PredictiveTarget]],
    *,
    config: WinHeadTrainConfig = WinHeadTrainConfig(),
) -> dict[str, float]:
    """Refit the constrained win decoder on frozen posterior means."""
    if not train_batches or not validation_batches:
        raise ValueError("training and validation batches cannot be empty")
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    model.to(device)
    train = _win_head_data(model, train_batches, device)
    validation = _win_head_data(model, validation_batches, device)
    optimizer = torch.optim.AdamW(
        model.win_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.win_head.state_dict())
    best_validation = float("inf")
    best_train = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(config.epochs):
        model.win_head.train()
        train_logits = model.win_head.logits(
            train[1], train[0], train[2], train[3]
        )
        train_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            train_logits, train[4]
        )
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

        model.win_head.eval()
        with torch.no_grad():
            validation_logits = model.win_head.logits(
                validation[1], validation[0], validation[2], validation[3]
            )
            validation_loss = float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    validation_logits, validation[4]
                )
            )
        epochs_run = epoch + 1
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_train = float(train_loss.detach())
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.win_head.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    model.win_head.load_state_dict(best_state)
    model.eval()
    return {
        "best_epoch": float(best_epoch),
        "epochs_run": float(epochs_run),
        "train_bce": best_train,
        "validation_bce": best_validation,
        "n_train_episodes": float(len(train[4])),
        "n_validation_episodes": float(len(validation[4])),
    }


__all__ = [
    "ActionTrainConfig",
    "VAETrainConfig",
    "WinHeadTrainConfig",
    "action_nll",
    "evaluate_action_policy",
    "resolve_device",
    "train_generative_world_model_step",
    "train_gameplay_rssm_step",
    "train_action_policy",
    "evaluate_continuous_vae",
    "fit_win_head",
    "train_continuous_vae",
]