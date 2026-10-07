"""Train one world model on the logged transitions of a simulated cohort and evaluate it teacher-forced and k-step.

Run either:
``python -m match3_simulator.world_modeling.train --model lewm --preset lewm-small --out runs/wm/lewm-small``
or ``python -m match3_simulator.world_modeling.train --model jepa-ema --preset jepa-ema-conv --out runs/wm/jepa-ema-conv``.
Add ``--wandb`` to log the validation curve and final test metrics to Weights & Biases.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from match3_simulator.learned_model.data import gameplay_transition_dataset_from_episodes, split_player_trajectories
from match3_simulator.release import ACCEPTED_SPEC_PATH
from match3_simulator.scm import DDA_GAINS, E_SIGMAS
from match3_simulator.spec import BENCHMARK_CONFIG
from match3_simulator.world_modeling.closed_loop_kstep import (
    DEFAULT_ANCHORS,
    DEFAULT_ROLLOUTS,
    STATE_SOURCES,
    evaluate_closed_loop_kstep,
)
from match3_simulator.world_modeling.cleared_mask import augment_batch
from match3_simulator.world_modeling.cohort import CohortSpec, load_or_simulate_cohort, release_dataset_players
from match3_simulator.world_modeling.kstep import DEFAULT_KS, evaluate_kstep
from match3_simulator.world_modeling.latent import LatentWorldModelConfig
from match3_simulator.world_modeling.registry import WorldModelConfig, build_world_model, config_kind
from match3_simulator.world_modeling.transformer import TransitionTransformerConfig

BATCH_KEYS = ("boards", "next_boards", "actions", "goal_colours", "moves_left", "goals_left", "next_moves_left", "cleared_masks", "refill_goal_cleared",
              "next_goals_left", "levels", "tiers", "served_difficulty", "step_mask",
              # WM-1 keys (special grids, terminal flag, settled-mask validity, signed residual); absent in cohorts simulated by the older path
              "specials", "next_specials", "terminal", "mask_valid", "residual_goal")

# Legacy presets retained for reproducibility.
# Preset names start with their --model value: transformer, jepa (stop-gradient target), jepa-ema (EMA target), lewm (shared target).
PRESETS: dict[str, WorldModelConfig] = {
    "transformer-small": TransitionTransformerConfig(d_model=128, n_layers=4, n_heads=4),
    "transformer-large": TransitionTransformerConfig(d_model=256, n_layers=6, n_heads=8),
    "jepa-conv": LatentWorldModelConfig(objective="jepa", encoder="conv", predictor_width=128, predictor_layers=4),
    "jepa-conv-large": LatentWorldModelConfig(objective="jepa", encoder="conv", predictor_width=256, predictor_layers=3, predictor_heads=8, latent_channels=64),
    "jepa-transformer": LatentWorldModelConfig(objective="jepa", encoder="transformer", encoder_layers=3, predictor_width=128, predictor_layers=6, changed_cell_weight=2.0),
    "jepa-transformer-large": LatentWorldModelConfig(objective="jepa", encoder="transformer", encoder_layers=2, predictor_width=256, predictor_layers=4, predictor_heads=8, latent_channels=64, changed_cell_weight=2.0),
    "jepa-ema-conv": LatentWorldModelConfig(objective="jepa", target="ema", ema_decay=0.99, encoder="conv", predictor_width=128, predictor_layers=4),
    "jepa-ema-transformer": LatentWorldModelConfig(objective="jepa", target="ema", ema_decay=0.99, encoder="transformer", encoder_layers=3, predictor_width=128, predictor_layers=6, changed_cell_weight=2.0),
    "lewm-small": LatentWorldModelConfig(objective="lewm", target="shared", sigreg_rows="states", encoder="transformer", encoder_layers=2, predictor_width=128, predictor_layers=2, sigreg_weight=20.0),
    "lewm-medium": LatentWorldModelConfig(objective="lewm", target="shared", sigreg_rows="states", encoder="transformer", encoder_layers=4, predictor_width=128, predictor_layers=4, sigreg_weight=20.0),
    "lewm-deep": LatentWorldModelConfig(objective="lewm", target="shared", sigreg_rows="states", encoder="transformer", encoder_layers=2, predictor_width=128, predictor_layers=6, sigreg_weight=1.0),
    "lewm-conv": LatentWorldModelConfig(objective="lewm", target="shared", sigreg_rows="states", encoder="conv", predictor_width=128, predictor_layers=4, sigreg_weight=10.0),
    # "xl" presets extend the largest measured configuration of each family.
    "transformer-xl": TransitionTransformerConfig(d_model=384, n_layers=8, n_heads=8),
    "transformer-xl-mask": TransitionTransformerConfig(d_model=384, n_layers=8, n_heads=8, cleared_head=True, goal_from_mask=True),
    # Tier 2 kernel fix: Tier 1 plus the k-step open-loop outcome-consistency loss (k in 3, 7, 12) on 8 episodes per update
    "transformer-xl-mask-kstep": TransitionTransformerConfig(d_model=384, n_layers=8, n_heads=8, cleared_head=True, goal_from_mask=True, kstep_horizons=(3, 7, 12), kstep_weight=1.0, kstep_episodes=8),
    "jepa-transformer-xl": LatentWorldModelConfig(objective="jepa", encoder="transformer", encoder_layers=3, predictor_width=192, predictor_layers=8, predictor_heads=6, changed_cell_weight=2.0),
    "jepa-ema-transformer-xl": LatentWorldModelConfig(objective="jepa", target="ema", ema_decay=0.99, encoder="transformer", encoder_layers=3, predictor_width=192, predictor_layers=8, predictor_heads=6, changed_cell_weight=2.0),
    "lewm-xl": LatentWorldModelConfig(objective="lewm", target="shared", sigreg_rows="states", encoder="transformer", encoder_layers=4, predictor_width=192, predictor_layers=8, predictor_heads=6, sigreg_weight=20.0),
}
MODELS: tuple[str, ...] = ("transformer", "jepa", "jepa-ema", "lewm")


@dataclass(frozen=True)
class TrainConfig:
    """Data, optimisation and evaluation settings shared by every world model."""

    label: str = "world-model"
    n_players: int = 512
    max_attempts: int = BENCHMARK_CONFIG.landmark_attempt
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 4201
    assignment: str = "natural"
    accepted_spec: str | None = str(ACCEPTED_SPEC_PATH)
    dataset: str | None = None
    goal_patience: int | None = 5
    augment_mirror: float = 0.0
    augment_colour: float = 0.0
    cache_dir: str | None = "runs/trajectory-cache"
    simulation_workers: int = 8
    batch_episodes: int = 32
    total_updates: int = 11400
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    eval_every: int = 570
    validation_batches: int = 20
    select_by: str = "loss"
    device: str = "cpu"
    ks: tuple[int, ...] = DEFAULT_KS
    max_test_episodes: int = 1500
    eval_batch_episodes: int = 64
    closed_loop_eval_anchors: int = DEFAULT_ANCHORS
    closed_loop_eval_rollouts: int = DEFAULT_ROLLOUTS
    closed_loop_eval_state_sources: tuple[str, ...] = STATE_SOURCES
    wandb: bool = False
    wandb_project: str = "match3-world-models"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_players < 3:
            raise ValueError("n_players must permit three player splits")
        if self.assignment not in ("natural", "randomized"):
            raise ValueError("assignment must be natural or randomized")
        if self.select_by not in ("loss", "cell_accuracy", "outcome"):
            raise ValueError("select_by must be loss, cell_accuracy or outcome")
        counts = (
            self.batch_episodes,
            self.total_updates,
            self.eval_every,
            self.validation_batches,
            self.max_test_episodes,
            self.eval_batch_episodes,
            self.closed_loop_eval_anchors,
            self.closed_loop_eval_rollouts,
            self.simulation_workers,
        )
        if min(counts) < 1:
            raise ValueError("counts must be positive")
        if self.closed_loop_eval_rollouts < 2:
            raise ValueError("closed_loop_eval_rollouts must be at least two")
        if (
            not self.closed_loop_eval_state_sources
            or len(set(self.closed_loop_eval_state_sources))
            != len(self.closed_loop_eval_state_sources)
            or any(
                source not in STATE_SOURCES
                for source in self.closed_loop_eval_state_sources
            )
        ):
            raise ValueError(
                f"closed_loop_eval_state_sources must be unique values from {STATE_SOURCES}"
            )
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.weight_decay < 0 or not self.ks or min(self.ks) < 1:
            raise ValueError("invalid optimizer or evaluation settings")


def model_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep the batch keys the world models consume.

    Inputs: a padded transition batch from GameplayTransitionDataset.batch.
    Outputs: the same dict restricted to BATCH_KEYS.
    """
    return {key: batch[key] for key in BATCH_KEYS if key in batch}


def _episodes_of(trajectories) -> tuple[list, list[int], list[int]]:
    """Flatten trajectories into episodes with their player and attempt ids.

    Inputs: list of PlayerTrajectory.
    Outputs: episodes, player ids, attempt ids (episodes without actions are skipped).
    """
    episodes, players, attempts = [], [], []
    for trajectory in trajectories:
        for record in trajectory.attempts:
            if record.episode.actions:
                episodes.append(record.episode)
                players.append(trajectory.player_id)
                attempts.append(record.attempt_id)
    return episodes, players, attempts


def teacher_forced_metrics(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]]) -> dict[str, object]:
    """Score next-board prediction under teacher forcing over several batches.

    Inputs: a world model; list of padded batches.
    Outputs: dict with loss, board_nll_per_cell, goal_nll, latent_mse, sigreg, cell_accuracy, changed_cell_accuracy, copy_cell_accuracy, skill_score and latent_diagnostics.
    """
    sums = {key: 0.0 for key in ("board_nll", "goal_nll", "counter_mse", "mask_nll", "special_nll", "terminal_nll", "kstep_mse", "loss", "cell_correct", "changed_correct", "changed",
                                 "cells", "transitions", "latent_mse", "sigreg", "special_correct", "special_ambiguous", "terminal_correct", "terminal_rows")}
    model.eval()
    with torch.no_grad():
        for batch in batches:
            out = model.objective(**model_batch(batch))
            mask = batch["step_mask"]
            n = float(mask.sum())
            correct = (out["board_logits"].argmax(-1) == batch["next_boards"]) & mask.unsqueeze(-1)
            changed = (batch["boards"] != batch["next_boards"]) & mask.unsqueeze(-1)
            sums["board_nll"] += float(out["board_nll"]) * n
            sums["goal_nll"] += float(out["goal_nll"]) * n
            sums["loss"] += float(out["loss"]) * n
            sums["latent_mse"] += float(out.get("latent_mse", 0.0)) * n
            sums["sigreg"] += float(out.get("sigreg", 0.0)) * n
            sums["counter_mse"] += float(out.get("counter_mse", 0.0)) * n
            sums["mask_nll"] += float(out.get("mask_nll", 0.0)) * n
            sums["special_nll"] += float(out.get("special_nll", 0.0)) * n
            sums["terminal_nll"] += float(out.get("terminal_nll", 0.0)) * n
            sums["kstep_mse"] += float(out.get("kstep_mse", 0.0)) * n
            sums["cell_correct"] += float(correct.sum())
            sums["changed_correct"] += float((correct & changed).sum())
            sums["changed"] += float(changed.sum())
            sums["cells"] += n * 64
            sums["transitions"] += n
            if out.get("special_logits") is not None and "next_specials" in batch:
                logits = out["special_logits"]
                ambiguous = (torch.isfinite(logits).sum(-1) > 1) & mask.unsqueeze(-1)
                predicted = logits.masked_fill(~torch.isfinite(logits), -1e4).argmax(-1)
                sums["special_correct"] += float(((predicted == batch["next_specials"]) & ambiguous).sum())
                sums["special_ambiguous"] += float(ambiguous.sum())
            if out.get("terminal_logits") is not None:
                target = batch["terminal"].bool() if "terminal" in batch else ((batch["next_goals_left"] <= 0) | (batch["next_moves_left"] <= 0))
                sums["terminal_correct"] += float((((out["terminal_logits"] > 0) == target) & mask).sum())
                sums["terminal_rows"] += n
    t = max(sums["transitions"], 1.0)
    copy = 1.0 - sums["changed"] / max(sums["cells"], 1.0)
    accuracy = sums["cell_correct"] / max(sums["cells"], 1.0)
    diagnostics = model.latent_diagnostics(model_batch(batches[0])) if hasattr(model, "latent_diagnostics") and batches else None
    return {
        "n_transitions": int(t), "loss": sums["loss"] / t, "board_nll_per_cell": sums["board_nll"] / t, "goal_nll": sums["goal_nll"] / t,
        "latent_mse": sums["latent_mse"] / t, "sigreg": sums["sigreg"] / t, "counter_mse": sums["counter_mse"] / t, "mask_nll": sums["mask_nll"] / t,
        "special_nll": sums["special_nll"] / t, "terminal_nll": sums["terminal_nll"] / t, "kstep_mse": sums["kstep_mse"] / t, "cell_accuracy": accuracy,
        "changed_cell_accuracy": sums["changed_correct"] / max(sums["changed"], 1.0), "copy_cell_accuracy": copy,
        "special_accuracy_ambiguous": sums["special_correct"] / sums["special_ambiguous"] if sums["special_ambiguous"] else None,
        "terminal_accuracy": sums["terminal_correct"] / sums["terminal_rows"] if sums["terminal_rows"] else None,
        "skill_score": (accuracy - copy) / (1.0 - copy) if copy < 1.0 else None, "latent_diagnostics": diagnostics,
    }


def _validation_log(metrics: dict[str, object]) -> dict[str, float]:
    """Flatten one validation record into the metrics logged to Weights & Biases.

    Inputs: a record of teacher_forced_metrics() with update, epoch and train_loss added.
    Outputs: dict of val/* and train/* scalars.
    """
    out = {"epoch": metrics["epoch"], "train/loss": metrics["train_loss"], "val/loss": metrics["loss"], "val/board_nll": metrics["board_nll_per_cell"],
           "val/goal_nll": metrics["goal_nll"], "val/cell_accuracy": metrics["cell_accuracy"], "val/changed_cell_accuracy": metrics["changed_cell_accuracy"],
           "val/skill_score": metrics["skill_score"], "val/latent_mse": metrics["latent_mse"], "val/sigreg": metrics["sigreg"], "val/counter_mse": metrics["counter_mse"], "val/mask_nll": metrics["mask_nll"],
           "val/special_nll": metrics.get("special_nll"), "val/terminal_nll": metrics.get("terminal_nll"), "val/special_accuracy_ambiguous": metrics.get("special_accuracy_ambiguous"),
           "val/terminal_accuracy": metrics.get("terminal_accuracy"), "val/outcome_loss": metrics.get("outcome_loss")}
    for key, value in (metrics.get("latent_diagnostics") or {}).items():
        out[f"val/latent/{key}"] = value
    return {k: v for k, v in out.items() if isinstance(v, (int, float)) and v is not None}


def run_summary(report: dict[str, object]) -> dict[str, float]:
    """Flatten the test and compute results of a run report into the scalars compared across runs.

    Inputs: a report dict as written to world-model.json.
    Outputs: dict of test/*, persistence/* and compute/* scalars (skill, macro-F1, changed-cell accuracy, change-detection F1 per horizon; board NLL; salience; parameters, time, memory).
    """
    out: dict[str, float] = {}
    horizons = report["test"]["kstep"]["decodings"]["argmax"]["horizons"]
    for k, block in horizons.items():
        for name, key in (("skill", "skill_score"), ("macro_f1", "macro_f1_colours"), ("cell_accuracy", "cell_accuracy"), ("changed_cell_accuracy", "changed_cell_accuracy")):
            if block.get(key) is not None:
                out[f"test/{name}_{k}"] = float(block[key])
        out[f"test/change_detection_f1_{k}"] = float(block["change_detection"]["f1"])
        out[f"persistence/cell_accuracy_{k}"] = float(block["persistence"]["cell_accuracy"])
        out[f"persistence/macro_f1_{k}"] = float(block["persistence"]["macro_f1_colours"])
    teacher = report["test"]["teacher_forced"]
    out["test/tf_board_nll"] = float(teacher["board_nll_per_cell"])
    out["test/tf_cell_accuracy"] = float(teacher["cell_accuracy"])
    out["test/tf_changed_cell_accuracy"] = float(teacher["changed_cell_accuracy"])
    for key, value in (teacher.get("latent_diagnostics") or {}).items():
        out[f"test/latent/{key}"] = float(value)
    # Closed-loop energy scores per state source ("model" = imagination, "engine" = teacher-forced control); lower is better.
    for source, block in (report["test"].get("closed_loop_kstep") or {}).items():
        for k, scores in block.get("horizons", {}).items():
            for name, value in scores.items():
                out[f"closed_loop/{source}/{name}_{k}"] = float(value)
    training = report["training"]
    out.update({"compute/parameters": float(report["parameters"]["total"]), "compute/train_seconds": float(training["seconds"]), "compute/eval_seconds": float(report["test"]["seconds"]),
                "compute/updates_per_second": float(training["updates_per_second"]), "compute/transitions_per_second": float(training["transitions_per_second"]),
                "compute/best_update": float(training["best_update"])})
    if training.get("peak_memory_bytes"):
        out["compute/peak_memory_mib"] = training["peak_memory_bytes"] / 2**20
    return out


def closed_loop_evaluation(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], config: TrainConfig) -> dict[str, object]:
    """Run the engine-in-the-loop k-step evaluation for every configured state source.

    Inputs: a world model; the test batches restricted to BATCH_KEYS; the training settings (anchors, rollouts, state sources, seed, ks).
    Outputs: dict state source -> closed-loop report (energy scores per horizon).
    """
    return {
        state_source: evaluate_closed_loop_kstep(
            model,
            batches,
            ks=tuple(config.ks),
            max_anchors=config.closed_loop_eval_anchors,
            rollouts_per_anchor=config.closed_loop_eval_rollouts,
            seed=config.seed + 17,
            state_source=state_source,
        )
        for state_source in config.closed_loop_eval_state_sources
    }


def cohort_spec_of(config: TrainConfig) -> CohortSpec:
    """Build the cohort specification of a run: the accepted benchmark's regime by default, the simulator's legacy defaults only when accepted_spec is None.

    Inputs: training settings (assignment selects the regime natural | randomized).
    Outputs: CohortSpec whose cache name encodes churn, mastery, assignment and the accepted-spec hash.
    """
    if config.accepted_spec is None:
        if config.dataset is not None:
            raise ValueError("a release dataset is always an accepted-benchmark cohort")
        gains = DDA_GAINS if config.assignment == "natural" else (0.0, 0.0, 0.0)
        return CohortSpec(n_players=config.n_players, seed=config.seed, max_attempts=config.max_attempts, dda_gains=tuple(gains), e_sigmas=tuple(E_SIGMAS), regime=config.assignment)
    if config.dataset is not None:
        info = release_dataset_players(config.dataset, config.assignment)
        return CohortSpec.from_accepted(n_players=len(info["player_ids"]), seed=info["seed"], max_attempts=info["max_attempts"], regime=config.assignment, path=config.accepted_spec, player_ids=info["player_ids"])
    return CohortSpec.from_accepted(n_players=config.n_players, seed=config.seed, max_attempts=config.max_attempts, regime=config.assignment, path=config.accepted_spec)


def split_cohort(config: TrainConfig, trajectories: list) -> tuple[list, list, list]:
    """Player-disjoint train / validation / test split: the release dataset's splits.json when training on a dataset, otherwise the seeded split."""
    if config.dataset is not None:
        splits = release_dataset_players(config.dataset, config.assignment)["splits"]
        by_id = {t.player_id: t for t in trajectories}
        return [by_id[i] for i in splits["train"]], [by_id[i] for i in splits["validation"]], [by_id[i] for i in splits["test"]]
    return split_player_trajectories(trajectories, validation_fraction=config.validation_fraction, test_fraction=config.test_fraction, seed=config.seed + 1)


def test_batches_of(config: TrainConfig, *, device: torch.device, progress: Callable[[str], None] | None = None) -> tuple[list[dict[str, torch.Tensor]], dict[str, object]]:
    """Rebuild the held-out test batches of a run from its training settings (same cohort, split and episode subset).

    Inputs: training settings; device; optional progress callback.
    Outputs: the padded test batches and the simulation record of the cohort.
    """
    emit = progress or (lambda _: None)
    trajectories, simulation_record = load_or_simulate_cohort(cohort_spec_of(config), cache_dir=config.cache_dir, workers=config.simulation_workers, progress=emit)
    _, _, test = split_cohort(config, trajectories)
    episodes, players, attempts = _episodes_of(test)
    dataset = gameplay_transition_dataset_from_episodes(episodes, player_ids=players, attempt_ids=attempts)
    keep = np.sort(np.random.default_rng(config.seed + 11).choice(dataset.episodes, size=min(config.max_test_episodes, len(dataset.episodes)), replace=False))
    return [dataset.batch(keep[s : s + config.eval_batch_episodes], device=device) for s in range(0, len(keep), config.eval_batch_episodes)], simulation_record


def train_world_model(config: TrainConfig, model_config: WorldModelConfig, *, output_dir: str | Path, progress: Callable[[str], None] | None = None) -> dict[str, object]:
    """Train a world model on the cohort's training players and evaluate it on the held-out players.

    Inputs: training settings; model configuration; output directory; optional progress callback.
    Outputs: the report dict, also written to <output_dir>/world-model.json next to best.pt (and logged to Weights & Biases when config.wandb is set).
    """
    emit = progress or (lambda _: None)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    started = time.perf_counter()
    run = None
    if config.wandb:
        import wandb  # optional dependency, imported only when logging is requested

        run = wandb.init(entity=config.wandb_entity, project=config.wandb_project, name=config.label, group=config.wandb_group, tags=list(config.wandb_tags),
                         config={"train": asdict(config), "model": asdict(model_config), "kind": config_kind(model_config)}, dir=str(output), reinit=True)
    trajectories, simulation_record = load_or_simulate_cohort(cohort_spec_of(config), cache_dir=config.cache_dir, workers=config.simulation_workers, progress=emit)
    train, validation, test = split_cohort(config, trajectories)
    datasets = {}
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        episodes, players, attempts = _episodes_of(split)
        datasets[name] = gameplay_transition_dataset_from_episodes(episodes, player_ids=players, attempt_ids=attempts)
    rng = np.random.default_rng(config.seed)
    train_ids = datasets["train"].episodes
    validation_ids = datasets["validation"].episodes
    rng_val = np.random.default_rng(config.seed + 7)
    val_batches = [datasets["validation"].batch(np.sort(ids), device=device)
                   for ids in np.array_split(rng_val.permutation(validation_ids), max(1, min(config.validation_batches, len(validation_ids) // config.batch_episodes or 1))) if len(ids)][: config.validation_batches]
    emit(f"{config.label}: {len(train_ids)} train episodes, {len(validation_ids)} validation, {len(datasets['test'].episodes)} test; budget {config.total_updates} updates")

    torch.manual_seed(config.seed)
    model = build_world_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    track_memory = device.type == "cuda"
    if track_memory:
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, object]] = []
    best: dict[str, object] = {"loss": float("inf"), "cell_accuracy": -1.0, "outcome": float("inf"), "update": 0}
    goal_guard = {"best": float("inf"), "stale": 0, "stopped_at": None}
    augment_generator = torch.Generator().manual_seed(config.seed + 23)
    updates, epoch, train_seconds, transitions_seen = 0, 0, 0.0, 0
    while updates < config.total_updates and goal_guard["stopped_at"] is None:
        epoch += 1
        order = rng.permutation(train_ids)
        for start in range(0, len(order) - config.batch_episodes + 1, config.batch_episodes):
            if updates >= config.total_updates or goal_guard["stopped_at"] is not None:
                break
            batch = model_batch(datasets["train"].batch(np.sort(order[start : start + config.batch_episodes]), device=device))
            if config.augment_mirror > 0 or config.augment_colour > 0:
                batch = augment_batch(batch, mirror_probability=config.augment_mirror, colour_probability=config.augment_colour, generator=augment_generator)
            torch.manual_seed(int(np.random.SeedSequence([config.seed, updates]).generate_state(1)[0]))
            model.train()
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            out = model.objective(**batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            post_step = getattr(model, "post_optimizer_step", None)  # EMA target encoder follows the online encoder after the step, never inside the loss
            if callable(post_step):
                post_step()
            if track_memory:
                torch.cuda.synchronize(device)
            train_seconds += time.perf_counter() - step_started
            updates += 1
            transitions_seen += int(batch["step_mask"].sum())
            if updates % config.eval_every == 0 or updates == config.total_updates:
                metrics = teacher_forced_metrics(model, val_batches)
                metrics.update({"update": updates, "epoch": epoch, "train_loss": float(out["loss"].detach())})
                history.append(metrics)
                outcome_loss = metrics["goal_nll"] + metrics.get("mask_nll", 0.0)  # the outcome channel: goal head plus, when present, the cleared-cell head
                metrics["outcome_loss"] = outcome_loss
                if config.select_by == "cell_accuracy":
                    improved = metrics["cell_accuracy"] > best["cell_accuracy"]
                elif config.select_by == "outcome":
                    improved = outcome_loss < best["outcome"]
                else:
                    improved = metrics["loss"] < best["loss"]
                if improved:
                    best = {"loss": metrics["loss"], "cell_accuracy": metrics["cell_accuracy"], "outcome": outcome_loss, "update": updates}
                if metrics["goal_nll"] < goal_guard["best"] - 1e-6:
                    goal_guard.update(best=metrics["goal_nll"], stale=0)
                else:
                    goal_guard["stale"] += 1
                    if config.goal_patience is not None and goal_guard["stale"] >= config.goal_patience:
                        goal_guard["stopped_at"] = updates
                        emit(f"{config.label}: goal-head validation NLL has not improved for {config.goal_patience} validations (best {goal_guard['best']:.4f}); early stop at update {updates}")
                    torch.save({"kind": config_kind(model_config), "update": updates, "model_config": asdict(model_config), "state_dict": model.state_dict()}, output / "best.pt")
                diagnostics = metrics.get("latent_diagnostics") or {}
                extra = f" | across-board var {diagnostics['across_board_variance']:.2e} salience {diagnostics['change_salience']:.2f} copy-ratio {diagnostics['copy_ratio']:.2f}" if diagnostics else ""
                emit(f"{config.label}: update {updates} val loss {metrics['loss']:.4f} board NLL {metrics['board_nll_per_cell']:.4f} cell acc {metrics['cell_accuracy']:.3f} changed {metrics['changed_cell_accuracy']:.3f}{extra} ({train_seconds:.0f}s)")
                if run is not None:
                    run.log({**_validation_log(metrics), "train/seconds": train_seconds, "best/update": best["update"]}, step=updates)
    peak = int(torch.cuda.max_memory_allocated(device)) if track_memory else None
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=False)["state_dict"])
    model.eval()
    keep = np.sort(np.random.default_rng(config.seed + 11).choice(datasets["test"].episodes, size=min(config.max_test_episodes, len(datasets["test"].episodes)), replace=False))
    test_batches = [datasets["test"].batch(keep[s : s + config.eval_batch_episodes], device=device) for s in range(0, len(keep), config.eval_batch_episodes)]
    eval_started = time.perf_counter()
    teacher_forced = teacher_forced_metrics(model, test_batches)
    evaluation_batches = [model_batch(b) for b in test_batches]
    kstep = evaluate_kstep(
        model,
        evaluation_batches,
        ks=tuple(config.ks),
        seed=config.seed + 13,
        n_colours=model_config.n_colours,
    )
    closed_loop_kstep = closed_loop_evaluation(model, evaluation_batches, config)
    eval_seconds = time.perf_counter() - eval_started
    report: dict[str, object] = {
        "label": config.label, "kind": config_kind(model_config), "config": json.loads(json.dumps(asdict(config))),
        "model_config": json.loads(json.dumps(asdict(model_config))), "simulation": json.loads(json.dumps(simulation_record)),
        "splits": {name: {"n_players": len(split), "n_episodes": int(len(datasets[name].episodes))} for name, split in (("train", train), ("validation", validation), ("test", test))},
        "parameters": model.parameter_counts(),
        "training": {"updates": updates, "epochs": epoch, "best_update": best["update"], "best_validation_loss": best["loss"], "select_by": config.select_by,
                     "goal_early_stop_update": goal_guard["stopped_at"], "goal_patience": config.goal_patience, "best_goal_nll": goal_guard["best"],
                     "history": history, "seconds": train_seconds, "updates_per_second": updates / max(train_seconds, 1e-9),
                     "transitions_per_second": transitions_seen / max(train_seconds, 1e-9), "peak_memory_bytes": peak},
        "test": {"n_episodes": int(len(keep)), "teacher_forced": teacher_forced, "kstep": kstep,
                 "closed_loop_kstep": closed_loop_kstep, "seconds": eval_seconds},
        "runtime_seconds": time.perf_counter() - started, "device": str(device),
    }
    (output / "world-model.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    emit(f"{config.label}: wrote {output / 'world-model.json'} (train {train_seconds:.0f}s, eval {eval_seconds:.0f}s)")
    if run is not None:
        summary = run_summary(report)
        run.log(summary, step=updates)
        run.summary.update(summary)
        run.save(str(output / "world-model.json"), base_path=str(output), policy="now")
        run.finish()
    return report


def model_config_from_args(args: argparse.Namespace) -> WorldModelConfig:
    """Build the model configuration from a preset and explicit size overrides.

    Inputs: parsed command-line arguments.
    Outputs: a TransitionTransformerConfig or LatentWorldModelConfig.
    """
    if args.preset is not None:
        base = PRESETS[args.preset]
    elif args.model == "transformer":
        base = TransitionTransformerConfig()
    elif args.model == "lewm":
        base = LatentWorldModelConfig(objective="lewm", target="shared", sigreg_rows="states")
    else:
        base = LatentWorldModelConfig(objective="jepa", target="ema" if args.model == "jepa-ema" else "stopgrad")
    overrides: dict[str, object] = {}
    if isinstance(base, TransitionTransformerConfig):
        for flag, field in (("width", "d_model"), ("layers", "n_layers"), ("heads", "n_heads"), ("decoder", "decoder")):
            if getattr(args, flag) is not None:
                overrides[field] = getattr(args, flag)
    else:
        for flag, field in (("width", "predictor_width"), ("layers", "predictor_layers"), ("heads", "predictor_heads"), ("encoder", "encoder"),
                            ("encoder_layers", "encoder_layers"), ("latent_channels", "latent_channels"), ("sigreg_weight", "sigreg_weight"),
                            ("sigreg_rows", "sigreg_rows"), ("changed_cell_weight", "changed_cell_weight"), ("colour_head_weight", "colour_head_weight"),
                            ("rollout", "rollout"), ("ema_decay", "ema_decay")):
            if getattr(args, flag) is not None:
                overrides[field] = getattr(args, flag)
        if args.no_residual:
            overrides["predictor_residual"] = False
    return replace(base, **overrides)


def main() -> None:
    """Parse arguments, train the requested world model and write its report.

    Inputs: command-line arguments.
    Outputs: none (files under --out).
    """
    parser = argparse.ArgumentParser(description="Train and evaluate one match-3 world model.")
    parser.add_argument("--model", choices=MODELS, required=True, help="transformer, jepa (stop-gradient target), jepa-ema (EMA target encoder) or lewm (shared target)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="validated configuration; explicit size flags override its fields")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--width", type=int, default=None, help="transformer d_model or latent predictor width")
    parser.add_argument("--layers", type=int, default=None, help="transformer layers or latent predictor layers")
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--decoder", choices=("independent", "structured"), default=None, help="transformer read-out")
    parser.add_argument("--encoder", choices=("conv", "transformer"), default=None, help="latent encoder")
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--sigreg-weight", type=float, default=None)
    parser.add_argument("--sigreg-rows", choices=("cells", "states"), default=None)
    parser.add_argument("--changed-cell-weight", type=float, default=None)
    parser.add_argument("--colour-head-weight", type=float, default=None)
    parser.add_argument("--no-residual", action="store_true", help="predict the next latent directly instead of the change")
    parser.add_argument("--rollout", choices=("latent", "reencode"), default=None)
    parser.add_argument("--ema-decay", type=float, default=None, help="decay of the EMA target encoder (jepa-ema only)")
    parser.add_argument("--players", type=int, default=512)
    parser.add_argument("--assignment", choices=("natural", "randomized"), default="natural", help="assignment regime of the accepted benchmark")
    parser.add_argument("--accepted-spec", default=str(ACCEPTED_SPEC_PATH), help="accepted_benchmark.json that fixes churn, mastery, assignment and quota calibration")
    parser.add_argument("--legacy-defaults", action="store_true", help="simulate with the simulator's legacy default parameters instead of the accepted benchmark (not for results)")
    parser.add_argument("--dataset", default=None, help="release.py output directory (players, seed and splits replace --players/--seed)")
    parser.add_argument("--augment-mirror", type=float, default=0.0, help="per-episode probability of horizontal mirroring in the training dataloader")
    parser.add_argument("--augment-colour", type=float, default=0.0, help="per-episode probability of a colour permutation (exchangeable-palette levels only)")
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--batch-episodes", type=int, default=32)
    parser.add_argument("--updates", type=int, default=11400)
    parser.add_argument("--eval-every", type=int, default=570)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--select-by", choices=("loss", "cell_accuracy", "outcome"), default="loss", help="checkpoint selection: total validation loss, cell accuracy, or the outcome channel (goal head + cleared-cell head)")
    parser.add_argument("--goal-patience", type=int, default=5, help="stop when the goal head's validation NLL has not improved for this many validations (0 disables)")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--max-test-episodes", type=int, default=1500)
    parser.add_argument("--closed-loop-eval-anchors", type=int, default=DEFAULT_ANCHORS)
    parser.add_argument("--closed-loop-eval-rollouts", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument(
        "--closed-loop-eval-state-sources",
        choices=STATE_SOURCES,
        nargs="+",
        default=list(STATE_SOURCES),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default="runs/trajectory-cache")
    parser.add_argument("--simulation-workers", type=int, default=8)
    parser.add_argument("--wandb", action="store_true", help="log validation curves and test results to Weights & Biases (off by default)")
    parser.add_argument("--wandb-project", default="match3-world-models")
    parser.add_argument("--wandb-entity", default=None, help="W&B team or user; empty string for the account default")
    parser.add_argument("--wandb-group", default=None, help="run group, e.g. the study and family")
    parser.add_argument("--wandb-tags", nargs="*", default=[], help="tags, e.g. family size seed")
    args = parser.parse_args()
    if args.preset is not None and not args.preset.startswith(args.model + "-"):
        parser.error(f"preset {args.preset} does not belong to model {args.model}")
    if args.preset is not None and args.model == "jepa" and args.preset.startswith("jepa-ema-"):
        parser.error(f"preset {args.preset} belongs to model jepa-ema")
    out = Path(args.out)
    if (out / "world-model.json").exists():
        print(f"{out / 'world-model.json'} exists; skipping")
        return
    config = TrainConfig(
        label=args.label or args.preset or f"{args.model}", n_players=args.players, assignment=args.assignment, accepted_spec=None if args.legacy_defaults else str(Path(args.accepted_spec).resolve()),
        dataset=args.dataset, augment_mirror=args.augment_mirror, augment_colour=args.augment_colour,
        seed=args.seed, cache_dir=args.cache_dir,
        simulation_workers=args.simulation_workers, batch_episodes=args.batch_episodes, total_updates=args.updates, eval_every=min(args.eval_every, args.updates),
        learning_rate=args.learning_rate, select_by=args.select_by, goal_patience=args.goal_patience or None, device=args.device, ks=tuple(args.ks), max_test_episodes=args.max_test_episodes,
        closed_loop_eval_anchors=args.closed_loop_eval_anchors, closed_loop_eval_rollouts=args.closed_loop_eval_rollouts,
        closed_loop_eval_state_sources=tuple(args.closed_loop_eval_state_sources),
        wandb=args.wandb, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity or None, wandb_group=args.wandb_group, wandb_tags=tuple(args.wandb_tags),
    )
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    def log(message: str) -> None:
        """Print a timestamped line and append it to the run log.

        Inputs: message text.
        Outputs: none.
        """
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    model_config = model_config_from_args(args)
    log(f"train config {json.dumps(asdict(config))}")
    log(f"model config {json.dumps(asdict(model_config))}")
    train_world_model(config, model_config, output_dir=out, progress=log)


if __name__ == "__main__":
    main()


__all__ = ["BATCH_KEYS", "MODELS", "PRESETS", "TrainConfig", "closed_loop_evaluation", "cohort_spec_of", "model_batch", "model_config_from_args", "run_summary", "teacher_forced_metrics", "test_batches_of", "train_world_model"]
