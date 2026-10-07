"""Train one WM-1 model with AdamW (lr 3e-4, weight decay 1e-4), 32 episodes per batch, and gradient clipping at 1.0.

The seed controls only (i) parameter initialisation, (ii) the minibatch order, and (iii) the model's own sampling. The data, split, and validation
batches use a constant seed and are fixed. Every run records processed examples (transitions), padded rows, tokens
(67 per state), equivalent dataset passes, cumulative training FLOPs (FLOPs per row times padded rows), wall time, updates per second, peak GPU memory, and the GPU.
``best.pt`` is the outcome-selected checkpoint; ``ckpt-equal-flops.pt`` is the state at the cell's equal-FLOPs update (wm1_configs.json); ``train-state.pt``
allows resuming an interrupted run. A non-finite loss writes STOP.md and raises.

    python -m match3_simulator.world_modeling.wm1_train --cell transformer-5.2M --seed 4201 --out runs/wm1/transformer-5.2M-4201.partial [--wandb]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from match3_simulator.world_modeling.registry import WorldModelConfig, build_world_model, config_from_dict, config_kind
from match3_simulator.world_modeling.train import BATCH_KEYS, _validation_log, model_batch, teacher_forced_metrics
from match3_simulator.world_modeling.wm1_configs import BATCH_EPISODES, TOTAL_UPDATES, config_for
from match3_simulator.world_modeling.wm1_data import EpisodeStore, Splits, load_splits
from match3_simulator.world_modeling.wm1_flops import count_training_flops
from match3_simulator.world_modeling.wm1_verify import provenance

TOKENS_PER_STATE = 67  # 64 tiles + action + task + counter tokens
VALIDATION_SEED = 7  # constant: the validation batches never depend on the run seed
SEEDS: tuple[int, ...] = (4201, 4202, 4203)


class NonFiniteLoss(RuntimeError):
    """Stop condition: the training loss became NaN or infinite."""


@dataclass(frozen=True)
class WM1TrainConfig:
    """The fixed WM-1 training contract plus run identity."""

    cell: str
    seed: int
    dataset_root: str = "data/release"
    cache_dir: str = "runs/wm1/data/natural"
    configs_path: str = "runs/wm1/wm1_configs.json"
    target: str = "settled"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_episodes: int = BATCH_EPISODES
    gradient_clip: float = 1.0
    total_updates: int = TOTAL_UPDATES
    eval_every: int = 285
    validation_batches: int = 40
    validation_batch_episodes: int = 32
    select_by: str = "outcome"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb: bool = False
    wandb_project: str = "match3-world-models"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()
    label: str | None = None
    config_overrides: str | None = None  # JSON dict applied on top of the cell's configuration (ablations: e.g. {"sigreg_weight": 20})

    def __post_init__(self) -> None:
        if self.select_by != "outcome":
            raise ValueError("WM-1 selects checkpoints by the validation outcome channel")
        if min(self.batch_episodes, self.total_updates, self.eval_every, self.validation_batches) < 1:
            raise ValueError("counts must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimiser settings")


def load_cell(configs_path: str | Path, cell: str, overrides: dict | None = None) -> tuple[WorldModelConfig, dict[str, object]]:
    """Inputs: wm1_configs.json path; cell name; optional overrides. Outputs: (model configuration, cell record)."""
    table = json.loads(Path(configs_path).read_text())
    if cell not in table["cells"]:
        raise KeyError(f"unknown cell {cell}; known: {sorted(table['cells'])}")
    record = table["cells"][cell]
    config = config_for(record["family"], record["shape"], **(overrides or {}))
    return config, record


def validation_episode_indices(store: EpisodeStore, *, batches: int, batch_episodes: int) -> list[np.ndarray]:
    """Fixed validation batches (constant seed, independent of the run seed).

    Inputs: validation store; number of batches; episodes per batch. Outputs: list of sorted episode-position arrays.
    """
    rng = np.random.default_rng(VALIDATION_SEED)
    order = rng.permutation(len(store))
    return [np.sort(order[i * batch_episodes : (i + 1) * batch_episodes]) for i in range(min(batches, len(store) // batch_episodes))]


def training_order(seed: int, epoch: int, n_episodes: int) -> np.ndarray:
    """Minibatch order of one pass: a permutation drawn from the run seed and the epoch index."""
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(n_episodes)


def train(config: WM1TrainConfig, *, output_dir: str | Path, progress=None) -> dict[str, object]:
    """Train one cell / seed under the contract and write best.pt, ckpt-equal-flops.pt, train-state.pt and training.json under output_dir.

    Inputs: WM1TrainConfig; output directory (a .partial run directory); optional progress callback.
    Outputs: the training record (also written to <output_dir>/training.json).
    """
    emit = progress or (lambda _: None)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    overrides = json.loads(config.config_overrides) if config.config_overrides else None
    model_config, cell = load_cell(config.configs_path, config.cell, overrides)
    label = config.label or f"{config.cell}-seed{config.seed}"
    splits = load_splits(config.dataset_root)
    train_store = EpisodeStore(config.cache_dir, "train", splits=splits, target=config.target)
    validation_store = EpisodeStore(config.cache_dir, "validation", splits=splits, target=config.target)
    val_indices = validation_episode_indices(validation_store, batches=config.validation_batches, batch_episodes=config.validation_batch_episodes)
    val_batches = [validation_store.batch(ids, device=device) for ids in val_indices]
    emit(f"{label}: {len(train_store)} train episodes / {train_store.n_rows} transitions ({train_store.n_players} players); validation {len(val_indices)} batches x {config.validation_batch_episodes} episodes")

    torch.manual_seed(config.seed)
    model = build_world_model(model_config).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    flops = count_training_flops(model, model_batch(train_store.batch(np.arange(4))))
    equal_flops_update = int(cell.get("equal_flops_update", config.total_updates))
    track_memory = device.type == "cuda"
    if track_memory:
        torch.cuda.reset_peak_memory_stats(device)
    state_path = output / "train-state.pt"
    history: list[dict[str, object]] = []
    best = {"outcome": float("inf"), "update": 0}
    counters = {"updates": 0, "epoch": 0, "transitions": 0, "padded_rows": 0, "train_seconds": 0.0, "flops": 0.0, "position": 0}
    history_started_here = True
    resumed_without_peak = False
    if state_path.exists():
        history_started_here = False
        saved = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, counters = saved["history"], saved["best"], saved["counters"]
        resumed_without_peak = counters.get("peak_memory") is None
        emit(f"{label}: resumed at update {counters['updates']} from {state_path}")

    def save_state() -> None:
        counters["wall_seconds"] = float(counters.get("wall_seconds_before_session", 0.0)) + (time.perf_counter() - session_started)
        if track_memory:
            counters["peak_memory"] = max(int(counters.get("peak_memory") or 0), int(torch.cuda.max_memory_allocated(device)))
            counters["peak_reserved"] = max(int(counters.get("peak_reserved") or 0), int(torch.cuda.max_memory_reserved(device)))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best": best, "counters": counters}, output / "train-state.pt.tmp")
        (output / "train-state.pt.tmp").replace(state_path)

    def checkpoint(path: Path) -> None:
        torch.save({"kind": config_kind(model_config), "update": counters["updates"], "model_config": asdict(model_config), "state_dict": model.state_dict(), "cell": config.cell, "seed": config.seed}, path)

    run = None
    if config.wandb:
        import wandb

        run = wandb.init(entity=config.wandb_entity, project=config.wandb_project, name=label, group=config.wandb_group or f"wm1/{cell['family']}", tags=list(config.wandb_tags),
                         config={"train": asdict(config), "model": asdict(model_config), "cell": {k: v for k, v in cell.items() if k not in ("config",)}}, dir=str(output), reinit=True,
                         resume="allow", id=None)

    def validate(final: bool = False) -> dict[str, object]:
        metrics = teacher_forced_metrics(model, val_batches)
        metrics["outcome_loss"] = float(metrics["goal_nll"]) + float(metrics["mask_nll"])
        metrics.update({"update": counters["updates"], "epoch": counters["epoch"], "train_loss": None if not math.isfinite(last_loss) else last_loss, "train_seconds": counters["train_seconds"],
                        "flops": counters["flops"], "transitions": counters["transitions"]})
        history.append(metrics)
        if metrics["outcome_loss"] < best["outcome"]:
            best.update(outcome=metrics["outcome_loss"], update=counters["updates"], loss=metrics["loss"])
            checkpoint(output / "best.pt")
        diagnostics = metrics.get("latent_diagnostics") or {}
        extra = f" | salience {diagnostics['change_salience']:.2f} copy-ratio {diagnostics['copy_ratio']:.2f}" if diagnostics else ""
        emit(f"{label}: update {counters['updates']} val loss {metrics['loss']:.4f} outcome {metrics['outcome_loss']:.4f} (mask {metrics['mask_nll']:.4f} goal {metrics['goal_nll']:.4f}) board NLL {metrics['board_nll_per_cell']:.4f} "
             f"special {metrics['special_nll']:.4f} terminal acc {metrics['terminal_accuracy'] if metrics['terminal_accuracy'] is None else round(metrics['terminal_accuracy'], 3)} cell acc {metrics['cell_accuracy']:.3f}{extra} "
             f"({counters['train_seconds']:.0f}s, {counters['updates'] / max(counters['train_seconds'], 1e-9):.2f} upd/s)")
        if run is not None:
            run.log({**_validation_log(metrics), "train/seconds": counters["train_seconds"], "train/flops": counters["flops"], "train/transitions": counters["transitions"], "best/update": best["update"]}, step=counters["updates"])
        return metrics

    last_loss = float("nan")
    n_train = len(train_store)
    started = time.perf_counter()
    session_started = started
    counters["wall_seconds_before_session"] = float(counters.get("wall_seconds", 0.0))
    if counters["updates"] == 0 and not history:
        validate()  # untrained reference point (update 0) so every per-head history starts at initialisation
    while counters["updates"] < config.total_updates:
        order = training_order(config.seed, counters["epoch"], n_train)
        start = counters["position"]
        while start + config.batch_episodes <= n_train and counters["updates"] < config.total_updates:
            batch = model_batch(train_store.batch(np.sort(order[start : start + config.batch_episodes]), device=device))
            torch.manual_seed(int(np.random.SeedSequence([config.seed, counters["updates"]]).generate_state(1)[0]))
            model.train()
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            out = model.objective(**batch)
            loss = out["loss"]
            if not torch.isfinite(loss):
                (output / "STOP.md").write_text(f"# STOP\n\nNon-finite loss at update {counters['updates'] + 1} of {label} (loss={float(loss)}).\nPer-head: " +
                                                ", ".join(f"{k}={float(v):.4f}" for k, v in out.items() if v is not None and getattr(v, 'ndim', 1) == 0) + "\n")
                raise NonFiniteLoss(f"{label}: non-finite loss at update {counters['updates'] + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            post_step = getattr(model, "post_optimizer_step", None)
            if callable(post_step):
                post_step()
            if track_memory:
                torch.cuda.synchronize(device)
            counters["train_seconds"] += time.perf_counter() - step_started
            counters["updates"] += 1
            rows = int(batch["boards"].shape[0] * batch["boards"].shape[1])
            counters["padded_rows"] += rows
            counters["transitions"] += int(batch["step_mask"].sum())
            counters["flops"] += flops.training_per_row * rows
            counters["position"] = start + config.batch_episodes
            last_loss = float(loss.detach())
            start += config.batch_episodes
            if counters["updates"] == equal_flops_update:
                checkpoint(output / "ckpt-equal-flops.pt")
                metrics = validate()
                metrics["equal_flops_checkpoint"] = True
            elif counters["updates"] % config.eval_every == 0 or counters["updates"] == config.total_updates:
                validate()
            if counters["updates"] % config.eval_every == 0:
                save_state()
        if start + config.batch_episodes > n_train:
            counters["epoch"] += 1
            counters["position"] = 0
    if not history or history[-1]["update"] != counters["updates"]:
        validate(final=True)
    save_state()
    peak = max(int(torch.cuda.max_memory_allocated(device)), int(counters.get("peak_memory") or 0)) if track_memory else None
    peak_reserved = max(int(torch.cuda.max_memory_reserved(device)), int(counters.get("peak_reserved") or 0)) if track_memory else None
    peak_source = "measured"
    if track_memory and resumed_without_peak and counters["updates"] > 0:
        # resumed from a state saved before peaks were persisted: fall back to the 200-update profile of the same cell (batch shapes are fixed, so the peak is stable)
        profile = Path("runs/wm1/profile") / config.cell / "training.json"
        if profile.exists():
            peak = int(json.loads(profile.read_text())["compute"]["peak_memory_allocated_bytes"] or peak)
            peak_source = "profile (run resumed after a post-training crash; see HISTORY.md)"
    record = {
        "label": label, "cell": config.cell, "seed": config.seed, "family": cell["family"], "band": cell["band"], "kind": config_kind(model_config), "status": "trained",
        "config": asdict(config), "model_config": json.loads(json.dumps(asdict(model_config))), "config_overrides": overrides,
        "parameters": {"trainable": trainable, "total": total_params, "frozen_target_encoder": total_params - trainable, "components": model.parameter_counts()},
        "optimizer": {"name": "AdamW", "learning_rate": config.learning_rate, "weight_decay": config.weight_decay, "betas": [0.9, 0.999], "scheduler": None, "gradient_clip": config.gradient_clip},
        "data": {"train_players": train_store.n_players, "train_episodes": len(train_store), "train_transitions": train_store.n_rows, "validation_players": validation_store.n_players,
                 "validation_batches": len(val_indices), "validation_batch_episodes": config.validation_batch_episodes, "validation_seed": VALIDATION_SEED, "target": config.target,
                 "train_summary": train_store.summary(), "validation_summary": validation_store.summary()},
        "compute": {"updates": counters["updates"], "epochs_started": counters["epoch"] + 1, "transitions_processed": counters["transitions"], "padded_rows_processed": counters["padded_rows"],
                    "tokens_processed": counters["padded_rows"] * TOKENS_PER_STATE, "tokens_per_state": TOKENS_PER_STATE, "dataset_passes": counters["transitions"] / train_store.n_rows,
                    "flops_training_per_row": flops.training_per_row, "flops_forward_per_row": flops.forward_per_row, "flops_counter_over_analytic": flops.counter_over_analytic,
                    "cumulative_training_flops": counters["flops"], "train_seconds": counters["train_seconds"], "wall_seconds": float(counters.get("wall_seconds_before_session", 0.0)) + (time.perf_counter() - started),
                    "updates_per_second": counters["updates"] / max(counters["train_seconds"], 1e-9), "transitions_per_second": counters["transitions"] / max(counters["train_seconds"], 1e-9),
                    "peak_memory_allocated_bytes": peak, "peak_memory_reserved_bytes": peak_reserved, "peak_memory_source": peak_source, "gpu": torch.cuda.get_device_name(device) if track_memory else None, "device": str(device),
                    "equal_flops_update": equal_flops_update},
        "selection": {"select_by": config.select_by, "best_update": best["update"], "best_outcome_loss": best["outcome"], "eval_every": config.eval_every, "history": history},
        "checkpoints": {"best": "best.pt", "equal_flops": "ckpt-equal-flops.pt" if (output / "ckpt-equal-flops.pt").exists() else None},
        "provenance": provenance(config.dataset_root),
    }
    (output / "training.json").write_text(json.dumps(_json_safe(record), indent=2, sort_keys=True, allow_nan=False, default=float) + "\n")
    emit(f"{label}: trained {counters['updates']} updates in {counters['train_seconds']:.0f}s ({record['compute']['updates_per_second']:.2f} upd/s), best outcome {best['outcome']:.4f} @ {best['update']}, "
         f"peak {peak / 2**30 if peak else 0:.2f} GiB, {counters['flops'] / 1e15:.3f} PFLOP")
    if run is not None:
        run.summary.update({"best/update": best["update"], "best/outcome_loss": best["outcome"], "compute/peak_memory_gib": (peak or 0) / 2**30, "compute/train_seconds": counters["train_seconds"],
                            "compute/updates_per_second": record["compute"]["updates_per_second"], "compute/cumulative_flops": counters["flops"], "params/trainable": trainable})
        run.finish()
    return record


def _json_safe(value):
    """Replace non-finite floats by None (strict JSON) recursively."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_checkpoint(path: str | Path, device: torch.device) -> torch.nn.Module:
    """Inputs: checkpoint path; device. Outputs: the model in eval mode."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_world_model(config_from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=TOTAL_UPDATES)
    parser.add_argument("--eval-every", type=int, default=285)
    parser.add_argument("--validation-batches", type=int, default=40)
    parser.add_argument("--configs", default="runs/wm1/wm1_configs.json")
    parser.add_argument("--cache", default="runs/wm1/data/natural")
    parser.add_argument("--root", default="data/release")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--label", default=None)
    parser.add_argument("--config-overrides", default=None, help="JSON dict of model-config overrides (ablations)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="match3-world-models")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    family, band = args.cell.rsplit("-", 1)
    tags = tuple(args.wandb_tags) if args.wandb_tags is not None else ("w0", "wm1", family, band, f"seed{args.seed}")
    config = WM1TrainConfig(cell=args.cell, seed=args.seed, dataset_root=args.root, cache_dir=args.cache, configs_path=args.configs, total_updates=args.updates, eval_every=min(args.eval_every, args.updates),
                            validation_batches=args.validation_batches, device=args.device, wandb=args.wandb, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity or None,
                            wandb_group=args.wandb_group or f"wm1/{family}", wandb_tags=tags, label=args.label, config_overrides=args.config_overrides)
    if (out / "training.json").exists():
        log(f"{out / 'training.json'} exists; skipping training")
        return
    log(f"train config {json.dumps(asdict(config))}")
    train(config, output_dir=out, progress=log)


if __name__ == "__main__":
    main()


__all__ = ["NonFiniteLoss", "SEEDS", "TOKENS_PER_STATE", "VALIDATION_SEED", "WM1TrainConfig", "load_cell", "load_checkpoint", "train", "training_order", "validation_episode_indices"]
