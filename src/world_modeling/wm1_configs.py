from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from match3_simulator.world_modeling.latent import LatentWorldModel, LatentWorldModelConfig
from match3_simulator.world_modeling.registry import WorldModelConfig, build_world_model
from match3_simulator.world_modeling.transformer import TransitionTransformer, TransitionTransformerConfig

FAMILIES: tuple[str, ...] = ("transformer", "jepa", "jepa-ema", "lewm")
LATENT_FAMILIES: tuple[str, ...] = ("jepa", "jepa-ema", "lewm")
BANDS: dict[str, int] = {"1.0M": 1_000_000, "5.2M": 5_200_000, "14.8M": 14_800_000}
TOLERANCE = 0.02
TOTAL_UPDATES = 11_400
BATCH_EPISODES = 32
LEWM_SIGREG_WEIGHT = 10.0 
# WM-1 head settings shared by every family
WM1_TRANSFORMER_HEADS = dict(cleared_head=True, goal_from_mask=True, use_specials=True, special_head=True, terminal_head=True, round0_clamp=True, residual_goal_min=-10, residual_goal_classes=32)
WM1_LATENT_HEADS = dict(cleared_head=True, goal_head="mask", use_specials=True, special_head=True, terminal_head=True, round0_clamp=True, residual_goal_min=-10, residual_goal_classes=32,
                        readout_detach=True, changed_cell_weight=2.0)


def _heads_for(width: int) -> int:
    heads = max(2, width // 48)
    while width % heads:
        heads -= 1
    return max(heads, 1)


def transformer_config(d_model: int, n_layers: int, **overrides) -> TransitionTransformerConfig:
    """Inputs: width and depth. Outputs: the WM-1 transformer configuration of that shape."""
    return TransitionTransformerConfig(d_model=d_model, n_layers=n_layers, n_heads=_heads_for(d_model), decoder_hidden_size=min(128, d_model), **{**WM1_TRANSFORMER_HEADS, **overrides})


def latent_config(family: str, *, encoder_layers: int, predictor_width: int, predictor_layers: int, latent_channels: int, **overrides) -> LatentWorldModelConfig:
    """Inputs: latent family and shape. Outputs: the WM-1 latent configuration (transformer encoder; family selects objective / target / SIGReg)."""
    if family == "jepa":
        objective = dict(objective="jepa", target="stopgrad", sigreg_rows="cells", sigreg_weight=1.0)
    elif family == "jepa-ema":
        objective = dict(objective="jepa", target="ema", ema_decay=0.99, sigreg_rows="cells", sigreg_weight=1.0)
    elif family == "lewm":
        objective = dict(objective="lewm", target="shared", sigreg_rows="states", sigreg_weight=LEWM_SIGREG_WEIGHT)
    else:
        raise ValueError(f"unknown latent family {family}")
    return LatentWorldModelConfig(encoder="transformer", encoder_layers=encoder_layers, predictor_width=predictor_width, predictor_layers=predictor_layers,
                                  predictor_heads=_heads_for(predictor_width), latent_channels=latent_channels, decoder_hidden_size=min(128, predictor_width),
                                  **{**objective, **WM1_LATENT_HEADS, **overrides})


def trainable_parameters(config: WorldModelConfig) -> tuple[int, int]:
    """Inputs: configuration. Outputs: (trainable, total) parameters of the built model (built on the meta device: exact counts, no memory)."""
    with torch.device("meta"):
        model = build_world_model(config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def _aspect_distance(family: str, shape: dict[str, int]) -> float:
    if family == "transformer":
        return abs(math.log(shape["d_model"] / shape["n_layers"]) - math.log(48.0))
    return abs(math.log(shape["predictor_width"] / shape["predictor_layers"]) - math.log(24.0)) + 0.15 * abs(shape["encoder_layers"] - 3) + 0.5 * abs(math.log(shape["latent_channels"] / 48.0))


def transformer_grid() -> list[dict[str, int]]:
    return [{"d_model": d, "n_layers": n} for d in range(64, 641, 8) for n in range(2, 13)]


def latent_grid() -> list[dict[str, int]]:
    return [{"encoder_layers": e, "predictor_width": w, "predictor_layers": n, "latent_channels": c}
            for e in (2, 3, 4) for w in range(96, 417, 16) for n in range(2, 13) for c in (32, 48, 64)]


def search_band(family: str, target: int, *, tolerance: float = TOLERANCE) -> dict[str, object]:
    """Pick the shape of a family inside the band closest to the reference aspect ratio.

    Inputs: family; target parameter count; tolerance. Outputs: dict with shape, trainable, total, deviation, in_band and the candidate count.
    """
    grid = transformer_grid() if family == "transformer" else latent_grid()
    builder = (lambda s: transformer_config(**s)) if family == "transformer" else (lambda s: latent_config(family, **s))
    candidates = []
    for shape in grid:
        trainable, total = trainable_parameters(builder(shape))
        deviation = (trainable - target) / target
        candidates.append((abs(deviation) <= tolerance, _aspect_distance(family, shape), abs(deviation), shape, trainable, total, deviation))
    in_band = [c for c in candidates if c[0]]
    if in_band:
        chosen = min(in_band, key=lambda c: (c[1], c[2]))
    else:
        chosen = min(candidates, key=lambda c: c[2])
    _, aspect, _, shape, trainable, total, deviation = chosen
    return {"shape": shape, "trainable": trainable, "total": total, "deviation": deviation, "in_band": bool(abs(deviation) <= tolerance), "candidates_in_band": len(in_band),
            "aspect_distance": aspect}


def config_for(family: str, shape: dict[str, int], **overrides) -> WorldModelConfig:
    """Inputs: family and shape (as stored in wm1_configs.json). Outputs: the configuration."""
    if family == "transformer":
        return transformer_config(**shape, **overrides)
    return latent_config(family, **shape, **overrides)


def expected_rows_per_update(lengths: np.ndarray, *, batch_episodes: int = BATCH_EPISODES, samples: int = 2000, seed: int = 0) -> float:
    """Expected padded rows of a random batch (batch_episodes x longest episode in the batch), estimated over seeded random batches.

    Inputs: episode lengths of the training split; batch size; samples; seed. Outputs: expected rows per update.
    """
    rng = np.random.default_rng(seed)
    n = len(lengths)
    totals = [batch_episodes * int(lengths[rng.choice(n, size=batch_episodes, replace=False)].max()) for _ in range(samples)]
    return float(np.mean(totals))


def build_table(*, flops_batch: dict[str, torch.Tensor] | None = None, rows_per_update: float | None = None) -> dict[str, object]:
    """Search all 12 cells and, when a batch is supplied, attach FLOPs per row / per update and the equal-FLOPs update.

    Inputs: optional padded CPU batch for FLOP counting; expected padded rows per update. Outputs: the configs table.
    """
    from match3_simulator.world_modeling.wm1_flops import count_training_flops

    started = time.perf_counter()
    cells: dict[str, dict[str, object]] = {}
    latent_shapes: dict[str, dict[str, object]] = {}
    for band, target in BANDS.items():
        cells[f"transformer-{band}"] = {"family": "transformer", "band": band, "target": target, **search_band("transformer", target)}
        shared = search_band("jepa", target)  # one shape shared by the three latent families
        latent_shapes[band] = shared
        for family in LATENT_FAMILIES:
            config = config_for(family, shared["shape"])
            trainable, total = trainable_parameters(config)
            cells[f"{family}-{band}"] = {"family": family, "band": band, "target": target, "shape": shared["shape"], "trainable": trainable, "total": total,
                                         "deviation": (trainable - target) / target, "in_band": abs(trainable - target) / target <= TOLERANCE,
                                         "candidates_in_band": shared["candidates_in_band"], "aspect_distance": shared["aspect_distance"], "frozen_target_encoder": total - trainable}
    for name, cell in cells.items():
        config = config_for(cell["family"], cell["shape"])
        model = build_world_model(config)
        counts = model.parameter_counts()
        assert counts["total"] == cell["trainable"], (name, counts["total"], cell["trainable"])
        cell["components"] = counts
        cell["config"] = json.loads(json.dumps(asdict(config)))
        cell["frozen_target_encoder"] = model.frozen_parameter_count() if hasattr(model, "frozen_parameter_count") else 0
        if flops_batch is not None:
            estimate = count_training_flops(model, flops_batch)
            cell["flops"] = {"forward_per_row": estimate.forward_per_row, "training_per_row": estimate.training_per_row, "analytic_forward_per_row": estimate.analytic_forward_per_row,
                             "counter_over_analytic": estimate.counter_over_analytic}
            if rows_per_update is not None:
                cell["flops"]["per_update"] = estimate.training_per_row * rows_per_update
                cell["flops"]["budget_total"] = cell["flops"]["per_update"] * TOTAL_UPDATES
    if flops_batch is not None and rows_per_update is not None:
        reference = min(cell["flops"]["budget_total"] for cell in cells.values())
        for cell in cells.values():
            cell["flops"]["equal_flops_reference"] = reference
            cell["equal_flops_update"] = int(min(TOTAL_UPDATES, math.floor(reference / cell["flops"]["per_update"])))
    return {"bands": BANDS, "tolerance": TOLERANCE, "total_updates": TOTAL_UPDATES, "batch_episodes": BATCH_EPISODES, "rows_per_update": rows_per_update,
            "lewm_sigreg_weight": LEWM_SIGREG_WEIGHT, "all_in_band": all(cell["in_band"] for cell in cells.values()), "cells": cells, "seconds": time.perf_counter() - started}


def format_table(table: dict[str, object]) -> str:
    lines = ["| cell | shape | trainable | total (+frozen EMA) | deviation | FLOPs/row train | FLOPs/update | equal-FLOPs update |", "|---|---|---|---|---|---|---|---|"]
    for name, cell in table["cells"].items():
        flops = cell.get("flops", {})
        lines.append(f"| {name} | {cell['shape']} | {cell['trainable']:,} | {cell['total']:,} | {100 * cell['deviation']:+.2f} % | "
                     f"{flops.get('training_per_row', float('nan')) / 1e6:.1f} M | {flops.get('per_update', float('nan')) / 1e12:.3f} T | {cell.get('equal_flops_update', '—')} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/wm1/wm1_configs.json")
    parser.add_argument("--cache", default="runs/wm1/data/natural", help="WM-1 shard caches (for FLOPs per update); omit FLOPs with --no-flops")
    parser.add_argument("--root", default="data/release")
    parser.add_argument("--no-flops", action="store_true")
    args = parser.parse_args()
    flops_batch = None
    rows_per_update = None
    if not args.no_flops:
        from match3_simulator.world_modeling.train import model_batch
        from match3_simulator.world_modeling.wm1_data import EpisodeStore, load_splits

        splits = load_splits(args.root)
        train_store = EpisodeStore(args.cache, "train", splits=splits)
        rows_per_update = expected_rows_per_update(train_store.lengths)
        flops_batch = model_batch(train_store.batch(np.arange(4)))
    table = build_table(flops_batch=flops_batch, rows_per_update=rows_per_update)
    print(format_table(table))
    out = Path(args.out)
    if out.exists():
        previous = json.loads(out.read_text())
        same = all(previous["cells"][k]["shape"] == v["shape"] and previous["cells"][k]["trainable"] == v["trainable"] for k, v in table["cells"].items())
        if not same:
            raise SystemExit(f"{out} exists with different cells; refusing to overwrite")
        print(f"{out} already holds the same cells")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_name(out.name + f".partial-{os.getpid()}")
        partial.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
        os.replace(partial, out)
        print(f"wrote {out}")
    if not table["all_in_band"]:
        missing = [k for k, v in table["cells"].items() if not v["in_band"]]
        raise SystemExit(f"STOP: cells outside ±{100 * TOLERANCE:.0f} %: {missing}")


if __name__ == "__main__":
    main()


__all__ = ["BANDS", "FAMILIES", "LATENT_FAMILIES", "LEWM_SIGREG_WEIGHT", "TOLERANCE", "TOTAL_UPDATES", "WM1_LATENT_HEADS", "WM1_TRANSFORMER_HEADS", "build_table", "config_for",
           "expected_rows_per_update", "format_table", "latent_config", "search_band", "trainable_parameters", "transformer_config"]
