"""
Training FLOPs per transition row of a world model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from match3_simulator.learned_model.tokens import N_CELLS
from match3_simulator.world_modeling.decoder import AutoregressiveBoardDecoder


@dataclass(frozen=True)
class FlopEstimate:
    forward_per_row: float
    training_per_row: float  # forward + backward (3x forward under the 2x-backward convention)
    analytic_forward_per_row: float
    counted_rows: int
    counter_over_analytic: float


def _analytic_linear(module: nn.Linear, rows: float) -> float:
    return 2.0 * rows * module.in_features * module.out_features


def analytic_forward_flops(model: nn.Module, rows: int, steps_per_episode: int) -> float:
    """Dense, attention and GRU matrix-product FLOPs of one forward pass over ``rows`` transition rows (episodes = rows / steps).

    Inputs: model; number of rows; steps per episode (the opening decoders run once per episode).
    Outputs: FLOPs (embeddings, layer norms, activations and softmaxes are ignored — they are < 3 % on these shapes).
    """
    total = 0.0
    counted: set[int] = set()

    def linear_rows(module: nn.Module, prefix: str) -> float:
        if prefix.startswith("initial_"):
            return rows / max(steps_per_episode, 1)
        return rows

    for name, module in model.named_modules():
        if id(module) in counted:
            continue
        if isinstance(module, nn.TransformerEncoderLayer):
            width = module.linear1.in_features
            tokens = 65 if name.startswith("encoder") or name.startswith("target_encoder") else 67
            per_token = _analytic_linear(module.linear1, 1) + _analytic_linear(module.linear2, 1) + 2.0 * width * (3 * width) + 2.0 * width * width  # ffn + qkv + out projections
            attention = 2.0 * 2.0 * tokens * tokens * width  # QK^T and AV
            factor = 0.0 if name.startswith("target_encoder") else 1.0
            total += factor * rows * (tokens * per_token + attention)
            for sub in module.modules():
                counted.add(id(sub))
        elif isinstance(module, nn.GRU):
            hidden = module.hidden_size
            total += rows * N_CELLS * 2.0 * 3 * (module.input_size * hidden + hidden * hidden)
            counted.add(id(module))
        elif isinstance(module, nn.Conv2d):
            total += rows * 2.0 * module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * N_CELLS / max(module.groups, 1)
            counted.add(id(module))
        elif isinstance(module, nn.Linear) and not name.startswith("target_encoder"):
            multiplier = N_CELLS if any(token in name for token in ("cell_in", "cell_out", "cleared_head_net", "colour_head", "special_head_net.network", "board_head", "structured.board.output", "cell_out_")) else 1.0
            if name.startswith("encoder.cell_out") or name.startswith("encoder.global_out"):
                multiplier = N_CELLS if "cell_out" in name else 1.0
            total += _analytic_linear(module, linear_rows(module, name) * multiplier)
            counted.add(id(module))
    return total


def count_training_flops(model: nn.Module, batch: dict[str, torch.Tensor]) -> FlopEstimate:
    """Count forward and forward+backward FLOPs of ``objective`` on a CPU copy for one padded batch and normalise per transition row.

    Inputs: model; padded batch (any device). Outputs: FlopEstimate (rows = batch_size * steps, padding included since compute is spent on it).
    """
    cpu_model = copy.deepcopy(model).cpu().train()
    cpu_batch = {key: value.detach().cpu() for key, value in batch.items()}
    rows = int(cpu_batch["boards"].shape[0] * cpu_batch["boards"].shape[1])
    steps = int(cpu_batch["boards"].shape[1])
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with FlopCounterMode(display=False) as forward_counter:
            with torch.no_grad():
                cpu_model.objective(**cpu_batch)
        forward = float(forward_counter.get_total_flops())
        cpu_model.zero_grad(set_to_none=True)
        with FlopCounterMode(display=False) as training_counter:
            out = cpu_model.objective(**cpu_batch)
            out["loss"].backward()
        training = float(training_counter.get_total_flops())
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    analytic = analytic_forward_flops(cpu_model, rows, steps)
    gru_missing = forward < 0.5 * analytic  # the CPU GRU kernel is not decomposed for the counter on some builds
    if gru_missing:
        gru = sum(rows * N_CELLS * 2.0 * 3 * (m.input_size * m.hidden_size + m.hidden_size * m.hidden_size) for m in cpu_model.modules() if isinstance(m, nn.GRU))
        forward += gru
        training += 3.0 * gru
    return FlopEstimate(forward_per_row=forward / rows, training_per_row=training / rows, analytic_forward_per_row=analytic / rows, counted_rows=rows,
                        counter_over_analytic=(forward / analytic) if analytic else float("nan"))


__all__ = ["FlopEstimate", "analytic_forward_flops", "count_training_flops"]
