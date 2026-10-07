"""k-step open-loop evaluation: from every logged state, roll the model forward with the logged actions and compare with the logged board k moves later.
Accuracy, macro-F1, changed-cell metrics and a persistence baseline are reported per horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from match3_simulator.learned_model.tokens import N_CELLS
from match3_simulator.scm import LEVELS

DEFAULT_KS: tuple[int, ...] = (1, 3, 7)


@dataclass
class KStepRollouts:
    """Generated and logged boards and counters at each horizon."""

    ks: tuple[int, ...]
    predicted: dict[int, np.ndarray] = field(default_factory=dict)
    target: dict[int, np.ndarray] = field(default_factory=dict)
    anchor: dict[int, np.ndarray] = field(default_factory=dict)
    predicted_goals: dict[int, np.ndarray] = field(default_factory=dict)
    target_goals: dict[int, np.ndarray] = field(default_factory=dict)
    n_anchors: int = 0
    n_dynamics_steps: int = 0


def _colour_support(levels: torch.Tensor) -> torch.Tensor:
    """Look up the colour count of each level.

    Inputs: levels (B,) long.
    Outputs: colour support (B,) long.
    """
    return torch.as_tensor([level.n_colours for level in LEVELS], device=levels.device)[levels.long()]


@torch.no_grad()
def kstep_rollouts(
    dynamics: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    stochastic: bool = False,
    seed: int = 0,
    chunk: int = 4096,
) -> KStepRollouts:
    """Roll the model open loop from every logged anchor of a padded batch for the largest horizon.

    Inputs: a model with encode_task / rollout_begin / rollout_sample; a padded transition batch; horizons; stochastic decoding flag; seed; anchors per chunk.
    Outputs: KStepRollouts with predicted, target and anchor boards and counters per horizon.
    """
    if not ks or min(ks) < 1:
        raise ValueError("ks must be positive horizons")
    ks = tuple(sorted(set(int(k) for k in ks)))
    device = next(dynamics.parameters()).device
    dynamics.eval()
    max_moves = int(dynamics.config.max_moves_left)
    states = torch.cat((batch["boards"], batch["next_boards"][:, -1:]), dim=1).to(device)
    moves = torch.cat((batch["moves_left"], batch["next_moves_left"][:, -1:]), dim=1).to(device)
    goals = torch.cat((batch["goals_left"], batch["next_goals_left"][:, -1:]), dim=1).to(device)
    actions = batch["actions"].to(device)
    lengths = batch["step_mask"].to(device).long().sum(dim=1)
    n_episodes, n_steps = actions.shape
    episode_index, anchor_t = torch.meshgrid(torch.arange(n_episodes, device=device), torch.arange(n_steps, device=device), indexing="ij")
    valid = anchor_t + 1 <= lengths.unsqueeze(1)
    episode_index, anchor_t = episode_index[valid], anchor_t[valid]
    task_context = dynamics.encode_task(batch["levels"].to(device), batch["tiers"].to(device), batch["served_difficulty"].to(device))
    goal_colours = batch["goal_colours"].to(device)[:, 0]
    support = _colour_support(batch["levels"].to(device))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    result = KStepRollouts(ks=ks, n_anchors=int(anchor_t.numel()))
    collected: dict[int, list[tuple[np.ndarray, ...]]] = {k: [] for k in ks}
    for start in range(0, anchor_t.numel(), chunk):
        rows_b, rows_t = episode_index[start : start + chunk], anchor_t[start : start + chunk]
        state = dynamics.rollout_begin(rows_b.numel(), task_context[rows_b])
        board, move, goal = states[rows_b, rows_t], moves[rows_b, rows_t], goals[rows_b, rows_t]
        generated_boards: dict[int, torch.Tensor] = {}
        generated_goals: dict[int, torch.Tensor] = {}
        for step in range(ks[-1]):
            index = rows_t + step
            alive = index + 1 <= lengths[rows_b]
            if not bool(alive.any()):
                break
            action = torch.where(alive, actions[rows_b, index.clamp_max(n_steps - 1)], torch.zeros_like(index))
            state, sampled = dynamics.rollout_sample(
                state, boards=board, goal_colours=goal_colours[rows_b], moves_left=move.clamp(0, max_moves), goals_left=goal.clamp_min(0),
                actions=action, task_context=task_context[rows_b], colour_support=support[rows_b], observe=step == 0, stochastic=stochastic, generator=generator,
            )
            result.n_dynamics_steps += int(alive.sum())
            board = torch.where(alive.unsqueeze(1), sampled["boards"], board)
            goal = torch.where(alive, torch.minimum(goal, sampled["goals_left"].long().clamp_min(0)), goal)
            move = torch.where(alive, (move - 1).clamp_min(0), move)
            if step + 1 in ks:
                generated_boards[step + 1] = board.clone()
                generated_goals[step + 1] = goal.clone()
        for k in ks:
            if k not in generated_boards:
                continue
            ok = rows_t + k <= lengths[rows_b]
            if not bool(ok.any()):
                continue
            b_ok, t_ok = rows_b[ok], rows_t[ok]
            collected[k].append((
                generated_boards[k][ok].cpu().numpy().astype(np.int64), states[b_ok, t_ok + k].cpu().numpy(), states[b_ok, t_ok].cpu().numpy(),
                generated_goals[k][ok].cpu().numpy(), goals[b_ok, t_ok + k].cpu().numpy(),
            ))
    for k in ks:
        if collected[k]:
            parts = list(zip(*collected[k]))
            result.predicted[k], result.target[k], result.anchor[k], result.predicted_goals[k], result.target_goals[k] = (np.concatenate(p) for p in parts)
    return result


def macro_f1_colours(pred: np.ndarray, target: np.ndarray, n_colours: int, mask: np.ndarray | None = None) -> float | None:
    """Average the per-colour F1 over the colours present in the evaluated cells.

    Inputs: predicted and target boards (rows, 64); number of colours; optional boolean cell mask.
    Outputs: macro-F1 or None when no cell is evaluated.
    """
    mask = np.ones_like(target, dtype=bool) if mask is None else mask
    if not mask.any():
        return None
    p, t = pred[mask], target[mask]
    scores = []
    for colour in range(n_colours):
        tp = float(np.sum((p == colour) & (t == colour)))
        fp = float(np.sum((p == colour) & (t != colour)))
        fn = float(np.sum((p != colour) & (t == colour)))
        if tp + fp + fn:
            scores.append(2 * tp / max(2 * tp + fp + fn, 1e-8))
    return float(np.mean(scores)) if scores else None


def binary_prf(pred_positive: np.ndarray, true_positive: np.ndarray) -> dict[str, float]:
    """Compute precision, recall and F1 of a binary prediction.

    Inputs: boolean arrays of predicted and true positives with the same shape.
    Outputs: dict with precision, recall, f1 and positive_rate.
    """
    tp = float(np.sum(pred_positive & true_positive))
    fp = float(np.sum(pred_positive & ~true_positive))
    fn = float(np.sum(~pred_positive & true_positive))
    return {"precision": tp / max(tp + fp, 1e-8), "recall": tp / max(tp + fn, 1e-8), "f1": 2 * tp / max(2 * tp + fp + fn, 1e-8),
            "positive_rate": float(true_positive.mean()) if true_positive.size else 0.0}


def skill_score(accuracy: float, baseline: float) -> float | None:
    """Scale accuracy against a baseline so that 0 equals the baseline and 1 is perfect.

    Inputs: model accuracy; baseline accuracy.
    Outputs: (accuracy - baseline) / (1 - baseline), or None when the baseline is perfect.
    """
    if baseline >= 1.0 - 1e-12:
        return None
    return float((accuracy - baseline) / (1.0 - baseline))


def _horizon_metrics(pred: np.ndarray, target: np.ndarray, anchor: np.ndarray, pred_goals: np.ndarray, target_goals: np.ndarray, n_colours: int) -> dict[str, object]:
    """Compute the metric block of one horizon.

    Inputs: predicted, target and anchor boards (rows, 64); predicted and target goals (rows,); number of colours.
    Outputs: dict of accuracy, F1, changed-cell, change-detection, persistence, skill and goal metrics.
    """
    correct = pred == target
    changed = target != anchor
    accuracy = float(correct.mean())
    persistence = float((anchor == target).mean())
    return {
        "n": int(pred.shape[0]),
        "change_rate": float(changed.mean()),
        "cell_accuracy": accuracy,
        "macro_f1_colours": macro_f1_colours(pred, target, n_colours),
        "board_exact_rate": float(correct.all(axis=1).mean()),
        "changed_cell_accuracy": float(correct[changed].mean()) if changed.any() else None,
        "changed_macro_f1_colours": macro_f1_colours(pred, target, n_colours, changed),
        "unchanged_cell_accuracy": float(correct[~changed].mean()) if (~changed).any() else None,
        "change_detection": binary_prf((pred != anchor).ravel(), changed.ravel()),
        "persistence": {"cell_accuracy": persistence, "macro_f1_colours": macro_f1_colours(anchor, target, n_colours)},
        "skill_score": skill_score(accuracy, persistence),
        "goals_left_mae": float(np.abs(pred_goals - target_goals).mean()),
        "goals_left_exact_rate": float((pred_goals == target_goals).mean()),
    }


def kstep_metrics(rollouts: KStepRollouts, *, n_colours: int) -> dict[str, object]:
    """Turn collected rollouts into per-horizon metric blocks.

    Inputs: KStepRollouts; number of colours.
    Outputs: dict with ks, n_anchors, n_dynamics_steps and horizons {"k1": block, ...}.
    """
    report: dict[str, object] = {"ks": list(rollouts.ks), "n_anchors": rollouts.n_anchors, "n_dynamics_steps": rollouts.n_dynamics_steps, "horizons": {}}
    for k in rollouts.ks:
        if k not in rollouts.predicted:
            report["horizons"][f"k{k}"] = {"n": 0}
            continue
        report["horizons"][f"k{k}"] = _horizon_metrics(rollouts.predicted[k], rollouts.target[k], rollouts.anchor[k], rollouts.predicted_goals[k], rollouts.target_goals[k], n_colours)
    return report


@torch.no_grad()
def evaluate_kstep(
    dynamics: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    seed: int = 0,
    decodings: tuple[str, ...] = ("argmax", "sampled"),
    n_colours: int = 6,
) -> dict[str, object]:
    """Evaluate k-step prediction over several batches for each decoding rule.

    Inputs: a world model; list of padded batches; horizons; seed; decoding rules; number of colours.
    Outputs: dict with ks and decodings {"argmax": metrics, "sampled": metrics}.
    """
    report: dict[str, object] = {"ks": list(ks), "decodings": {}}
    for decoding in decodings:
        merged = KStepRollouts(ks=tuple(sorted(set(ks))))
        parts: dict[int, list[KStepRollouts]] = {k: [] for k in merged.ks}
        for index, batch in enumerate(batches):
            roll = kstep_rollouts(dynamics, batch, ks=ks, stochastic=decoding == "sampled", seed=seed + index)
            merged.n_anchors += roll.n_anchors
            merged.n_dynamics_steps += roll.n_dynamics_steps
            for k in merged.ks:
                if k in roll.predicted:
                    parts[k].append(roll)
        for k in merged.ks:
            if parts[k]:
                for name in ("predicted", "target", "anchor", "predicted_goals", "target_goals"):
                    getattr(merged, name)[k] = np.concatenate([getattr(r, name)[k] for r in parts[k]])
        report["decodings"][decoding] = kstep_metrics(merged, n_colours=n_colours)
    return report


__all__ = ["DEFAULT_KS", "KStepRollouts", "binary_prf", "evaluate_kstep", "kstep_metrics", "kstep_rollouts", "macro_f1_colours", "skill_score"]
