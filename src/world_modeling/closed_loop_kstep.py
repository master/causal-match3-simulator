"""K-step evaluation with engine-selected uniform legal actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from match3_simulator.board import legal_moves, resolve_move
from match3_simulator.learned_model.tokens import N_CELLS, action_to_index
from match3_simulator.scm import LEVELS
from match3_simulator.spec import LevelContext, State
from match3_simulator.world_modeling.decoder import torch_legal_mask
from match3_simulator.world_modeling.kstep import DEFAULT_KS

DEFAULT_ANCHORS = 256
DEFAULT_ROLLOUTS = 4
ANCHOR_CHUNK = 32
STATE_SOURCES: tuple[str, ...] = ("model", "engine")


@dataclass
class ClosedLoopKStepRollouts:
    ks: tuple[int, ...]
    state_source: str
    predicted: dict[int, np.ndarray] = field(default_factory=dict)
    target: dict[int, np.ndarray] = field(default_factory=dict)
    predicted_goals: dict[int, np.ndarray] = field(default_factory=dict)
    target_goals: dict[int, np.ndarray] = field(default_factory=dict)
    n_anchors: int = 0
    rollouts_per_anchor: int = 0


def _anchors(
    batches: list[dict[str, torch.Tensor]],
    *,
    max_anchors: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "boards",
            "goal_colours",
            "moves_left",
            "goals_left",
            "levels",
            "tiers",
            "served_difficulty",
        )
    }
    for batch in batches:
        mask = batch["step_mask"].detach().cpu().bool()
        steps = mask.shape[1]
        for name in ("boards", "goal_colours", "moves_left", "goals_left"):
            collected[name].append(batch[name].detach().cpu()[mask])
        for name in ("levels", "tiers", "served_difficulty"):
            values = batch[name].detach().cpu().unsqueeze(1).expand(-1, steps)
            collected[name].append(values[mask])
    anchors = {name: torch.cat(parts) for name, parts in collected.items()}
    legal = torch.cat(
        [
            torch_legal_mask(anchors["boards"][start : start + 512]).any(dim=1)
            for start in range(0, anchors["boards"].shape[0], 512)
        ]
    )
    valid = (
        (anchors["moves_left"] > 0)
        & (anchors["goals_left"] > 0)
        & legal
    )
    anchors = {name: values[valid] for name, values in anchors.items()}
    if not anchors["boards"].shape[0]:
        raise ValueError("evaluation batches contain no non-terminal legal states")
    count = min(max_anchors, anchors["boards"].shape[0])
    indices = np.sort(
        np.random.default_rng(seed).choice(
            anchors["boards"].shape[0], size=count, replace=False
        )
    )
    selected = torch.as_tensor(indices, dtype=torch.long)
    return {name: values[selected] for name, values in anchors.items()}


def _draw(level: LevelContext, rng: np.random.Generator):
    probabilities = level.weights()

    def draw(n: int, _tag: str) -> np.ndarray:
        return rng.choice(
            level.n_colours, size=n, p=probabilities
        ).astype(np.int8)

    return draw


def _masked_state(
    current: dict[str, torch.Tensor],
    advanced: dict[str, torch.Tensor],
    active: torch.Tensor,
) -> dict[str, torch.Tensor]:
    updated = {}
    for name, values in advanced.items():
        previous = current.get(name)
        if previous is None:
            updated[name] = values
            continue
        shape = (active.shape[0],) + (1,) * (values.ndim - 1)
        updated[name] = torch.where(active.view(shape), values, previous)
    return updated


@torch.no_grad()
def closed_loop_kstep_rollouts(
    dynamics: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    max_anchors: int = DEFAULT_ANCHORS,
    rollouts_per_anchor: int = DEFAULT_ROLLOUTS,
    seed: int = 0,
    state_source: str = "model",
) -> ClosedLoopKStepRollouts:
    if not batches:
        raise ValueError("batches must be non-empty")
    if not ks or min(ks) < 1:
        raise ValueError("ks must be positive horizons")
    if min(max_anchors, rollouts_per_anchor) < 1:
        raise ValueError("evaluation sizes must be positive")
    if rollouts_per_anchor < 2:
        raise ValueError("rollouts_per_anchor must be at least two")
    if state_source not in STATE_SOURCES:
        raise ValueError(f"state_source must be one of {STATE_SOURCES}")
    ks = tuple(sorted(set(int(k) for k in ks)))
    anchors = _anchors(batches, max_anchors=max_anchors, seed=seed)
    device = next(dynamics.parameters()).device
    dynamics.eval()
    result = ClosedLoopKStepRollouts(
        ks=ks,
        state_source=state_source,
        n_anchors=anchors["boards"].shape[0],
        rollouts_per_anchor=rollouts_per_anchor,
    )
    collected = {
        k: {"predicted": [], "target": [], "predicted_goals": [], "target_goals": []}
        for k in ks
    }
    for start in range(0, result.n_anchors, ANCHOR_CHUNK):
        stop = min(start + ANCHOR_CHUNK, result.n_anchors)
        chunk = {name: values[start:stop] for name, values in anchors.items()}
        n_anchors = stop - start
        transition_seed = int(
            np.random.SeedSequence([seed, 2, start]).generate_state(1)[0]
        )
        transition_generator = torch.Generator(device=device).manual_seed(
            transition_seed
        )
        repeat = lambda values: values.repeat_interleave(rollouts_per_anchor, dim=0).to(device)
        predicted_boards = repeat(chunk["boards"])
        goal_colours = repeat(chunk["goal_colours"])
        predicted_moves_left = repeat(chunk["moves_left"])
        predicted_goals_left = repeat(chunk["goals_left"])
        levels = repeat(chunk["levels"])
        tiers = repeat(chunk["tiers"])
        served = repeat(chunk["served_difficulty"])
        context = dynamics.encode_task(levels, tiers, served)
        colour_support = torch.as_tensor(
            [LEVELS[int(index)].n_colours for index in levels.detach().cpu()],
            dtype=torch.long,
            device=device,
        )
        model_state = dynamics.rollout_begin(predicted_boards.shape[0], context)
        model_active = torch.ones(
            predicted_boards.shape[0], dtype=torch.bool, device=device
        )

        engine_states = [
            State(
                board=chunk["boards"][anchor].numpy().reshape(8, 8).copy(),
                moves_left=int(chunk["moves_left"][anchor]),
                goals_left=int(chunk["goals_left"][anchor]),
                goal_colour=int(chunk["goal_colours"][anchor]),
            )
            for anchor in range(n_anchors)
            for _ in range(rollouts_per_anchor)
        ]
        engine_levels = [
            LEVELS[int(chunk["levels"][anchor])]
            for anchor in range(n_anchors)
            for _ in range(rollouts_per_anchor)
        ]
        engine_action_rngs = []
        engine_refill_rngs = []
        for anchor in range(n_anchors):
            for replicate in range(rollouts_per_anchor):
                anchor_id = start + anchor
                engine_action_rngs.append(
                    np.random.default_rng(
                        np.random.SeedSequence([seed, 3, anchor_id, replicate])
                    )
                )
                engine_refill_rngs.append(
                    np.random.default_rng(
                        np.random.SeedSequence([seed, 4, anchor_id, replicate])
                    )
                )

        for step in range(ks[-1]):
            engine_boards = torch.as_tensor(
                np.stack([state.board.reshape(-1) for state in engine_states]),
                dtype=torch.long,
                device=device,
            )
            engine_moves_left = torch.as_tensor(
                [state.moves_left for state in engine_states],
                dtype=torch.long,
                device=device,
            )
            engine_goals_left = torch.as_tensor(
                [state.goals_left for state in engine_states],
                dtype=torch.long,
                device=device,
            )
            engine_actions = []
            action_indices = np.zeros(len(engine_states), dtype=np.int64)
            engine_active_values = np.zeros(len(engine_states), dtype=bool)
            for index, state in enumerate(engine_states):
                if state.terminal:
                    engine_actions.append(None)
                    continue
                moves = legal_moves(state.board)
                if not moves:
                    engine_actions.append(None)
                    continue
                action = moves[
                    int(engine_action_rngs[index].integers(len(moves)))
                ]
                engine_actions.append(action)
                action_indices[index] = action_to_index(action)
                engine_active_values[index] = True
            engine_active = torch.as_tensor(
                engine_active_values, dtype=torch.bool, device=device
            )
            actions = torch.as_tensor(
                action_indices, dtype=torch.long, device=device
            )
            if state_source == "engine":
                input_boards = engine_boards
                input_moves_left = engine_moves_left
                input_goals_left = engine_goals_left
                model_active = engine_active
                observe = True
            else:
                input_boards = predicted_boards
                input_moves_left = predicted_moves_left
                input_goals_left = predicted_goals_left
                model_active &= (
                    engine_active
                    & (predicted_moves_left > 0)
                    & (predicted_goals_left > 0)
                )
                observe = step == 0
            advanced_state, sampled = dynamics.rollout_sample(
                model_state,
                boards=input_boards,
                goal_colours=goal_colours,
                moves_left=input_moves_left,
                goals_left=input_goals_left,
                actions=actions,
                task_context=context,
                colour_support=colour_support,
                observe=observe,
                stochastic=True,
                generator=transition_generator,
            )
            model_state = _masked_state(model_state, advanced_state, model_active)
            predicted_boards = torch.where(
                model_active.unsqueeze(1), sampled["boards"], predicted_boards
            )
            predicted_goals_left = torch.where(
                model_active,
                sampled["goals_left"].long(),
                predicted_goals_left,
            )
            predicted_moves_left = torch.where(
                model_active,
                (input_moves_left - 1).clamp_min(0),
                predicted_moves_left,
            )

            for index, action in enumerate(engine_actions):
                if action is None:
                    continue
                engine_states[index], _ = resolve_move(
                    engine_states[index],
                    action,
                    engine_levels[index],
                    _draw(engine_levels[index], engine_refill_rngs[index]),
                )

            horizon = step + 1
            if horizon not in collected:
                continue
            shape = (n_anchors, rollouts_per_anchor)
            collected[horizon]["predicted"].append(
                predicted_boards.detach().cpu().numpy().reshape(*shape, N_CELLS)
            )
            collected[horizon]["predicted_goals"].append(
                predicted_goals_left.detach().cpu().numpy().reshape(shape)
            )
            collected[horizon]["target"].append(
                np.stack([state.board.reshape(-1) for state in engine_states]).reshape(
                    *shape, N_CELLS
                )
            )
            collected[horizon]["target_goals"].append(
                np.asarray(
                    [state.goals_left for state in engine_states], dtype=np.int64
                ).reshape(shape)
            )

    for k in ks:
        result.predicted[k] = np.concatenate(collected[k]["predicted"])
        result.target[k] = np.concatenate(collected[k]["target"])
        result.predicted_goals[k] = np.concatenate(
            collected[k]["predicted_goals"]
        )
        result.target_goals[k] = np.concatenate(collected[k]["target_goals"])
    return result


def _energy_score(cross_distances: np.ndarray, model_distances: np.ndarray) -> float:
    rollouts = model_distances.shape[1]
    cross = cross_distances.mean(axis=(1, 2))
    within = model_distances.sum(axis=(1, 2)) / (
        rollouts * (rollouts - 1)
    )
    return float((cross - 0.5 * within).mean())


def closed_loop_kstep_metrics(
    rollouts: ClosedLoopKStepRollouts,
) -> dict[str, object]:
    report: dict[str, object] = {
        "ks": list(rollouts.ks),
        "state_source": rollouts.state_source,
        "n_anchors": rollouts.n_anchors,
        "rollouts_per_anchor": rollouts.rollouts_per_anchor,
        "horizons": {},
    }
    for k in rollouts.ks:
        predicted = rollouts.predicted[k]
        target = rollouts.target[k]
        board_distances = (
            predicted[:, :, None, :] != target[:, None, :, :]
        ).mean(axis=-1)
        model_board_distances = (
            predicted[:, :, None, :] != predicted[:, None, :, :]
        ).mean(axis=-1)
        goal_distances = np.abs(
            rollouts.predicted_goals[k][:, :, None]
            - rollouts.target_goals[k][:, None, :]
        )
        model_goal_distances = np.abs(
            rollouts.predicted_goals[k][:, :, None]
            - rollouts.predicted_goals[k][:, None, :]
        )
        report["horizons"][f"k{k}"] = {
            "board_energy_score": _energy_score(
                board_distances, model_board_distances
            ),
            "goals_left_energy_score": _energy_score(
                goal_distances, model_goal_distances
            ),
        }
    return report


@torch.no_grad()
def evaluate_closed_loop_kstep(
    dynamics: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    max_anchors: int = DEFAULT_ANCHORS,
    rollouts_per_anchor: int = DEFAULT_ROLLOUTS,
    seed: int = 0,
    state_source: str = "model",
) -> dict[str, object]:
    return closed_loop_kstep_metrics(
        closed_loop_kstep_rollouts(
            dynamics,
            batches,
            ks=ks,
            max_anchors=max_anchors,
            rollouts_per_anchor=rollouts_per_anchor,
            seed=seed,
            state_source=state_source,
        )
    )


__all__ = [
    "DEFAULT_ANCHORS",
    "DEFAULT_ROLLOUTS",
    "STATE_SOURCES",
    "ClosedLoopKStepRollouts",
    "closed_loop_kstep_metrics",
    "closed_loop_kstep_rollouts",
    "evaluate_closed_loop_kstep",
]
