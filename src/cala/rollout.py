from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from match3_simulator.scm import LEVELS, TIER_MOVE_BUDGETS
from match3_simulator.spec import State
from match3_simulator.world_modeling.cala_support import _EngineRows, legal_masks
from match3_simulator.world_modeling.decoder import torch_legal_mask
from match3_simulator.experiments.guard import GuardViolation
from match3_simulator.experiments.rollout_utils import Rows, _sample_actions, chunk_generator, margin_of

PolicyFn4 = Callable[..., torch.Tensor]


class ContextRecorder:
    """Wraps a policy_fn and records the identity of the context rows it was called with at every step (fixed-context verification)."""

    def __init__(self, policy_fn: PolicyFn4, context: torch.Tensor):
        self.policy_fn = policy_fn
        self.context = context
        self.seen: list[torch.Tensor] = []

    def __call__(self, boards, goal_colour, moves_left, goals_left, legal, idx, specials=None):
        self.seen.append(self.context[idx].detach().clone())
        return self.policy_fn(boards, goal_colour, moves_left, goals_left, legal, idx, specials)

    def fixed(self) -> bool:
        return all(torch.equal(self.seen[0][: s.shape[0]], s) for s in self.seen) if self.seen else True


@torch.no_grad()
def imagine_rows4(kernel: torch.nn.Module, policy_fn: PolicyFn4, rows: Rows, *, seed: int, temperature: float = 1.0, chunk: int = 4096, progress=None) -> dict[str, np.ndarray | int | float]:
    """As ``cala.rollout.imagine_rows``; policy_fn(boards, goal_colour, moves_left, goals_left, legal, idx, specials)."""
    device = next(kernel.parameters()).device
    n = len(rows)
    started = time.perf_counter()
    won = np.zeros(n, np.int8)
    final_moves = np.zeros(n, np.float64)
    final_goals = np.zeros(n, np.float64)
    n_actions = 0
    stripe_states = 0
    max_moves = int(kernel.config.max_moves_left)
    for ci, start in enumerate(range(0, n, chunk)):
        idx = np.arange(start, min(start + chunk, n))
        m = len(idx)
        boards = torch.as_tensor(rows.board[idx], dtype=torch.long, device=device)
        specials = torch.zeros_like(boards)
        moves = torch.as_tensor(rows.moves[idx], dtype=torch.long, device=device)
        goals = torch.as_tensor(rows.goals[idx], dtype=torch.long, device=device)
        colours = torch.as_tensor(rows.goal_colour[idx], dtype=torch.long, device=device)
        levels = torch.as_tensor(rows.level[idx], dtype=torch.long, device=device)
        task = kernel.encode_task(levels, torch.as_tensor(rows.tier[idx], dtype=torch.long, device=device), torch.as_tensor(rows.served[idx], dtype=torch.float32, device=device))
        support = torch.as_tensor([LEVELS[int(l)].n_colours for l in rows.level[idx]], dtype=torch.long, device=device)
        carried = kernel.rollout_begin(m, task)
        generator = chunk_generator(device, seed, ci)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        for step in range(int(rows.moves[idx].max())):
            legal = torch_legal_mask(boards)
            active = (goals > 0) & (moves > 0) & legal.any(dim=1)
            if not bool(active.any()):
                break
            legal_in = legal.clone()
            legal_in[~active, 0] = True
            logp = policy_fn(boards, colours, moves.clamp(max=max_moves), goals, legal_in, idx_t, specials)
            actions = _sample_actions(logp, legal, active, temperature=temperature, generator=generator)
            carried, sampled = kernel.rollout_sample(carried, boards=boards, goal_colours=colours, moves_left=moves.clamp(max=max_moves), goals_left=goals, actions=actions, task_context=task,
                                                     colour_support=support, specials=specials, observe=step == 0, stochastic=True, generator=generator)
            stripe_states += int(((specials != 0).any(dim=1) & active).sum())
            boards = torch.where(active.unsqueeze(1), sampled["boards"].long(), boards)
            specials = torch.where(active.unsqueeze(1), sampled["specials"].long(), specials)
            goals = torch.where(active, torch.minimum(goals, sampled["goals_left"].long().clamp_min(0)), goals)
            moves = torch.where(active, moves - 1, moves)
            n_actions += int(active.sum())
        won[idx] = (goals <= 0).cpu().numpy().astype(np.int8)
        final_moves[idx] = moves.cpu().numpy()
        final_goals[idx] = goals.cpu().numpy()
        if progress:
            progress(f"imagine chunk {ci}: rows {start}-{start + m}, win {won[idx].mean():.3f}")
    return {"won": won, "margin": margin_of(rows.moves.astype(np.float64), rows.goals.astype(np.float64), final_moves, final_goals), "final_moves": final_moves, "final_goals": final_goals,
            "n_actions": n_actions, "stripe_states": stripe_states, "illegal": 0, "seconds": time.perf_counter() - started}


@torch.no_grad()
def engine_rows4(policy_fn: PolicyFn4, rows: Rows, *, seed: int, device: torch.device, temperature: float = 1.0, workers: int = 1, max_moves: int = max(TIER_MOVE_BUDGETS), progress=None) -> dict[str, np.ndarray | int | float]:
    """As ``cala.rollout.engine_rows`` with the engine states' special grids handed to the policy."""
    n = len(rows)
    started = time.perf_counter()
    won = np.zeros(n, np.int8)
    final_moves = np.zeros(n, np.float64)
    final_goals = np.zeros(n, np.float64)
    n_actions = 0
    stripe_states = 0
    for li, level in enumerate(LEVELS):
        idx = np.flatnonzero(rows.level == li)
        if not len(idx):
            continue
        states = [State(board=rows.board[i].reshape(8, 8).astype(np.int8).copy(), moves_left=int(rows.moves[i]), goals_left=int(rows.goals[i]), goal_colour=int(rows.goal_colour[i])) for i in idx]
        engine = _EngineRows(level, states, [int(s) for s in rows.refill_seed[idx]], workers=workers)
        generator = chunk_generator(device, seed, li)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        try:
            for _ in range(int(rows.moves[idx].max())):
                boards_np = engine.boards()
                legal_np = legal_masks(boards_np)
                active_np = ~engine.terminal() & legal_np.any(axis=1)
                if not active_np.any():
                    break
                legal_in = legal_np.copy()
                legal_in[~active_np, 0] = True
                current = engine.current
                specials_np = np.stack([s.specials.reshape(-1) for s in current])
                stripe_states += int(((specials_np != 0).any(axis=1) & active_np).sum())
                boards = torch.as_tensor(boards_np, dtype=torch.long, device=device)
                legal = torch.as_tensor(legal_np, device=device)
                active = torch.as_tensor(active_np, device=device)
                logp = policy_fn(boards, torch.as_tensor([s.goal_colour for s in current], dtype=torch.long, device=device), torch.as_tensor([min(s.moves_left, max_moves) for s in current], dtype=torch.long, device=device),
                                 torch.as_tensor([s.goals_left for s in current], dtype=torch.long, device=device), torch.as_tensor(legal_in, device=device), idx_t, torch.as_tensor(specials_np, dtype=torch.long, device=device))
                actions = _sample_actions(logp, legal, active, temperature=temperature, generator=generator).cpu().numpy()
                engine.advance(np.flatnonzero(active_np), actions)
                n_actions += int(active_np.sum())
        finally:
            engine.close()
        won[idx] = np.asarray([int(s.won) for s in engine.current], np.int8)
        final_moves[idx] = [s.moves_left for s in engine.current]
        final_goals[idx] = [s.goals_left for s in engine.current]
        if progress:
            progress(f"engine {level.name}: {len(idx)} rows, win {won[idx].mean():.3f} ({time.perf_counter() - started:.0f}s)")
    return {"won": won, "margin": margin_of(rows.moves.astype(np.float64), rows.goals.astype(np.float64), final_moves, final_goals), "final_moves": final_moves, "final_goals": final_goals,
            "n_actions": n_actions, "stripe_states": stripe_states, "illegal": 0, "seconds": time.perf_counter() - started}


def assert_kernel_has_no_context_input(kernel: torch.nn.Module) -> None:
    """Static check: the frozen kernel's rollout / forward signatures carry no player-context argument."""
    import inspect

    for name in ("rollout_sample", "forward", "features", "encode_task"):
        fn = getattr(kernel, name, None)
        if fn is None:
            continue
        params = set(inspect.signature(fn).parameters)
        bad = {p for p in params if any(tag in p.lower() for tag in ("skill", "context", "player", "z_"))} - {"task_context"}
        if bad:
            raise GuardViolation(f"kernel.{name} exposes a player-context input {sorted(bad)}")


__all__ = ["ContextRecorder", "assert_kernel_has_no_context_input", "engine_rows4", "imagine_rows4"]
