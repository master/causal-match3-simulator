from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from match3_simulator.calibrate import goal_count_for_E
from match3_simulator.retention import ChurnSchedule, MasteryConfig, mastery_mismatch_hazard, update_mastery
from match3_simulator.scm import LEVELS, TIER_MOVE_BUDGETS
from match3_simulator.spec import State
from match3_simulator.world_modeling.cala_support import _EngineRows, legal_masks
from match3_simulator.world_modeling.cohort import use_accepted_calibration
from match3_simulator.world_modeling.decoder import torch_legal_mask
from match3_simulator.world_modeling.wm1_train import load_checkpoint
from match3_simulator.experiments.guard import GuardViolation

PolicyFn = Callable[..., torch.Tensor]
KERNELS: dict[str, str] = {"transformer-5.2M": "runs/wm1/transformer-5.2M-{seed}/best.pt", "transformer-14.8M": "runs/wm1/transformer-14.8M-{seed}/best.pt",
                           "lewm-5.2M-reencode": "runs/wm1/variants/lewm-5.2M-{seed}-reencode/best.pt", "lewm-14.8M-reencode": "runs/wm1/variants/lewm-14.8M-{seed}-reencode/best.pt"}


def kernel_path(name: str, seed: int, *, workspace: str | Path = ".") -> Path:
    if name not in KERNELS:
        raise ValueError(f"unknown kernel {name}; known {sorted(KERNELS)}")
    return Path(workspace) / KERNELS[name].format(seed=seed)


def load_kernel(name: str, seed: int, *, device: torch.device, workspace: str | Path = ".") -> torch.nn.Module:
    """Inputs: kernel name; WM-1 seed; device. Outputs: the frozen WM-1 world model in eval mode (parameters frozen)."""
    model = load_checkpoint(kernel_path(name, seed, workspace=workspace), device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def quota_for(level: np.ndarray, served: np.ndarray) -> np.ndarray:
    """Calibrated goal quota per row under the accepted table. Inputs: level index (N,), served e (N,). Outputs: (N,) int64."""
    use_accepted_calibration()
    return np.asarray([goal_count_for_E(LEVELS[int(l)].name, float(e)) for l, e in zip(level, served)], dtype=np.int64)


@dataclass
class Rows:
    """Flat game table. All arrays have length N; ``context`` is (N, width)."""

    board: np.ndarray
    level: np.ndarray
    tier: np.ndarray
    served: np.ndarray
    goals: np.ndarray
    moves: np.ndarray
    goal_colour: np.ndarray
    context: np.ndarray
    refill_seed: np.ndarray
    player: np.ndarray
    mastery_before: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.level)
        for name in ("board", "tier", "served", "goals", "moves", "goal_colour", "context", "refill_seed", "player", "mastery_before"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"{name} does not align with level")
        if self.board.shape != (n, 64):
            raise ValueError("board must be (N, 64)")
        if np.any(self.goals <= 0) or np.any(self.moves <= 0):
            raise ValueError("quota and move budget must be positive")

    def __len__(self) -> int:
        return int(len(self.level))


def make_rows(target: dict[str, np.ndarray], *, served: np.ndarray, context: np.ndarray, refill_seed: np.ndarray, quota: np.ndarray | None = None) -> Rows:
    """Rows from ``Logged.target_rows`` fields (one entry per game row; repeat players upstream for draws / replicates).

    Inputs: target dict (board, level, tier, goal_colour, move_budget, mastery_before); served e per row; context per row; refill seed per row; optional quota (default: calibrated from e).
    """
    level = np.asarray(target["level"], dtype=np.int64)
    served = np.asarray(served, dtype=np.float64)
    goals = quota_for(level, served) if quota is None else np.asarray(quota, dtype=np.int64)
    return Rows(board=np.asarray(target["board"], dtype=np.int64), level=level, tier=np.asarray(target["tier"], dtype=np.int64), served=served, goals=goals,
                moves=np.asarray(target["move_budget"], dtype=np.int64), goal_colour=np.asarray(target["goal_colour"], dtype=np.int64), context=np.asarray(context, dtype=np.float32),
                refill_seed=np.asarray(refill_seed, dtype=np.int64), player=np.asarray(target["player_id"], dtype=np.int64), mastery_before=np.asarray(target["mastery_before"], dtype=np.float64))


def margin_of(init_moves: np.ndarray, init_goals: np.ndarray, moves: np.ndarray, goals: np.ndarray) -> np.ndarray:
    won = goals <= 0
    return np.where(won, moves / init_moves, -goals / np.maximum(init_goals, 1)).astype(np.float64)


def _sample_actions(logp: torch.Tensor, legal: torch.Tensor, active: torch.Tensor, *, temperature: float, generator: torch.Generator) -> torch.Tensor:
    if temperature != 1.0:
        logp = torch.log_softmax(logp / float(temperature), dim=1)
    actions = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(1)
    if not bool(legal[torch.arange(len(actions), device=actions.device), actions][active].all()):
        raise GuardViolation("realized illegal action in a rollout")
    return actions


def chunk_generator(device: torch.device, seed: int, chunk_index: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(np.random.SeedSequence([int(seed), int(chunk_index)]).generate_state(1)[0]))


@torch.no_grad()
def imagine_rows(kernel: torch.nn.Module, policy_fn: PolicyFn, rows: Rows, *, seed: int, temperature: float = 1.0, chunk: int = 4096, progress=None) -> dict[str, np.ndarray | int | float]:
    """Play every row through the frozen kernel with the learned policy.

    Inputs: kernel (WM-1 world model); policy_fn(boards, goal_colour, moves_left, goals_left, legal, idx) -> log-probabilities (n, 128) where idx are the table rows of the chunk;
    rows; seed (generator re-seeded per chunk from (seed, chunk index)); sampling temperature; chunk size.
    Outputs: won (N,) int8, margin (N,), final_moves, final_goals (N,), n_actions (int), illegal (int, always 0 or raised), seconds.
    """
    device = next(kernel.parameters()).device
    n = len(rows)
    started = time.perf_counter()
    won = np.zeros(n, np.int8)
    final_moves = np.zeros(n, np.float64)
    final_goals = np.zeros(n, np.float64)
    n_actions = 0
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
            logp = policy_fn(boards, colours, moves.clamp(max=max_moves), goals, legal_in, idx_t)
            actions = _sample_actions(logp, legal, active, temperature=temperature, generator=generator)
            carried, sampled = kernel.rollout_sample(carried, boards=boards, goal_colours=colours, moves_left=moves.clamp(max=max_moves), goals_left=goals, actions=actions, task_context=task,
                                                     colour_support=support, specials=specials, observe=step == 0, stochastic=True, generator=generator)
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
            "n_actions": n_actions, "illegal": 0, "seconds": time.perf_counter() - started}


@torch.no_grad()
def engine_rows(policy_fn: PolicyFn, rows: Rows, *, seed: int, device: torch.device, temperature: float = 1.0, workers: int = 1, max_moves: int = max(TIER_MOVE_BUDGETS), progress=None) -> dict[str, np.ndarray | int | float]:
    """Play every row through the real engine (``causal._EngineRows``: resolve_move with the level's refill draw seeded per row) with the learned policy.

    Inputs: policy_fn as in imagine_rows; rows; seed for the policy generator (per level group); torch device of the policy; temperature; engine worker processes.
    Outputs: as imagine_rows. Refills use each row's refill_seed, so the same table at another e shares its refill stream.
    """
    n = len(rows)
    started = time.perf_counter()
    won = np.zeros(n, np.int8)
    final_moves = np.zeros(n, np.float64)
    final_goals = np.zeros(n, np.float64)
    n_actions = 0
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
                boards = torch.as_tensor(boards_np, dtype=torch.long, device=device)
                legal = torch.as_tensor(legal_np, device=device)
                active = torch.as_tensor(active_np, device=device)
                logp = policy_fn(boards, torch.as_tensor([s.goal_colour for s in current], dtype=torch.long, device=device), torch.as_tensor([min(s.moves_left, max_moves) for s in current], dtype=torch.long, device=device),
                                 torch.as_tensor([s.goals_left for s in current], dtype=torch.long, device=device), torch.as_tensor(legal_in, device=device), idx_t)
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
            "n_actions": n_actions, "illegal": 0, "seconds": time.perf_counter() - started}


def churn_of(result: dict[str, np.ndarray], rows: Rows, *, churn_fn: Callable[..., np.ndarray], attempt: int = 20, mastery: MasteryConfig) -> dict[str, np.ndarray]:
    """Push a rollout result through the experience update and a churn function.

    Inputs: rollout result (won, margin); rows (mastery_before, level); churn_fn(completion, margin, mastery_after, level, attempt) -> probability; attempt; mastery config.
    Outputs: dict mastery_after (N,), churn (N,).
    """
    won = result["won"].astype(np.float64)
    after = update_mastery(rows.mastery_before, won, mastery)
    p = churn_fn(won, np.clip(result["margin"], -1.0, 1.0), after, rows.level, np.full(len(rows), attempt))
    return {"mastery_after": after, "churn": np.asarray(p, dtype=np.float64)}


def accepted_churn_fn(churn: ChurnSchedule, *, warmup_scale: float = 0.005, landmark: int = 20) -> Callable[..., np.ndarray]:
    """The accepted hazard as a churn_fn (the 'hazard-substituted' diagnostic row; never inside an estimator)."""

    def call(completion, margin, mastery_after, level, attempt):
        level = np.asarray(level)
        out = np.zeros(len(level), np.float64)
        for li, name in enumerate(churn.level_names):
            sel = level == li
            if sel.any():
                out[sel] = mastery_mismatch_hazard(np.asarray(mastery_after)[sel], churn.for_level(name), completion_margin=np.asarray(margin)[sel])
        return out * np.where(np.asarray(attempt) < landmark, warmup_scale, 1.0)

    return call


__all__ = ["KERNELS", "Rows", "accepted_churn_fn", "chunk_generator", "churn_of", "engine_rows", "imagine_rows", "kernel_path", "load_kernel", "make_rows", "margin_of", "quota_for"]
