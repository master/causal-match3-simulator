from __future__ import annotations

from collections.abc import Sequence
import os

import numpy as np

from match3_simulator.board import resolve_move
from match3_simulator.learned_model.rollout import LearnedDynamicsRollout
from match3_simulator.learned_model.tokens import N_CELLS, index_to_action
from match3_simulator.spec import LevelContext, State
from match3_simulator.world_modeling.closed_loop_kstep import _draw

#: dataset generation seed of the accepted release (``release-manifest.json`` ``seed``); the simulator seeds player ``p`` with SeedSequence([seed, p, 0])
DATASET_SEED = 12011


# ---------------------------------------------------------------- learned_model.tokens.legal_masks (workspace addition) ----


def legal_masks(boards: np.ndarray) -> np.ndarray:
    """Return the (N, 128) legal-swap masks of many flattened 8x8 boards at once (batched equivalent of legal_mask; boards contain no EMPTY cells).

    Inputs: boards (N, 64) or (N, 8, 8) colour indices.
    Outputs: boolean array (N, 128).
    """
    import torch  # local import keeps the module importable without torch at module load

    from match3_simulator.world_modeling.decoder import torch_legal_mask

    array = np.asarray(boards).reshape(-1, N_CELLS)
    with torch.no_grad():
        return torch_legal_mask(torch.as_tensor(array, dtype=torch.long)).numpy()


# ---------------------------------------------------------------- world_modeling.wm1_action_sensitivity.player_skill ----


def player_skill(player_id: int, *, seed: int = DATASET_SEED) -> tuple[float, ...]:
    """Re-derive a player's true skill exactly as retention.simulate_player_trajectory does (simulator oracle, never the dataset)."""
    import pyro

    from match3_simulator.scm import sample_K

    pyro.set_rng_seed(int(np.random.SeedSequence([seed, int(player_id), 0]).generate_state(1)[0]))
    return tuple(float(v) for v in sample_K().values)


# ---------------------------------------------------------------- world_modeling.causal._EngineRows ----


def _engine_rows_worker(conn, level: LevelContext, initial_states: Sequence[State], refill_seeds: Sequence[int]) -> None:
    """Worker process owning a contiguous block of engine games: applies the actions it is sent and returns the new states (see _EngineRows)."""
    draws = [_draw(level, np.random.default_rng(np.random.SeedSequence([int(s), 11]))) for s in refill_seeds]
    current = [s.copy() for s in initial_states]
    while True:
        # forked siblings inherit each other's pipe ends, so a dead parent does not close this pipe: poll and leave when re-parented
        while not conn.poll(5.0):
            if os.getppid() == 1:
                return
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            break
        out = []
        for local, index in message:
            current[local], _ = resolve_move(current[local], index_to_action(int(index)), level, draws[local])
            out.append((local, current[local]))
        conn.send(out)
    conn.close()


class _EngineRows:
    """Advance many engine games one action each, in-process or partitioned over persistent worker processes.

    Every row's refill draw is a private numpy Generator seeded from its refill seed, so partitioning rows over processes changes nothing:
    the boards, counters, state histories and action lists are identical to the sequential path. The main process keeps the full histories.
    """

    def __init__(self, level: LevelContext, initial_states: Sequence[State], refill_seeds: Sequence[int], *, workers: int = 1):
        self.level = level
        self.n = len(initial_states)
        self.current = [s.copy() for s in initial_states]
        self.states = [[s.copy()] for s in initial_states]
        self.actions: list[list] = [[] for _ in initial_states]
        self.workers = max(1, min(int(workers), self.n // 64 or 1))
        self._processes: list = []
        self._connections: list = []
        self._chunks: list[tuple[int, int]] = []
        if self.workers > 1:
            import multiprocessing as mp

            context = mp.get_context("fork")
            bounds = np.linspace(0, self.n, self.workers + 1).astype(int)
            for low, high in zip(bounds[:-1], bounds[1:]):
                parent, child = context.Pipe()
                process = context.Process(target=_engine_rows_worker, args=(child, level, list(initial_states[low:high]), [int(x) for x in refill_seeds[low:high]]), daemon=True)
                process.start()
                child.close()
                self._processes.append(process)
                self._connections.append(parent)
                self._chunks.append((int(low), int(high)))
        else:
            self._draws = [_draw(level, np.random.default_rng(np.random.SeedSequence([int(s), 11]))) for s in refill_seeds]

    def boards(self) -> np.ndarray:
        return np.stack([s.board.reshape(-1) for s in self.current])

    def terminal(self) -> np.ndarray:
        return np.asarray([s.terminal for s in self.current], dtype=bool)

    def advance(self, rows: np.ndarray, action_indices: np.ndarray) -> None:
        """Apply action_indices[row] to every row in rows."""
        if self.workers == 1:
            for row in rows:
                action = index_to_action(int(action_indices[row]))
                self.current[row], _ = resolve_move(self.current[row], action, self.level, self._draws[row])
                self.actions[row].append(action)
                self.states[row].append(self.current[row])
            return
        active = np.zeros(self.n, dtype=bool)
        active[np.asarray(rows, dtype=int)] = True
        for connection, (low, high) in zip(self._connections, self._chunks):
            connection.send([(row - low, int(action_indices[row])) for row in range(low, high) if active[row]])
        for connection, (low, high) in zip(self._connections, self._chunks):
            for local, state in connection.recv():
                row = low + local
                self.current[row] = state
                self.actions[row].append(index_to_action(int(action_indices[row])))
                self.states[row].append(state)

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send(None)
                connection.close()
            except (OSError, ValueError):
                pass
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        self._connections, self._processes = [], []

    def rollouts(self) -> list[LearnedDynamicsRollout]:
        return [LearnedDynamicsRollout(tuple(s), tuple(a)) for s, a in zip(self.states, self.actions)]


__all__ = ["DATASET_SEED", "legal_masks", "player_skill", "_EngineRows", "_engine_rows_worker"]
