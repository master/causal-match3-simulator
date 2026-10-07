"""ENGINE_ABLATION — exact cleared masks by re-simulating the release players with the simulator (off by default; **not part of the 36 fits and not deployable**:
it needs the engine's cascade steps, which the release does not ship).

``build_shard_cache(..., target="engine")`` re-simulates the shard's train + validation players with ``CohortSpec.from_accepted(seed=12011, regime="natural",
player_ids=...)`` (the dataset's configuration), cross-checks every logged row (board_before, action, board_after, specials) against the shipped npz — any
mismatch is a hard failure — and takes the mask from ``learned_model.cleared.cleared_mask_from_transition`` (exact) with the refill-goal count as the residual.
Caches are stamped ``target = "engine"`` so the training loop can never mix them with the canonical settled target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from match3_simulator.world_modeling.cleared import cleared_mask_from_transition
from match3_simulator.learned_model.tokens import N_CELLS, action_to_index
from match3_simulator.spec import NO_SPECIAL
from match3_simulator.world_modeling.cohort import CohortSpec, load_or_simulate_cohort

ENGINE_ABLATION = True
DATASET_SEED = 12011


def derive_rows_engine(arrays: dict[str, np.ndarray], attempts: dict[tuple[int, int], dict[str, object]], *, shard_dir: Path, cache_dir: str | Path = "runs/trajectory-cache-ro", workers: int = 8) -> dict[str, np.ndarray]:
    """Exact masks for every row of a shard from a re-simulation of its players (cross-checked row by row)."""
    players = tuple(int(p) for p in np.unique(arrays["player_id"]))
    spec = CohortSpec.from_accepted(n_players=len(players), seed=DATASET_SEED, regime="natural", player_ids=players)
    trajectories, _ = load_or_simulate_cohort(spec, cache_dir=str(cache_dir), workers=workers)
    by_player = {t.player_id: t for t in trajectories}
    n = len(arrays["action_index"])
    out = {"cleared_mask": np.zeros((n, N_CELLS), dtype=np.int8), "m_per_column": np.zeros((n, 8), dtype=np.int8), "derived_goal": np.zeros(n, dtype=np.int16), "residual_goal": np.zeros(n, dtype=np.int16),
           "terminal": np.zeros(n, dtype=bool), "mask_valid": np.ones(n, dtype=bool), "stripe_present": np.zeros(n, dtype=bool), "stripe_activated": np.zeros(n, dtype=np.int8), "stripe_created": np.zeros(n, dtype=bool),
           "round0_activations": np.zeros(n, dtype=np.int8), "inferred_activations": np.zeros(n, dtype=np.int8), "consistent": np.ones(n, dtype=bool), "reshuffled_attempt": np.zeros(n, dtype=bool)}
    for i in range(n):
        record = next(r for r in by_player[int(arrays["player_id"][i])].attempts if r.attempt_id == int(arrays["attempt_id"][i]))
        episode = record.episode
        step = int(arrays["step_id"][i])
        state, nxt, transition = episode.states[step], episode.states[step + 1], episode.transitions[step]
        if not (np.array_equal(state.board, arrays["board_before"][i]) and np.array_equal(nxt.board, arrays["board_after"][i]) and action_to_index(transition.action) == int(arrays["action_index"][i])
                and np.array_equal(state.specials, arrays["specials_before"][i]) and np.array_equal(nxt.specials, arrays["specials_after"][i])):
            raise RuntimeError(f"re-simulated row differs from the shipped npz (player {arrays['player_id'][i]}, attempt {arrays['attempt_id'][i]}, step {step})")
        mask, s_goal, refill_goal, _ = cleared_mask_from_transition(transition, state.goal_colour)
        out["cleared_mask"][i] = mask
        out["derived_goal"][i] = s_goal
        out["residual_goal"][i] = int(arrays["goals_left"][i]) - int(arrays["goals_left_next"][i]) - s_goal
        out["terminal"][i] = int(arrays["goals_left_next"][i]) <= 0 or int(arrays["moves_left_next"][i]) <= 0
        out["mask_valid"][i] = not transition.reshuffled
        out["reshuffled_attempt"][i] = episode.reshuffles > 0
        out["stripe_present"][i] = bool((state.specials != NO_SPECIAL).any()) or bool(transition.created_specials)
        out["stripe_activated"][i] = len(transition.activated_specials)
        out["stripe_created"][i] = bool(transition.created_specials)
        out["round0_activations"][i] = len(transition.steps[0].activated_specials) if transition.steps else 0
    return out


__all__ = ["ENGINE_ABLATION", "derive_rows_engine"]
