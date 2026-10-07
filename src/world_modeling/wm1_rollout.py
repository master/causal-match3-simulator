"""WM evaluation rollouts on validation players, horizons 1 / 3 / 7, three modes:

A. logged-action replay (open loop): start from every logged state, feed the logged actions, roll the model from its own predicted states
   (boards, specials, counters); per horizon the logged board is scored (skill, macro-F1, changed-cell accuracy, change-detection F1, colour
   accuracy, goals MAE, special-kind accuracy, terminal accuracy) and the teacher-forced NLL of the logged next state from the model's own state.
   A logged action can be illegal on the model's *imagined* board (step >= 1). The action is still applied, and the event is counted.
   Each horizon reports the share of anchors with at least one illegal step and the illegal rate per step. Every metric block is also reported
   for anchors whose logged actions were all legal on the imagined boards (``all_legal_anchors``).
B. engine-conditioned (one-step control): the model receives the engine's true state and the engine's action at every step and predicts one step;
   board / goals-left energy scores against the engine's next states at the horizons, one-step NLL and accuracies.
C. autonomous imagination: model state at every step, actions sampled uniformly from the legal swaps of the **decoded model board**; 512 fixed
   anchors x 4 model rollouts against 4 paired engine rollouts (same anchors, same action-rng seeds) -> board / goals energy scores, terminal rate,
   terminal-head accuracy, and the flat generated transitions for the §7 contract checks (invalid-state rate).

Energy score (per anchor, samples X from the model, Y from the reference): ES = mean_{i,j} d(X_i, Y_j) - 0.5 mean_{i!=j} d(X_i, X_j); d = Hamming
fraction over the 64 cells for boards, |difference| for goals left; lower is better. Skill = 1 - err_model / err_persistence with persistence = "board unchanged".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import torch

from match3_simulator.board import legal_moves, resolve_move
from match3_simulator.learned_model.tokens import N_CELLS, action_to_index, index_to_action
from match3_simulator.scm import LEVELS
from match3_simulator.spec import State
from match3_simulator.world_modeling.closed_loop_kstep import _draw, _energy_score
from match3_simulator.world_modeling.decoder import torch_legal_mask
from match3_simulator.world_modeling.kstep import binary_prf, macro_f1_colours, skill_score

DEFAULT_KS: tuple[int, ...] = (1, 3, 7)
IMAGINATION_ANCHORS = 512
IMAGINATION_ROLLOUTS = 4
CHUNK = 2048


def colour_support_of(levels: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor([level.n_colours for level in LEVELS], device=levels.device)[levels.long()]


def _sample_uniform_legal(legal: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Inputs: legal (N, 128) bool with at least one True per active row; generator. Outputs: actions (N,) long (0 where no legal move)."""
    weights = legal.to(torch.float32)
    any_legal = legal.any(dim=1)
    weights[~any_legal, 0] = 1.0
    return torch.multinomial(weights, 1, generator=generator).squeeze(1)


# ------------------------------------------------------------------- mode A ----


@dataclass
class ReplayRollouts:
    ks: tuple[int, ...]
    decoding: str
    data: dict[int, dict[str, list[np.ndarray]]] = field(default_factory=dict)
    n_anchors: int = 0
    n_dynamics_steps: int = 0

    def concatenated(self, k: int) -> dict[str, np.ndarray]:
        return {name: np.concatenate(parts) for name, parts in self.data[k].items() if parts}


@torch.no_grad()
def replay_rollouts(model: torch.nn.Module, batch: dict[str, torch.Tensor], *, ks: tuple[int, ...] = DEFAULT_KS, stochastic: bool, seed: int, chunk: int = CHUNK) -> ReplayRollouts:
    """Mode A over one padded batch (all logged anchors).

    Inputs: model; padded batch with the WM-1 keys (+ optional stripe_present / stripe_activated per step); horizons; decoding; seed; anchors per chunk.
    Outputs: ReplayRollouts with, per horizon, predicted / target / anchor boards and specials, goals, board_nll, special_nll, terminal predictions and strata keys.
    """
    ks = tuple(sorted(set(int(k) for k in ks)))
    device = next(model.parameters()).device
    model.eval()
    max_moves = int(model.config.max_moves_left)
    boards_all = torch.cat((batch["boards"], batch["next_boards"][:, -1:]), dim=1).to(device)
    specials_all = torch.cat((batch["specials"], batch["next_specials"][:, -1:]), dim=1).to(device)
    moves_all = torch.cat((batch["moves_left"], batch["next_moves_left"][:, -1:]), dim=1).to(device)
    goals_all = torch.cat((batch["goals_left"], batch["next_goals_left"][:, -1:]), dim=1).to(device)
    actions = batch["actions"].to(device)
    lengths = batch["step_mask"].to(device).long().sum(dim=1)
    n_episodes, n_steps = actions.shape
    episode_index, anchor_t = torch.meshgrid(torch.arange(n_episodes, device=device), torch.arange(n_steps, device=device), indexing="ij")
    valid = anchor_t + 1 <= lengths.unsqueeze(1)
    episode_index, anchor_t = episode_index[valid], anchor_t[valid]
    context = model.encode_task(batch["levels"].to(device), batch["tiers"].to(device), batch["served_difficulty"].to(device))
    goal_colours = batch["goal_colours"].to(device)[:, 0]
    support = colour_support_of(batch["levels"].to(device))
    levels = batch["levels"].to(device)
    stripe_present = batch.get("stripe_present")
    stripe_activated = batch.get("stripe_activated")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    result = ReplayRollouts(ks=ks, decoding="sampled" if stochastic else "argmax", n_anchors=int(anchor_t.numel()))
    for k in ks:
        result.data[k] = {name: [] for name in ("predicted", "target", "anchor", "predicted_specials", "target_specials", "anchor_specials", "predicted_goals", "target_goals",
                                                 "board_nll", "special_nll", "special_ambiguous", "terminal_prob", "terminal_target", "level", "stripe_present", "stripe_activated",
                                                 "all_legal", "illegal_steps")}
    result.illegal_by_step = getattr(result, "illegal_by_step", {})
    for start in range(0, anchor_t.numel(), chunk):
        rows_b, rows_t = episode_index[start : start + chunk], anchor_t[start : start + chunk]
        state = model.rollout_begin(rows_b.numel(), context[rows_b])
        board, special, move, goal = boards_all[rows_b, rows_t], specials_all[rows_b, rows_t], moves_all[rows_b, rows_t], goals_all[rows_b, rows_t]
        generated: dict[int, dict[str, torch.Tensor]] = {}
        all_legal = torch.ones(rows_b.numel(), dtype=torch.bool, device=device)
        illegal_steps = torch.zeros(rows_b.numel(), dtype=torch.long, device=device)
        for step in range(ks[-1]):
            index = rows_t + step
            alive = index + 1 <= lengths[rows_b]
            if not bool(alive.any()):
                break
            safe = index.clamp_max(n_steps - 1)
            action = torch.where(alive, actions[rows_b, safe], torch.zeros_like(index))
            legal_now = torch_legal_mask(board).gather(1, action.unsqueeze(1)).squeeze(1)  # legality of the logged action on the model's current board
            illegal = alive & ~legal_now
            all_legal &= ~illegal
            illegal_steps += illegal.long()
            bucket = result.illegal_by_step.setdefault(step, [0, 0])
            bucket[0] += int(illegal.sum())
            bucket[1] += int(alive.sum())
            targets = {"next_boards": boards_all[rows_b, safe + 1], "next_specials": specials_all[rows_b, safe + 1], "next_moves_left": moves_all[rows_b, safe + 1], "next_goals_left": goals_all[rows_b, safe + 1]}
            state, sampled = model.rollout_sample(state, boards=board, goal_colours=goal_colours[rows_b], moves_left=move.clamp(0, max_moves), goals_left=goal.clamp_min(0), actions=action,
                                                  task_context=context[rows_b], colour_support=support[rows_b], specials=special, observe=step == 0, stochastic=stochastic, generator=generator, targets=targets)
            result.n_dynamics_steps += int(alive.sum())
            board = torch.where(alive.unsqueeze(1), sampled["boards"], board)
            special = torch.where(alive.unsqueeze(1), sampled["specials"], special)
            goal = torch.where(alive, torch.minimum(goal, sampled["goals_left"].long().clamp_min(0)), goal)
            move = torch.where(alive, (move - 1).clamp_min(0), move)
            if step + 1 in ks:
                generated[step + 1] = {"boards": board.clone(), "specials": special.clone(), "goals": goal.clone(), "board_nll": sampled.get("board_nll"), "special_nll": sampled.get("special_nll"),
                                       "special_ambiguous": sampled.get("special_ambiguous_cells"),
                                       "terminal_prob": torch.sigmoid(sampled["terminal_logit"]) if sampled.get("terminal_logit") is not None else None,
                                       "all_legal": all_legal.clone(), "illegal_steps": illegal_steps.clone()}
        for k, block in generated.items():
            ok = rows_t + k <= lengths[rows_b]
            if not bool(ok.any()):
                continue
            b_ok, t_ok = rows_b[ok], rows_t[ok]
            data = result.data[k]
            data["predicted"].append(block["boards"][ok].cpu().numpy().astype(np.int64))
            data["target"].append(boards_all[b_ok, t_ok + k].cpu().numpy())
            data["anchor"].append(boards_all[b_ok, t_ok].cpu().numpy())
            data["predicted_specials"].append(block["specials"][ok].cpu().numpy().astype(np.int64))
            data["target_specials"].append(specials_all[b_ok, t_ok + k].cpu().numpy())
            data["anchor_specials"].append(specials_all[b_ok, t_ok].cpu().numpy())
            data["predicted_goals"].append(block["goals"][ok].cpu().numpy())
            data["target_goals"].append(goals_all[b_ok, t_ok + k].cpu().numpy())
            if block["board_nll"] is not None:
                data["board_nll"].append(block["board_nll"][ok].cpu().numpy())
            if block["special_nll"] is not None:
                data["special_nll"].append(block["special_nll"][ok].cpu().numpy())
                data["special_ambiguous"].append(block["special_ambiguous"][ok].cpu().numpy())
            if block["terminal_prob"] is not None:
                data["terminal_prob"].append(block["terminal_prob"][ok].cpu().numpy())
                data["terminal_target"].append(((goals_all[b_ok, t_ok + k] <= 0) | (moves_all[b_ok, t_ok + k] <= 0)).cpu().numpy())
            data["level"].append(levels[b_ok].cpu().numpy())
            data["all_legal"].append(block["all_legal"][ok].cpu().numpy())
            data["illegal_steps"].append(block["illegal_steps"][ok].cpu().numpy())
            if stripe_present is not None:
                data["stripe_present"].append(stripe_present.to(device)[b_ok, t_ok].cpu().numpy())
                data["stripe_activated"].append((stripe_activated.to(device)[b_ok, t_ok] > 0).cpu().numpy())
    return result


def horizon_metrics(d: dict[str, np.ndarray], n_colours: int, select: np.ndarray | None = None) -> dict[str, object]:
    """Metric block of one horizon (optionally on a subset of rows).

    Inputs: concatenated arrays of a horizon; number of colours; optional boolean row selector. Outputs: dict of metrics (None where undefined).
    """
    if select is not None:
        d = {name: values[select] for name, values in d.items() if values.shape[:1] == select.shape}
    pred, target, anchor = d["predicted"], d["target"], d["anchor"]
    n = int(pred.shape[0])
    if n == 0:
        return {"n": 0}
    correct = pred == target
    changed = target != anchor
    accuracy = float(correct.mean())
    persistence = float((anchor == target).mean())
    out = {
        "n": n, "change_rate": float(changed.mean()), "cell_accuracy": accuracy, "colour_accuracy": accuracy, "macro_f1_colours": macro_f1_colours(pred, target, n_colours),
        "board_exact_rate": float(correct.all(axis=1).mean()), "changed_cell_accuracy": float(correct[changed].mean()) if changed.any() else None,
        "changed_macro_f1_colours": macro_f1_colours(pred, target, n_colours, changed), "unchanged_cell_accuracy": float(correct[~changed].mean()) if (~changed).any() else None,
        "change_detection": binary_prf((pred != anchor).ravel(), changed.ravel()), "persistence": {"cell_accuracy": persistence, "macro_f1_colours": macro_f1_colours(anchor, target, n_colours)},
        "skill_score": skill_score(accuracy, persistence), "goals_left_mae": float(np.abs(d["predicted_goals"] - d["target_goals"]).mean()),
        "goals_left_exact_rate": float((d["predicted_goals"] == d["target_goals"]).mean()),
    }
    if "predicted_specials" in d and d["predicted_specials"].size:
        ps, ts = d["predicted_specials"], d["target_specials"]
        stripe_cells = (ts != 0) | (ps != 0)
        out["special_accuracy"] = float((ps == ts).mean())
        out["special_accuracy_stripe_cells"] = float((ps == ts)[stripe_cells].mean()) if stripe_cells.any() else None
        out["special_exact_board_rate"] = float((ps == ts).all(axis=1).mean())
        out["special_count_mae"] = float(np.abs((ps != 0).sum(axis=1) - (ts != 0).sum(axis=1)).mean())
    if "board_nll" in d and d["board_nll"].size:
        out["board_nll_per_cell"] = float(d["board_nll"].mean())
    if "special_nll" in d and d["special_nll"].size:
        ambiguous = d["special_ambiguous"] > 0
        out["special_nll_ambiguous"] = float(d["special_nll"][ambiguous].mean()) if ambiguous.any() else None
        out["special_ambiguous_row_rate"] = float(ambiguous.mean())
    if "terminal_prob" in d and d["terminal_prob"].size:
        predicted = d["terminal_prob"] >= 0.5
        truth = d["terminal_target"].astype(bool)
        out["terminal_accuracy"] = float((predicted == truth).mean())
        out["terminal_rate_target"] = float(truth.mean())
        out["terminal_brier"] = float(((d["terminal_prob"] - truth) ** 2).mean())
    if "all_legal" in d and d["all_legal"].size:
        legal = d["all_legal"].astype(bool)
        out["illegal_logged_action"] = {"handling": "applied anyway on the imagined board", "anchors_with_any_illegal_step": int((~legal).sum()),
                                        "anchor_rate": float((~legal).mean()), "illegal_steps_per_anchor": float(d["illegal_steps"].mean())}
    return out


def replay_metrics(rollouts: list[ReplayRollouts], *, n_colours: int) -> dict[str, object]:
    """Merge chunks / batches of one decoding and compute per-horizon metrics, stratified by level and stripe presence / activation at the anchor."""
    ks = rollouts[0].ks
    report: dict[str, object] = {"ks": list(ks), "decoding": rollouts[0].decoding, "n_anchors": sum(r.n_anchors for r in rollouts), "n_dynamics_steps": sum(r.n_dynamics_steps for r in rollouts), "horizons": {}}
    by_step: dict[int, list[int]] = {}
    for roll in rollouts:
        for step, (illegal, alive) in getattr(roll, "illegal_by_step", {}).items():
            bucket = by_step.setdefault(int(step), [0, 0])
            bucket[0] += illegal
            bucket[1] += alive
    report["illegal_logged_action_by_step"] = {f"step{step}": {"illegal": v[0], "alive": v[1], "rate": (v[0] / v[1] if v[1] else None)} for step, v in sorted(by_step.items())}
    for k in ks:
        merged: dict[str, list[np.ndarray]] = {}
        for roll in rollouts:
            for name, parts in roll.data[k].items():
                merged.setdefault(name, []).extend(parts)
        d = {name: np.concatenate(parts) for name, parts in merged.items() if parts}
        if "predicted" not in d:
            report["horizons"][f"k{k}"] = {"n": 0}
            continue
        block = horizon_metrics(d, n_colours)
        block["by_level"] = {LEVELS[i].name: horizon_metrics(d, n_colours, d["level"] == i) for i in range(len(LEVELS))}
        if "all_legal" in d:
            legal = d["all_legal"].astype(bool)
            block["all_legal_anchors"] = horizon_metrics(d, n_colours, legal)
            block["all_legal_anchors"]["by_level"] = {LEVELS[i].name: horizon_metrics(d, n_colours, legal & (d["level"] == i)) for i in range(len(LEVELS))}
            block["any_illegal_anchors"] = horizon_metrics(d, n_colours, ~legal)
        if "stripe_present" in d:
            block["by_stripe"] = {"no_stripe": horizon_metrics(d, n_colours, ~d["stripe_present"].astype(bool)), "stripe_present": horizon_metrics(d, n_colours, d["stripe_present"].astype(bool)),
                                  "stripe_activated": horizon_metrics(d, n_colours, d["stripe_activated"].astype(bool))}
        report["horizons"][f"k{k}"] = block
    return report


@torch.no_grad()
def evaluate_replay(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], *, ks: tuple[int, ...] = DEFAULT_KS, seed: int, decodings: tuple[str, ...] = ("argmax", "sampled")) -> dict[str, object]:
    """Mode A over several batches for each decoding rule."""
    out: dict[str, object] = {"ks": list(ks), "decodings": {}}
    n_colours = int(model.config.n_colours)
    for decoding in decodings:
        rolls = [replay_rollouts(model, batch, ks=ks, stochastic=decoding == "sampled", seed=seed + index) for index, batch in enumerate(batches)]
        out["decodings"][decoding] = replay_metrics(rolls, n_colours=n_colours)
    return out


# ------------------------------------------------------------------- anchors ----


def collect_anchors(batches: list[dict[str, torch.Tensor]], *, max_anchors: int, seed: int) -> dict[str, torch.Tensor]:
    """Non-terminal logged states with a legal move, sampled without replacement (fixed seed) from the batches; includes specials and level / tier / e."""
    names = ("boards", "specials", "goal_colours", "moves_left", "goals_left")
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in names + ("levels", "tiers", "served_difficulty")}
    for batch in batches:
        mask = batch["step_mask"].detach().cpu().bool()
        steps = mask.shape[1]
        for name in names:
            collected[name].append(batch[name].detach().cpu()[mask])
        for name in ("levels", "tiers", "served_difficulty"):
            collected[name].append(batch[name].detach().cpu().unsqueeze(1).expand(-1, steps)[mask])
    anchors = {name: torch.cat(parts) for name, parts in collected.items()}
    legal = torch.cat([torch_legal_mask(anchors["boards"][s : s + 1024]).any(dim=1) for s in range(0, anchors["boards"].shape[0], 1024)])
    keep = (anchors["moves_left"] > 0) & (anchors["goals_left"] > 0) & legal
    anchors = {name: values[keep] for name, values in anchors.items()}
    count = min(max_anchors, anchors["boards"].shape[0])
    chosen = torch.as_tensor(np.sort(np.random.default_rng(seed).choice(anchors["boards"].shape[0], size=count, replace=False)), dtype=torch.long)
    return {name: values[chosen] for name, values in anchors.items()}


def _engine_state(anchors: dict[str, torch.Tensor], i: int) -> State:
    return State(board=anchors["boards"][i].numpy().reshape(8, 8).astype(np.int8).copy(), moves_left=int(anchors["moves_left"][i]), goals_left=int(anchors["goals_left"][i]),
                 goal_colour=int(anchors["goal_colours"][i]), specials=anchors["specials"][i].numpy().reshape(8, 8).astype(np.int8).copy())


# ------------------------------------------------------------------- mode B ----


@torch.no_grad()
def evaluate_engine_conditioned(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], *, ks: tuple[int, ...] = DEFAULT_KS, max_anchors: int = IMAGINATION_ANCHORS,
                                rollouts_per_anchor: int = IMAGINATION_ROLLOUTS, seed: int, chunk: int = 64) -> dict[str, object]:
    """Mode B: at every step the model sees the engine's true state (board, specials, counters) and the engine's uniform-random legal action and predicts one step.

    Outputs: per horizon board / goals-left energy scores of the one-step predictions against the engine's next state, one-step board NLL, colour accuracy,
    special accuracy, terminal accuracy (of the terminal head against the engine's terminal status).
    """
    ks = tuple(sorted(set(int(k) for k in ks)))
    device = next(model.parameters()).device
    model.eval()
    anchors = collect_anchors(batches, max_anchors=max_anchors, seed=seed)
    n_anchors = anchors["boards"].shape[0]
    per_k = {k: {"predicted": [], "target": [], "predicted_goals": [], "target_goals": []} for k in ks}
    one_step = {"board_nll": [], "cell_correct": [], "special_correct": [], "special_stripe_correct": [], "special_stripe_cells": [], "terminal_correct": [], "goal_abs_error": [], "n": 0}
    for start in range(0, n_anchors, chunk):
        stop = min(start + chunk, n_anchors)
        n = stop - start
        repeat = lambda values: values[start:stop].repeat_interleave(rollouts_per_anchor, dim=0).to(device)
        goal_colours, levels, tiers, served = repeat(anchors["goal_colours"]), repeat(anchors["levels"]), repeat(anchors["tiers"]), repeat(anchors["served_difficulty"])
        context = model.encode_task(levels, tiers, served)
        support = colour_support_of(levels)
        generator = torch.Generator(device=device).manual_seed(int(np.random.SeedSequence([seed, 2, start]).generate_state(1)[0]))
        engine = [_engine_state(anchors, start + a) for a in range(n) for _ in range(rollouts_per_anchor)]
        engine_levels = [LEVELS[int(anchors["levels"][start + a])] for a in range(n) for _ in range(rollouts_per_anchor)]
        action_rngs = [np.random.default_rng(np.random.SeedSequence([seed, 3, start + a, r])) for a in range(n) for r in range(rollouts_per_anchor)]
        refill_rngs = [np.random.default_rng(np.random.SeedSequence([seed, 4, start + a, r])) for a in range(n) for r in range(rollouts_per_anchor)]
        model_state = model.rollout_begin(len(engine), context)
        for step in range(ks[-1]):
            boards = torch.as_tensor(np.stack([s.board.reshape(-1) for s in engine]), dtype=torch.long, device=device)
            specials = torch.as_tensor(np.stack([s.specials.reshape(-1) for s in engine]), dtype=torch.long, device=device)
            moves = torch.as_tensor([s.moves_left for s in engine], dtype=torch.long, device=device)
            goals = torch.as_tensor([s.goals_left for s in engine], dtype=torch.long, device=device)
            actions_np = np.zeros(len(engine), dtype=np.int64)
            active = np.zeros(len(engine), dtype=bool)
            chosen = []
            for i, state in enumerate(engine):
                if state.terminal:
                    chosen.append(None)
                    continue
                moves_list = legal_moves(state.board)
                if not moves_list:
                    chosen.append(None)
                    continue
                action = moves_list[int(action_rngs[i].integers(len(moves_list)))]
                chosen.append(action)
                actions_np[i] = action_to_index(action)
                active[i] = True
            if not active.any():
                break
            for i, action in enumerate(chosen):
                if action is not None:
                    engine[i], _ = resolve_move(engine[i], action, engine_levels[i], _draw(engine_levels[i], refill_rngs[i]))
            next_boards = torch.as_tensor(np.stack([s.board.reshape(-1) for s in engine]), dtype=torch.long, device=device)
            next_specials = torch.as_tensor(np.stack([s.specials.reshape(-1) for s in engine]), dtype=torch.long, device=device)
            next_moves = torch.as_tensor([s.moves_left for s in engine], dtype=torch.long, device=device)
            next_goals = torch.as_tensor([s.goals_left for s in engine], dtype=torch.long, device=device)
            actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
            model_state, sampled = model.rollout_sample(model_state, boards=boards, goal_colours=goal_colours, moves_left=moves, goals_left=goals, actions=actions, task_context=context,
                                                        colour_support=support, specials=specials, observe=True, stochastic=True, generator=generator,
                                                        targets={"next_boards": next_boards, "next_specials": next_specials, "next_moves_left": next_moves, "next_goals_left": next_goals})
            act = torch.as_tensor(active, device=device)
            one_step["n"] += int(act.sum())
            one_step["board_nll"].append(sampled["board_nll"][act].cpu().numpy())
            one_step["cell_correct"].append((sampled["boards"] == next_boards)[act].float().mean(dim=1).cpu().numpy())
            one_step["special_correct"].append((sampled["specials"] == next_specials)[act].float().mean(dim=1).cpu().numpy())
            stripe_cells = ((next_specials != 0) | (sampled["specials"] != 0)) & act.unsqueeze(1)
            one_step["special_stripe_correct"].append(((sampled["specials"] == next_specials) & stripe_cells).sum().item())
            one_step["special_stripe_cells"].append(stripe_cells.sum().item())
            one_step["goal_abs_error"].append((sampled["goals_left"].long() - next_goals).abs()[act].float().cpu().numpy())
            if sampled.get("terminal_logit") is not None:
                truth = (next_goals <= 0) | (next_moves <= 0)
                one_step["terminal_correct"].append(((sampled["terminal_logit"] > 0) == truth)[act].float().cpu().numpy())
            horizon = step + 1
            if horizon in per_k:
                shape = (n, rollouts_per_anchor)
                pred_boards = torch.where(act.unsqueeze(1), sampled["boards"], next_boards)
                pred_goals = torch.where(act, sampled["goals_left"].long(), next_goals)
                per_k[horizon]["predicted"].append(pred_boards.cpu().numpy().reshape(*shape, N_CELLS))
                per_k[horizon]["target"].append(next_boards.cpu().numpy().reshape(*shape, N_CELLS))
                per_k[horizon]["predicted_goals"].append(pred_goals.cpu().numpy().reshape(shape))
                per_k[horizon]["target_goals"].append(next_goals.cpu().numpy().reshape(shape))
    report: dict[str, object] = {"ks": list(ks), "n_anchors": n_anchors, "rollouts_per_anchor": rollouts_per_anchor, "horizons": {}, "one_step": {}}
    for k in ks:
        if not per_k[k]["predicted"]:
            report["horizons"][f"k{k}"] = {"n": 0}
            continue
        predicted = np.concatenate(per_k[k]["predicted"]); target = np.concatenate(per_k[k]["target"])
        pg = np.concatenate(per_k[k]["predicted_goals"]); tg = np.concatenate(per_k[k]["target_goals"])
        report["horizons"][f"k{k}"] = {
            "board_energy_score": _energy_score((predicted[:, :, None, :] != target[:, None, :, :]).mean(axis=-1), (predicted[:, :, None, :] != predicted[:, None, :, :]).mean(axis=-1)),
            "goals_left_energy_score": _energy_score(np.abs(pg[:, :, None] - tg[:, None, :]), np.abs(pg[:, :, None] - pg[:, None, :])),
            "n_anchors": int(predicted.shape[0]),
        }
    if one_step["n"]:
        report["one_step"] = {"n": one_step["n"], "board_nll_per_cell": float(np.concatenate(one_step["board_nll"]).mean()), "colour_accuracy": float(np.concatenate(one_step["cell_correct"]).mean()),
                              "special_accuracy": float(np.concatenate(one_step["special_correct"]).mean()),
                              "special_accuracy_stripe_cells": float(sum(one_step["special_stripe_correct"]) / max(sum(one_step["special_stripe_cells"]), 1)) if sum(one_step["special_stripe_cells"]) else None,
                              "goals_left_mae": float(np.concatenate(one_step["goal_abs_error"]).mean()),
                              "terminal_accuracy": float(np.concatenate(one_step["terminal_correct"]).mean()) if one_step["terminal_correct"] else None}
    return report


# ------------------------------------------------------------------- mode C ----


@dataclass
class ImaginationResult:
    report: dict[str, object]
    transitions: dict[str, np.ndarray]  # flat generated transitions for the contract checks
    trajectories: dict[str, np.ndarray]  # per-step boards / specials / goals / moves for the raw npz


@torch.no_grad()
def imagine(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], *, ks: tuple[int, ...] = DEFAULT_KS, max_anchors: int = IMAGINATION_ANCHORS, rollouts_per_anchor: int = IMAGINATION_ROLLOUTS,
            seed: int, engine_reference: bool = True) -> ImaginationResult:
    """Mode C: autonomous imagination from fixed anchors with uniform-random legal actions on the decoded model board, against paired engine rollouts.

    Outputs: ImaginationResult with per-horizon board / goals energy scores, terminal rates (model vs engine), terminal-head accuracy against the model's own
    counter-derived terminal flag, retries / fallback rates, and every generated transition (for wm1_contracts).
    """
    ks = tuple(sorted(set(int(k) for k in ks)))
    device = next(model.parameters()).device
    model.eval()
    anchors = collect_anchors(batches, max_anchors=max_anchors, seed=seed)
    n_anchors = anchors["boards"].shape[0]
    rows = n_anchors * rollouts_per_anchor
    repeat = lambda values: values.repeat_interleave(rollouts_per_anchor, dim=0).to(device)
    goal_colours, levels, tiers, served = repeat(anchors["goal_colours"]), repeat(anchors["levels"]), repeat(anchors["tiers"]), repeat(anchors["served_difficulty"])
    context = model.encode_task(levels, tiers, served)
    support = colour_support_of(levels)
    max_moves = int(model.config.max_moves_left)
    boards, specials = repeat(anchors["boards"]), repeat(anchors["specials"])
    moves, goals = repeat(anchors["moves_left"]), repeat(anchors["goals_left"])
    active = torch.ones(rows, dtype=torch.bool, device=device)
    state = model.rollout_begin(rows, context)
    action_generator = torch.Generator(device=device).manual_seed(int(np.random.SeedSequence([seed, 31]).generate_state(1)[0]))
    transition_generator = torch.Generator(device=device).manual_seed(int(np.random.SeedSequence([seed, 32]).generate_state(1)[0]))
    flat: dict[str, list[np.ndarray]] = {name: [] for name in ("boards", "specials", "actions", "moves_left", "goals_left", "next_boards", "next_specials", "next_moves_left", "next_goals_left",
                                                                 "cleared_mask", "colour_support", "terminal_prob", "goal_delta", "retries", "fallback", "deadlocked", "level", "step")}
    traj = {"boards": [boards.cpu().numpy().astype(np.int8)], "specials": [specials.cpu().numpy().astype(np.int8)], "goals_left": [goals.cpu().numpy().astype(np.int16)],
            "moves_left": [moves.cpu().numpy().astype(np.int16)], "active": [active.cpu().numpy()]}
    horizon_states: dict[int, dict[str, np.ndarray]] = {}
    terminal_head = {"correct": 0.0, "n": 0}
    for step in range(ks[-1]):
        legal = torch_legal_mask(boards)
        active &= (goals > 0) & (moves > 0) & legal.any(dim=1)
        if not bool(active.any()):
            for k in ks:
                if k > step and k not in horizon_states:
                    horizon_states[k] = {"boards": boards.cpu().numpy(), "goals": goals.cpu().numpy(), "terminal": ((goals <= 0) | (moves <= 0)).cpu().numpy()}
            break
        actions = _sample_uniform_legal(legal, action_generator)
        state, sampled = model.rollout_sample(state, boards=boards, goal_colours=goal_colours, moves_left=moves.clamp(0, max_moves), goals_left=goals, actions=actions, task_context=context,
                                              colour_support=support, specials=specials, observe=step == 0, stochastic=True, generator=transition_generator)
        next_boards = torch.where(active.unsqueeze(1), sampled["boards"], boards)
        next_specials = torch.where(active.unsqueeze(1), sampled["specials"], specials)
        next_goals = torch.where(active, sampled["goals_left"].long(), goals)
        next_moves = torch.where(active, (moves - 1).clamp_min(0), moves)
        idx = active.cpu().numpy()
        flat["boards"].append(boards.cpu().numpy()[idx]); flat["specials"].append(specials.cpu().numpy()[idx]); flat["actions"].append(actions.cpu().numpy()[idx])
        flat["moves_left"].append(moves.cpu().numpy()[idx]); flat["goals_left"].append(goals.cpu().numpy()[idx])
        flat["next_boards"].append(next_boards.cpu().numpy()[idx]); flat["next_specials"].append(next_specials.cpu().numpy()[idx])
        flat["next_moves_left"].append(next_moves.cpu().numpy()[idx]); flat["next_goals_left"].append(next_goals.cpu().numpy()[idx])
        flat["cleared_mask"].append(sampled["cleared_mask"].cpu().numpy()[idx]); flat["colour_support"].append(support.cpu().numpy()[idx])
        flat["goal_delta"].append(sampled["goal_delta"].cpu().numpy()[idx]); flat["retries"].append(sampled["retries"].cpu().numpy()[idx]); flat["fallback"].append(sampled["fallback"].cpu().numpy()[idx])
        flat["deadlocked"].append((sampled["deadlocked"] if sampled["deadlocked"] is not None else torch.zeros_like(active)).cpu().numpy()[idx])
        flat["level"].append(levels.cpu().numpy()[idx]); flat["step"].append(np.full(int(idx.sum()), step))
        if sampled.get("terminal_logit") is not None:
            prob = torch.sigmoid(sampled["terminal_logit"])
            flat["terminal_prob"].append(prob.cpu().numpy()[idx])
            truth = (next_goals <= 0) | (next_moves <= 0)
            terminal_head["correct"] += float((((prob >= 0.5) == truth) & active).sum()); terminal_head["n"] += int(active.sum())
        else:
            flat["terminal_prob"].append(np.full(int(idx.sum()), np.nan))
        boards, specials, goals, moves = next_boards, next_specials, next_goals, next_moves
        traj["boards"].append(boards.cpu().numpy().astype(np.int8)); traj["specials"].append(specials.cpu().numpy().astype(np.int8)); traj["goals_left"].append(goals.cpu().numpy().astype(np.int16))
        traj["moves_left"].append(moves.cpu().numpy().astype(np.int16)); traj["active"].append(active.cpu().numpy())
        if step + 1 in ks:
            horizon_states[step + 1] = {"boards": boards.cpu().numpy(), "goals": goals.cpu().numpy(), "terminal": ((goals <= 0) | (moves <= 0)).cpu().numpy()}
    transitions = {name: np.concatenate(parts) if parts else np.zeros((0,)) for name, parts in flat.items()}
    report: dict[str, object] = {"ks": list(ks), "n_anchors": n_anchors, "rollouts_per_anchor": rollouts_per_anchor, "n_generated_transitions": int(transitions["actions"].shape[0]),
                                 "terminal_head_accuracy_self": terminal_head["correct"] / terminal_head["n"] if terminal_head["n"] else None,
                                 "retry_rate": float((transitions["retries"] > 0).mean()) if transitions["retries"].size else None,
                                 "fallback_rate": float(transitions["fallback"].mean()) if transitions["fallback"].size else None,
                                 "deadlock_rate": float(transitions["deadlocked"].mean()) if transitions["deadlocked"].size else None, "horizons": {}}
    if engine_reference:
        engine_states = [_engine_state(anchors, a) for a in range(n_anchors) for _ in range(rollouts_per_anchor)]
        engine_levels = [LEVELS[int(anchors["levels"][a])] for a in range(n_anchors) for _ in range(rollouts_per_anchor)]
        action_rngs = [np.random.default_rng(np.random.SeedSequence([seed, 3, a, r])) for a in range(n_anchors) for r in range(rollouts_per_anchor)]
        refill_rngs = [np.random.default_rng(np.random.SeedSequence([seed, 4, a, r])) for a in range(n_anchors) for r in range(rollouts_per_anchor)]
        engine_horizons: dict[int, dict[str, np.ndarray]] = {}
        for step in range(ks[-1]):
            for i, s in enumerate(engine_states):
                if s.terminal:
                    continue
                moves_list = legal_moves(s.board)
                if not moves_list:
                    continue
                engine_states[i], _ = resolve_move(s, moves_list[int(action_rngs[i].integers(len(moves_list)))], engine_levels[i], _draw(engine_levels[i], refill_rngs[i]))
            if step + 1 in ks:
                engine_horizons[step + 1] = {"boards": np.stack([s.board.reshape(-1) for s in engine_states]).astype(np.int64), "goals": np.asarray([s.goals_left for s in engine_states]),
                                             "terminal": np.asarray([s.terminal for s in engine_states])}
        for k in ks:
            if k not in horizon_states:
                continue
            shape = (n_anchors, rollouts_per_anchor)
            pm = horizon_states[k]["boards"].reshape(*shape, N_CELLS); pe = engine_horizons[k]["boards"].reshape(*shape, N_CELLS)
            gm = horizon_states[k]["goals"].reshape(shape); ge = engine_horizons[k]["goals"].reshape(shape)
            report["horizons"][f"k{k}"] = {
                "board_energy_score": _energy_score((pm[:, :, None, :] != pe[:, None, :, :]).mean(axis=-1), (pm[:, :, None, :] != pm[:, None, :, :]).mean(axis=-1)),
                "goals_left_energy_score": _energy_score(np.abs(gm[:, :, None] - ge[:, None, :]), np.abs(gm[:, :, None] - gm[:, None, :])),
                "terminal_rate_model": float(horizon_states[k]["terminal"].mean()), "terminal_rate_engine": float(engine_horizons[k]["terminal"].mean()),
                "goals_left_mean_model": float(gm.mean()), "goals_left_mean_engine": float(ge.mean()),
                "by_level": {LEVELS[i].name: {"n": int((anchors["levels"].numpy() == i).sum()),
                                              "board_energy_score": _energy_score((pm[:, :, None, :] != pe[:, None, :, :]).mean(axis=-1)[anchors["levels"].numpy() == i], (pm[:, :, None, :] != pm[:, None, :, :]).mean(axis=-1)[anchors["levels"].numpy() == i]) if (anchors["levels"].numpy() == i).any() else None,
                                              "goals_left_energy_score": _energy_score(np.abs(gm[:, :, None] - ge[:, None, :])[anchors["levels"].numpy() == i], np.abs(gm[:, :, None] - gm[:, None, :])[anchors["levels"].numpy() == i]) if (anchors["levels"].numpy() == i).any() else None}
                             for i in range(len(LEVELS))},
            }
    trajectories = {name: np.stack(parts, axis=1) for name, parts in traj.items()}
    trajectories["anchor_level"] = anchors["levels"].numpy().astype(np.int8)
    return ImaginationResult(report=report, transitions=transitions, trajectories=trajectories)


# ------------------------------------------------------------------- latency ----


@torch.no_grad()
def measure_latency(model: torch.nn.Module, batch: dict[str, torch.Tensor], *, sizes: tuple[int, ...] = (1, 256), repeats: int = 20) -> dict[str, object]:
    """Median wall time of one rollout step (rollout_sample with stochastic decoding) at the given batch sizes, CUDA-synchronised.

    Inputs: model; a padded batch to draw states from; batch sizes; repeats. Outputs: dict size -> {median_ms_per_step, median_ms_per_state}.
    """
    device = next(model.parameters()).device
    model.eval()
    mask = batch["step_mask"]
    boards = batch["boards"][mask].to(device); specials = batch["specials"][mask].to(device); actions = batch["actions"][mask].to(device)
    goal_colours = batch["goal_colours"][mask].to(device); moves = batch["moves_left"][mask].to(device); goals = batch["goals_left"][mask].to(device)
    steps = mask.shape[1]
    levels = batch["levels"].to(device).unsqueeze(1).expand(-1, steps)[mask.to(device)]
    tiers = batch["tiers"].to(device).unsqueeze(1).expand(-1, steps)[mask.to(device)]
    served = batch["served_difficulty"].to(device).unsqueeze(1).expand(-1, steps)[mask.to(device)]
    out = {}
    for size in sizes:
        take = torch.arange(min(size, boards.shape[0]), device=device)
        if take.numel() < size:
            take = take.repeat(int(np.ceil(size / take.numel())))[:size]
        context = model.encode_task(levels[take], tiers[take], served[take])
        support = colour_support_of(levels[take])
        generator = torch.Generator(device=device).manual_seed(0)
        times = []
        for repeat in range(repeats + 3):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            state = model.rollout_begin(size, context)
            model.rollout_sample(state, boards=boards[take], goal_colours=goal_colours[take], moves_left=moves[take], goals_left=goals[take], actions=actions[take], task_context=context,
                                 colour_support=support, specials=specials[take], observe=True, stochastic=True, generator=generator)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if repeat >= 3:  # warm-up excluded
                times.append(time.perf_counter() - started)
        median = float(np.median(times))
        out[str(size)] = {"median_ms_per_step": 1e3 * median, "median_ms_per_state": 1e3 * median / size, "repeats": repeats}
    return out


__all__ = ["CHUNK", "DEFAULT_KS", "IMAGINATION_ANCHORS", "IMAGINATION_ROLLOUTS", "ImaginationResult", "ReplayRollouts", "collect_anchors", "colour_support_of", "evaluate_engine_conditioned",
           "evaluate_replay", "horizon_metrics", "imagine", "measure_latency", "replay_metrics", "replay_rollouts"]
