"""Evaluation cells: roll one trained arm over (player, posterior draw s, replicate b, candidate e) through a frozen kernel or the engine and save the
per-row arrays every estimator reads (``arrays.npz`` + ``cell.json``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from match3_simulator.retention import ChurnSchedule, MasteryConfig
from match3_simulator.spec import BENCHMARK_CONFIG
from match3_simulator.world_modeling.wm1_data import load_splits
from match3_simulator.experiments.arm_base import CalaArm, load_arm
from match3_simulator.experiments.contexts import Arm
from match3_simulator.experiments.data_base import LANDMARK, HandcraftedFeatures, Logged
from match3_simulator.experiments.guard import EvaluationOnly, phase
from match3_simulator.experiments.rollout_utils import accepted_churn_fn, churn_of, engine_rows, imagine_rows, load_kernel, make_rows
from match3_simulator.experiments.training_utils import ArmInputs, accepted

GRID: tuple[float, ...] = tuple(float(e) for e in BENCHMARK_CONFIG.e_grid)


@dataclass(frozen=True)
class CellSpec:
    arm_run: str
    dynamics: str  # kernel name or "engine"
    kernel_seed: int
    regime: str
    population: str = "test"
    S: int = 1
    B: int = 1
    seed: int = 4201
    temperature: float = 1.0
    grid: tuple[float, ...] = GRID
    include_logged: bool = True
    n_players: int | None = None
    chunk: int = 8192
    workers: int = 8
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.S < 1 or self.B < 1:
            raise ValueError("S and B must be positive")
        if self.population not in ("test", "validation"):
            raise ValueError("population must be test or validation")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    @property
    def arm(self) -> str:
        return json.loads((Path(self.arm_run) / "training.json").read_text())["arm"]

    @property
    def dynamics_label(self) -> str:
        return "engine" if self.dynamics == "engine" else f"{self.dynamics}-{self.kernel_seed}"

    def label(self) -> str:
        return f"{self.arm}-{self.regime}-{self.seed}__{self.dynamics_label}__{self.population}_S{self.S}_B{self.B}_T{self.temperature:g}" + (f"_off{self.seed_offset}" if self.seed_offset else "") + (f"_n{self.n_players}" if self.n_players else "")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_trained_arm(run_dir: str | Path, *, device: torch.device, churn: ChurnSchedule, mastery: MasteryConfig) -> tuple[CalaArm, dict[str, object]]:
    model, meta = load_arm(Path(run_dir) / "best.pt", churn=churn, mastery=mastery, device=device)
    training = json.loads((Path(run_dir) / "training.json").read_text())
    return model, training


def arm_inputs_for(model: CalaArm, training: dict[str, object], logged: Logged) -> ArmInputs:
    features = HandcraftedFeatures.from_json(training["features"]) if training.get("features") else None
    return ArmInputs(model.arm, logged, features=features, oracle_consumer="oracle_k" if model.arm is Arm.ORACLE_K else None)


def draw_eps(player_ids: np.ndarray, S: int, seed: int, d: int) -> np.ndarray:
    """Standard-normal draws eps[i, s] from SeedSequence([seed, player_id, s]). Outputs: (P, S, d) float32."""
    out = np.empty((len(player_ids), S, d), dtype=np.float32)
    for i, pid in enumerate(player_ids):
        for s in range(S):
            out[i, s] = np.random.default_rng(np.random.SeedSequence([int(seed), int(pid), int(s)])).standard_normal(d)
    return out


@torch.no_grad()
def player_contexts(model: CalaArm, inputs: ArmInputs, idx: np.ndarray, *, S: int, seed: int, device: torch.device, batch: int = 256) -> tuple[np.ndarray, np.ndarray | None]:
    """Contexts (P, S, width) inferred once from the strict prefix at the landmark; eps (P, S, d) for the variational arm else None."""
    logged = inputs.logged
    pids = logged.player_ids[idx]
    eps = draw_eps(pids, S, seed, model.config.latent_dimensions) if model.arm is Arm.VAR_Z else None
    out = []
    for start in range(0, len(idx), batch):
        ci = inputs.context_inputs(idx[start : start + batch], LANDMARK, device)
        e = None if eps is None else torch.as_tensor(eps[start : start + batch], device=device)
        ctx, _, _ = model.context(prefix=ci.get("prefix"), base=ci["base"], extra=ci.get("extra"), S=S, eps=e)
        out.append(ctx.cpu().numpy().astype(np.float32))
    return np.concatenate(out), eps


def run_cell(spec: CellSpec, *, out_dir: str | Path, device: torch.device, progress=None, dataset_root: str = "data/release", cache_dir: str = "runs/cala/data") -> Path:
    """Run one cell (skips when its arrays exist). Outputs: the cell directory."""
    emit = progress or (lambda _: None)
    cell_dir = Path(out_dir) / spec.label()
    if (cell_dir / "arrays.npz").exists():
        emit(f"cell {spec.label()} exists; skipping")
        return cell_dir
    started = time.perf_counter()
    churn, mastery, _, _ = accepted(spec.regime)
    logged = Logged(cache_dir, spec.regime, splits=load_splits(dataset_root), mastery_config=mastery)
    model, training = load_trained_arm(spec.arm_run, device=device, churn=churn, mastery=mastery)
    S = spec.S if model.arm is Arm.VAR_Z else 1
    token = EvaluationOnly("oracle_k") if model.arm is Arm.ORACLE_K else None
    if token is not None:
        token.__enter__()
    try:
        with phase("evaluate"):
            inputs = arm_inputs_for(model, training, logged)
            idx = logged.active_at(LANDMARK, spec.population)
            if spec.n_players:
                idx = idx[: spec.n_players]
            target = logged.target_rows(idx, LANDMARK)
            P = len(idx)
            ctx, eps = player_contexts(model, inputs, idx, S=S, seed=spec.seed, device=device)
            grid = np.asarray(spec.grid, dtype=np.float64)
            served = np.tile(grid, (P, 1))
            if spec.include_logged:
                served = np.concatenate([served, target["E"][:, None]], axis=1)
            C = served.shape[1]
            p_i, s_i, b_i = np.meshgrid(np.arange(P), np.arange(S), np.arange(spec.B), indexing="ij")
            p_i, s_i, b_i = p_i.ravel(), s_i.ravel(), b_i.ravel()
            refill = np.asarray([np.random.SeedSequence([spec.seed, int(target["player_id"][p]), int(s), int(b)]).generate_state(1)[0] for p, s, b in zip(p_i, s_i, b_i)], dtype=np.int64)
            row_ctx = torch.as_tensor(ctx[p_i, s_i], device=device)
            policy = lambda b, c, m, g, l, i: model.policy(board=b, goal_colour=c, moves_left=m, goals_left=g, skill=row_ctx[i], legal_actions=l)
            retention = model.retention_fn(row_ctx)
            hazard = accepted_churn_fn(churn)
            kernel = None if spec.dynamics == "engine" else load_kernel(spec.dynamics, spec.kernel_seed, device=device)
            shape = (P, S, spec.B, C)
            arrays = {k: np.zeros(shape, dtype=np.float32) for k in ("margin", "mastery_after", "churn", "churn_hazard")}
            arrays["won"] = np.zeros(shape, dtype=np.int8)
            quota = np.zeros((P, C), dtype=np.int64)
            timings, n_actions = [], 0
            stream_seed = int(np.random.SeedSequence([spec.seed, spec.seed_offset, 5]).generate_state(1)[0])
            for c in range(C):
                t0 = time.perf_counter()
                rows = make_rows({k: v[p_i] for k, v in target.items()}, served=served[p_i, c], context=ctx[p_i, s_i], refill_seed=refill)
                quota[:, c] = rows.goals.reshape(P, S, spec.B)[:, 0, 0]
                if kernel is None:
                    res = engine_rows(policy, rows, seed=stream_seed, device=device, temperature=spec.temperature, workers=spec.workers)
                else:
                    res = imagine_rows(kernel, policy, rows, seed=stream_seed, temperature=spec.temperature, chunk=spec.chunk)
                learned = churn_of(res, rows, churn_fn=retention, mastery=mastery)
                haz = churn_of(res, rows, churn_fn=hazard, mastery=mastery)
                arrays["won"][..., c] = res["won"].reshape(P, S, spec.B)
                arrays["margin"][..., c] = res["margin"].reshape(P, S, spec.B)
                arrays["mastery_after"][..., c] = learned["mastery_after"].reshape(P, S, spec.B)
                arrays["churn"][..., c] = learned["churn"].reshape(P, S, spec.B)
                arrays["churn_hazard"][..., c] = haz["churn"].reshape(P, S, spec.B)
                n_actions += int(res["n_actions"])
                timings.append(time.perf_counter() - t0)
                emit(f"{spec.label()} candidate {c + 1}/{C} (e={'logged' if c == len(grid) else f'{served[0, c]:+.2f}'}): win {res['won'].mean():.3f}, churn {learned['churn'].mean():.4f} ({timings[-1]:.1f}s)")
    finally:
        if token is not None:
            token.__exit__(None, None, None)
    cell_dir.mkdir(parents=True, exist_ok=True)
    payload = {"player_ids": target["player_id"], "level": target["level"], "tier": target["tier"], "E_logged": target["E"], "mastery_before": target["mastery_before"], "served": served.astype(np.float32), "quota": quota,
               "grid": grid, "context": ctx, **arrays}
    if eps is not None:
        payload["eps"] = eps
    partial = cell_dir / f"arrays.partial-{os.getpid()}"
    np.savez_compressed(partial, **payload)
    meta = {"spec": spec.to_dict(), "label": spec.label(), "arm": model.arm.value, "context_width": model.width, "S_effective": S, "players": int(P), "candidates": int(C), "has_logged_candidate": bool(spec.include_logged),
            "n_actions": n_actions, "illegal": 0, "seconds": time.perf_counter() - started, "seconds_per_candidate": timings, "rows_per_candidate": int(P * S * spec.B),
            "oracle_accesses": len(inputs.oracle_accesses), "arm_checkpoint_sha256": __import__("hashlib").sha256((Path(spec.arm_run) / "best.pt").read_bytes()).hexdigest(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (cell_dir / "cell.json").write_text(json.dumps(meta, indent=1) + "\n")
    os.replace(str(partial) + ".npz", cell_dir / "arrays.npz")
    emit(f"cell {spec.label()} done: {P} players x {S} x {spec.B} x {C} candidates, {n_actions} legal actions, {time.perf_counter() - started:.0f}s")
    return cell_dir


def load_cell(cell_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(cell_dir) / "arrays.npz", allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    return arrays, json.loads((Path(cell_dir) / "cell.json").read_text())


def level_slices(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from match3_simulator.scm import LEVELS

    return {level.name: np.flatnonzero(arrays["level"] == li) for li, level in enumerate(LEVELS)}


__all__ = ["CellSpec", "GRID", "arm_inputs_for", "draw_eps", "level_slices", "load_cell", "load_trained_arm", "player_contexts", "run_cell"]
