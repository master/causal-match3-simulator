from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from match3_simulator.world_modeling.wm1_data import load_splits
from match3_simulator.experiments.cell_utils import CellSpec, GRID, draw_eps
from match3_simulator.experiments.contexts import Arm
from match3_simulator.experiments.data_base import LANDMARK, HandcraftedFeatures
from match3_simulator.experiments.guard import GuardViolation, phase
from match3_simulator.experiments.rollout_utils import accepted_churn_fn, churn_of, load_kernel, make_rows
from match3_simulator.experiments.training_utils import ArmInputs, accepted
from match3_simulator.experiments.arm import CalaArm4, load_arm4
from match3_simulator.experiments.data import Logged4
from match3_simulator.experiments.rollout import assert_kernel_has_no_context_input, imagine_rows4

POSTERIORS = ("sample", "mean")
ARRAY_KEYS = ("won", "margin", "mastery_after", "churn", "churn_hazard")


@dataclass(frozen=True)
class CellSpec4(CellSpec):
    posterior: str = "sample"
    variant: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.posterior not in POSTERIORS:
            raise ValueError(f"posterior must be one of {POSTERIORS}")

    def label(self) -> str:
        base = super().label()
        return base + ("_mean" if self.posterior == "mean" else "")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_trained_arm4(run_dir: str | Path, *, device: torch.device, churn, mastery) -> tuple[CalaArm4, dict[str, object]]:
    model, _ = load_arm4(Path(run_dir) / "best.pt", churn=churn, mastery=mastery, device=device)
    training = json.loads((Path(run_dir) / "training.json").read_text())
    return model, training


def arm_inputs_for4(model: CalaArm4, training: dict[str, object], logged: Logged4, *, allow_oracle: bool = False) -> ArmInputs:
    """``allow_oracle`` (2026-10-01): the frozen oracle-K reference arm may be *analysed* (support mask / W for saved cells, under an EvaluationOnly token held
    by the caller) so that the equal-budget tables can carry the 'Oracle K' and 'No history' rows; it is still refused for CaLA-4 training and rollouts."""
    if model.arm is Arm.ORACLE_K and not allow_oracle:
        raise GuardViolation("oracle arm not part of the CaLA-4 study")
    features = HandcraftedFeatures.from_json(training["features"]) if training.get("features") else None
    return ArmInputs(model.arm, logged, features=features, oracle_consumer="oracle_k" if model.arm is Arm.ORACLE_K else None)


@torch.no_grad()
def player_contexts4(model: CalaArm4, inputs: ArmInputs, idx: np.ndarray, *, S: int, seed: int, device: torch.device, posterior: str = "sample", batch: int = 256) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    """Contexts (P, S, width) inferred once from the strict prefix at the landmark. Outputs also eps (P, S, d) or None and posterior info (mean, log_scale) for the Z arms."""
    logged = inputs.logged
    pids = logged.player_ids[idx]
    is_var = model.arm is Arm.VAR_Z
    eps = None
    if is_var:
        eps = np.zeros((len(pids), S, model.config.latent_dimensions), np.float32) if posterior == "mean" else draw_eps(pids, S, seed, model.config.latent_dimensions)
    out, means, scales = [], [], []
    for start in range(0, len(idx), batch):
        ci = inputs.context_inputs(idx[start : start + batch], LANDMARK, device)
        e = None if eps is None else torch.as_tensor(eps[start : start + batch], device=device)
        ctx, _, info = model.context(prefix=ci.get("prefix"), base=ci["base"], extra=ci.get("extra"), S=S, eps=e)
        out.append(ctx.cpu().numpy().astype(np.float32))
        if "mean" in info:
            means.append(info["mean"].cpu().numpy())
        if "log_scale" in info:
            scales.append(info["log_scale"].cpu().numpy())
    info_out = {}
    if means:
        info_out["posterior_mean"] = np.concatenate(means)
    if scales:
        info_out["posterior_log_scale"] = np.concatenate(scales)
    return np.concatenate(out), eps, info_out


def block_path(cell_dir: Path, b_start: int, b_end: int) -> Path:
    return cell_dir / f"arrays.block-b{b_start:03d}-{b_end:03d}.npz"


def run_cell4(spec: CellSpec4, *, out_dir: str | Path, device: torch.device, cache_dir: str, dataset_root: str = "data/release", block: int | None = None, progress=None, stop_file: str | Path | None = None,
              player_ids: np.ndarray | None = None, logged: Logged4 | None = None) -> Path:
    """Run one cell in replicate blocks (default: one block = the whole cell). Skips finished blocks; merges into ``arrays.npz`` + ``cell.json``. Outputs: the cell directory."""
    emit = progress or (lambda _: None)
    cell_dir = Path(out_dir) / spec.label()
    if (cell_dir / "arrays.npz").exists():
        emit(f"cell {spec.label()} exists; skipping")
        return cell_dir
    cell_dir.mkdir(parents=True, exist_ok=True)
    block = block or spec.B
    blocks = [(b, min(b + block, spec.B)) for b in range(0, spec.B, block)]
    todo = [(a, b) for a, b in blocks if not block_path(cell_dir, a, b).exists()]
    churn, mastery, _, _ = accepted(spec.regime)
    logged = logged if logged is not None else Logged4(cache_dir, spec.regime, splits=load_splits(dataset_root), mastery_config=mastery)
    model, training = load_trained_arm4(spec.arm_run, device=device, churn=churn, mastery=mastery)
    S = spec.S if (model.arm is Arm.VAR_Z and spec.posterior == "sample") else 1
    if spec.dynamics == "engine":
        raise ValueError("CaLA-4 cells run through the frozen kernel; engine cells keep the frozen path")
    kernel = load_kernel(spec.dynamics, spec.kernel_seed, device=device)
    assert_kernel_has_no_context_input(kernel)
    with phase("evaluate"):
        inputs = arm_inputs_for4(model, training, logged)
        idx = logged.active_at(LANDMARK, spec.population)
        if player_ids is not None:
            keep = set(int(p) for p in player_ids)
            idx = np.asarray([i for i in idx if int(logged.player_ids[i]) in keep], dtype=np.int64)
        if spec.n_players:
            idx = idx[: spec.n_players]
        target = logged.target_rows(idx, LANDMARK)
        P = len(idx)
        ctx, eps, info = player_contexts4(model, inputs, idx, S=S, seed=spec.seed, device=device, posterior=spec.posterior)
        grid = np.asarray(spec.grid, dtype=np.float64)
        served = np.tile(grid, (P, 1))
        if spec.include_logged:
            served = np.concatenate([served, target["E"][:, None]], axis=1)
        C = served.shape[1]
        hazard = accepted_churn_fn(churn)
        meta_common = {"spec": spec.to_dict(), "label": spec.label(), "arm": model.arm.value, "variant": model.config.variant, "context_width": model.width, "S_effective": S, "posterior": spec.posterior if model.arm is Arm.VAR_Z else "n/a",
                       "players": int(P), "candidates": int(C), "has_logged_candidate": bool(spec.include_logged), "kernel": spec.dynamics_label, "arm_checkpoint_sha256": hashlib.sha256((Path(spec.arm_run) / "best.pt").read_bytes()).hexdigest(),
                       "oracle_accesses": 0, "arm_config": model.config.to_dict(), "parameters": model.parameter_counts()}
        for b_start, b_end in todo:
            if stop_file is not None and Path(stop_file).exists():
                emit(f"stop file {stop_file} present; leaving before block b{b_start}-{b_end}")
                return cell_dir
            started = time.perf_counter()
            Bb = b_end - b_start
            p_i, s_i, b_i = np.meshgrid(np.arange(P), np.arange(S), np.arange(b_start, b_end), indexing="ij")
            p_i, s_i, b_i = p_i.ravel(), s_i.ravel(), b_i.ravel()
            refill = np.asarray([np.random.SeedSequence([spec.seed, int(target["player_id"][p]), int(s), int(b)]).generate_state(1)[0] for p, s, b in zip(p_i, s_i, b_i)], dtype=np.int64)
            row_ctx = torch.as_tensor(ctx[p_i, s_i], device=device)
            policy = model.policy_fn(row_ctx)
            retention = model.retention_fn(row_ctx)
            shape = (P, S, Bb, C)
            arrays = {k: np.zeros(shape, dtype=np.float32) for k in ("margin", "mastery_after", "churn", "churn_hazard")}
            arrays["won"] = np.zeros(shape, dtype=np.int8)
            quota = np.zeros((P, C), dtype=np.int64)
            timings, n_actions, stripe_states = [], 0, 0
            stream_seed = int(np.random.SeedSequence([spec.seed, spec.seed_offset, 5]).generate_state(1)[0]) if len(blocks) == 1 else int(np.random.SeedSequence([spec.seed, spec.seed_offset, 5, int(b_start)]).generate_state(1)[0])
            for c in range(C):
                t0 = time.perf_counter()
                rows = make_rows({k: v[p_i] for k, v in target.items()}, served=served[p_i, c], context=ctx[p_i, s_i], refill_seed=refill)
                quota[:, c] = rows.goals.reshape(P, S, Bb)[:, 0, 0]
                res = imagine_rows4(kernel, policy, rows, seed=stream_seed, temperature=spec.temperature, chunk=spec.chunk)
                learned = churn_of(res, rows, churn_fn=retention, mastery=mastery)
                haz = churn_of(res, rows, churn_fn=hazard, mastery=mastery)
                arrays["won"][..., c] = res["won"].reshape(P, S, Bb)
                arrays["margin"][..., c] = res["margin"].reshape(P, S, Bb)
                arrays["mastery_after"][..., c] = learned["mastery_after"].reshape(P, S, Bb)
                arrays["churn"][..., c] = learned["churn"].reshape(P, S, Bb)
                arrays["churn_hazard"][..., c] = haz["churn"].reshape(P, S, Bb)
                n_actions += int(res["n_actions"])
                stripe_states += int(res["stripe_states"])
                timings.append(time.perf_counter() - t0)
                emit(f"{spec.label()} b{b_start}-{b_end} candidate {c + 1}/{C} (e={'logged' if c == len(grid) else f'{served[0, c]:+.2f}'}): win {res['won'].mean():.3f}, churn {learned['churn'].mean():.4f} ({timings[-1]:.1f}s)")
            payload = {"player_ids": target["player_id"], "level": target["level"], "tier": target["tier"], "E_logged": target["E"], "mastery_before": target["mastery_before"], "served": served.astype(np.float32), "quota": quota,
                       "grid": grid, "context": ctx, "b_start": np.int64(b_start), "b_end": np.int64(b_end), **arrays, **info}
            if eps is not None:
                payload["eps"] = eps
            path = block_path(cell_dir, b_start, b_end)
            partial = cell_dir / f"{path.stem}.partial-{os.getpid()}"
            np.savez_compressed(partial, **payload)
            os.replace(str(partial) + ".npz", path)
            meta = {**meta_common, "block": [b_start, b_end], "rows_per_candidate": int(P * S * Bb), "n_actions": n_actions, "stripe_states_seen_by_policy": stripe_states, "illegal": 0, "seconds": time.perf_counter() - started,
                    "seconds_per_candidate": timings, "stream_seed": stream_seed, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
            emit(f"block {path.name} done: {P} x {S} x {Bb} x {C}, {n_actions} legal actions, {stripe_states} stripe states seen, {meta['seconds']:.0f}s")
    merge_blocks4(cell_dir, expected_B=spec.B, progress=emit)
    return cell_dir


def merge_blocks4(cell_dir: str | Path, *, expected_B: int | None = None, progress=None) -> Path | None:
    """Concatenate contiguous finished blocks into ``arrays.npz`` + ``cell.json`` (readable by ``cala.cells.load_cell``). Outputs: arrays path or None."""
    emit = progress or (lambda _: None)
    cell_dir = Path(cell_dir)
    blocks = sorted(cell_dir.glob("arrays.block-b*.npz"))
    metas = [json.loads(p.with_suffix(".json").read_text()) for p in blocks]
    order = sorted(range(len(blocks)), key=lambda i: metas[i]["block"][0])
    chain, expect = [], 0
    for i in order:
        if metas[i]["block"][0] != expect:
            break
        chain.append(i)
        expect = metas[i]["block"][1]
    if not chain:
        return None
    if expected_B is not None and expect != expected_B:
        emit(f"{cell_dir.name}: {expect} / {expected_B} replicates finished; not merging yet")
        return None
    loaded = [np.load(blocks[i], allow_pickle=False) for i in chain]
    first = loaded[0]
    payload = {k: np.asarray(first[k]) for k in first.files if k not in ARRAY_KEYS and k not in ("b_start", "b_end")}
    for k in ARRAY_KEYS:
        payload[k] = np.concatenate([np.asarray(z[k]) for z in loaded], axis=2)
    B_total = int(payload["won"].shape[2])
    m0 = metas[chain[0]]
    meta = {k: v for k, v in m0.items() if k not in ("block", "rows_per_candidate", "n_actions", "seconds", "seconds_per_candidate", "stream_seed", "created_at", "stripe_states_seen_by_policy")}
    meta.update({"B_merged": B_total, "blocks": [metas[i]["block"] for i in chain], "rows_per_candidate": int(m0["players"] * m0["S_effective"] * B_total), "n_actions": int(sum(metas[i]["n_actions"] for i in chain)),
                 "stripe_states_seen_by_policy": int(sum(metas[i]["stripe_states_seen_by_policy"] for i in chain)), "illegal": 0, "seconds": float(sum(metas[i]["seconds"] for i in chain)),
                 "seconds_per_candidate": [float(np.sum(x)) for x in zip(*[metas[i]["seconds_per_candidate"] for i in chain])], "stream_seeds": [metas[i]["stream_seed"] for i in chain],
                 "trajectories_per_player_candidate": int(m0["S_effective"] * B_total), "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    partial = cell_dir / f"arrays.partial-{os.getpid()}"
    np.savez_compressed(partial, **payload)
    os.replace(str(partial) + ".npz", cell_dir / "arrays.npz")
    (cell_dir / "cell.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    emit(f"merged {cell_dir.name}: B = {B_total} from {len(chain)} block(s) ({meta['seconds']:.0f}s GPU)")
    return cell_dir / "arrays.npz"


def effective_trajectories(cell_dir: str | Path) -> dict[str, int]:
    """Verified counts from the saved arrays, never from labels. Outputs: players, S, B, candidates, trajectories_per_player_candidate."""
    with np.load(Path(cell_dir) / "arrays.npz", allow_pickle=False) as z:
        P, S, B, C = z["churn"].shape
    return {"players": int(P), "S": int(S), "B": int(B), "candidates": int(C), "trajectories_per_player_candidate": int(S * B)}


__all__ = ["ARRAY_KEYS", "CellSpec4", "GRID", "POSTERIORS", "arm_inputs_for4", "effective_trajectories", "load_trained_arm4", "merge_blocks4", "player_contexts4", "run_cell4"]
