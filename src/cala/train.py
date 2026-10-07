"""Train one CaLA arm."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from match3_simulator.retention import update_mastery
from match3_simulator.world_modeling.decoder import torch_legal_mask
from match3_simulator.world_modeling.wm1_data import load_splits
from match3_simulator.experiments.contexts import Arm
from match3_simulator.experiments.data_base import LANDMARK
from match3_simulator.experiments.guard import EvaluationOnly, GuardViolation, phase
from match3_simulator.experiments.rollout_utils import load_kernel, make_rows
from match3_simulator.experiments.training_utils import ArmInputs, CalaTrainConfig, accepted, fit_propensity, retention_calibration
from match3_simulator.experiments.arm import CalaArm4, CalaArm4Config, save_arm4
from match3_simulator.experiments.data import Logged4, stripe_state_mask
from match3_simulator.experiments.rollout import imagine_rows4

os.environ.setdefault("WANDB_MODE", "offline")
STATUS4 = "cala4_arch_run"
DEFAULT_CACHE = "runs/cala-4/CURRENT/data"


@dataclass(frozen=True)
class CalaTrainConfig4(CalaTrainConfig):
    encoder: str = "current"
    policy_specials: bool = False
    policy_special_features: bool = False
    cnn_channels: int = 32
    cnn_layers: int = 3
    frozen_config_source: str | None = None
    frozen_config_sha256: str | None = None

    @property
    def variant(self) -> str:
        return CalaArm4Config(arm=self.arm, feature_width=1 if self.arm == "handcrafted" else 0, encoder=self.encoder, policy_specials=self.policy_specials, policy_special_features=self.policy_special_features).variant

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def frozen_train_config(regime: str, seed: int, *, cache_dir: str = DEFAULT_CACHE, arms_root: str = "runs/cala", **overrides) -> CalaTrainConfig4:
    """The frozen study's training configuration for (regime, seed) with the CaLA-4 switches applied. Records the source file and the SHA-256 of its config block."""
    src = Path(arms_root) / regime / "arms" / f"var_z-{seed}" / "training.json"
    frozen = json.loads(src.read_text())["config"]
    digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    names = {f.name for f in fields(CalaTrainConfig4)}
    base = {k: v for k, v in frozen.items() if k in names}
    base["temperature_grid"] = tuple(base["temperature_grid"])
    base.update({"cache_dir": cache_dir, "wandb": False, "status": STATUS4, "label": None, "frozen_config_source": str(src), "frozen_config_sha256": digest})
    base.update(overrides)
    return CalaTrainConfig4(**base)


def arm_config_of(config: CalaTrainConfig4, feature_width: int) -> CalaArm4Config:
    return CalaArm4Config(arm=config.arm, feature_width=feature_width, latent_dimensions=config.latent_dimensions, encoder_hidden=config.encoder_hidden, policy_d_model=config.policy_d_model, policy_layers=config.policy_layers,
                          policy_heads=config.policy_heads, kl_weight=config.kl_weight, l2_weight=config.l2_weight, retention_weight=config.retention_weight, aux_weight=config.aux_weight,
                          encoder=config.encoder, policy_specials=config.policy_specials, policy_special_features=config.policy_special_features, cnn_channels=config.cnn_channels, cnn_layers=config.cnn_layers)


def make_batch4(logged: Logged4, inputs: ArmInputs, idx: np.ndarray, a: np.ndarray, *, mastery, device: torch.device) -> dict[str, torch.Tensor]:
    """``cala.train.make_batch`` + ``t_specials`` (M, 64) of the target attempt's logged transitions."""
    b = inputs.context_inputs(idx, a, device)
    r, o, t = logged.target_rows(idx, a), logged.outcome_rows(idx, a), logged.transition_rows(idx, a)
    tb = torch.as_tensor(t["boards"], device=device)
    f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    i64 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)
    b.update({"level": i64(r["level"]), "tier": i64(r["tier"]), "attempt": i64(r["attempt"]), "E": f32(r["E"]), "R": f32(o["R"]), "Q": f32(o["Q"]), "C": f32(o["C"]),
              "mastery_after": f32(update_mastery(r["mastery_before"], o["R"].astype(np.int64), mastery)), "evidence": f32(o["evidence"]),
              "summaries": f32(np.stack([o["mean_legal"] / 10.0, o["mean_goal_clear"]], axis=1)),
              "t_boards": tb, "t_actions": i64(t["actions"]), "t_moves": i64(t["moves_left"]), "t_goals": i64(t["goals_left"]), "t_colour": i64(t["goal_colour"]), "t_legal": torch_legal_mask(tb), "t_row": i64(t["row"]),
              "t_specials": i64(t["specials"])})
    return b


def sample_rows(logged, players, n, rng):
    idx = rng.choice(players, size=n, replace=True)
    a = (rng.random(n) * logged.n_attempts[idx]).astype(np.int64) + 1
    return idx, a


def fixed_validation_rows(logged, n, seed):
    return sample_rows(logged, logged.indices("validation"), n, np.random.default_rng(np.random.SeedSequence([seed, 991])))


@torch.no_grad()
def evaluate_rows4(model: CalaArm4, logged: Logged4, inputs: ArmInputs, idx, a, *, mastery, device, batch: int = 96) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n_rows = n_tr = 0
    for start in range(0, len(idx), batch):
        b = make_batch4(logged, inputs, idx[start : start + batch], a[start : start + batch], mastery=mastery, device=device)
        out = model.objective(b, generator=torch.Generator(device=device).manual_seed(0))
        rows, trans = float(out["n_rows"]), float(out["n_transitions"])
        for k, v in out.items():
            if k in ("n_rows", "n_transitions"):
                continue
            sums[k] = sums.get(k, 0.0) + float(v) * (trans if k == "action" else rows)
        n_rows += rows
        n_tr += trans
    model.train()
    return {k: v / (n_tr if k == "action" else n_rows) for k, v in sums.items()}


@torch.no_grad()
def policy_metrics4(model: CalaArm4, logged: Logged4, inputs: ArmInputs, idx, a, *, device, batch: int = 64, seed: int = 0) -> dict[str, object]:
    """``cala.train.policy_metrics`` + the same metrics restricted to stripe states (>= 1 special on the board) and to stripe-free states."""
    from match3_simulator.learned_model.tokens import ACTION_SLOTS, IN_BOUNDS

    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    nll, top1, top5, conf, correct, illegal_mass, stripe = [], [], [], [], [], [], []
    illegal_realised, n = 0, 0
    sampled_hist = np.zeros(ACTION_SLOTS)
    logged_hist = np.zeros(ACTION_SLOTS)
    in_bounds = torch.as_tensor(IN_BOUNDS, device=device)
    for start in range(0, len(idx), batch):
        ci = inputs.context_inputs(idx[start : start + batch], a[start : start + batch], device)
        ctx, _, _ = model.context(prefix=ci.get("prefix"), base=ci["base"], extra=ci.get("extra"), S=1, generator=gen)
        t = logged.transition_rows(idx[start : start + batch], a[start : start + batch])
        if not len(t["row"]):
            continue
        tb = torch.as_tensor(t["boards"], device=device)
        ts = torch.as_tensor(t["specials"], device=device)
        legal = torch_legal_mask(tb)
        args = dict(board=tb, goal_colour=torch.as_tensor(t["goal_colour"], device=device), moves_left=torch.as_tensor(t["moves_left"], device=device), goals_left=torch.as_tensor(t["goals_left"], device=device),
                    skill=ctx[torch.as_tensor(t["row"], device=device), 0], specials=ts)
        logp = model.policy(**args, legal_actions=legal)
        actions = torch.as_tensor(t["actions"], device=device)
        nll.append(-logp.gather(1, actions[:, None]).squeeze(1))
        rank = (logp > logp.gather(1, actions[:, None])).sum(dim=1)
        top1.append(rank == 0)
        top5.append(rank < 5)
        conf.append(logp.max(dim=1).values.exp())
        correct.append(logp.argmax(dim=1) == actions)
        stripe.append(torch.as_tensor(stripe_state_mask(ts), device=device))
        unmasked = model.policy(**args, legal_actions=in_bounds.unsqueeze(0).expand(len(tb), -1))
        illegal_mass.append((unmasked.exp() * (~legal).float()).sum(dim=1))
        draw = torch.multinomial(logp.exp(), 1, generator=gen).squeeze(1)
        illegal_realised += int((~legal.gather(1, draw[:, None]).squeeze(1)).sum())
        sampled_hist += np.bincount(draw.cpu().numpy(), minlength=ACTION_SLOTS)
        logged_hist += np.bincount(actions.cpu().numpy(), minlength=ACTION_SLOTS)
        n += len(tb)
    cat = lambda xs: torch.cat(xs).float().cpu().numpy()
    nll_v, conf_v, corr_v, t1, t5, st = cat(nll), cat(conf), cat(correct), cat(top1), cat(top5), cat(stripe).astype(bool)
    bins = np.clip((conf_v * 10).astype(int), 0, 9)
    ece = float(sum(abs(corr_v[bins == k].mean() - conf_v[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any()))
    ent = lambda h: float(-(h[h > 0] / h.sum() * np.log(h[h > 0] / h.sum())).sum()) if h.sum() else 0.0
    sub = lambda m: {"n_transitions": int(m.sum()), "action_nll": float(nll_v[m].mean()) if m.any() else None, "top1": float(t1[m].mean()) if m.any() else None, "top5": float(t5[m].mean()) if m.any() else None}
    model.train()
    return {"n_transitions": int(n), "action_nll": float(nll_v.mean()), "top1": float(t1.mean()), "top5": float(t5.mean()), "ece": ece, "illegal_mass_before_mask": float(cat(illegal_mass).mean()),
            "illegal_rate_after_mask": illegal_realised / max(n, 1), "illegal_after_mask": illegal_realised, "entropy_sampled_actions": ent(sampled_hist), "entropy_logged_actions": ent(logged_hist),
            "distinct_sampled_actions": int((sampled_hist > 0).sum()), "stripe_states": sub(st), "stripe_free_states": sub(~st), "stripe_state_fraction": float(st.mean())}


@torch.no_grad()
def fit_temperature4(model: CalaArm4, logged: Logged4, inputs: ArmInputs, config: CalaTrainConfig4, *, device, progress=None) -> dict[str, object]:
    """``cala.train.fit_temperature`` with the special-aware rollout (identical criterion, players, replicates, kernel and grid)."""
    emit = progress or (lambda _: None)
    tr = logged.active_at(LANDMARK, "train")
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 77]))
    idx = np.sort(rng.choice(tr, size=min(config.temperature_players, len(tr)), replace=False))
    target, outcome = logged.target_rows(idx, LANDMARK), logged.outcome_rows(idx, LANDMARK)
    ci = inputs.context_inputs(idx, LANDMARK, device)
    eps = torch.as_tensor(np.stack([np.random.default_rng(np.random.SeedSequence([config.seed, int(p), 0])).standard_normal(config.latent_dimensions) for p in target["player_id"]]), dtype=torch.float32, device=device)[:, None, :]
    ctx, _, _ = model.context(prefix=ci.get("prefix"), base=ci["base"], extra=ci.get("extra"), S=1, eps=eps if model.arm is Arm.VAR_Z else None)
    ctx = ctx[:, 0]
    B = config.temperature_replicates
    rep = np.repeat(np.arange(len(idx)), B)
    rows = make_rows({k: v[rep] for k, v in target.items()}, served=target["E"][rep], context=ctx.cpu().numpy()[rep], quota=target["served_goal_count"][rep],
                     refill_seed=np.asarray([np.random.SeedSequence([config.seed, int(p), 0, b]).generate_state(1)[0] for p, b in zip(target["player_id"][rep], np.tile(np.arange(B), len(idx)))]))
    ctx_rows = ctx[torch.as_tensor(rep, device=device)]
    policy = model.policy_fn(ctx_rows)
    kernel = load_kernel(config.temperature_kernel, config.temperature_kernel_seed, device=device)
    logged_q = np.clip(outcome["Q"], -1, 1)
    qs = np.linspace(0.1, 0.9, 9)
    ref = {"win": float(outcome["R"].mean()), "margin_mean": float(logged_q.mean()), "margin_sd": float(logged_q.std()), "margin_quantiles": np.quantile(logged_q, qs).tolist()}
    rows_out = []
    for T in config.temperature_grid:
        res = imagine_rows4(kernel, policy, rows, seed=int(np.random.SeedSequence([config.seed, 78]).generate_state(1)[0]), temperature=float(T))
        q = res["margin"]
        row = {"temperature": float(T), "win": float(res["won"].mean()), "margin_mean": float(q.mean()), "margin_sd": float(q.std()), "margin_quantiles": np.quantile(q, qs).tolist(), "stripe_states": int(res["stripe_states"])}
        row["score"] = abs(row["win"] - ref["win"]) + abs(row["margin_mean"] - ref["margin_mean"]) + float(np.abs(np.asarray(row["margin_quantiles"]) - np.asarray(ref["margin_quantiles"])).mean())
        rows_out.append(row)
        emit(f"temperature {T}: win {row['win']:.3f} (logged {ref['win']:.3f}), margin {row['margin_mean']:+.3f} (logged {ref['margin_mean']:+.3f}), score {row['score']:.4f}")
    best = min(rows_out, key=lambda r: r["score"])
    return {"fitted_temperature": best["temperature"], "grid": list(config.temperature_grid), "rows": rows_out, "logged": ref, "players": int(len(idx)), "replicates": B,
            "kernel": f"{config.temperature_kernel}-{config.temperature_kernel_seed}", "criterion": "|d win| + |d mean margin| + mean |d margin deciles| on training players at their logged e and quota (frozen protocol)"}


def train4(config: CalaTrainConfig4, out: str | Path, *, progress=None) -> dict[str, object]:
    emit = progress or (lambda _: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    splits = load_splits(config.dataset_root)
    churn, mastery, _, _ = accepted(config.regime)
    logged = Logged4(config.cache_dir, config.regime, splits=splits, mastery_config=mastery)
    arm = Arm(config.arm)
    if arm is Arm.ORACLE_K:
        raise GuardViolation("the CaLA-4 study trains deployable arms only")
    inputs = ArmInputs(arm, logged)
    arm_config = arm_config_of(config, inputs.feature_width)
    model = CalaArm4(arm_config, churn=churn, mastery=mastery).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    gen = torch.Generator(device=device).manual_seed(config.seed)
    train_players = logged.indices("train")
    val_idx, val_a = fixed_validation_rows(logged, config.validation_rows, config.seed)
    label = config.label or f"{config.arm}-{config.regime}-{config.seed}__{arm_config.variant}"
    history: list[dict[str, object]] = []
    best = {"update": -1, "total": math.inf}
    since_best = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with phase("train"):
        model.train()

        def validate(update: int) -> None:
            nonlocal since_best
            v = evaluate_rows4(model, logged, inputs, val_idx, val_a, mastery=mastery, device=device)
            rec = {"update": update, "seconds": time.perf_counter() - started, **{f"val_{k}": v_ for k, v_ in v.items()}}
            history.append(rec)
            emit(f"update {update}: val total {v['total']:.4f} action {v['action']:.4f} retention {v['retention']:.4f} penalty {v['penalty']:.4f}")
            if v["total"] < best["total"] - 1e-5:
                best.update({"update": update, "total": v["total"], "components": v})
                save_arm4(model, out / "best.pt", extra={"update": update, "validation": v, "train_config": config.to_dict(), "label": label})
                since_best = 0
            else:
                since_best += 1

        validate(0)
        for update in range(1, config.updates + 1):
            idx, a = sample_rows(logged, train_players, config.batch_rows, rng)
            batch = make_batch4(logged, inputs, idx, a, mastery=mastery, device=device)
            out_ = model.objective(batch, generator=gen)
            loss = out_["total"]
            if not torch.isfinite(loss):
                raise GuardViolation(f"non-finite loss at update {update}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if update % config.eval_every == 0 or update == config.updates:
                validate(update)
                if since_best >= config.patience:
                    emit(f"early stop at update {update} (best {best['update']})")
                    break
    wall = time.perf_counter() - started
    updates_done = history[-1]["update"]
    state = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    with phase("select"):
        val_players = logged.indices("validation")
        v_idx, v_a = np.repeat(val_players, LANDMARK), np.tile(np.arange(1, LANDMARK + 1), len(val_players))
        keep = v_a <= logged.n_attempts[v_idx]
        pm_val = policy_metrics4(model, logged, inputs, v_idx[keep][: config.metric_players * 20], v_a[keep][: config.metric_players * 20], device=device, seed=config.seed)
        te_players = logged.indices("test")
        t_idx, t_a = np.repeat(te_players, LANDMARK - 1), np.tile(np.arange(1, LANDMARK), len(te_players))
        keep = t_a <= np.minimum(logged.n_attempts[t_idx], LANDMARK - 1)
        pm_test = policy_metrics4(model, logged, inputs, t_idx[keep][: config.metric_players * 19], t_a[keep][: config.metric_players * 19], device=device, seed=config.seed)
        calibration = retention_calibration(model, logged, inputs, mastery=mastery, churn=churn, device=device, seed=config.seed)
        temperature = fit_temperature4(model, logged, inputs, config, device=device, progress=emit)
        prop, prop_summary = fit_propensity(model, logged, inputs, device=device)
    if pm_val["illegal_after_mask"] or pm_test["illegal_after_mask"]:
        raise GuardViolation("realised illegal action after masking")
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    record = {
        "status": config.status, "label": label, "arm": config.arm, "regime": config.regime, "seed": config.seed, "kernel_regime": "kernel frozen (WM-1 transformer-14.8M), never retrained", "config": config.to_dict(), "arm_config": arm_config.to_dict(),
        "variant": arm_config.variant, "context_width": model.width, "parameters": model.parameter_counts(),
        "data": {"train_players": int(len(train_players)), "validation_players": int(len(logged.indices("validation"))), "test_players": int(len(logged.indices("test"))), "test_players_active_at_landmark": int(len(logged.active_at(LANDMARK, "test"))),
                 "validation_rows": int(len(val_idx)), "cache_schema": logged.header.get("cache_schema")},
        "compute": {"updates": int(updates_done), "wall_seconds": wall, "updates_per_second": updates_done / max(wall, 1e-9), "rows_per_update": config.batch_rows, "peak_memory_allocated_bytes": peak, "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "selection": {"criterion": "validation total loss (frozen protocol)", "best_update": best["update"], "best_total": best["total"], "components": best.get("components"), "early_stopped": since_best >= config.patience},
        "history": history, "policy_metrics": {"validation_all_attempts": pm_val, "test_prefix_attempts_1_19": pm_test}, "retention_calibration": calibration, "temperature": temperature, "propensity": prop_summary,
        "oracle_fields_accessed": {"consumer": None, "n_accesses": 0, "fields": []}, "features": None if inputs.features is None else inputs.features.to_json(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "propensity.json").write_text(json.dumps(prop.to_json()) + "\n")
    (out / "training.json").write_text(json.dumps(record, indent=1, default=float, allow_nan=False) + "\n")
    emit(f"trained {label}: best update {best['update']} (val total {best['total']:.4f}); policy NLL {pm_val['action_nll']:.3f} top1 {pm_val['top1']:.3f} (stripe states NLL {pm_val['stripe_states']['action_nll']}); "
         f"retention Brier {calibration['models']['learned_head']['landmark']['brier']:.4f} pass={calibration['passed']}; T={temperature['fitted_temperature']}; params {record['parameters']['total']}; {wall:.0f}s")
    return record


__all__ = ["CalaTrainConfig4", "DEFAULT_CACHE", "STATUS4", "arm_config_of", "evaluate_rows4", "fit_temperature4", "frozen_train_config", "make_batch4", "policy_metrics4", "train4"]


def finalize_run4(run_dir: str | Path, *, progress=None) -> Path:
    """``cala.artifacts.finalize_run`` with the CaLA-4 status accepted (same schema, provenance and checkpoint-hash validation; atomic rename; never overwrites)."""
    from match3_simulator.experiments.artifacts import ArtifactError, assemble_run, validate_run

    emit = progress or (lambda _: None)
    run = Path(run_dir)
    if not run.name.endswith(".partial"):
        raise ArtifactError(f"{run} is not a .partial run directory")
    final = run.with_name(run.name[: -len(".partial")])
    if final.exists():
        raise ArtifactError(f"refusing to overwrite {final}")
    report = assemble_run(run)
    if report["status"] != STATUS4:
        raise ArtifactError(f"expected status {STATUS4}")
    report_check = {**report, "status": "cala_causal_run"}  # the frozen validator's fixed status set; every other check is applied unchanged
    validate_run(report_check, run)
    tmp = run / "run_report.json.partial"
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False, default=float) + "\n")
    tmp.replace(run / "run_report.json")
    os.rename(run, final)
    emit(f"finalized {final}")
    return final
