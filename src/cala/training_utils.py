"""Utilities for training a CaLA arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from match3_simulator.release import ACCEPTED_SPEC_PATH, load_accepted_spec, resolved_configs
from match3_simulator.retention import ChurnSchedule, MasteryConfig, update_mastery
from match3_simulator.scm import LEVELS
from match3_simulator.world_modeling.decoder import torch_legal_mask
from match3_simulator.world_modeling.wm1_data import Splits, load_splits
from match3_simulator.experiments.arm_base import CalaArm, CalaArmConfig, save_arm
from match3_simulator.experiments.contexts import Arm
from match3_simulator.experiments.data_base import LANDMARK, HandcraftedFeatures, Logged, raw_handcrafted
from match3_simulator.experiments.guard import EvaluationOnly, GuardViolation, oracle_skills, phase
from match3_simulator.experiments.propensity import Propensity
from match3_simulator.experiments.rollout_utils import accepted_churn_fn, imagine_rows, load_kernel, make_rows

os.environ.setdefault("WANDB_MODE", "offline")
LEVEL_NAMES = tuple(level.name for level in LEVELS)


@dataclass(frozen=True)
class CalaTrainConfig:
    arm: str = "var_z"
    regime: str = "natural"
    seed: int = 4201
    updates: int = 6000
    batch_rows: int = 48
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 200
    validation_rows: int = 768
    patience: int = 6
    kl_weight: float = 1.0
    l2_weight: float = 1e-3
    aux_weight: float = 0.1
    retention_weight: float = 1.0
    latent_dimensions: int = 4
    encoder_hidden: int = 64
    policy_d_model: int = 128
    policy_layers: int = 4
    policy_heads: int = 4
    temperature_grid: tuple[float, ...] = (0.5, 0.7, 0.85, 1.0, 1.25)
    temperature_players: int = 200
    temperature_replicates: int = 2
    temperature_kernel: str = "transformer-5.2M"
    temperature_kernel_seed: int = 4201
    metric_players: int = 360
    dataset_root: str = "data/release"
    cache_dir: str = "runs/cala/data"
    device: str = "cuda"
    label: str | None = None
    wandb: bool = False
    status: str = "cala_causal_run"

    def __post_init__(self) -> None:
        Arm(self.arm)
        if self.regime not in ("natural", "randomized"):
            raise ValueError("regime must be natural or randomized")
        if self.updates < 1 or self.batch_rows < 1 or self.eval_every < 1 or self.patience < 1:
            raise ValueError("updates, batch_rows, eval_every and patience must be positive")
        if any(t <= 0 for t in self.temperature_grid) or 1.0 not in self.temperature_grid:
            raise ValueError("temperature_grid must be positive and contain 1.0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def accepted(regime: str) -> tuple[ChurnSchedule, MasteryConfig, tuple[float, ...], tuple[float, ...]]:
    spec = load_accepted_spec(ACCEPTED_SPEC_PATH)
    _, churn, mastery, gains, sigmas = resolved_configs(spec, regime)
    return churn, mastery, gains, sigmas


class ArmInputs:
    """Arm-specific side inputs: standardised handcrafted features (fitted on training rows) or the oracle skill table (token-gated)."""

    def __init__(self, arm: Arm, logged: Logged, *, features: HandcraftedFeatures | None = None, oracle_consumer: str | None = None):
        self.arm = arm
        self.logged = logged
        self.features = features
        self.oracle: np.ndarray | None = None
        self.oracle_accesses: list[dict[str, object]] = []
        if arm is Arm.HANDCRAFTED and features is None:
            tr = logged.indices("train")
            rows_idx, rows_a = np.repeat(tr, LANDMARK), np.tile(np.arange(1, LANDMARK + 1), len(tr))
            self.features = HandcraftedFeatures.fit(raw_handcrafted(logged, rows_idx, rows_a))
        if arm is Arm.ORACLE_K:
            if oracle_consumer is None:
                raise GuardViolation("the oracle-K arm needs a named EvaluationOnly consumer")
            with EvaluationOnly(oracle_consumer) as token:
                self.oracle = oracle_skills(logged.player_ids).astype(np.float32)
                self.oracle_accesses = list(token.accesses)

    @property
    def feature_width(self) -> int:
        return 0 if self.features is None else self.features.width

    def extra(self, idx: np.ndarray, a: np.ndarray) -> np.ndarray | None:
        if self.arm is Arm.HANDCRAFTED:
            return self.features.transform(raw_handcrafted(self.logged, idx, a))
        if self.arm is Arm.ORACLE_K:
            return self.oracle[idx]
        return None

    def context_inputs(self, idx: np.ndarray, a: np.ndarray | int, device: torch.device) -> dict[str, torch.Tensor]:
        """prefix / base / extra tensors for the context module."""
        idx = np.asarray(idx)
        a = np.broadcast_to(np.asarray(a), idx.shape)
        out = {"base": torch.as_tensor(self.logged.base_context(idx, a), device=device)}
        if self.arm.uses_encoder:
            out["prefix"] = self.logged.prefix_batch(idx, a, device)
        extra = self.extra(idx, a)
        if extra is not None:
            out["extra"] = torch.as_tensor(extra, device=device)
        return out


def make_batch(logged: Logged, inputs: ArmInputs, idx: np.ndarray, a: np.ndarray, *, mastery: MasteryConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Full training batch for (player, target attempt) rows (train / validation players only)."""
    b = inputs.context_inputs(idx, a, device)
    r, o, t = logged.target_rows(idx, a), logged.outcome_rows(idx, a), logged.transition_rows(idx, a)
    tb = torch.as_tensor(t["boards"], device=device)
    f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    i64 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)
    b.update({"level": i64(r["level"]), "tier": i64(r["tier"]), "attempt": i64(r["attempt"]), "E": f32(r["E"]), "R": f32(o["R"]), "Q": f32(o["Q"]), "C": f32(o["C"]),
              "mastery_after": f32(update_mastery(r["mastery_before"], o["R"].astype(np.int64), mastery)), "evidence": f32(o["evidence"]),
              "summaries": f32(np.stack([o["mean_legal"] / 10.0, o["mean_goal_clear"]], axis=1)),
              "t_boards": tb, "t_actions": i64(t["actions"]), "t_moves": i64(t["moves_left"]), "t_goals": i64(t["goals_left"]), "t_colour": i64(t["goal_colour"]), "t_legal": torch_legal_mask(tb), "t_row": i64(t["row"])})
    return b


def sample_rows(logged: Logged, players: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Uniform players, uniform logged attempt per player (rows with at least one transition)."""
    idx = rng.choice(players, size=n, replace=True)
    a = (rng.random(n) * logged.n_attempts[idx]).astype(np.int64) + 1
    return idx, a


def fixed_validation_rows(logged: Logged, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return sample_rows(logged, logged.indices("validation"), n, np.random.default_rng(np.random.SeedSequence([seed, 991])))


@torch.no_grad()
def evaluate_rows(model: CalaArm, logged: Logged, inputs: ArmInputs, idx: np.ndarray, a: np.ndarray, *, mastery: MasteryConfig, device: torch.device, batch: int = 96) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n_rows = n_tr = 0
    for start in range(0, len(idx), batch):
        b = make_batch(logged, inputs, idx[start : start + batch], a[start : start + batch], mastery=mastery, device=device)
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


# --- policy metrics --------------------------------------------------------------------------------------------------------

@torch.no_grad()
def policy_metrics(model: CalaArm, logged: Logged, inputs: ArmInputs, idx: np.ndarray, a: np.ndarray, *, device: torch.device, batch: int = 64, seed: int = 0) -> dict[str, object]:
    """Action NLL, top-1 / top-5, ECE (10 bins on the argmax confidence), pre-mask illegal probability mass, realised illegal rate after masking, entropy of sampled vs logged actions."""
    from match3_simulator.learned_model.tokens import ACTION_SLOTS, IN_BOUNDS

    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    nll, top1, top5, conf, correct, illegal_mass, illegal_realised, n = [], [], [], [], [], [], 0, 0
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
        legal = torch_legal_mask(tb)
        args = dict(board=tb, goal_colour=torch.as_tensor(t["goal_colour"], device=device), moves_left=torch.as_tensor(t["moves_left"], device=device), goals_left=torch.as_tensor(t["goals_left"], device=device),
                    skill=ctx[torch.as_tensor(t["row"], device=device), 0])
        logp = model.policy(**args, legal_actions=legal)
        actions = torch.as_tensor(t["actions"], device=device)
        nll.append(-logp.gather(1, actions[:, None]).squeeze(1))
        rank = (logp > logp.gather(1, actions[:, None])).sum(dim=1)
        top1.append(rank == 0)
        top5.append(rank < 5)
        conf.append(logp.max(dim=1).values.exp())
        correct.append(logp.argmax(dim=1) == actions)
        unmasked = model.policy(**args, legal_actions=in_bounds.unsqueeze(0).expand(len(tb), -1))
        illegal_mass.append((unmasked.exp() * (~legal).float()).sum(dim=1))
        draw = torch.multinomial(logp.exp(), 1, generator=gen).squeeze(1)
        illegal_realised += int((~legal.gather(1, draw[:, None]).squeeze(1)).sum())
        sampled_hist += np.bincount(draw.cpu().numpy(), minlength=ACTION_SLOTS)
        logged_hist += np.bincount(actions.cpu().numpy(), minlength=ACTION_SLOTS)
        n += len(tb)
    cat = lambda xs: torch.cat(xs).float().cpu().numpy()
    nll_v, conf_v, corr_v = cat(nll), cat(conf), cat(correct)
    bins = np.clip((conf_v * 10).astype(int), 0, 9)
    ece = float(sum(abs(corr_v[bins == k].mean() - conf_v[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any()))
    ent = lambda h: float(-(h[h > 0] / h.sum() * np.log(h[h > 0] / h.sum())).sum()) if h.sum() else 0.0
    model.train()
    return {"n_transitions": int(n), "action_nll": float(nll_v.mean()), "top1": float(cat(top1).mean()), "top5": float(cat(top5).mean()), "ece": ece, "illegal_mass_before_mask": float(cat(illegal_mass).mean()),
            "illegal_rate_after_mask": illegal_realised / max(n, 1), "illegal_after_mask": illegal_realised, "entropy_sampled_actions": ent(sampled_hist), "entropy_logged_actions": ent(logged_hist),
            "distinct_sampled_actions": int((sampled_hist > 0).sum())}


# --- retention calibration ----------------------------------------------------------------------------------------------------

def _calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, object]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)) if len(p) >= bins else np.linspace(0, 1, bins + 1)
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    rel = []
    ece = 0.0
    for k in range(bins):
        sel = which == k
        if sel.any():
            rel.append({"bin": k, "n": int(sel.sum()), "mean_p": float(p[sel].mean()), "mean_y": float(y[sel].mean())})
            ece += abs(p[sel].mean() - y[sel].mean()) * sel.mean()
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece), "reliability": rel, "mean_p": float(p.mean()), "mean_y": float(y.mean()), "n": int(len(p))}


@torch.no_grad()
def retention_calibration(model: CalaArm, logged: Logged, inputs: ArmInputs, *, mastery: MasteryConfig, churn: ChurnSchedule, device: torch.device, seed: int = 0, batch: int = 128) -> dict[str, object]:
    """Learned head vs per-level constant (train mean churn) vs accepted hazard on validation rows: all logged attempts and the landmark rows; by level, tier, completion, margin bin."""
    model.eval()
    val = logged.indices("validation")
    rows_idx = np.repeat(val, LANDMARK)
    rows_a = np.tile(np.arange(1, LANDMARK + 1), len(val))
    keep = rows_a <= logged.n_attempts[rows_idx]
    rows_idx, rows_a = rows_idx[keep], rows_a[keep]
    gen = torch.Generator(device=device).manual_seed(seed)
    p_head = np.zeros(len(rows_idx))
    for start in range(0, len(rows_idx), batch):
        i, a = rows_idx[start : start + batch], rows_a[start : start + batch]
        b = make_batch(logged, inputs, i, a, mastery=mastery, device=device)
        ctx, _, _ = model.contexts(b, S=1, generator=gen)
        p_head[start : start + batch] = model.retention.probabilities(completion=b["R"], margin=b["Q"], mastery_after=b["mastery_after"], level=b["level"], context=ctx[:, 0], attempt=b["attempt"]).cpu().numpy()
    r, o = logged.target_rows(rows_idx, rows_a), logged.outcome_rows(rows_idx, rows_a)
    y = o["C"].astype(np.float64)
    m_after = update_mastery(r["mastery_before"], o["R"].astype(np.int64), mastery)
    p_hazard = accepted_churn_fn(churn)(o["R"], np.clip(o["Q"], -1, 1), m_after, r["level"], r["attempt"])
    tr = logged.indices("train")
    tr_idx, tr_a = np.repeat(tr, LANDMARK), np.tile(np.arange(1, LANDMARK + 1), len(tr))
    keep_tr = tr_a <= logged.n_attempts[tr_idx]
    tr_r, tr_o = logged.target_rows(tr_idx[keep_tr], tr_a[keep_tr]), logged.outcome_rows(tr_idx[keep_tr], tr_a[keep_tr])
    const = np.zeros((len(LEVELS), 2))
    for li in range(len(LEVELS)):
        for pre in (0, 1):
            sel = (tr_r["level"] == li) & ((tr_r["attempt"] < LANDMARK) == bool(pre))
            const[li, pre] = tr_o["C"][sel].mean() if sel.any() else 0.0
    p_const = const[r["level"], (r["attempt"] < LANDMARK).astype(int)]
    out: dict[str, object] = {"n_rows": int(len(y)), "models": {}}
    for name, p in (("learned_head", p_head), ("per_level_constant", p_const), ("accepted_hazard", p_hazard)):
        block = {"all_attempts": _calibration(p, y)}
        land = r["attempt"] == LANDMARK
        block["landmark"] = _calibration(p[land], y[land])
        block["by_level_landmark"] = {LEVEL_NAMES[li]: _calibration(p[land & (r["level"] == li)], y[land & (r["level"] == li)], bins=5) for li in range(len(LEVELS)) if (land & (r["level"] == li)).any()}
        block["by_tier_landmark"] = {int(t): _calibration(p[land & (r["tier"] == t)], y[land & (r["tier"] == t)], bins=5) for t in range(3) if (land & (r["tier"] == t)).any()}
        block["by_completion_landmark"] = {int(v): _calibration(p[land & (o["R"] == v)], y[land & (o["R"] == v)], bins=5) for v in (0, 1) if (land & (o["R"] == v)).any()}
        q = np.clip(o["Q"], -1, 1)
        mbins = np.digitize(q, [-0.5, -0.2, 0.0, 0.2, 0.5])
        block["by_margin_bin_landmark"] = {str(k): {"margin_range": f"bin {k}", **_calibration(p[land & (mbins == k)], y[land & (mbins == k)], bins=4)} for k in range(6) if (land & (mbins == k)).any()}
        block["by_difficulty_landmark"] = {str(k): _calibration(p[land & (np.digitize(r["E"], [-1, 0, 1]) == k)], y[land & (np.digitize(r["E"], [-1, 0, 1]) == k)], bins=4) for k in range(4) if (land & (np.digitize(r["E"], [-1, 0, 1]) == k)).any()}
        out["models"][name] = block
    head, cst = out["models"]["learned_head"]["landmark"], out["models"]["per_level_constant"]["landmark"]
    out["passed"] = bool(head["brier"] <= cst["brier"] * 1.02 + 1e-4 and head["ece"] <= cst["ece"] + 0.02)
    out["criterion"] = "landmark Brier <= 1.02 x per-level constant + 1e-4 and landmark ECE <= constant ECE + 0.02 (validation players)"
    model.train()
    return out


# --- temperature ------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def fit_temperature(model: CalaArm, logged: Logged, inputs: ArmInputs, config: CalaTrainConfig, *, device: torch.device, progress=None) -> dict[str, object]:
    """One sampling temperature per arm on training players active at the landmark: roll the policy through the temperature kernel at each player's logged e
    (logged quota) and match win rate and margin distribution (mean, sd, nine quantiles) to the logged outcomes."""
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
    policy = lambda b, c, m, g, l, i: model.policy(board=b, goal_colour=c, moves_left=m, goals_left=g, skill=ctx_rows[i], legal_actions=l)
    kernel = load_kernel(config.temperature_kernel, config.temperature_kernel_seed, device=device)
    logged_q = np.clip(outcome["Q"], -1, 1)
    qs = np.linspace(0.1, 0.9, 9)
    ref = {"win": float(outcome["R"].mean()), "margin_mean": float(logged_q.mean()), "margin_sd": float(logged_q.std()), "margin_quantiles": np.quantile(logged_q, qs).tolist()}
    rows_out = []
    for T in config.temperature_grid:
        res = imagine_rows(kernel, policy, rows, seed=int(np.random.SeedSequence([config.seed, 78]).generate_state(1)[0]), temperature=float(T))
        q = res["margin"]
        row = {"temperature": float(T), "win": float(res["won"].mean()), "margin_mean": float(q.mean()), "margin_sd": float(q.std()), "margin_quantiles": np.quantile(q, qs).tolist()}
        row["score"] = abs(row["win"] - ref["win"]) + abs(row["margin_mean"] - ref["margin_mean"]) + float(np.abs(np.asarray(row["margin_quantiles"]) - np.asarray(ref["margin_quantiles"])).mean())
        rows_out.append(row)
        emit(f"temperature {T}: win {row['win']:.3f} (logged {ref['win']:.3f}), margin {row['margin_mean']:+.3f} (logged {ref['margin_mean']:+.3f}), score {row['score']:.4f}")
    best = min(rows_out, key=lambda r: r["score"])
    return {"fitted_temperature": best["temperature"], "grid": list(config.temperature_grid), "rows": rows_out, "logged": ref, "players": int(len(idx)), "replicates": B,
            "kernel": f"{config.temperature_kernel}-{config.temperature_kernel_seed}", "criterion": "|d win| + |d mean margin| + mean |d margin deciles| on training players at their logged e and quota"}


# --- propensity --------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def arm_W(model: CalaArm, logged: Logged, inputs: ArmInputs, idx: np.ndarray, *, device: torch.device, batch: int = 256) -> np.ndarray:
    """W_i = (B_i, context_i) at the landmark with the posterior *mean* for the variational arm. Outputs: (N, width)."""
    out = []
    for start in range(0, len(idx), batch):
        ci = inputs.context_inputs(idx[start : start + batch], LANDMARK, device)
        if model.arm is Arm.VAR_Z:
            mean, _ = model.context.encoder(**ci["prefix"])
            out.append(torch.cat((ci["base"], mean), dim=-1).cpu().numpy())
        else:
            ctx, _, _ = model.context(prefix=ci.get("prefix"), base=ci["base"], extra=ci.get("extra"), S=1)
            out.append(ctx[:, 0].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def fit_propensity(model: CalaArm, logged: Logged, inputs: ArmInputs, *, device: torch.device) -> tuple[Propensity, dict[str, object]]:
    idx = np.concatenate([logged.active_at(LANDMARK, "train"), logged.active_at(LANDMARK, "validation")])
    W = arm_W(model, logged, inputs, idx, device=device)
    target = logged.target_rows(idx, LANDMARK)
    prop = Propensity.fit(W, target["E"], target["level"])
    g = prop.density(target["E"], W, target["level"], floor=False)
    return prop, {"n_rows": int(len(idx)), "sigma": prop.sigma.tolist(), "mean_log_density": float(np.log(np.maximum(g, 1e-12)).mean()), "floored_fraction": float((g < prop.g_min).mean()),
                  "coefficient_norm_per_level": np.linalg.norm(prop.coef[:, 1:], axis=1).tolist(), "note": "randomized regime: coefficients are a diagnostic (assignment ignores W by design)"}


# --- main loop ----------------------------------------------------------------------------------------------------------------

def train(config: CalaTrainConfig, out: str | Path, *, progress=None) -> dict[str, object]:
    emit = progress or (lambda _: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    splits = load_splits(config.dataset_root)
    churn, mastery, gains, sigmas = accepted(config.regime)
    logged = Logged(config.cache_dir, config.regime, splits=splits, mastery_config=mastery)
    arm = Arm(config.arm)
    inputs = ArmInputs(arm, logged, oracle_consumer="oracle_k" if arm is Arm.ORACLE_K else None)
    arm_config = CalaArmConfig(arm=config.arm, feature_width=inputs.feature_width, latent_dimensions=config.latent_dimensions, encoder_hidden=config.encoder_hidden, policy_d_model=config.policy_d_model,
                               policy_layers=config.policy_layers, policy_heads=config.policy_heads, kl_weight=config.kl_weight, l2_weight=config.l2_weight, retention_weight=config.retention_weight, aux_weight=config.aux_weight)
    model = CalaArm(arm_config, churn=churn, mastery=mastery).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    gen = torch.Generator(device=device).manual_seed(config.seed)
    train_players = logged.indices("train")
    val_idx, val_a = fixed_validation_rows(logged, config.validation_rows, config.seed)
    label = config.label or f"{config.arm}-{config.regime}-{config.seed}"
    run = None
    if config.wandb:
        import wandb

        run = wandb.init(project="match3-cala", mode="offline", dir=str(out), name=label, group=f"cala/{config.regime}", tags=["cala", config.arm, config.regime, f"seed{config.seed}"], config=config.to_dict())
    history: list[dict[str, object]] = []
    best = {"update": -1, "total": math.inf}
    since_best = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    token = EvaluationOnly("oracle_k") if arm is Arm.ORACLE_K else None
    if token is not None:
        token.__enter__()
    try:
        with phase("train"):
            model.train()
            def validate(update: int) -> None:
                nonlocal since_best
                v = evaluate_rows(model, logged, inputs, val_idx, val_a, mastery=mastery, device=device)
                rec = {"update": update, "seconds": time.perf_counter() - started, **{f"val_{k}": v_ for k, v_ in v.items()}}
                history.append(rec)
                if run is not None:
                    run.log(rec, step=update)
                emit(f"update {update}: val total {v['total']:.4f} action {v['action']:.4f} retention {v['retention']:.4f} penalty {v['penalty']:.4f}")
                if v["total"] < best["total"] - 1e-5:
                    best.update({"update": update, "total": v["total"], "components": v})
                    save_arm(model, out / "best.pt", extra={"update": update, "validation": v, "train_config": config.to_dict(), "label": label})
                    since_best = 0
                else:
                    since_best += 1
            validate(0)
            for update in range(1, config.updates + 1):
                idx, a = sample_rows(logged, train_players, config.batch_rows, rng)
                batch = make_batch(logged, inputs, idx, a, mastery=mastery, device=device)
                out_ = model.objective(batch, generator=gen)
                loss = out_["total"]
                if not torch.isfinite(loss):
                    raise GuardViolation(f"non-finite loss at update {update}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                if run is not None and update % 20 == 0:
                    run.log({f"train_{k}": float(v_) for k, v_ in out_.items()}, step=update)
                if update % config.eval_every == 0 or update == config.updates:
                    validate(update)
                    if since_best >= config.patience:
                        emit(f"early stop at update {update} (best {best['update']})")
                        break
        wall = time.perf_counter() - started
        updates_done = history[-1]["update"]
        # reload the best checkpoint for the downstream fits
        state = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        model.eval()
        with phase("select"):
            val_players = logged.indices("validation")
            v_idx, v_a = np.repeat(val_players, LANDMARK), np.tile(np.arange(1, LANDMARK + 1), len(val_players))
            keep = v_a <= logged.n_attempts[v_idx]
            pm_val = policy_metrics(model, logged, inputs, v_idx[keep][: config.metric_players * 20], v_a[keep][: config.metric_players * 20], device=device, seed=config.seed)
            te_players = logged.indices("test")
            t_idx, t_a = np.repeat(te_players, LANDMARK - 1), np.tile(np.arange(1, LANDMARK), len(te_players))
            keep = t_a <= np.minimum(logged.n_attempts[t_idx], LANDMARK - 1)
            pm_test = policy_metrics(model, logged, inputs, t_idx[keep][: config.metric_players * 19], t_a[keep][: config.metric_players * 19], device=device, seed=config.seed)
            calibration = retention_calibration(model, logged, inputs, mastery=mastery, churn=churn, device=device, seed=config.seed)
            temperature = fit_temperature(model, logged, inputs, config, device=device, progress=emit)
            prop, prop_summary = fit_propensity(model, logged, inputs, device=device)
    finally:
        if token is not None:
            token.__exit__(None, None, None)
        if run is not None:
            run.finish()
    if pm_val["illegal_after_mask"] or pm_test["illegal_after_mask"]:
        raise GuardViolation("realised illegal action after masking")
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    record = {
        "status": config.status, "label": label, "arm": config.arm, "regime": config.regime, "seed": config.seed, "kernel_regime": "kernel frozen (WM-1), never retrained", "config": config.to_dict(), "arm_config": arm_config.to_dict(),
        "context_width": model.width, "parameters": model.parameter_counts(), "data": {"train_players": int(len(train_players)), "validation_players": int(len(logged.indices('validation'))), "test_players": int(len(logged.indices('test'))),
                                                                                       "test_players_active_at_landmark": int(len(logged.active_at(LANDMARK, 'test'))), "validation_rows": int(len(val_idx))},
        "compute": {"updates": int(updates_done), "wall_seconds": wall, "updates_per_second": updates_done / max(wall, 1e-9), "rows_per_update": config.batch_rows, "peak_memory_allocated_bytes": peak, "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "selection": {"criterion": "validation total loss", "best_update": best["update"], "best_total": best["total"], "components": best.get("components"), "early_stopped": since_best >= config.patience},
        "history": history, "policy_metrics": {"validation_all_attempts": pm_val, "test_prefix_attempts_1_19": pm_test}, "retention_calibration": calibration, "temperature": temperature, "propensity": prop_summary,
        "oracle_fields_accessed": {"consumer": "oracle_k" if inputs.oracle is not None else None, "n_accesses": len(inputs.oracle_accesses), "fields": sorted({x["field"] for x in inputs.oracle_accesses})},
        "features": None if inputs.features is None else inputs.features.to_json(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "propensity.json").write_text(json.dumps(prop.to_json()) + "\n")
    (out / "training.json").write_text(json.dumps(record, indent=1, default=float, allow_nan=False) + "\n")
    emit(f"trained {label}: best update {best['update']} (val total {best['total']:.4f}); policy NLL {pm_val['action_nll']:.3f} top1 {pm_val['top1']:.3f}; retention Brier {calibration['models']['learned_head']['landmark']['brier']:.4f} "
         f"(const {calibration['models']['per_level_constant']['landmark']['brier']:.4f}, hazard {calibration['models']['accepted_hazard']['landmark']['brier']:.4f}) pass={calibration['passed']}; T={temperature['fitted_temperature']}; {wall:.0f}s")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=[a.value for a in Arm])
    parser.add_argument("--regime", default="natural")
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overrides", default=None, help="JSON dict of CalaTrainConfig overrides")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    overrides = json.loads(args.overrides) if args.overrides else {}
    config = CalaTrainConfig(arm=args.arm, regime=args.regime, seed=args.seed, wandb=args.wandb, **overrides)
    log_path = Path(args.out) / "train.log"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    try:
        train(config, args.out, progress=log)
    except GuardViolation as exc:
        (Path(args.out) / "STOP.md").write_text(f"# STOP\n\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{exc}\n")
        log(f"STOP: {exc}")
        raise


if __name__ == "__main__":
    main()


__all__ = ["ArmInputs", "CalaTrainConfig", "accepted", "arm_W", "fit_propensity", "fit_temperature", "make_batch", "policy_metrics", "retention_calibration", "train"]
