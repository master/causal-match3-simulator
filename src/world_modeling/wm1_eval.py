"""
Evaluation of a trained WM model:
    python -m match3_simulator.world_modeling.wm1_eval --run runs/wm1/transformer-5.2M-4201.partial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from match3_simulator.world_modeling.wm1_contracts import run_contracts
from match3_simulator.world_modeling.wm1_data import EpisodeStore, load_splits
from match3_simulator.world_modeling.wm1_rollout import DEFAULT_KS, IMAGINATION_ANCHORS, IMAGINATION_ROLLOUTS, evaluate_engine_conditioned, evaluate_replay, imagine, measure_latency, replay_rollouts
from match3_simulator.world_modeling.wm1_train import load_checkpoint

EVAL_EPISODES = 2000
EVAL_BATCH_EPISODES = 64
EVAL_SEED = 1301  # constant: the evaluation subset never depends on the run seed


def evaluation_batches(store: EpisodeStore, *, episodes: int = EVAL_EPISODES, batch_episodes: int = EVAL_BATCH_EPISODES, device: torch.device, seed: int = EVAL_SEED) -> list[dict[str, torch.Tensor]]:
    """Fixed seeded subset of validation episodes, padded in batches (all keys, including the strata keys)."""
    keep = np.sort(np.random.default_rng(seed).choice(len(store), size=min(episodes, len(store)), replace=False))
    return [store.batch(keep[s : s + batch_episodes], device=device) for s in range(0, len(keep), batch_episodes)]


def _raw_replay(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], *, ks: tuple[int, ...], seed: int, max_batches: int = 4) -> dict[str, np.ndarray]:
    """Compressed raw k-step trajectories (sampled decoding) of the first batches for the artifact."""
    out: dict[str, list[np.ndarray]] = {}
    for index, batch in enumerate(batches[:max_batches]):
        roll = replay_rollouts(model, batch, ks=ks, stochastic=True, seed=seed + index)
        for k in ks:
            d = roll.concatenated(k)
            for name in ("predicted", "target", "anchor", "predicted_specials", "target_specials", "predicted_goals", "target_goals", "level"):
                if name in d:
                    out.setdefault(f"k{k}_{name}", []).append(d[name].astype(np.int16 if name != "level" else np.int8))
    return {name: np.concatenate(parts) for name, parts in out.items()}


def evaluate_checkpoint(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], *, ks: tuple[int, ...] = DEFAULT_KS, seed: int, anchors: int = IMAGINATION_ANCHORS,
                        rollouts: int = IMAGINATION_ROLLOUTS, progress=None, with_raw: bool = True) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run every §6 mode, the §7 contracts and the latency measurement on one checkpoint.

    Inputs: model (eval mode); evaluation batches; horizons; seed; anchors and rollouts of modes B / C; progress; keep raw trajectories.
    Outputs: (evaluation dict, raw arrays for the trajectories npz).
    """
    emit = progress or (lambda _: None)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    timings = {}
    t0 = time.perf_counter()
    replay = evaluate_replay(model, batches, ks=ks, seed=seed)
    timings["replay"] = time.perf_counter() - t0
    emit(f"mode A done ({timings['replay']:.0f}s): skill k1/k3/k7 = " + "/".join(f"{replay['decodings']['argmax']['horizons'][f'k{k}'].get('skill_score', float('nan')):.3f}" for k in ks))
    t0 = time.perf_counter()
    engine_conditioned = evaluate_engine_conditioned(model, batches, ks=ks, max_anchors=anchors, rollouts_per_anchor=rollouts, seed=seed + 1)
    timings["engine_conditioned"] = time.perf_counter() - t0
    emit(f"mode B done ({timings['engine_conditioned']:.0f}s): board ES k7 {engine_conditioned['horizons'].get('k7', {}).get('board_energy_score', float('nan')):.4f}")
    t0 = time.perf_counter()
    imagination = imagine(model, batches, ks=ks, max_anchors=anchors, rollouts_per_anchor=rollouts, seed=seed + 2)
    timings["imagination"] = time.perf_counter() - t0
    emit(f"mode C done ({timings['imagination']:.0f}s): board ES k7 {imagination.report['horizons'].get('k7', {}).get('board_energy_score', float('nan')):.4f}, "
         f"goals ES k7 {imagination.report['horizons'].get('k7', {}).get('goals_left_energy_score', float('nan')):.4f}")
    t0 = time.perf_counter()
    contracts = run_contracts(model, imagination.transitions, batches, seed=seed + 3)
    timings["contracts"] = time.perf_counter() - t0
    emit(f"contracts done ({timings['contracts']:.0f}s): invalid-state rate {contracts['invalid_state_rate']:.5f}, mutation learned pass {contracts['mutation']['learned']['pass_rate']:.3f}, all passed {contracts['all_passed']}")
    latency = measure_latency(model, batches[0])
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    evaluation = {"ks": list(ks), "seed": seed, "n_batches": len(batches), "n_episodes": int(sum(int(b["step_mask"].shape[0]) for b in batches)), "n_transitions": int(sum(int(b["step_mask"].sum()) for b in batches)),
                  "replay": replay, "engine_conditioned": engine_conditioned, "imagination": imagination.report, "contracts": contracts, "latency": latency,
                  "eval_peak_memory_allocated_bytes": peak, "timings_seconds": timings, "seconds": time.perf_counter() - started}
    raw: dict[str, np.ndarray] = {}
    if with_raw:
        raw = {f"imagination_{k}": v for k, v in imagination.trajectories.items()}
        raw.update({f"replay_{k}": v for k, v in _raw_replay(model, batches, ks=ks, seed=seed + 4).items()})
    return evaluation, raw


def evaluate_run(run_dir: str | Path, *, device: str, episodes: int = EVAL_EPISODES, anchors: int = IMAGINATION_ANCHORS, rollouts: int = IMAGINATION_ROLLOUTS, root: str = "data/release",
                 cache: str = "runs/wm1/data/natural", progress=None, with_equal_flops: bool = True) -> dict[str, object]:
    """Evaluate best.pt (and ckpt-equal-flops.pt when present) of a run directory; writes evaluation.json and trajectories.npz."""
    emit = progress or (lambda _: None)
    run = Path(run_dir)
    if (run / "evaluation.json").exists():
        emit(f"{run / 'evaluation.json'} exists; skipping")
        return json.loads((run / "evaluation.json").read_text())
    training = json.loads((run / "training.json").read_text())
    dev = torch.device(device)
    splits = load_splits(root)
    store = EpisodeStore(cache, "validation", splits=splits, target=training["config"]["target"])
    batches = evaluation_batches(store, episodes=episodes, device=dev)
    emit(f"{training['label']}: evaluating on {sum(int(b['step_mask'].shape[0]) for b in batches)} validation episodes / {sum(int(b['step_mask'].sum()) for b in batches)} transitions ({store.n_players} players)")
    seed = 90000 + int(training["seed"])  # evaluation randomness is tied to the run seed but never to the training data order
    model = load_checkpoint(run / "best.pt", dev)
    evaluation, raw = evaluate_checkpoint(model, batches, seed=seed, anchors=anchors, rollouts=rollouts, progress=emit)
    result = {"label": training["label"], "cell": training["cell"], "seed": training["seed"], "checkpoints": {"best": evaluation}, "protocol": {"episodes": episodes, "batch_episodes": EVAL_BATCH_EPISODES, "subset_seed": EVAL_SEED,
              "anchors": anchors, "rollouts_per_anchor": rollouts, "ks": list(DEFAULT_KS), "device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None}}
    if with_equal_flops and (run / "ckpt-equal-flops.pt").exists():
        model = load_checkpoint(run / "ckpt-equal-flops.pt", dev)
        equal, _ = evaluate_checkpoint(model, batches, seed=seed, anchors=anchors, rollouts=rollouts, progress=lambda m: emit("[equal-flops] " + m), with_raw=False)
        result["checkpoints"]["equal_flops"] = equal
    partial = run / "trajectories.npz.partial.npz"
    np.savez_compressed(partial, **raw)
    partial.replace(run / "trajectories.npz")
    (run / "evaluation.json.partial").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=float) + "\n")
    (run / "evaluation.json.partial").replace(run / "evaluation.json")
    emit(f"{training['label']}: wrote {run / 'evaluation.json'} ({evaluation['seconds']:.0f}s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--anchors", type=int, default=IMAGINATION_ANCHORS)
    parser.add_argument("--rollouts", type=int, default=IMAGINATION_ROLLOUTS)
    parser.add_argument("--root", default="data/release")
    parser.add_argument("--cache", default="runs/wm1/data/natural")
    parser.add_argument("--no-equal-flops", action="store_true")
    args = parser.parse_args()
    log_path = Path(args.run) / "eval.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    evaluate_run(args.run, device=args.device, episodes=args.episodes, anchors=args.anchors, rollouts=args.rollouts, root=args.root, cache=args.cache, progress=log, with_equal_flops=not args.no_equal_flops)


if __name__ == "__main__":
    main()


__all__ = ["EVAL_BATCH_EPISODES", "EVAL_EPISODES", "EVAL_SEED", "evaluate_checkpoint", "evaluate_run", "evaluation_batches"]
