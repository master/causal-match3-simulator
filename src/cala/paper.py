"""Run the experiment pipeline and write its outputs under ``runs/``."""

from __future__ import annotations
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import torch

PACKAGE = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE / "configs/paper_protocol.json"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return read(PROTOCOL_PATH)


def write(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(p)


def arm_path(policy, regime, arm, seed):
    return Path("runs/arms") / policy / regime / f"{arm}-{seed}"


def train_config(policy, regime, arm, seed):
    name = (
        f"{regime}-{arm}-{seed}"
        if policy == "current"
        else f"{arm}-current-special-{regime}-{seed}"
    )
    return PACKAGE / "configs/training" / policy / (name + ".json")


def training_jobs(group="main"):
    if group != "main":
        raise ValueError("This package supports only the special-policy main comparison")
    p = protocol()
    return [dict(policy="special", regime=r, arm=a, seed=s)
            for r in p["regimes"] for s in p["seeds"] for a in p["arms"]]


def train_job(job, *, root="data/release", device="cuda"):
    if job["policy"] != "special":
        raise ValueError("Only special-policy training is supported")
    from match3_simulator.experiments.training_utils import CalaTrainConfig, train
    from match3_simulator.experiments.artifacts import finalize_run
    from match3_simulator.experiments.rollout_utils import kernel_path
    from match3_simulator.experiments.train import (
        CalaTrainConfig4,
        train4,
        finalize_run4,
    )

    p = protocol()
    src = train_config(**job)
    raw = read(src)
    if (
        raw["updates"] != p["training"]["max_updates"]
        or raw["kl_weight"] != p["training"]["kl_weight"]
    ):
        raise ValueError("Training config disagrees with the paper protocol")
    final = arm_path(**job)
    out = final.with_name(final.name + ".partial")
    if final.exists() or out.exists():
        raise FileExistsError(f"Refusing to overwrite {final} or its partial run")
    temp = kernel_path(raw["temperature_kernel"], raw["temperature_kernel_seed"])
    if not temp.is_file():
        raise FileNotFoundError(
            f"Temperature calibration requires {temp} before training"
        )
    raw.update(dataset_root=root, device=device, wandb=False)
    cls, runner, seal = (
        (CalaTrainConfig, train, finalize_run)
        if job["policy"] == "current"
        else (CalaTrainConfig4, train4, finalize_run4)
    )
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(unknown)
    raw["temperature_grid"] = tuple(raw["temperature_grid"])
    config = cls(**raw)
    out.mkdir(parents=True)
    write(out / "requested_config.json", config.to_dict())
    runner(config, out, progress=print)
    return seal(out, progress=print)


def rollout_jobs(group="main"):
    if group != "main":
        raise ValueError("Capacity comparison is excluded from this special-policy subset")
    p = protocol()
    jobs = []
    kernels = (
        [p["world_model"]["main"]]
        if group == "main"
        else [
            "transformer-5.2M",
            "transformer-14.8M",
            "lewm-5.2M-reencode",
            "lewm-14.8M-reencode",
        ]
    )
    arms = p["arms"] if group == "main" else ["naive", "var_z", "oracle_k"]
    for regime in p["regimes"]:
        for seed in p["seeds"]:
            for arm in arms:
                for kernel in kernels:
                    jobs.append(
                        dict(
                            group=group,
                            regime=regime,
                            seed=seed,
                            arm=arm,
                            kernel=kernel,
                            kernel_seed=4201,
                            policy=(
                                p["main_wm_policy"][arm]
                                if group == "main"
                                else "current"
                            ),
                        )
                    )
    return jobs


def job_key(job):
    return "__".join(
        str(job[k])
        for k in ["group", "regime", "arm", "seed", "kernel", "kernel_seed", "policy"]
    )


def fitted_temperature(run):
    extended = Path(run) / "temperature-extended.json"
    return (
        float(read(extended)["fitted_temperature"])
        if extended.exists()
        else float(
            read(Path(run) / "training.json")["temperature"]["fitted_temperature"]
        )
    )


def freeze(path, *, root="data/release", groups=("main",)):
    """Freeze actual checkpoint/config/nuisance identities before test prediction.

    It records validation decisions; it does not declare that every historical
    convergence/retention criterion passed. Raw validation evidence stays local.
    """
    from match3_simulator.world_modeling.wm1_verify import verify
    from match3_simulator.experiments.rollout_utils import kernel_path

    out = Path(path)
    if out.exists():
        raise FileExistsError(out)
    verify(root)
    assets = {}
    jobs = []
    for group in groups:
        for job in rollout_jobs(group):
            run = arm_path(job["policy"], job["regime"], job["arm"], job["seed"])
            tr = read(run / "training.json")
            temp = fitted_temperature(run)
            if tr["config"]["updates"] != 12000 or tr["config"]["kl_weight"] != 0.01:
                raise ValueError(f"Wrong protocol: {run}")
            files = [
                run / name
                for name in (
                    "best.pt",
                    "training.json",
                    "propensity.json",
                    "run_report.json",
                )
            ]
            checkpoint = kernel_path(job["kernel"], job["kernel_seed"])
            files.append(checkpoint)
            evaluation = checkpoint.parent / "evaluation.json"
            ev = read(evaluation)["checkpoints"]["best"]["contracts"]
            if (
                ev["invalid_states"] != 0
                or not ev["mutation"]["learned"]["passed"]
                or not ev["mutation"]["support"]["passed"]
            ):
                raise ValueError(
                    f"World-model state/mutation contracts failed: {checkpoint}"
                )
            files.append(evaluation)
            if (run / "temperature-extended.json").exists():
                files.append(run / "temperature-extended.json")
            for f in files:
                assets[str(f)] = sha(f)
            jobs.append(
                {
                    **job,
                    "temperature": temp,
                    "validation_retention_pass": tr["retention_calibration"]["passed"],
                }
            )
    payload = {
        "schema": 1,
        "protocol_sha256": sha(PROTOCOL_PATH),
        "dataset_root": str(root),
        "splits_sha256": sha(Path(root) / "splits.json"),
        "assets": assets,
        "jobs": jobs,
        "purpose": "configuration/checkpoint freeze; no test-outcome access",
        "historical_action_gate": "not evaluated in this release",
    }
    write(out, payload)
    return payload


def verify_freeze(path):
    f = read(path)
    if f["protocol_sha256"] != sha(PROTOCOL_PATH):
        raise ValueError("Protocol changed after freeze")
    if f["splits_sha256"] != sha(Path(f["dataset_root"]) / "splits.json"):
        raise ValueError("Split changed after freeze")
    for name, digest in f["assets"].items():
        if sha(name) != digest:
            raise ValueError(f"Frozen asset changed: {name}")
    return f


def rollout(job, *, freeze_path, device="cuda", chunk=4096):
    from match3_simulator.experiments.cell_utils import CellSpec, run_cell
    from match3_simulator.experiments.cells import CellSpec4, run_cell4

    frozen = verify_freeze(freeze_path)
    matches = [j for j in frozen["jobs"] if job_key(j) == job_key(job)]
    if len(matches) != 1:
        raise ValueError("Job is not in the frozen comparison")
    job = matches[0]
    run = arm_path(job["policy"], job["regime"], job["arm"], job["seed"])
    spec = dict(
        arm_run=str(run),
        dynamics=job["kernel"],
        kernel_seed=job["kernel_seed"],
        regime=job["regime"],
        population="test",
        S=16 if job["arm"] == "var_z" else 1,
        B=8 if job["arm"] == "var_z" else 128,
        seed=job["seed"],
        temperature=job["temperature"],
        include_logged=True,
        chunk=chunk,
    )
    out = (
        Path("runs/cells")
        / job["group"]
        / job["regime"]
        / f"{job['policy']}-{job['arm']}-{job['seed']}"
    )
    if job["policy"] == "current":
        path = run_cell(
            CellSpec(**spec),
            out_dir=out,
            device=torch.device(device),
            cache_dir="runs/data/schema1",
            dataset_root=frozen["dataset_root"],
            progress=print,
        )
    else:
        path = run_cell4(
            CellSpec4(**spec, posterior="sample"),
            out_dir=out,
            device=torch.device(device),
            cache_dir="runs/data/schema2",
            dataset_root=frozen["dataset_root"],
            block=32,
            progress=print,
        )
    with __import__("numpy").load(path / "arrays.npz") as arr:
        if arr["churn"].shape[1] * arr["churn"].shape[2] != 128:
            raise ValueError("Effective rollout budget is not 128")
    write(
        path / "paper_job.json",
        {
            "job": job,
            "freeze_sha256": sha(freeze_path),
            "kernel_sha256": frozen["assets"][
                str(
                    __import__(
                        "match3_simulator.experiments.rollout_utils",
                        fromlist=["kernel_path"],
                    ).kernel_path(job["kernel"], job["kernel_seed"])
                )
            ],
        },
    )
    return path
