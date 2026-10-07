"""Train/evaluate one of 27 paper world models."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=["train", "evaluate"], default="train")
    p.add_argument("--job", type=int)
    p.add_argument("--list", action="store_true")
    p.add_argument("--root", default="data/release")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    from match3_simulator.experiments.paper import protocol, PACKAGE, write, read, sha

    spec = protocol()
    jobs = [
        dict(cell=f"{family}-{band}", seed=seed)
        for family in spec["world_model"]["families"]
        for band in spec["world_model"]["bands"]
        for seed in spec["seeds"]
    ]
    if a.list:
        print(json.dumps([dict(job=i, **j) for i, j in enumerate(jobs)], indent=2))
        return
    if a.job is None or not 0 <= a.job < len(jobs):
        p.error("choose a valid --job from --list")
    job = jobs[a.job]
    source = Path("runs/wm1") / f"{job['cell']}-{job['seed']}"
    from match3_simulator.world_modeling.wm1_train import WM1TrainConfig, train
    from match3_simulator.world_modeling.wm1_eval import evaluate_run

    if a.action == "train":
        if source.exists():
            raise FileExistsError(source)
        cfg = WM1TrainConfig(
            **job,
            dataset_root=a.root,
            device=a.device,
            configs_path=str(PACKAGE / "configs/wm1_configs.json"),
            total_updates=spec["world_model"]["training_updates"],
        )
        train(cfg, output_dir=source, progress=print)
        return
    run = source
    if job["cell"].startswith("lewm-"):
        # This is an inference setting on the same learned weights, not a new fit.
        import torch

        run = Path("runs/wm1/variants") / (source.name + "-reencode")
        if not run.exists():
            run.mkdir(parents=True)
            for name in ("best.pt", "ckpt-equal-flops.pt"):
                src = source / name
                if src.exists():
                    obj = torch.load(src, map_location="cpu", weights_only=False)
                    obj["model_config"]["rollout"] = "reencode"
                    torch.save(obj, run / name)
            training = read(source / "training.json")
            training["model_config"]["rollout"] = "reencode"
            training["label"] += "-reencode"
            write(run / "training.json", training)
            write(
                run / "variant.json",
                {
                    "source_sha256": sha(source / "best.pt"),
                    "change": "model_config.rollout = reencode; weights unchanged",
                },
            )
    result = evaluate_run(run, device=a.device, root=a.root, progress=print)
    write(
        run / "paper_validation.json",
        {
            "checkpoint_sha256": sha(run / "best.pt"),
            "contracts": result["checkpoints"]["best"]["contracts"],
            "historical_action_sensitivity_gate": "not evaluated in this release",
            "scope": "prediction and imagination metrics; not a claim of passing every historical gate",
        },
    )


if __name__ == "__main__":
    main()
