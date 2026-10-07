# CaLA: Causal Latent Adjustment for World Models under Adaptive Policies

## Structure

The repository has the following structure:

| Folder / file | Contents |
|---|---|
| `engine/` | Original match-3 simulator with the accepted calibration and benchmark |
| `src/world_modeling/` | World-modeling components (Transformer, LeJEPA, and LeWM) |
| `src/cala/` | Variational CaLA and other adjustment arms, policies, rollouts, and evaluation |
| `scripts/` | Scripts that reproduce the experiments and export tables and figures |
| `GENERATE_DATA.md` | Data-generation and preparation instructions |

## Setup

Use Python >=3.10 and a CUDA-enabled PyTorch installation for training.
From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run all commands below from the repository root. Start with
`GENERATE_DATA.md` to prepare the data and the two engine references.

## Run the experiments

The commands operate on one job at a time. `--list` prints every job and its
index; repeat the command with the corresponding `--job` indices.

```bash
# World models: (3 families x 3 sizes x 3 seeds).
python scripts/02_world_models.py --list
python scripts/02_world_models.py --action train --job 0 --device cuda
python scripts/02_world_models.py --action evaluate --job 0 --device cuda

# Special-aware adjustment/policy fits.
python scripts/03_train_adjustment.py --list
python scripts/03_train_adjustment.py --job 0 --device cuda

# Freeze checkpoints, validation evaluations, nuisances and configuration hashes.
python scripts/04_freeze.py --groups main

# Special-policy imagined cells and evaluation.
python scripts/05_rollouts.py --group main --list
python scripts/05_rollouts.py --group main --job 0 --device cuda
python scripts/06_evaluate.py --route WM --group main --job 0

```

## Generate paper outputs

To regenerate all tables used in the paper, run:

```bash
python scripts/07_tables_figures.py --group main
python scripts/08_descriptive_tables.py benchmark
python scripts/08_descriptive_tables.py world-models
python scripts/08_descriptive_tables.py data
```
