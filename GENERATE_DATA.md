# Generate and prepare data

Run commands from the repository root after `python -m pip install -e .`.

## Generate a release with the original engine

```bash
python -m match3_simulator.release --out data/generated --workers 10
```

The command uses the original accepted benchmark and calibration. It writes
player-level splits, natural and randomized regimes, attempt histories,
transition records (including boards and special tiles), privileged reference
fields, a manifest, and quality checks. Do not use `--no-transitions`:
world-model training needs the transition records. Inspect generation checks
before using the release.

Then place the accepted payload in
`data/release/`, including `release-manifest.json`, `splits.json`, `qc.json`,
`accepted_benchmark.json` and all natural/randomized shard files.

```bash
python scripts/00_prepare_data.py --root data/release --workers 10
```

This checks hashes and structural consistency, then builds the natural WM
cache and both CaLA history caches. Test outcomes remain locked for
prediction.

## Generate the Table 1 benchmark reference

To reproduce the benchmark reference used in Table 1, run:

```bash
python scripts/01_engine_reference.py benchmark --stage equivalence --workers 10
python scripts/01_engine_reference.py benchmark --stage calibrate --workers 10
python scripts/01_engine_reference.py benchmark --stage validate --workers 10
python scripts/01_engine_reference.py benchmark --stage audit --workers 10
```

The calibration phase uses seed 8077. Validation uses seeds 4409, 5519,
6637, 7753, and 8861.

## Generate the decision-table reference

To generate the decision-table reference used as the paper's gold reference,
run:

```bash
for regime in natural randomized; do
  python scripts/01_engine_reference.py logged --regime "$regime" --split validation --workers 10
  python scripts/01_engine_reference.py logged --regime "$regime" --split test --workers 10
done
```

These gold arrays use eligible players from the requested split, along with
their logged level and attempt-20 opening, including special tiles.
