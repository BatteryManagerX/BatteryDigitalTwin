# Battery Digital Twin

This repository prepares vehicle battery telemetry and evaluates three model components: a reduced-order model (ROM), a stateless state-of-charge (SOC) predictor, and a Condition Restorer (CR) for missing current measurements. Run the commands below from the repository root, in the order shown.

## Installation

Use Python 3.10 or newer and install the packages with:

```bash
python -m pip install -r requirements.txt
```

Every command-line script supports `--help` for its full option list.

## 1. Prepare and split the data

If `data/split_data/` has already been prepared with its manifests and normalization file, start at the ROM section.

Download the raw CSV files from [Hugging Face (placeholder link)](https://huggingface.co/datasets/your-username/battery-digital-twin-raw-csv) and put them in `data/raw_data/`, one file per vehicle. Replace this URL with the published dataset link before release. `data/`, `checkpoints/`, and `results/` are ignored by Git.

The first script filters and cleans the raw telemetry, constructs contiguous 10-second sequences, and assigns operating conditions C1–C4:

```bash
python scripts/data_process/data_process.py --input-dir data/raw_data --output-dir data/processed_data
```

Both arguments are required. `data/processed_data/` must not already exist. The output contains `sequence_*.csv`, `normalization_parameters.json`, `processing_summary.json`, and provenance and condition manifests under `_reports/`.

The second script splits those sequences by operating condition and smooths quantized SOC values:

```bash
python scripts/data_process/split_data.py
```

It reads `data/processed_data/` and writes `data/split_data/` by default. The default split selects 1,000 training trajectories per condition with seed `20260830`; the remaining trajectories become the test set. To use a ratio instead, run, for example, `python scripts/data_process/split_data.py --train-ratio 0.8`. `--train-ratio` and `--train-per-condition` are mutually exclusive. Other options are `--input-dir`, `--output-dir`, `--seed`, and `--workers`. The command refuses an existing output directory unless `--overwrite` is supplied; use that flag only when replacing the existing split is intended.

The resulting files are:

```text
data/split_data/
  train/sequence_*.csv
  test/sequence_*.csv
  metadata/train_manifest.csv
  metadata/test_manifest.csv
  normalization_parameters.json
  split_config.json
  trajectory_labels.csv
```

The normalization statistics are fitted on training records only. All following scripts use `data/split_data` as their default data source.

For the input CSV schema and processing rules, see [Data processing details](docs/DATA_PROCESSING.md).

## 2. Train and test the ROMs

The PNGV and single-particle model (SPM) have separate fitting and inference commands:

```bash
python scripts/rom/fit_pngv.py
python scripts/rom/fit_spm.py

python scripts/rom/infer_pngv.py
python scripts/rom/infer_spm.py -T 100
```

The fitting scripts read `data/split_data/train/` and `metadata/train_manifest.csv`. Each samples up to 10 sequences per condition and 500 contiguous points per sequence by default. Both accept `--data-dir`, `--manifest`, `--sequences-per-condition`, `--max-points-per-sequence`, `--max-nfev`, `--seed`, and `--output`. If optimization does not converge, no checkpoint is saved unless `--save-on-failure` is explicitly set.

| Script | Default checkpoint | Default test result |
| --- | --- | --- |
| `fit_pngv.py` / `infer_pngv.py` | `checkpoints/rom/pngv.npz` | `results/rom/pngv_infer_test.db` |
| `fit_spm.py` / `infer_spm.py` | `checkpoints/rom/spm.npz` | `results/rom/spm_infer_t100_test.db` |

The inference scripts use `data/split_data/test/` and `metadata/test_manifest.csv` by default. They accept `--test-dir`, `--manifest`, `--checkpoint`, `--output`, and `--limit`. PNGV runs autoregressively from the initial frame. SPM uses ground-truth SOC to reset its state every `-T` steps; `100` is the default. Keep `-T` and the output database name aligned when comparing settings.

The shared `calibration_common.py` and `inference_common.py` files are helper modules, not command-line entry points.

## 3. Train and test the Stateless SOC Predictor

Train on the split training set:

```bash
python scripts/soc_predictor/train_stateless_soc_pred.py
```

The script reads the training manifest and training-only normalization statistics. By default, it holds out 20% of source vehicles within the training split for validation (`--validation-fraction 0.2`, `--split-group-column source_file`). It samples variable-length windows from each trajectory. Useful overrides include `--epochs`, `--batch-size`, `--min-window`, `--max-window`, `--validation-interval`, `--num-workers`, `--device`, and `--no-amp`. Use `--split-group-column none` to split validation trajectories by condition instead. Resume with `--resume checkpoints/stateless_soc_predictor/last.pt`.

Training writes `best.pt`, `last.pt`, `history.csv`, the training and validation manifests, and `training_config.json` under `checkpoints/stateless_soc_predictor/` by default. To use a different split or output location, set `--data-dir`, `--manifest`, `--normalization`, and `--output-dir`.

Run blockwise test inference with the best checkpoint:

```bash
python scripts/soc_predictor/infer_stateless_soc_pred.py
```

This reads the split test set and writes `results/stateless_soc_predictor/stateless_soc_pred_infer_test.db`. Options include `--checkpoint`, `--test-dir`, `--manifest`, `--output`, `--min-window`, `--max-window`, `--overlap`, `--device`, `--no-amp`, and `--limit`. Window lengths default to the checkpoint's training settings; overlap defaults to the minimum window length. Predictions start at zero-based step `min_window - 1`, so earlier steps have no prediction rows.

## 4. Train and test the Condition Restorer

CR reconstructs short missing `rate` segments from observed context. Its training loss passes the reconstructed rate through a frozen differentiable SPM, so fit `checkpoints/rom/spm.npz` before training CR:

```bash
python scripts/condition_restorer/train_cr.py
```

Defaults are 12 preceding frames (`--pre-len`), 6 following frames (`--sub-len`), and missing lengths from 1 to 6 (`--max-missing-len`). The script reads `data/split_data/train/`, its manifest, and the training-only normalization file. It writes `cr_best.pt`, `cr_last.pt`, and `cr_history.csv` under `checkpoints/condition_restorer/`. Options include `--spm-checkpoint`, `--data-dir`, `--manifest`, `--normalization`, `--output-dir`, `--epochs`, `--batch-size`, `--samples-per-trajectory`, `--validation-fraction`, `--voltage-loss-weight`, `--device`, and `--no-amp`. Resume with `--resume checkpoints/condition_restorer/cr_last.pt`. Set `--voltage-loss-weight 0` to disable the auxiliary voltage loss.

### Three-module test and random-current ROM baseline

Use `infer_cr_rom_soc_pred.py` to test CR, a ROM, and the Stateless SOC Predictor together under simulated packet loss:

```bash
python scripts/condition_restorer/infer_cr_rom_soc_pred.py --rom spm -T 100 --loss-count 5 --max-loss-length 6
```

`--rom pngv` selects the PNGV ROM instead. The script requires `--rom`, `--correction-interval` (or `-T`), `--loss-count`, and `--max-loss-length`. It reads the test split, `cr_best.pt`, `stateless_soc_predictor/best.pt`, and the selected ROM checkpoint by default. `--max-loss-length` must be at most 6 and no greater than the CR checkpoint's trained maximum. The selected SPM checkpoint must match the one used to train CR.

This command writes **two** databases for the same sampled loss intervals:

| Result | Default path with `--rom spm -T 100` | Behavior |
| --- | --- | --- |
| Three-module result | `results/condition_restorer/cr_spm_soc_pred_t100_infer_test.db` | CR restores missing rate, the ROM advances state, and the SOC predictor makes periodic corrections. |
| Random-current ROM baseline | `results/condition_restorer/rom_spm_random_current_t100_infer_test.db` | The ROM replaces missing rate with random values sampled from that trajectory's observed rate range; CR and the SOC predictor are not used. |

For PNGV, replace `spm` with `pngv` in the paths. Use `--output` and `--rom-output` to change the two destinations. Other options are `--rom-checkpoint`, `--cr-checkpoint`, `--soc-checkpoint`, `--test-dir`, `--manifest`, `--min-window`, `--max-window`, `--seed`, `--device`, `--no-amp`, and `--limit`. Loss intervals are sampled within the first 2,048 frames, with deterministic per-trajectory seeds.

### CR + SPM test without the SOC predictor

Use the separate CR + SPM mode to evaluate rate restoration and ROM prediction without periodic SOC predictor corrections:

```bash
python scripts/condition_restorer/infer_cr_rom.py
```

It simulates five nonoverlapping loss intervals of up to six frames per trajectory by default, uses CR to restore their rates, and writes `results/condition_restorer/cr_spm_infer_test.db`. Options include `--cr-checkpoint`, `--rom-checkpoint`, `--test-dir`, `--manifest`, `--output`, `--loss-count`, `--max-loss-length`, `--seed`, `--device`, `--no-amp`, and `--limit`. This command supports SPM only and does not produce a random-current baseline; that baseline is written by the three-module command above.

## Reading inference results

Inference outputs are SQLite databases with `run_metadata`, `trajectory_status`, and `predictions` tables. `predictions` contains `sequence_file`, zero-based `step`, `acqtime`, `rate`, `soc_gt`, `soc_pred`, `voltage_gt`, and `voltage_pred`. The `rate` column contains the input actually used for that result, including restored or random values in the packet-loss tests. The stateless SOC predictor leaves `voltage_pred` empty. Existing databases can be resumed for unfinished trajectories when their saved run metadata matches the current configuration. Use a new output path for a different test configuration.
