# Vehicle telemetry data processing

This document describes [data_process.py](../scripts/data_process/data_process.py) and [split_data.py](../scripts/data_process/split_data.py). The first script converts raw vehicle CSV files into cleaned, variable-length 10-second sequences and metadata. The second creates training and test partitions and smooths SOC. `data_process.py` uses only the Python standard library; `split_data.py` also requires NumPy and pandas.

## Quick start

Run both scripts from the repository root or pass absolute paths:

```bash
python scripts/data_process/data_process.py --input-dir data/raw_data --output-dir data/processed_data
python scripts/data_process/split_data.py
```

The input directory must contain the raw vehicle CSV files directly, with one file per vehicle. `data_process.py` requires a new output path. Both scripts write to temporary sibling directories and move finished output into place after processing. They do not alter their inputs. `split_data.py` also refuses an existing output path unless `--overwrite` is supplied.

## Required input schema

Each input CSV must contain the following columns. Additional columns are ignored.

| Column | Use |
| --- | --- |
| acqtime | Timestamp, parsed by Python's datetime.fromisoformat |
| soc | State of charge; copied to output |
| cell_battery_voltage | String containing 102 whitespace-separated cell voltages, optionally enclosed in square brackets |
| total_current | Current measurement used during preprocessing |
| vehicle_status | Categorical field; copied to output |
| charge_state | Categorical field; copied to output and used for condition labeling |
| speed | Speed; copied to output and used for condition labeling |

The source files may contain a UTF-8 byte-order mark. Every data row must have the same number of CSV fields as its header. File names are processed in lexicographic order; the file name is retained in the sequence manifest for provenance.

## Processing rules

### 1. Select vehicles by dominant sampling interval

For each raw file, the script computes the difference between adjacent timestamps in the file's existing row order. It converts each difference to an integer number of seconds, counts the frequencies, and selects the most frequent interval. Ties are resolved in favor of the smaller interval.

Only entire files whose dominant interval is 10 seconds continue through the pipeline. Files dominated by another interval are listed in the exclusion report. This step does not downsample 2-second files, and a selected 10-second file may still contain gaps or irregular timestamps.

### 2. Derive fields and remove invalid rows

The output schema is:

| Column | Transformation |
| --- | --- |
| acqtime | Copied |
| soc | Copied |
| voltage | Arithmetic mean of the 102 cell voltages, formatted to six decimal places |
| rate | Current-related feature used by the models |
| vehicle_status | Copied |
| charge_state | Copied |
| speed | Copied |

An entire row is rejected if any of these checks fails:

1. The voltage string must contain exactly 102 numeric, finite cell values.
2. Every cell voltage must lie within the inclusive range [2.5, 3.7] V.
3. The derived rate must be finite and lie within the inclusive range [-2, 2].
4. acqtime, soc, vehicle_status, charge_state, and speed must be nonempty.

Rejected rows are removed. The pipeline does not replace, clip, or interpolate values. The cleaning report assigns one reason to each rejected row using the order above. Invalid timestamp syntax and nonnumeric nonempty total_current values stop processing with an error. Numeric soc and speed values must also be finite in every retained sequence.

### 3. Build sequences

The script scans the cleaned rows for each selected vehicle in their original order. A row continues the current segment when the integer timestamp difference from the previous cleaned row is 10 seconds. Any other difference starts a new segment. Removing an invalid row can therefore create a gap and split a segment.

Only complete segments with more than 1,000 rows are retained. The output contains variable-length, nonoverlapping sequences; it does not cut long segments into fixed 1,000-row windows. Sequence files are numbered globally from sequence_000001.csv in source-file order.

### 4. Record normalization statistics

For soc, voltage, rate, and speed, the script records the minimum, maximum, mean, and population standard deviation over all retained sequence rows. The JSON file includes min-max and z-score formulas. The sequence CSV values themselves remain unnormalized. acqtime, vehicle_status, and charge_state are not normalization targets.

The provided statistics summarize the complete processed dataset. For model evaluation, fit transformation parameters on the training portion after defining the split, then apply those same parameters to validation and test data.

### 5. Derive sequence features and condition labels

Each sequence receives descriptive features including SOC change, rate statistics, whole-sequence mean speed, moving speed, and charge-state proportions. A row counts as moving when speed > 1 and as high-speed when speed >= 60. Whole-sequence mean speed includes stopped rows.

The four condition labels are generated from these features:

| Label | Meaning | Assignment |
| --- | --- | --- |
| C1 | Charging or low activity | All sequences with charge_state = 1 for at least 50% of rows, then the lowest-mean-speed noncharging sequences needed to fill the first balanced group |
| C2 | Low-speed operation | First third of the remaining noncharging sequences by whole-sequence mean speed |
| C3 | Medium-speed operation | Middle third of the remaining noncharging sequences |
| C4 | High-speed operation | Final third of the remaining noncharging sequences |

The target group sizes differ by at most one. Ties in speed use moving ratio and then sequence file name. If charging sequences alone exceed the first group's target size, C1 retains all of them and the remaining sequences are balanced across C2–C4. The boundaries are data dependent. These are rule-derived labels, not annotations supplied in the raw CSV. Labeling writes metadata files and does not change the sequence CSVs.

## Output layout

The output directory contains:

    processed_data/
      sequence_000001.csv
      sequence_000002.csv
      ...
      normalization_parameters.json
      operating_condition_parameters.json
      processing_summary.json
      _reports/
        time_interval_report.csv
        included_10s_main_interval_files.csv
        excluded_non_10s_main_interval_files.csv
        voltage_current_cleaning_report.csv
        voltage_current_cleaning_summary.txt
        sequence_file_summary.csv
        sequence_manifest.csv
        sequence_condition_features.csv
        operating_condition_manifest.csv
        sequence_length_distribution.json
        operating_conditions/
          C1_charging_or_low_activity.csv
          C2_low_speed_operation.csv
          C3_medium_speed_operation.csv
          C4_high_speed_operation.csv

The _reports/sequence_manifest.csv schema links each output file to its source vehicle, source segment number, record indexes, timestamps, length, and duration. In newly generated output, source_start_record and source_end_record count rows after cleaning; source_start_raw_record and source_end_raw_record count data rows in the raw CSV, starting at 1. The two raw-record columns provide a direct trace back to the input, since the standalone script does not persist an intermediate cleaned file.

The operating condition manifest joins each sequence name with its label and descriptive features. The normalization and condition-parameter JSON files preserve the parameters used to interpret the data. The length-distribution report includes estimates for possible 1,000-row windows; those windows are not created by this script. `split_data.py` uses this manifest for the next stage.

## Training/test split and SOC smoothing

`split_data.py` reads `data/processed_data/_reports/operating_condition_manifest.csv` and the corresponding sequence files. It randomly selects training trajectories within each of C1–C4. By default it selects 1,000 per condition, using seed `20260830`; all remaining trajectories form the test set. Pass `--train-per-condition N` to change the count or `--train-ratio R` to select a fraction instead. These two options cannot be combined. `--input-dir`, `--output-dir`, `--seed`, and `--workers` can also be set explicitly.

The script smooths each trajectory's quantized SOC by interpolating between plateau centers and applying bounded endpoint extrapolation. It writes separate sequence CSV files into `data/split_data/train/` and `data/split_data/test/`. SOC remains in 0–100 percent units and the CSV values are not normalized. The script computes `data/split_data/normalization_parameters.json` from training records only, then records the allocation in `split_config.json`.

```text
split_data/
  train/sequence_*.csv
  test/sequence_*.csv
  metadata/train_manifest.csv
  metadata/test_manifest.csv
  normalization_parameters.json
  split_config.json
  trajectory_labels.csv
```

The split is by trajectory, so a source vehicle can contribute different trajectories to both sets. Model scripts use the manifests, sequences, and training-only normalization file under `data/split_data/` by default.

## Counts for the supplied dataset

These figures describe the 30-file dataset used for this project. Other input directories will produce different counts.

| Stage | Files or sequences | Rows |
| --- | ---: | ---: |
| Raw input | 30 files | 50,922,766 |
| Dominant 10-second files selected | 26 files | 33,004,492 |
| After invalid-row removal | 26 files | 32,358,824 |
| Retained sequences | 4,742 sequences | 8,537,845 |

The four excluded 2-second files are vehicle20.csv, vehicle23.csv, vehicle35.csv, and vehicle38.csv, containing 17,918,274 rows in total. Cleaning removes 645,668 rows, or 1.9563% of selected rows. Segmentation produces 487,598 candidate segments and discards 482,856 short segments containing 23,820,979 rows. The final sequences contain 26.38% of cleaned rows and 16.77% of all raw rows. Retained lengths range from 1,001 to 9,740 rows, with a median of 1,535.

| Condition | Sequences | Rows |
| --- | ---: | ---: |
| C1: charging or low activity | 1,186 | 2,433,630 |
| C2: low-speed operation | 1,186 | 2,179,760 |
| C3: medium-speed operation | 1,185 | 2,118,705 |
| C4: high-speed operation | 1,185 | 1,805,750 |
