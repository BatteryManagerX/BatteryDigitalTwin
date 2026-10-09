import argparse
import csv
import json
import math
import os
import random
import shutil
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "processed_data"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "split_data"
DEFAULT_TRAIN_PER_CONDITION = 1000
DEFAULT_RANDOM_SEED = 20260830
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
NUMERIC_FIELDS = ("soc", "voltage", "rate", "speed")
CONDITION_CODES = ("C1", "C2", "C3", "C4")


def read_source_manifest(input_dir):
    manifest_path = input_dir / "_reports" / "operating_condition_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Condition manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Condition manifest has no header: {manifest_path}")
        required = {"sequence_file", "condition_code", "condition_name", "condition_name_zh", "records"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"Condition manifest is missing fields: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Condition manifest has no data rows: {manifest_path}")
    names = []
    groups = {code: [] for code in CONDITION_CODES}
    for row in rows:
        name = row["sequence_file"]
        if not name or "/" in name or "\\" in name or Path(name).name != name or not name.endswith(".csv"):
            raise ValueError(f"Invalid sequence file name in manifest: {name!r}")
        if row["condition_code"] not in groups:
            raise ValueError(f"Unexpected condition code for {name}: {row['condition_code']!r}")
        try:
            records = int(row["records"])
        except ValueError as error:
            raise ValueError(f"Invalid record count for {name}: {row['records']!r}") from error
        if records <= 0:
            raise ValueError(f"Nonpositive record count for {name}: {records}")
        names.append(name)
        groups[row["condition_code"]].append(row)
    if len(names) != len(set(names)):
        raise ValueError("Condition manifest contains duplicate sequence files")
    missing_groups = [code for code in CONDITION_CODES if not groups[code]]
    if missing_groups:
        raise ValueError(f"No sequences found for conditions: {', '.join(missing_groups)}")
    actual_names = {path.name for path in input_dir.glob("sequence_*.csv")}
    if actual_names != set(names):
        missing_files = sorted(set(names) - actual_names)
        extra_files = sorted(actual_names - set(names))
        raise ValueError(
            f"Sequence files and manifest differ; missing={missing_files[:5]}, "
            f"extra={extra_files[:5]}"
        )
    return fieldnames, rows, groups


def choose_train_names(groups, train_ratio, train_per_condition, seed):
    generator = random.Random(seed)
    selected = set()
    planned_counts = {}
    for code in CONDITION_CODES:
        candidates = sorted(row["sequence_file"] for row in groups[code])
        if train_per_condition is None:
            train_count = math.floor(len(candidates) * train_ratio + 0.5)
            train_count = max(1, min(len(candidates) - 1, train_count))
        else:
            train_count = train_per_condition
        if not 1 <= train_count < len(candidates):
            raise ValueError(
                f"{code} has {len(candidates)} sequences and cannot allocate "
                f"{train_count} to train while retaining a test set"
            )
        selected.update(generator.sample(candidates, train_count))
        planned_counts[code] = train_count
    return selected, planned_counts


def empty_stats():
    return {
        field: {
            "count": 0,
            "minimum": math.inf,
            "maximum": -math.inf,
            "sum": 0.0,
            "sum_squares": 0.0,
        }
        for field in NUMERIC_FIELDS
    }


def smooth_quantized_soc(soc, elapsed_seconds):
    soc = np.asarray(soc, dtype=np.float64)
    elapsed_seconds = np.asarray(elapsed_seconds, dtype=np.float64)
    if soc.ndim != 1 or elapsed_seconds.shape != soc.shape or soc.size == 0:
        raise ValueError("SOC and time must be nonempty one-dimensional arrays of equal length")
    if not np.isfinite(soc).all() or not np.isfinite(elapsed_seconds).all():
        raise ValueError("SOC or time contains non-finite values")
    if np.any(np.diff(elapsed_seconds) <= 0):
        raise ValueError("acqtime must be strictly increasing without duplicates")

    change_starts = np.r_[0, np.flatnonzero(np.diff(soc) != 0) + 1]
    if change_starts.size == 1:
        return soc.copy()
    change_ends = np.r_[change_starts[1:] - 1, soc.size - 1]
    centers = (elapsed_seconds[change_starts] + elapsed_seconds[change_ends]) / 2.0
    levels = soc[change_starts]
    smoothed = np.interp(elapsed_seconds, centers, levels)
    quantization_step = float(np.min(np.abs(np.diff(np.unique(soc)))))

    start_mask = elapsed_seconds < centers[0]
    start_slope = (levels[1] - levels[0]) / (centers[1] - centers[0])
    start_values = levels[0] + start_slope * (elapsed_seconds[start_mask] - centers[0])
    smoothed[start_mask] = np.clip(
        start_values,
        max(0.0, levels[0] - quantization_step / 2.0),
        min(100.0, levels[0] + quantization_step / 2.0),
    )

    end_mask = elapsed_seconds > centers[-1]
    end_slope = (levels[-1] - levels[-2]) / (centers[-1] - centers[-2])
    end_values = levels[-1] + end_slope * (elapsed_seconds[end_mask] - centers[-1])
    smoothed[end_mask] = np.clip(
        end_values,
        max(0.0, levels[-1] - quantization_step / 2.0),
        min(100.0, levels[-1] + quantization_step / 2.0),
    )
    return smoothed


def smooth_and_analyze(task):
    source, destination, is_train, expected_records = task
    frame = pd.read_csv(source)
    required = {"acqtime", *NUMERIC_FIELDS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing fields in {source}: {', '.join(sorted(missing))}")
    records = len(frame)
    if records != expected_records:
        raise ValueError(
            f"Record count mismatch in {source}: manifest={expected_records}, actual={records}"
        )

    timestamps = pd.to_datetime(frame["acqtime"], errors="raise")
    elapsed_seconds = (
        (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
    )
    try:
        smoothed_soc = smooth_quantized_soc(
            frame["soc"].to_numpy(dtype=np.float64), elapsed_seconds
        )
    except ValueError as error:
        raise ValueError(f"Invalid trajectory in {source}: {error}") from error
    if smoothed_soc.min() < 0.0 or smoothed_soc.max() > 100.0:
        raise ValueError(f"Smoothed SOC is outside [0, 100] in {source}")
    frame["soc"] = smoothed_soc
    frame.to_csv(destination, index=False)
    if not is_train:
        return source.name, destination.stat().st_size, records, None

    stats = empty_stats()
    for field in NUMERIC_FIELDS:
        values = frame[field].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {field} in {source}")
        field_stats = stats[field]
        field_stats["count"] = records
        field_stats["minimum"] = float(values.min())
        field_stats["maximum"] = float(values.max())
        field_stats["sum"] = float(values.sum(dtype=np.float64))
        field_stats["sum_squares"] = float(np.square(values).sum(dtype=np.float64))
    return source.name, destination.stat().st_size, records, stats


def merge_stats(target, source):
    for field in NUMERIC_FIELDS:
        current = target[field]
        incoming = source[field]
        current["count"] += incoming["count"]
        current["minimum"] = min(current["minimum"], incoming["minimum"])
        current["maximum"] = max(current["maximum"], incoming["maximum"])
        current["sum"] += incoming["sum"]
        current["sum_squares"] += incoming["sum_squares"]


def normalization_document(stats, train_records):
    fields = {}
    for field, values in stats.items():
        count = values["count"]
        minimum = values["minimum"]
        maximum = values["maximum"]
        mean = values["sum"] / count
        variance = max(0.0, values["sum_squares"] / count - mean * mean)
        fields[field] = {
            "count": count,
            "min": minimum,
            "max": maximum,
            "range": maximum - minimum,
            "mean": mean,
            "std_population": math.sqrt(variance),
        }
    return {
        "fit_scope": "training_set_only",
        "train_record_count": train_records,
        "default_method": "min_max",
        "min_max": {
            "normalize_formula": "x_normalized = (x - min) / (max - min)",
            "inverse_formula": "x = x_normalized * (max - min) + min",
        },
        "z_score": {
            "normalize_formula": "x_standardized = (x - mean) / std_population",
            "inverse_formula": "x = x_standardized * std_population + mean",
        },
        "fields": fields,
        "field_notes": {
            "rate": "rate = total_current / 125",
            "vehicle_status": "categorical; not normalized",
            "charge_state": "categorical; not normalized",
            "acqtime": "timestamp/index; not normalized",
        },
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, document):
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(document, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def run(input_dir, output_dir, train_ratio, train_per_condition, seed, workers, overwrite):
    fieldnames, rows, groups = read_source_manifest(input_dir)
    train_names, planned_counts = choose_train_names(
        groups, train_ratio, train_per_condition, seed
    )
    labeled_rows = [
        {
            "split": "train" if row["sequence_file"] in train_names else "test",
            **row,
        }
        for row in sorted(rows, key=lambda item: item["sequence_file"])
    ]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        train_dir = stage / "train"
        test_dir = stage / "test"
        metadata_dir = stage / "metadata"
        train_dir.mkdir()
        test_dir.mkdir()
        metadata_dir.mkdir()
        output_fields = ["split", *fieldnames]
        write_csv(stage / "trajectory_labels.csv", labeled_rows, output_fields)
        write_csv(
            metadata_dir / "train_manifest.csv",
            [row for row in labeled_rows if row["split"] == "train"],
            output_fields,
        )
        write_csv(
            metadata_dir / "test_manifest.csv",
            [row for row in labeled_rows if row["split"] == "test"],
            output_fields,
        )
        tasks = []
        for row in labeled_rows:
            source = input_dir / row["sequence_file"]
            is_train = row["split"] == "train"
            destination = (train_dir if is_train else test_dir) / source.name
            tasks.append((source, destination, is_train, int(row["records"])))
        train_stats = empty_stats()
        train_records = 0
        test_records = 0
        written_bytes = 0
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            for completed, result in enumerate(
                executor.map(smooth_and_analyze, tasks), start=1
            ):
                _, size, records, file_stats = result
                written_bytes += size
                if file_stats is None:
                    test_records += records
                else:
                    train_records += records
                    merge_stats(train_stats, file_stats)
                if completed % 500 == 0 or completed == len(tasks):
                    print(f"Smoothed {completed:,}/{len(tasks):,} sequences", flush=True)
        write_json(
            stage / "normalization_parameters.json",
            normalization_document(train_stats, train_records),
        )
        conditions = {}
        for code in CONDITION_CODES:
            condition_rows = groups[code]
            train_count = planned_counts[code]
            conditions[code] = {
                "name": condition_rows[0]["condition_name"],
                "name_zh": condition_rows[0]["condition_name_zh"],
                "total": len(condition_rows),
                "train": train_count,
                "test": len(condition_rows) - train_count,
            }
        config = {
            "method": "stratified_random_split_by_trajectory",
            "soc_smoothing": {
                "method": "linear_interpolation_between_plateau_centers",
                "endpoints": "linear_extrapolation_clipped_to_half_quantization_step",
            },
            "source_directory": str(input_dir),
            "output_directory": str(output_dir),
            "random_seed": seed,
            "split_mode": "fixed_count_per_condition" if train_per_condition is not None else "ratio_per_condition",
            "requested_train_ratio": train_ratio if train_per_condition is None else None,
            "requested_train_per_condition": train_per_condition,
            "train_per_condition": train_per_condition,
            "train_sequences": len(train_names),
            "test_sequences": len(rows) - len(train_names),
            "train_records": train_records,
            "test_records": test_records,
            "conditions": conditions,
            "normalization_fit_scope": "training_set_only",
            "leakage_note": (
                "No sequence appears in both splits. Different sequences from the same "
                "source vehicle may appear in both splits."
            ),
        }
        write_json(stage / "split_config.json", config)
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Output directory appeared during processing: {output_dir}")
            backup = output_dir.with_name(f".{output_dir.name}.backup-{uuid.uuid4().hex}")
            os.replace(output_dir, backup)
            try:
                os.replace(stage, output_dir)
            except BaseException:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(stage, output_dir)
        print(
            f"Done: train={len(train_names):,} sequences / {train_records:,} records; "
            f"test={len(rows) - len(train_names):,} sequences / {test_records:,} records; "
            f"written={written_bytes / (1024 ** 2):.2f} MiB; output={output_dir}",
            flush=True,
        )
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Split processed sequences and smooth SOC in each trajectory."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output directory"
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--train-ratio", type=float)
    split_group.add_argument("--train-per-condition", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    train_ratio = args.train_ratio
    train_per_condition = args.train_per_condition
    if train_ratio is None and train_per_condition is None:
        train_per_condition = DEFAULT_TRAIN_PER_CONDITION
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        parser.error("Input and output directories must not contain each other")
    if output_dir.exists() and not args.overwrite:
        parser.error(f"Output path already exists: {output_dir}; use --overwrite")
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"Output path is not a directory: {output_dir}")
    if train_ratio is not None and not 0 < train_ratio < 1:
        parser.error("--train-ratio must be strictly between 0 and 1")
    if train_per_condition is not None and train_per_condition <= 0:
        parser.error("--train-per-condition must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    run(
        input_dir,
        output_dir,
        train_ratio,
        train_per_condition,
        args.seed,
        args.workers,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
