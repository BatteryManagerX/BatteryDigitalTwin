import argparse
import csv
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


SOURCE_FIELDS = (
    "acqtime",
    "soc",
    "cell_battery_voltage",
    "total_current",
    "vehicle_status",
    "charge_state",
    "speed",
)
OUTPUT_FIELDS = (
    "acqtime",
    "soc",
    "voltage",
    "rate",
    "vehicle_status",
    "charge_state",
    "speed",
)
NUMERIC_FIELDS = ("soc", "voltage", "rate", "speed")
OTHER_REQUIRED_FIELDS = ("acqtime", "soc", "vehicle_status", "charge_state", "speed")
REASON_FIELDS = (
    "invalid_voltage_length",
    "invalid_voltage_number",
    "nonfinite_voltage",
    "cell_voltage_below_range",
    "cell_voltage_above_range",
    "cell_voltage_below_and_above_range",
    "invalid_rate",
    "rate_below_range",
    "rate_above_range",
    "missing_other_required_field",
)
INTERVAL_FIELDS = (
    "file",
    "records",
    "intervals",
    "main_interval_seconds",
    "main_interval_count",
    "main_interval_percent",
    "interval_10s_count",
    "interval_10s_percent",
    "zero_interval_count",
    "negative_interval_count",
    "other_positive_interval_count",
    "top_intervals",
)
CONDITIONS = (
    ("C1", "charging_or_low_activity", "充电或低活动工况"),
    ("C2", "low_speed_operation", "低速运行工况"),
    ("C3", "medium_speed_operation", "中速运行工况"),
    ("C4", "high_speed_operation", "高速运行工况"),
)
CONDITION_SUMMARY_FIELDS = (
    "mean_speed",
    "mean_moving_speed",
    "moving_ratio",
    "high_speed_ratio",
    "mean_rate",
    "soc_delta",
)
LENGTH_BINS = (
    ("1001-1250", 1001, 1250),
    ("1251-1500", 1251, 1500),
    ("1501-2000", 1501, 2000),
    ("2001-3000", 2001, 3000),
    ("3001-5000", 3001, 5000),
    ("5001-7000", 5001, 7000),
    ("7001-9000", 7001, 9000),
    (">9000", 9001, math.inf),
)
CELL_COUNT = 102
MIN_CELL_VOLTAGE = 2.5
MAX_CELL_VOLTAGE = 3.7
CURRENT_DIVISOR = Decimal(125)
MIN_RATE = -2.0
MAX_RATE = 2.0
REQUIRED_INTERVAL_SECONDS = 10
MINIMUM_SEQUENCE_LENGTH_EXCLUSIVE = 1000
CHARGING_RATIO_THRESHOLD = 0.5


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, document):
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(document, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def parse_timestamp(value, path, record_number):
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError as error:
        raise ValueError(
            f"Invalid acqtime in {path}, record {record_number}: {value!r}"
        ) from error


def scan_intervals(path):
    intervals = Counter()
    previous_time = None
    records = 0
    with path.open("r", encoding="utf-8-sig", newline="", buffering=8 * 1024 * 1024) as input_file:
        reader = csv.reader(input_file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty input file: {path}")
        missing = [field for field in SOURCE_FIELDS if field not in header]
        if missing:
            raise ValueError(f"Missing fields in {path}: {', '.join(missing)}")
        time_index = header.index("acqtime")
        for records, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(f"Malformed record in {path}, record {records}")
            timestamp = parse_timestamp(row[time_index], path, records)
            if previous_time is not None:
                seconds = int((timestamp - previous_time).total_seconds())
                intervals[seconds] += 1
            previous_time = timestamp
    if not intervals:
        raise ValueError(f"At least two data records are required: {path}")
    main_interval = min(intervals, key=lambda seconds: (-intervals[seconds], seconds))
    interval_count = sum(intervals.values())
    main_count = intervals[main_interval]
    top_intervals = "; ".join(
        f"{seconds}s:{count}"
        for seconds, count in sorted(intervals.items(), key=lambda item: (-item[1], item[0]))[:10]
    )
    return {
        "file": path.name,
        "records": records,
        "intervals": interval_count,
        "main_interval_seconds": main_interval,
        "main_interval_count": main_count,
        "main_interval_percent": f"{main_count / interval_count * 100:.6f}",
        "interval_10s_count": intervals[REQUIRED_INTERVAL_SECONDS],
        "interval_10s_percent": f"{intervals[REQUIRED_INTERVAL_SECONDS] / interval_count * 100:.6f}",
        "zero_interval_count": intervals[0],
        "negative_interval_count": sum(count for seconds, count in intervals.items() if seconds < 0),
        "other_positive_interval_count": sum(
            count for seconds, count in intervals.items()
            if seconds > 0 and seconds != REQUIRED_INTERVAL_SECONDS
        ),
        "top_intervals": top_intervals,
    }


def calculate_rate(value, path, record_number):
    value = value.strip()
    if not value:
        return ""
    try:
        return format(Decimal(value) / CURRENT_DIVISOR, "f")
    except InvalidOperation as error:
        raise ValueError(
            f"Invalid total_current in {path}, record {record_number}: {value!r}"
        ) from error


def clean_record(row, indexes, path, record_number):
    rate_text = calculate_rate(row[indexes["total_current"]], path, record_number)
    voltage_text = row[indexes["cell_battery_voltage"]].strip()
    parts = voltage_text.removeprefix("[").removesuffix("]").split()
    if len(parts) != CELL_COUNT:
        return None, "invalid_voltage_length", None
    try:
        cell_voltages = [float(part) for part in parts]
    except ValueError:
        return None, "invalid_voltage_number", None
    if not all(math.isfinite(value) for value in cell_voltages):
        return None, "nonfinite_voltage", None
    below_range = min(cell_voltages) < MIN_CELL_VOLTAGE
    above_range = max(cell_voltages) > MAX_CELL_VOLTAGE
    if below_range and above_range:
        return None, "cell_voltage_below_and_above_range", None
    if below_range:
        return None, "cell_voltage_below_range", None
    if above_range:
        return None, "cell_voltage_above_range", None
    try:
        rate = float(rate_text)
    except ValueError:
        return None, "invalid_rate", None
    if not math.isfinite(rate):
        return None, "invalid_rate", None
    if rate < MIN_RATE:
        return None, "rate_below_range", format(rate, ".12g")
    if rate > MAX_RATE:
        return None, "rate_above_range", format(rate, ".12g")
    if any(not row[indexes[field]].strip() for field in OTHER_REQUIRED_FIELDS):
        return None, "missing_other_required_field", None
    cleaned = [
        row[indexes["acqtime"]],
        row[indexes["soc"]],
        f"{sum(cell_voltages) / CELL_COUNT:.6f}",
        rate_text,
        row[indexes["vehicle_status"]],
        row[indexes["charge_state"]],
        row[indexes["speed"]],
    ]
    return cleaned, None, None


def empty_numeric_stats():
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


def add_numeric_stats(stats, row):
    for field, text in zip(OUTPUT_FIELDS, row):
        if field not in NUMERIC_FIELDS:
            continue
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {field}: {text!r}")
        field_stats = stats[field]
        field_stats["count"] += 1
        field_stats["minimum"] = min(field_stats["minimum"], value)
        field_stats["maximum"] = max(field_stats["maximum"], value)
        field_stats["sum"] += value
        field_stats["sum_squares"] += value * value


def merge_numeric_stats(target, source):
    for field in NUMERIC_FIELDS:
        current = target[field]
        incoming = source[field]
        current["count"] += incoming["count"]
        current["minimum"] = min(current["minimum"], incoming["minimum"])
        current["maximum"] = max(current["maximum"], incoming["maximum"])
        current["sum"] += incoming["sum"]
        current["sum_squares"] += incoming["sum_squares"]


def process_selected_file(path, output_dir, sequence_index):
    counts = Counter()
    rejected_rate_values = Counter()
    stats = empty_numeric_stats()
    manifest = []
    candidate_segments = 0
    discarded_segments = 0
    discarded_records = 0
    retained_records = 0
    buffer = []
    segment_length = 0
    segment_start_time = None
    segment_end_time = None
    segment_start_record = 0
    segment_end_record = 0
    segment_start_raw_record = 0
    segment_end_raw_record = 0
    previous_time = None
    output_file = None
    writer = None
    sequence_name = None

    def begin_segment(row, timestamp, cleaned_number, raw_number):
        nonlocal buffer, segment_length, segment_start_time, segment_end_time
        nonlocal segment_start_record, segment_end_record
        nonlocal segment_start_raw_record, segment_end_raw_record, candidate_segments
        candidate_segments += 1
        buffer = [row]
        segment_length = 1
        segment_start_time = timestamp
        segment_end_time = timestamp
        segment_start_record = cleaned_number
        segment_end_record = cleaned_number
        segment_start_raw_record = raw_number
        segment_end_raw_record = raw_number

    def append_row(row, timestamp, cleaned_number, raw_number):
        nonlocal segment_length, segment_end_time, segment_end_record, segment_end_raw_record
        nonlocal output_file, writer, sequence_name, sequence_index
        segment_length += 1
        segment_end_time = timestamp
        segment_end_record = cleaned_number
        segment_end_raw_record = raw_number
        if writer is None:
            buffer.append(row)
            if segment_length > MINIMUM_SEQUENCE_LENGTH_EXCLUSIVE:
                sequence_index += 1
                sequence_name = f"sequence_{sequence_index:06d}.csv"
                output_file = (output_dir / sequence_name).open(
                    "w", encoding="utf-8-sig", newline="", buffering=8 * 1024 * 1024
                )
                writer = csv.writer(output_file)
                writer.writerow(OUTPUT_FIELDS)
                writer.writerows(buffer)
                for buffered_row in buffer:
                    add_numeric_stats(stats, buffered_row)
                buffer.clear()
        else:
            writer.writerow(row)
            add_numeric_stats(stats, row)

    def finish_segment():
        nonlocal output_file, writer, discarded_segments, discarded_records
        nonlocal retained_records, segment_length
        if segment_length == 0:
            return
        if writer is None:
            discarded_segments += 1
            discarded_records += segment_length
            buffer.clear()
        else:
            output_file.close()
            output_file = None
            writer = None
            retained_records += segment_length
            manifest.append({
                "sequence_file": sequence_name,
                "source_file": path.name,
                "source_segment_index": candidate_segments,
                "source_start_record": segment_start_record,
                "source_end_record": segment_end_record,
                "source_start_raw_record": segment_start_raw_record,
                "source_end_raw_record": segment_end_raw_record,
                "start_time": segment_start_time.isoformat(sep=" "),
                "end_time": segment_end_time.isoformat(sep=" "),
                "records": segment_length,
                "duration_seconds": int((segment_end_time - segment_start_time).total_seconds()),
            })
        segment_length = 0

    try:
        with path.open("r", encoding="utf-8-sig", newline="", buffering=8 * 1024 * 1024) as input_file:
            reader = csv.reader(input_file)
            header = next(reader)
            indexes = {field: header.index(field) for field in SOURCE_FIELDS}
            for raw_number, raw_row in enumerate(reader, start=1):
                counts["input_records"] += 1
                if len(raw_row) != len(header):
                    raise ValueError(f"Malformed record in {path}, record {raw_number}")
                cleaned, reason, rejected_rate = clean_record(raw_row, indexes, path, raw_number)
                if reason is not None:
                    counts[reason] += 1
                    if rejected_rate is not None:
                        rejected_rate_values[rejected_rate] += 1
                    continue
                counts["output_records"] += 1
                cleaned_number = counts["output_records"]
                timestamp = parse_timestamp(cleaned[0], path, raw_number)
                if previous_time is None:
                    begin_segment(cleaned, timestamp, cleaned_number, raw_number)
                elif int((timestamp - previous_time).total_seconds()) == REQUIRED_INTERVAL_SECONDS:
                    append_row(cleaned, timestamp, cleaned_number, raw_number)
                else:
                    finish_segment()
                    begin_segment(cleaned, timestamp, cleaned_number, raw_number)
                previous_time = timestamp
            finish_segment()
    finally:
        if output_file is not None:
            output_file.close()

    rejected_records = counts["input_records"] - counts["output_records"]
    if rejected_records != sum(counts[field] for field in REASON_FIELDS):
        raise AssertionError(f"Rejection counts do not balance for {path}")
    if counts["output_records"] != retained_records + discarded_records:
        raise AssertionError(f"Sequence record counts do not balance for {path}")
    cleaning = {
        "file": path.name,
        "input_records": counts["input_records"],
        "output_records": counts["output_records"],
        "rejected_records": rejected_records,
        **{field: counts[field] for field in REASON_FIELDS},
        "rejected_rate_values": "; ".join(
            f"{value}:{count}" for value, count in rejected_rate_values.most_common()
        ),
    }
    summary = {
        "source_file": path.name,
        "input_records": counts["output_records"],
        "candidate_segments": candidate_segments,
        "retained_sequences": len(manifest),
        "retained_records": retained_records,
        "discarded_segments": discarded_segments,
        "discarded_records": discarded_records,
    }
    return sequence_index, cleaning, summary, manifest, stats, rejected_rate_values


def normalization_document(stats, input_dir, output_dir, sequences, records):
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
        "scope": {
            "source_directory": str(input_dir),
            "sequence_directory": str(output_dir),
            "retained_sequences": sequences,
            "retained_records": records,
            "required_interval_seconds": REQUIRED_INTERVAL_SECONDS,
            "minimum_sequence_length_exclusive": MINIMUM_SEQUENCE_LENGTH_EXCLUSIVE,
        },
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


def percentile(sorted_values, proportion):
    position = (len(sorted_values) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def analyze_sequence(path, metadata):
    records = 0
    speeds = []
    rate_sum = 0.0
    rate_minimum = math.inf
    rate_maximum = -math.inf
    negative_rate = 0
    positive_rate = 0
    moving = 0
    high_speed = 0
    moving_speed_sum = 0.0
    charge_states = Counter()
    vehicle_statuses = Counter()
    start_soc = None
    end_soc = None
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for records, row in enumerate(reader, start=1):
            soc = float(row["soc"])
            rate = float(row["rate"])
            speed = float(row["speed"])
            if records == 1:
                start_soc = soc
            end_soc = soc
            speeds.append(speed)
            rate_sum += rate
            rate_minimum = min(rate_minimum, rate)
            rate_maximum = max(rate_maximum, rate)
            negative_rate += rate < -0.02
            positive_rate += rate > 0.02
            if speed > 1.0:
                moving += 1
                moving_speed_sum += speed
            high_speed += speed >= 60.0
            charge_states[row["charge_state"]] += 1
            vehicle_statuses[row["vehicle_status"]] += 1
    if records != metadata["records"]:
        raise ValueError(f"Sequence length mismatch: {path}")
    speeds.sort()
    dominant_charge_state, dominant_charge_count = charge_states.most_common(1)[0]
    dominant_vehicle_status, dominant_vehicle_count = vehicle_statuses.most_common(1)[0]
    charge_state_1 = charge_states["1.0"] + charge_states["1"]
    charge_state_2 = charge_states["2.0"] + charge_states["2"]
    charge_state_3 = charge_states["3.0"] + charge_states["3"]
    return {
        "sequence_file": path.name,
        "records": records,
        "start_soc": start_soc,
        "end_soc": end_soc,
        "soc_delta": end_soc - start_soc,
        "mean_rate": rate_sum / records,
        "min_rate": rate_minimum,
        "max_rate": rate_maximum,
        "negative_rate_ratio": negative_rate / records,
        "positive_rate_ratio": positive_rate / records,
        "mean_speed": sum(speeds) / records,
        "mean_moving_speed": moving_speed_sum / moving if moving else 0.0,
        "speed_p50": percentile(speeds, 0.5),
        "speed_p90": percentile(speeds, 0.9),
        "max_speed": speeds[-1],
        "moving_ratio": moving / records,
        "high_speed_ratio": high_speed / records,
        "charge_state_1_ratio": charge_state_1 / records,
        "charge_state_2_ratio": charge_state_2 / records,
        "charge_state_3_ratio": charge_state_3 / records,
        "dominant_charge_state": dominant_charge_state,
        "dominant_charge_state_ratio": dominant_charge_count / records,
        "dominant_vehicle_status": dominant_vehicle_status,
        "dominant_vehicle_status_ratio": dominant_vehicle_count / records,
        "source_file": metadata["source_file"],
        "start_time": metadata["start_time"],
        "end_time": metadata["end_time"],
    }


def balanced_sizes(total, groups):
    base, remainder = divmod(total, groups)
    return [base + (index < remainder) for index in range(groups)]


def summarize_condition(rows):
    summary = {
        "sequence_count": len(rows),
        "record_count": sum(int(row["records"]) for row in rows),
        "source_file_count": len({row["source_file"] for row in rows}),
        "dominant_charging_sequence_count": sum(
            float(row["charge_state_1_ratio"]) >= CHARGING_RATIO_THRESHOLD
            for row in rows
        ),
    }
    for field in CONDITION_SUMMARY_FIELDS:
        values = [float(row[field]) for row in rows]
        summary[field] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
        }
    return summary


def classify_conditions(features, output_dir, final_output_dir):
    target_sizes = balanced_sizes(len(features), len(CONDITIONS))
    charging = [
        row for row in features
        if float(row["charge_state_1_ratio"]) >= CHARGING_RATIO_THRESHOLD
    ]
    noncharging = [
        row for row in features
        if float(row["charge_state_1_ratio"]) < CHARGING_RATIO_THRESHOLD
    ]
    noncharging.sort(
        key=lambda row: (
            float(row["mean_speed"]),
            float(row["moving_ratio"]),
            row["sequence_file"],
        )
    )
    first_noncharging_count = max(0, target_sizes[0] - len(charging))
    groups = [charging + noncharging[:first_noncharging_count]]
    remaining = noncharging[first_noncharging_count:]
    other_sizes = (
        target_sizes[1:] if len(charging) <= target_sizes[0]
        else balanced_sizes(len(remaining), len(CONDITIONS) - 1)
    )
    offset = 0
    for size in other_sizes:
        groups.append(remaining[offset:offset + size])
        offset += size
    assigned = {}
    for condition, group in zip(CONDITIONS, groups):
        for row in group:
            name = row["sequence_file"]
            if name in assigned:
                raise AssertionError(f"Duplicate condition assignment: {name}")
            assigned[name] = condition
    if len(assigned) != len(features):
        raise AssertionError("Some sequences were not assigned a condition")
    output_rows = []
    for row in sorted(features, key=lambda item: item["sequence_file"]):
        code, name, name_zh = assigned[row["sequence_file"]]
        output_rows.append({
            "sequence_file": row["sequence_file"],
            "condition_code": code,
            "condition_name": name,
            "condition_name_zh": name_zh,
            **{key: value for key, value in row.items() if key != "sequence_file"},
        })
    report_dir = output_dir / "_reports"
    fieldnames = list(output_rows[0])
    write_csv(report_dir / "operating_condition_manifest.csv", fieldnames, output_rows)
    for code, name, _ in CONDITIONS:
        selected = [row for row in output_rows if row["condition_code"] == code]
        write_csv(report_dir / "operating_conditions" / f"{code}_{name}.csv", fieldnames, selected)
    noncharging_groups = [
        sorted(
            [
                row for row in group
                if float(row["charge_state_1_ratio"]) < CHARGING_RATIO_THRESHOLD
            ],
            key=lambda row: (float(row["mean_speed"]), row["sequence_file"]),
        )
        for group in groups
    ]
    boundaries = []
    for left, right in zip(noncharging_groups, noncharging_groups[1:]):
        if left and right:
            boundaries.append(
                (float(left[-1]["mean_speed"]) + float(right[0]["mean_speed"])) / 2
            )
    condition_summaries = {
        code: {"name": name, "name_zh": name_zh, **summarize_condition(group)}
        for (code, name, name_zh), group in zip(CONDITIONS, groups)
    }
    parameters = {
        "method": "balanced_rule_based_sequence_classification",
        "sequence_count": len(features),
        "target_group_sizes": target_sizes,
        "rules": {
            "charging_definition": f"charge_state_1_ratio >= {CHARGING_RATIO_THRESHOLD}",
            "C1": "All dominant-charging sequences, plus the lowest-mean-speed non-charging sequences until the balanced target size is reached.",
            "C2_to_C4": "Remaining non-charging sequences ranked by whole-sequence mean_speed and divided into three balanced groups.",
            "noncharging_mean_speed_boundaries": boundaries,
            "tie_breaker": "moving_ratio, then sequence_file",
        },
        "conditions": condition_summaries,
        "notes": {
            "charge_state_1": "Charge state 1 is treated as charging.",
            "mean_speed": "Mean speed includes stopped periods.",
            "labels_location": str(final_output_dir / "_reports" / "operating_condition_manifest.csv"),
            "data_files_are_not_modified": True,
        },
    }
    write_json(output_dir / "operating_condition_parameters.json", parameters)
    return output_rows, parameters


def summarize_lengths(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": percentile(ordered, 0.05),
        "p10": percentile(ordered, 0.10),
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.5),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.9),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "std_population": statistics.pstdev(ordered),
    }


def write_length_distribution(classified, report_dir):
    lengths = [int(row["records"]) for row in classified]
    total_records = sum(lengths)
    histogram = []
    for label, lower, upper in LENGTH_BINS:
        selected = [length for length in lengths if lower <= length <= upper]
        histogram.append({
            "range": label,
            "sequences": len(selected),
            "sequence_percent": len(selected) / len(lengths) * 100,
            "records": sum(selected),
            "record_percent": sum(selected) / total_records * 100,
        })
    by_condition = defaultdict(list)
    for row in classified:
        by_condition[row["condition_code"]].append(int(row["records"]))
    nonoverlap_windows = sum(length // 1000 for length in lengths)
    overlapping_windows = sum((length - 1000) // 500 + 1 for length in lengths)
    document = {
        "all_sequences": summarize_lengths(lengths),
        "total_records": total_records,
        "histogram": histogram,
        "by_condition": {
            code: summarize_lengths(values) for code, values in sorted(by_condition.items())
        },
        "window_estimates": {
            "window_length": 1000,
            "nonoverlap_stride_1000": {
                "windows": nonoverlap_windows,
                "records_used": nonoverlap_windows * 1000,
                "source_record_coverage_percent": nonoverlap_windows * 1000 / total_records * 100,
            },
            "overlap_stride_500": {
                "windows": overlapping_windows,
                "note": "Overlapping observations are counted more than once.",
            },
        },
    }
    write_json(report_dir / "sequence_length_distribution.json", document)


def write_cleaning_reports(cleaning_rows, rejected_rates, report_dir):
    numeric_fields = (
        "input_records",
        "output_records",
        "rejected_records",
        *REASON_FIELDS,
    )
    overall = {
        "file": "ALL",
        **{
            field: sum(int(row[field]) for row in cleaning_rows)
            for field in numeric_fields
        },
        "rejected_rate_values": "; ".join(
            f"{value}:{count}" for value, count in rejected_rates.most_common()
        ),
    }
    write_csv(
        report_dir / "voltage_current_cleaning_report.csv",
        list(overall),
        [*cleaning_rows, overall],
    )
    input_records = overall["input_records"]
    rejected_records = overall["rejected_records"]
    summary = [
        f"Input files: {len(cleaning_rows)}",
        f"Input records: {input_records:,}",
        f"Output records: {overall['output_records']:,}",
        f"Rejected records: {rejected_records:,} ({rejected_records / input_records * 100:.6f}%)",
        f"Voltage rule: exactly {CELL_COUNT} finite cell values, each in [{MIN_CELL_VOLTAGE}, {MAX_CELL_VOLTAGE}] V",
        f"Output voltage: arithmetic mean of the {CELL_COUNT} cell values",
        f"Rate rule: [{MIN_RATE}, {MAX_RATE}] (equivalent total_current: [-250, 250])",
        *(f"{field}: {overall[field]:,}" for field in REASON_FIELDS),
        f"Rejected rate values: {overall['rejected_rate_values']}",
    ]
    (report_dir / "voltage_current_cleaning_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    return overall


def write_sequence_reports(file_summaries, manifest, report_dir):
    summary_fields = (
        "source_file",
        "input_records",
        "candidate_segments",
        "retained_sequences",
        "retained_records",
        "discarded_segments",
        "discarded_records",
    )
    manifest_fields = (
        "sequence_file",
        "source_file",
        "source_segment_index",
        "source_start_record",
        "source_end_record",
        "source_start_raw_record",
        "source_end_raw_record",
        "start_time",
        "end_time",
        "records",
        "duration_seconds",
    )
    write_csv(report_dir / "sequence_file_summary.csv", summary_fields, file_summaries)
    write_csv(report_dir / "sequence_manifest.csv", manifest_fields, manifest)


def run(input_dir, output_dir, sources, interval_rows):
    selected = [
        row for row in interval_rows
        if row["main_interval_seconds"] == REQUIRED_INTERVAL_SECONDS
    ]
    excluded = [
        row for row in interval_rows
        if row["main_interval_seconds"] != REQUIRED_INTERVAL_SECONDS
    ]
    if not selected:
        raise ValueError("No input file has a 10-second dominant interval")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        report_dir = stage / "_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        write_csv(report_dir / "time_interval_report.csv", INTERVAL_FIELDS, interval_rows)
        write_csv(
            report_dir / "included_10s_main_interval_files.csv",
            INTERVAL_FIELDS,
            selected,
        )
        write_csv(
            report_dir / "excluded_non_10s_main_interval_files.csv",
            INTERVAL_FIELDS,
            excluded,
        )
        interval_by_name = {row["file"]: row for row in interval_rows}
        selected_names = {row["file"] for row in selected}
        sequence_index = 0
        cleaning_rows = []
        file_summaries = []
        manifest = []
        numeric_stats = empty_numeric_stats()
        rejected_rates = Counter()
        for path in sources:
            if path.name not in selected_names:
                continue
            sequence_index, cleaning, summary, file_manifest, file_stats, file_rates = (
                process_selected_file(path, stage, sequence_index)
            )
            if cleaning["input_records"] != interval_by_name[path.name]["records"]:
                raise ValueError(f"Input row count changed between passes: {path}")
            cleaning_rows.append(cleaning)
            file_summaries.append(summary)
            manifest.extend(file_manifest)
            merge_numeric_stats(numeric_stats, file_stats)
            rejected_rates.update(file_rates)
            print(
                f"Processed {path.name}: {cleaning['output_records']:,} clean rows, "
                f"{summary['retained_sequences']:,} sequences",
                flush=True,
            )
        if not manifest:
            raise ValueError("No sequence longer than 1000 records was found")
        write_sequence_reports(file_summaries, manifest, report_dir)
        cleaning_total = write_cleaning_reports(cleaning_rows, rejected_rates, report_dir)
        retained_records = sum(int(row["records"]) for row in manifest)
        normalization = normalization_document(
            numeric_stats, input_dir, output_dir, len(manifest), retained_records
        )
        write_json(stage / "normalization_parameters.json", normalization)
        features = [
            analyze_sequence(stage / row["sequence_file"], row)
            for row in manifest
        ]
        write_csv(
            report_dir / "sequence_condition_features.csv",
            list(features[0]),
            features,
        )
        classified, condition_parameters = classify_conditions(features, stage, output_dir)
        write_length_distribution(classified, report_dir)
        candidate_segments = sum(int(row["candidate_segments"]) for row in file_summaries)
        discarded_segments = sum(int(row["discarded_segments"]) for row in file_summaries)
        discarded_records = sum(int(row["discarded_records"]) for row in file_summaries)
        selected_records = sum(int(row["records"]) for row in selected)
        if selected_records != cleaning_total["input_records"]:
            raise AssertionError("Selected input counts do not balance")
        if retained_records + discarded_records != cleaning_total["output_records"]:
            raise AssertionError("Cleaned record counts do not balance")
        summary = {
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "raw_files": len(sources),
            "raw_records": sum(int(row["records"]) for row in interval_rows),
            "selected_10s_files": len(selected),
            "excluded_non_10s_files": len(excluded),
            "excluded_file_names": [row["file"] for row in excluded],
            "selected_records": selected_records,
            "rejected_records": cleaning_total["rejected_records"],
            "cleaned_records": cleaning_total["output_records"],
            "candidate_segments": candidate_segments,
            "discarded_short_segments": discarded_segments,
            "discarded_short_segment_records": discarded_records,
            "retained_sequences": len(manifest),
            "retained_records": retained_records,
            "condition_sequence_counts": {
                code: condition_parameters["conditions"][code]["sequence_count"]
                for code, _, _ in CONDITIONS
            },
        }
        write_json(stage / "processing_summary.json", summary)
        if output_dir.exists():
            raise FileExistsError(f"Output directory appeared during processing: {output_dir}")
        os.replace(stage, output_dir)
        print(
            f"Done: {len(manifest):,} sequences, {retained_records:,} records; "
            f"output: {output_dir}",
            flush=True,
        )
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Build cleaned, labeled 10-second sequences from raw vehicle CSV files."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    if output_dir.exists():
        parser.error(f"Output path already exists: {output_dir}")
    sources = sorted(input_dir.glob("*.csv"))
    if not sources:
        parser.error(f"No CSV files found in: {input_dir}")
    interval_rows = []
    for path in sources:
        row = scan_intervals(path)
        interval_rows.append(row)
        print(
            f"Scanned {path.name}: {row['records']:,} rows, "
            f"dominant interval {row['main_interval_seconds']}s",
            flush=True,
        )
    run(input_dir, output_dir, sources, interval_rows)


if __name__ == "__main__":
    main()
