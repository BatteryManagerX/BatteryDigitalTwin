import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1"


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(manifest_path: Path, expected_split: str) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 {expected_split} 清单: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype=str)
    required = {"sequence_file", "condition_code"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"清单缺少字段 {sorted(missing)}: {manifest_path}")
    if manifest.empty:
        raise ValueError(f"清单没有轨迹: {manifest_path}")
    if "split" in manifest and not manifest["split"].eq(expected_split).all():
        raise ValueError(f"清单包含非 {expected_split} 轨迹: {manifest_path}")
    if manifest[list(required)].isna().any().any():
        raise ValueError(f"清单包含空文件名或工况: {manifest_path}")
    names = manifest["sequence_file"]
    if names.duplicated().any():
        raise ValueError(f"清单包含重复轨迹文件名: {manifest_path}")
    if any(
        not name.endswith(".csv") or Path(name).name != name
        or "/" in name or "\\" in name or name in (".", "..")
        for name in names
    ):
        raise ValueError(f"清单包含非法轨迹文件名: {manifest_path}")
    return manifest[["sequence_file", "condition_code"]]


def _test_source_metadata(test_dir: Path, manifest_path: Path) -> Dict[str, str]:
    return {
        "test_dir": str(test_dir.resolve()),
        "test_manifest_path": str(manifest_path.resolve()),
        "test_manifest_sha256": _checkpoint_sha256(manifest_path),
    }


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trajectory_status (
            sequence_file TEXT PRIMARY KEY,
            condition_code TEXT NOT NULL,
            frame_count INTEGER NOT NULL,
            completed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            sequence_file TEXT NOT NULL,
            step INTEGER NOT NULL,
            acqtime TEXT NOT NULL,
            rate REAL NOT NULL,
            soc_gt REAL NOT NULL,
            soc_pred REAL NOT NULL,
            voltage_gt REAL NOT NULL,
            voltage_pred REAL,
            PRIMARY KEY (sequence_file, step)
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_step
        ON predictions(step);
        """
    )


def _validate_run_metadata(connection: sqlite3.Connection, metadata: Dict[str, str]) -> None:
    existing = dict(connection.execute("SELECT key, value FROM run_metadata"))
    if existing:
        mismatches = {
            key: (existing.get(key), value)
            for key, value in metadata.items()
            if existing.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: 数据库={old!r}, 当前={new!r}"
                for key, (old, new) in mismatches.items()
            )
            raise ValueError(f"现有数据库与当前推理配置不匹配: {details}")
        return

    connection.executemany(
        "INSERT INTO run_metadata(key, value) VALUES (?, ?)",
        metadata.items()
    )
    connection.commit()


def _read_sequence(path: Path) -> pd.DataFrame:
    required_columns = ["acqtime", "soc", "voltage", "rate"]
    frame = pd.read_csv(path, usecols=required_columns)
    frame = frame.dropna(subset=required_columns)
    frame = frame.sort_values("acqtime").drop_duplicates("acqtime").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"轨迹没有有效帧: {path}")

    frame["soc"] = frame["soc"].astype(float) / 100.0
    frame["voltage"] = frame["voltage"].astype(float)
    frame["rate"] = frame["rate"].astype(float)
    if not np.isfinite(frame[["soc", "voltage", "rate"]].to_numpy()).all():
        raise ValueError(f"轨迹包含非有限数值: {path}")
    if not frame["soc"].between(0.0, 1.0).all():
        raise ValueError(f"SOC 应为 0–100 百分比: {path}")
    return frame


def _write_trajectory(connection: sqlite3.Connection, sequence_file: str,
                      condition_code: str, frame: pd.DataFrame,
                      soc_pred: np.ndarray, voltage_pred: np.ndarray) -> None:
    rows = zip(
        [sequence_file] * len(frame),
        range(len(frame)),
        frame["acqtime"].astype(str),
        frame["rate"].to_numpy(dtype=float),
        frame["soc"].to_numpy(dtype=float),
        soc_pred,
        frame["voltage"].to_numpy(dtype=float),
        voltage_pred
    )
    completed_at = datetime.now(timezone.utc).isoformat()

    with connection:
        connection.execute(
            "DELETE FROM predictions WHERE sequence_file = ?",
            (sequence_file,)
        )
        connection.executemany(
            """
            INSERT INTO predictions(
                sequence_file, step, acqtime, rate,
                soc_gt, soc_pred, voltage_gt, voltage_pred
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO trajectory_status(
                sequence_file, condition_code, frame_count, completed_at
            ) VALUES (?, ?, ?, ?)
            """,
            (sequence_file, condition_code, len(frame), completed_at)
        )


def _write_soc_trajectory(connection: sqlite3.Connection, sequence_file: str,
                          condition_code: str, frame: pd.DataFrame,
                          steps: np.ndarray, soc_pred: np.ndarray) -> None:
    selected = frame.iloc[steps]
    rows = zip(
        [sequence_file] * len(steps),
        steps.astype(int).tolist(),
        selected["acqtime"].astype(str),
        selected["rate"].to_numpy(dtype=float),
        selected["soc"].to_numpy(dtype=float),
        soc_pred,
        selected["voltage"].to_numpy(dtype=float),
        [None] * len(steps)
    )
    completed_at = datetime.now(timezone.utc).isoformat()

    with connection:
        connection.execute(
            "DELETE FROM predictions WHERE sequence_file = ?",
            (sequence_file,)
        )
        connection.executemany(
            """
            INSERT INTO predictions(
                sequence_file, step, acqtime, rate,
                soc_gt, soc_pred, voltage_gt, voltage_pred
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO trajectory_status(
                sequence_file, condition_code, frame_count, completed_at
            ) VALUES (?, ?, ?, ?)
            """,
            (sequence_file, condition_code, len(frame), completed_at)
        )


def run_soc_test_inference(
    model_name: str,
    checkpoint_path: Path,
    output_path: Path,
    test_dir: Path,
    manifest_path: Path,
    predict: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]],
    metadata: Optional[Dict[str, str]] = None,
    limit: int = None
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型权重: {checkpoint_path}")

    if not test_dir.is_dir():
        raise FileNotFoundError(f"找不到测试数据目录: {test_dir}")
    manifest = _read_manifest(manifest_path, expected_split="test")
    if limit is not None:
        manifest = manifest.iloc[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output_path)
    try:
        _initialize_database(connection)
        run_metadata = {
            "schema_version": SCHEMA_VERSION,
            "model_name": model_name,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
            "soc_scale": "0_to_1",
            "current_condition": "rate",
            "inference_mode": "causal_prefix_from_min_window",
            "voltage_prediction": "not_available"
        }
        run_metadata.update(_test_source_metadata(test_dir, manifest_path))
        if metadata:
            run_metadata.update(metadata)
        _validate_run_metadata(connection, run_metadata)

        completed = {
            row[0]
            for row in connection.execute("SELECT sequence_file FROM trajectory_status")
        }
        pending = manifest[~manifest["sequence_file"].isin(completed)]
        print(
            f"测试轨迹共 {len(manifest)} 条，已完成 {len(manifest) - len(pending)} 条，"
            f"待推理 {len(pending)} 条"
        )

        for progress, row in enumerate(pending.itertuples(index=False), start=1):
            frame = _read_sequence(test_dir / row.sequence_file)
            steps, soc_pred = predict(frame)
            steps = np.asarray(steps, dtype=np.int64)
            soc_pred = np.asarray(soc_pred, dtype=float)
            if len(steps) != len(soc_pred) or len(steps) == 0:
                raise ValueError(f"模型输出为空或长度不一致: {row.sequence_file}")
            if steps.min() < 0 or steps.max() >= len(frame) or np.any(np.diff(steps) <= 0):
                raise ValueError(f"模型输出 step 非法: {row.sequence_file}")
            if not np.isfinite(soc_pred).all():
                raise ValueError(f"模型输出包含非有限值: {row.sequence_file}")

            _write_soc_trajectory(
                connection, row.sequence_file, row.condition_code,
                frame, steps, soc_pred
            )
            if progress == 1 or progress % 25 == 0 or progress == len(pending):
                print(f"[{progress}/{len(pending)}] 已完成 {row.sequence_file}")

        summary = {
            "trajectory_count": connection.execute(
                "SELECT COUNT(*) FROM trajectory_status"
            ).fetchone()[0],
            "prediction_count": connection.execute(
                "SELECT COUNT(*) FROM predictions"
            ).fetchone()[0]
        }
        print(f"推理完成: {json.dumps(summary, ensure_ascii=False)}")
        print(f"结果数据库: {output_path}")
    finally:
        connection.close()


def run_test_inference(
    model_name: str,
    checkpoint_path: Path,
    output_path: Path,
    test_dir: Path,
    manifest_path: Path,
    predict: Callable[..., Tuple[np.ndarray, np.ndarray]],
    metadata: Optional[Dict[str, str]] = None,
    limit: int = None,
    pass_soc_ground_truth: bool = False
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型权重: {checkpoint_path}")

    if not test_dir.is_dir():
        raise FileNotFoundError(f"找不到测试数据目录: {test_dir}")
    manifest = _read_manifest(manifest_path, expected_split="test")
    if limit is not None:
        manifest = manifest.iloc[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output_path)
    try:
        _initialize_database(connection)
        run_metadata = {
            "schema_version": SCHEMA_VERSION,
            "model_name": model_name,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
            "soc_scale": "0_to_1",
            "current_condition": "rate",
            "inference_mode": "autoregressive_from_initial_frame"
        }
        run_metadata.update(_test_source_metadata(test_dir, manifest_path))
        if metadata:
            run_metadata.update(metadata)
        _validate_run_metadata(connection, run_metadata)

        completed = {
            row[0]
            for row in connection.execute("SELECT sequence_file FROM trajectory_status")
        }
        pending = manifest[~manifest["sequence_file"].isin(completed)]
        print(
            f"测试轨迹共 {len(manifest)} 条，已完成 {len(manifest) - len(pending)} 条，"
            f"待推理 {len(pending)} 条"
        )

        for progress, row in enumerate(pending.itertuples(index=False), start=1):
            sequence_path = test_dir / row.sequence_file
            frame = _read_sequence(sequence_path)
            current = frame["rate"].to_numpy(dtype=float)
            predict_args = [
                float(frame["soc"].iloc[0]),
                float(frame["voltage"].iloc[0]),
                current
            ]
            if pass_soc_ground_truth:
                predict_args.append(frame["soc"].to_numpy(dtype=float))
            soc_pred, voltage_pred = predict(*predict_args)
            if len(soc_pred) != len(frame) or len(voltage_pred) != len(frame):
                raise ValueError(f"模型输出长度与轨迹不一致: {row.sequence_file}")
            if not np.isfinite(soc_pred).all() or not np.isfinite(voltage_pred).all():
                raise ValueError(f"模型输出包含非有限值: {row.sequence_file}")

            _write_trajectory(
                connection,
                row.sequence_file,
                row.condition_code,
                frame,
                np.asarray(soc_pred, dtype=float),
                np.asarray(voltage_pred, dtype=float)
            )
            if progress == 1 or progress % 25 == 0 or progress == len(pending):
                print(f"[{progress}/{len(pending)}] 已完成 {row.sequence_file}")

        summary = {
            "trajectory_count": connection.execute(
                "SELECT COUNT(*) FROM trajectory_status"
            ).fetchone()[0],
            "prediction_count": connection.execute(
                "SELECT COUNT(*) FROM predictions"
            ).fetchone()[0]
        }
        print(f"推理完成: {json.dumps(summary, ensure_ascii=False)}")
        print(f"结果数据库: {output_path}")
    finally:
        connection.close()
