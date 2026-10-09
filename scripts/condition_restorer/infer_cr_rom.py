import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "split_data"
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from models.condition_restorer.condition_restorer import ConditionRestorer
from models.rom.spm import SPMModel
from scripts.rom.inference_common import (
	SCHEMA_VERSION,
	_checkpoint_sha256,
	_initialize_database,
	_read_sequence,
	_validate_run_metadata,
	_write_trajectory,
)


LOSS_HORIZON = 2048


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run ConditionRestorer + SPM inference with simulated packet loss."
	)
	parser.add_argument(
		"--cr-checkpoint", type=Path,
		default=ROOT / "checkpoints" / "condition_restorer" / "cr_best.pt",
	)
	parser.add_argument(
		"--rom-checkpoint", type=Path,
		default=ROOT / "checkpoints" / "rom" / "spm.npz",
	)
	parser.add_argument(
		"--output", type=Path,
		default=ROOT / "results" / "condition_restorer" / "cr_spm_infer_test.db",
	)
	parser.add_argument("--test-dir", type=Path, default=DATA_ROOT / "test")
	parser.add_argument(
		"--manifest", type=Path,
		default=DATA_ROOT / "metadata" / "test_manifest.csv",
	)
	parser.add_argument("--loss-count", type=int, default=5)
	parser.add_argument("--max-loss-length", type=int, default=6)
	parser.add_argument("--seed", type=int, default=20260901)
	parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
	parser.add_argument("--no-amp", action="store_true")
	parser.add_argument("--limit", type=int, default=None)
	return parser.parse_args()


def select_device(requested: str) -> torch.device:
	if requested == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(requested)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("指定了 CUDA，但当前环境无法使用 CUDA")
	return device


def validate_args(args: argparse.Namespace) -> None:
	if args.loss_count <= 0:
		raise ValueError("--loss-count 必须为正整数")
	if not 1 <= args.max_loss_length < 7:
		raise ValueError("--max-loss-length 必须在 [1, 6] 内")
	if args.limit is not None and args.limit <= 0:
		raise ValueError("--limit 必须为正整数")
	for path, label in (
		(args.cr_checkpoint, "CR checkpoint"),
		(args.rom_checkpoint, "ROM checkpoint"),
		(args.manifest, "测试清单"),
	):
		if not path.exists():
			raise FileNotFoundError(f"找不到{label}: {path}")


def load_cr(checkpoint_path: Path, device: torch.device):
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	if checkpoint.get("model_name") != "condition_restorer":
		raise ValueError("CR checkpoint 的 model_name 不是 condition_restorer")
	model_config = checkpoint.get("model_config")
	if not model_config:
		raise ValueError("CR checkpoint 缺少 model_config")
	if checkpoint.get("input_channels") != {
		"pre_seq": ["rate", "voltage", "soc_percent"],
		"sub_seq": ["rate", "voltage"],
	}:
		raise ValueError("CR checkpoint 的输入通道定义不兼容")
	if checkpoint.get("output_channel") != "rate":
		raise ValueError("CR checkpoint 的输出通道不是 rate")
	model = ConditionRestorer(**model_config)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device)
	model.eval()
	return model, model_config, checkpoint


def trajectory_rng(seed: int, sequence_file: str) -> np.random.Generator:
	digest = hashlib.sha256(f"{seed}:{sequence_file}".encode("utf-8")).digest()
	return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def sample_loss_intervals(frame_count: int, loss_count: int, max_loss_length: int,
						  pre_len: int, sub_len: int,
						  rng: np.random.Generator) -> list[tuple[int, int]]:
	latest_end = min(frame_count - sub_len, LOSS_HORIZON)
	if latest_end <= pre_len:
		raise ValueError("轨迹过短，无法在 2048 步内提供 CR 所需的前后文")

	intervals = []
	for _ in range(loss_count):
		accepted = None
		for _ in range(10_000):
			length = int(rng.integers(1, max_loss_length + 1))
			if latest_end - length < pre_len:
				continue
			start = int(rng.integers(pre_len, latest_end - length + 1))
			end = start + length
			if all(
				end + sub_len <= other_start - pre_len
				or start - pre_len >= other_end + sub_len
				for other_start, other_end in intervals
			):
				accepted = (start, end)
				break
		if accepted is None:
			raise ValueError(
				f"无法在前 {LOSS_HORIZON} 步放置 {loss_count} 次互不污染上下文的丢包；"
				"请减少 --loss-count 或 --max-loss-length"
			)
		intervals.append(accepted)
	return sorted(intervals)


class CRSPMPredictor:
	def __init__(self, cr_model: ConditionRestorer, rom_model: SPMModel,
				 model_config: dict, device: torch.device, use_amp: bool):
		self.cr_model = cr_model
		self.rom_model = rom_model
		self.pre_len = int(model_config["pre_len"])
		self.sub_len = int(model_config["sub_len"])
		self.max_missing_len = int(model_config["max_missing_len"])
		self.device = device
		self.use_amp = use_amp

	def restore(self, start: int, end: int, source_rate: np.ndarray,
				source_voltage: np.ndarray, source_soc: np.ndarray,
				effective_rate: np.ndarray) -> np.ndarray:
		pre = np.column_stack((
			effective_rate[start - self.pre_len:start],
			source_voltage[start - self.pre_len:start],
			source_soc[start - self.pre_len:start] * 100.0,
		)).astype(np.float32)
		sub = np.column_stack((
			source_rate[end:end + self.sub_len],
			source_voltage[end:end + self.sub_len],
		)).astype(np.float32)
		missing_len = end - start
		with torch.inference_mode(), torch.autocast(
			device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp,
		):
			prediction = self.cr_model(
				torch.from_numpy(pre).unsqueeze(0).to(self.device),
				torch.from_numpy(sub).unsqueeze(0).to(self.device),
				torch.tensor([missing_len], dtype=torch.long, device=self.device),
			)
		restored = prediction[0, :missing_len].float().cpu().numpy()
		if not np.isfinite(restored).all():
			raise RuntimeError(f"CR 在 [{start}, {end}) 输出了非有限 rate")
		return restored

	def __call__(self, frame: pd.DataFrame, intervals: list[tuple[int, int]]):
		source_rate = frame["rate"].to_numpy(dtype=np.float64)
		source_voltage = frame["voltage"].to_numpy(dtype=np.float64)
		source_soc = frame["soc"].to_numpy(dtype=np.float64)
		frame_count = len(frame)
		effective_rate = source_rate.copy()
		soc_pred = np.empty(frame_count, dtype=np.float64)
		voltage_pred = np.empty(frame_count, dtype=np.float64)
		soc_pred[0] = float(frame["soc"].iloc[0])
		voltage_pred[0] = float(frame["voltage"].iloc[0])
		self.rom_model.reset_state(soc_pred[0])

		interval_by_start = {start: end for start, end in intervals}
		if 0 in interval_by_start:
			raise ValueError("丢包不能从初始帧开始")
		for step in range(frame_count - 1):
			next_state = self.rom_model._state_transition(
				self.rom_model.state, effective_rate[step], self.rom_model.params,
			)
			next_step = step + 1
			if next_step in interval_by_start:
				end = interval_by_start[next_step]
				effective_rate[next_step:end] = self.restore(
					next_step, end, source_rate, source_voltage, source_soc,
					effective_rate,
				)
			self.rom_model.state = next_state.copy()
			soc_pred[next_step] = next_state[0]
			voltage_pred[next_step] = self.rom_model._measurement(
				next_state, effective_rate[next_step], self.rom_model.params,
			)
		return soc_pred, voltage_pred, effective_rate


def main() -> None:
	args = parse_args()
	validate_args(args)
	device = select_device(args.device)
	cr_model, model_config, cr_checkpoint = load_cr(args.cr_checkpoint, device)
	if args.max_loss_length > int(model_config["max_missing_len"]):
		raise ValueError(
			"--max-loss-length 超过 CR checkpoint 的 max_missing_len="
			f"{model_config['max_missing_len']}"
		)
	rom_sha256 = _checkpoint_sha256(args.rom_checkpoint)
	trained_rom_sha256 = cr_checkpoint.get("spm_checkpoint_sha256")
	if trained_rom_sha256 and trained_rom_sha256 != rom_sha256:
		raise ValueError("推理 SPM checkpoint 与 CR 训练时使用的 SPM 不一致")
	rom_model = SPMModel.load_checkpoint(args.rom_checkpoint)
	use_amp = device.type == "cuda" and not args.no_amp
	predictor = CRSPMPredictor(cr_model, rom_model, model_config, device, use_amp)

	manifest = pd.read_csv(args.manifest, usecols=["sequence_file", "condition_code"])
	if args.limit is not None:
		manifest = manifest.iloc[:args.limit]
	args.output.parent.mkdir(parents=True, exist_ok=True)
	connection = sqlite3.connect(args.output)
	try:
		_initialize_database(connection)
		metadata = {
			"schema_version": SCHEMA_VERSION,
			"model_name": "condition_restorer_spm",
			"checkpoint_path": str(args.rom_checkpoint.resolve()),
			"checkpoint_sha256": rom_sha256,
			"cr_checkpoint_path": str(args.cr_checkpoint.resolve()),
			"cr_checkpoint_sha256": _checkpoint_sha256(args.cr_checkpoint),
			"soc_scale": "0_to_1",
			"current_condition": "rate_with_cr_restoration_at_packet_losses",
			"inference_mode": "spm_autoregressive_with_condition_restoration",
			"loss_count_per_trajectory": str(args.loss_count),
			"max_loss_length": str(args.max_loss_length),
			"loss_length_distribution": "discrete_uniform_1_to_max_inclusive",
			"loss_horizon_steps_exclusive": str(LOSS_HORIZON),
			"loss_seed": str(args.seed),
			"loss_seed_scope": "sha256(global_seed:sequence_file)",
			"stored_rate": "effective_rom_input_observed_or_cr_restored",
			"cr_pre_context": "observed_soc_voltage_and_effective_rate",
			"cr_sub_context": "observed_rate_and_voltage_after_loss",
			"cr_training_global_step": str(cr_checkpoint.get("global_step", "unknown")),
		}
		_validate_run_metadata(connection, metadata)
		completed = {
			row[0] for row in connection.execute("SELECT sequence_file FROM trajectory_status")
		}
		pending = manifest[~manifest["sequence_file"].isin(completed)]
		print(
			f"Device: {device}; AMP: {use_amp}; loss count: {args.loss_count}; "
			f"max loss length: {args.max_loss_length}; horizon: {LOSS_HORIZON}"
		)
		print(
			f"测试轨迹共 {len(manifest)} 条，已完成 {len(manifest) - len(pending)} 条，"
			f"待推理 {len(pending)} 条"
		)
		for progress, row in enumerate(pending.itertuples(index=False), start=1):
			frame = _read_sequence(args.test_dir / row.sequence_file)
			intervals = sample_loss_intervals(
				len(frame), args.loss_count, args.max_loss_length,
				predictor.pre_len, predictor.sub_len,
				trajectory_rng(args.seed, row.sequence_file),
			)
			soc_pred, voltage_pred, effective_rate = predictor(frame, intervals)
			output_frame = frame.copy()
			output_frame["rate"] = effective_rate
			_write_trajectory(
				connection, row.sequence_file, row.condition_code, output_frame,
				soc_pred, voltage_pred,
			)
			if progress == 1 or progress % 25 == 0 or progress == len(pending):
				interval_text = ", ".join(f"[{start},{end})" for start, end in intervals)
				print(f"[{progress}/{len(pending)}] {row.sequence_file}: {interval_text}")

		summary = {
			"trajectory_count": connection.execute(
				"SELECT COUNT(*) FROM trajectory_status"
			).fetchone()[0],
			"prediction_count": connection.execute(
				"SELECT COUNT(*) FROM predictions"
			).fetchone()[0],
		}
		print(f"推理完成: {json.dumps(summary, ensure_ascii=False)}")
		print(f"结果数据库: {args.output}")
	finally:
		connection.close()


if __name__ == "__main__":
	main()
