import argparse
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
from models.rom.pngv import PNGVModel
from models.rom.spm import SPMModel
from models.soc_predictor.stateless_soc_pred import StatelessSoCPredictor
from scripts.condition_restorer.infer_cr_rom import (
	LOSS_HORIZON,
	load_cr,
	sample_loss_intervals,
	trajectory_rng,
)
from scripts.rom.inference_common import (
	SCHEMA_VERSION,
	_checkpoint_sha256,
	_initialize_database,
	_read_sequence,
	_validate_run_metadata,
	_write_trajectory,
)
from scripts.soc_predictor.infer_stateless_soc_pred import (
	checkpoint_training_value,
	load_model,
	select_device,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run ConditionRestorer + PNGV/SPM + SOC predictor inference with "
			"simulated packet loss."
		)
	)
	parser.add_argument("--rom", choices=("pngv", "spm"), required=True)
	parser.add_argument(
		"--rom-checkpoint", type=Path, default=None,
		help="Defaults to checkpoints/rom/<rom>.npz",
	)
	parser.add_argument(
		"--soc-checkpoint", type=Path,
		default=ROOT / "checkpoints" / "stateless_soc_predictor" / "best.pt",
	)
	parser.add_argument(
		"--cr-checkpoint", type=Path,
		default=ROOT / "checkpoints" / "condition_restorer" / "cr_best.pt",
	)
	parser.add_argument(
		"--output", type=Path, default=None,
		help=(
			"Defaults to results/condition_restorer/"
			"cr_<rom>_soc_pred_t<T>_infer_test.db"
		),
	)
	parser.add_argument(
		"--rom-output", type=Path, default=None,
		help=(
			"Defaults to the same directory as --output, with a "
			"rom_<rom>_random_current suffix"
		),
	)
	parser.add_argument("--correction-interval", "-T", type=int, required=True)
	parser.add_argument("--loss-count", type=int, required=True)
	parser.add_argument("--max-loss-length", type=int, required=True)
	parser.add_argument("--test-dir", type=Path, default=DATA_ROOT / "test")
	parser.add_argument(
		"--manifest", type=Path,
		default=DATA_ROOT / "metadata" / "test_manifest.csv",
	)
	parser.add_argument("--min-window", type=int, default=None)
	parser.add_argument("--max-window", type=int, default=None)
	parser.add_argument("--seed", type=int, default=20260901)
	parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
	parser.add_argument("--no-amp", action="store_true")
	parser.add_argument("--limit", type=int, default=None)
	return parser.parse_args()


def validate_args(args: argparse.Namespace, rom_checkpoint: Path,
				  output_path: Path, rom_output_path: Path) -> None:
	if args.correction_interval <= 0:
		raise ValueError("--correction-interval/-T 必须为正整数")
	if args.loss_count <= 0:
		raise ValueError("--loss-count 必须为正整数")
	if not 1 <= args.max_loss_length < 7:
		raise ValueError("--max-loss-length 必须在 [1, 6] 内")
	if args.limit is not None and args.limit <= 0:
		raise ValueError("--limit 必须为正整数")
	for path, label in (
		(rom_checkpoint, "ROM checkpoint"),
		(args.soc_checkpoint, "SOC predictor checkpoint"),
		(args.cr_checkpoint, "CR checkpoint"),
		(args.manifest, "测试清单"),
	):
		if not path.exists():
			raise FileNotFoundError(f"找不到{label}: {path}")
	for path, label in ((output_path, "--output"), (rom_output_path, "--rom-output")):
		if path.suffix.lower() != ".db":
			raise ValueError(f"{label} 必须是 .db 文件")


class OnlineSOCCorrector:
	def __init__(self, model: StatelessSoCPredictor, checkpoint: dict,
				 device: torch.device, min_window: int, max_window: int,
				 use_amp: bool):
		normalization = checkpoint.get("normalization")
		if not normalization:
			raise ValueError("SOC predictor checkpoint 缺少 normalization")
		self.model = model
		self.device = device
		self.min_window = min_window
		self.max_window = max_window
		self.use_amp = use_amp
		self.calibration_steps = int(model.calibration_steps)
		self.soc_min = float(normalization["soc_min"])
		self.soc_max = float(normalization["soc_max"])
		self.soc_range = self.soc_max - self.soc_min
		self.rate_to_current = float(normalization.get("rate_to_current", 125.0))
		if self.soc_range <= 0 or self.rate_to_current <= 0:
			raise ValueError("SOC predictor checkpoint 的归一化参数无效")

	def normalize_soc(self, soc_fraction: float) -> float:
		return (soc_fraction * 100.0 - self.soc_min) / self.soc_range

	def predict(self, target: int, anchor_step: int, anchor_soc: float,
				source_soc: np.ndarray, effective_rate: np.ndarray,
				effective_voltage: np.ndarray) -> float | None:
		window_start = max(anchor_step, target - self.max_window + 1)
		window_length = target - window_start + 1
		if window_length < self.min_window:
			return None

		if window_start == anchor_step:
			soc0 = anchor_soc
		else:
			soc0 = float(source_soc[window_start])
		voltage = torch.from_numpy(
			effective_voltage[window_start:target + 1].astype(np.float32)
		).unsqueeze(0).to(self.device)
		current = torch.from_numpy(
			(
				effective_rate[window_start:target + 1] * self.rate_to_current
			).astype(np.float32)
		).unsqueeze(0).to(self.device)
		soc0 = torch.tensor(
			[self.normalize_soc(soc0)], dtype=torch.float32, device=self.device,
		)
		with torch.inference_mode(), torch.autocast(
			device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp,
		):
			prediction = self.model(
				voltage, current, soc0,
				torch.tensor([window_length], dtype=torch.long, device=self.device),
			)
			prediction_normalized = float(
				prediction["soc"][0, -1].float().cpu().item()
			)
		corrected_soc = (
			prediction_normalized * self.soc_range + self.soc_min
		) / 100.0
		if not np.isfinite(corrected_soc):
			raise RuntimeError(f"SOC predictor 在 step={target} 输出了非有限值")
		return float(np.clip(corrected_soc, 0.0, 1.0))


class ThreeModelPredictor:
	def __init__(self, cr_model: ConditionRestorer, rom_model,
				 soc_corrector: OnlineSOCCorrector, rom_name: str,
				 cr_config: dict, device: torch.device, use_amp: bool):
		self.cr_model = cr_model
		self.rom_model = rom_model
		self.soc_corrector = soc_corrector
		self.rom_name = rom_name
		self.pre_len = int(cr_config["pre_len"])
		self.sub_len = int(cr_config["sub_len"])
		self.max_missing_len = int(cr_config["max_missing_len"])
		self.device = device
		self.use_amp = use_amp

	def restore_rate(self, start: int, end: int, source_rate: np.ndarray,
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

	def initialize_rom(self, initial_soc: float, initial_voltage: float,
					   initial_rate: float) -> np.ndarray:
		if self.rom_name == "pngv":
			ocv = self.rom_model._ocv_from_soc(initial_soc, self.rom_model.params)
			vp = ocv - initial_voltage - self.rom_model.params[4] * initial_rate
			return np.array([initial_soc, vp], dtype=np.float64)
		self.rom_model.reset_state(initial_soc)
		return self.rom_model.get_state()

	def transition(self, state: np.ndarray, rate: float) -> np.ndarray:
		if self.rom_name == "pngv":
			return self.rom_model._state_transition_pngv(
				state, rate, self.rom_model.params,
			)
		return self.rom_model._state_transition(state, rate, self.rom_model.params)

	def measurement(self, state: np.ndarray, rate: float) -> float:
		if self.rom_name == "pngv":
			return float(self.rom_model._measurement_pngv(
				state, rate, self.rom_model.params,
			))
		return float(self.rom_model._measurement(
			state, rate, self.rom_model.params,
		))

	def __call__(self, frame: pd.DataFrame, intervals: list[tuple[int, int]],
				 correction_interval: int):
		source_rate = frame["rate"].to_numpy(dtype=np.float64)
		source_voltage = frame["voltage"].to_numpy(dtype=np.float64)
		source_soc = frame["soc"].to_numpy(dtype=np.float64)
		effective_rate = source_rate.copy()
		effective_voltage = source_voltage.copy()
		soc_pred = np.empty(len(frame), dtype=np.float64)
		voltage_pred = np.empty(len(frame), dtype=np.float64)
		soc_pred[0] = source_soc[0]
		voltage_pred[0] = source_voltage[0]
		state = self.initialize_rom(soc_pred[0], voltage_pred[0], source_rate[0])

		interval_by_start = {start: end for start, end in intervals}
		interval_ends = {end for _, end in intervals}
		loss_mask = np.zeros(len(frame), dtype=bool)
		for start, end in intervals:
			loss_mask[start:end] = True
		anchor_step = 0
		anchor_soc = soc_pred[0]
		next_correction_step = correction_interval
		correction_count = 0

		for step in range(len(frame) - 1):
			next_step = step + 1
			if next_step in interval_by_start:
				end = interval_by_start[next_step]
				effective_rate[next_step:end] = self.restore_rate(
					next_step, end, source_rate, source_voltage, source_soc,
					effective_rate,
				)

			state = self.transition(state, effective_rate[step])
			if next_step in interval_ends:
				anchor_step = next_step
				anchor_soc = float(state[0])
				next_correction_step = next_step + correction_interval
			if (
				next_step == next_correction_step
				and not loss_mask[next_step]
				and next_step not in interval_ends
			):
				corrected_soc = self.soc_corrector.predict(
					next_step, anchor_step, anchor_soc, source_soc,
					effective_rate, effective_voltage,
				)
				if corrected_soc is not None:
					state[0] = corrected_soc
					correction_count += 1
				next_correction_step += correction_interval

			soc_pred[next_step] = state[0]
			voltage_pred[next_step] = self.measurement(
				state, effective_rate[next_step],
			)
			if loss_mask[next_step]:
				effective_voltage[next_step] = voltage_pred[next_step]

		if not all(np.isfinite(values).all() for values in (
			soc_pred, voltage_pred, effective_rate, effective_voltage,
		)):
			raise RuntimeError("三模型联合推理输出包含非有限值")
		return soc_pred, voltage_pred, effective_rate, correction_count


class RandomCurrentROMPredictor:
	def __init__(self, rom_model, rom_name: str):
		self.rom_model = rom_model
		self.rom_name = rom_name

	def initialize_rom(self, initial_soc: float, initial_voltage: float,
					   initial_rate: float) -> np.ndarray:
		if self.rom_name == "pngv":
			ocv = self.rom_model._ocv_from_soc(initial_soc, self.rom_model.params)
			vp = ocv - initial_voltage - self.rom_model.params[4] * initial_rate
			return np.array([initial_soc, vp], dtype=np.float64)
		self.rom_model.reset_state(initial_soc)
		return self.rom_model.get_state()

	def transition(self, state: np.ndarray, rate: float) -> np.ndarray:
		if self.rom_name == "pngv":
			return self.rom_model._state_transition_pngv(
				state, rate, self.rom_model.params,
			)
		return self.rom_model._state_transition(state, rate, self.rom_model.params)

	def measurement(self, state: np.ndarray, rate: float) -> float:
		if self.rom_name == "pngv":
			return float(self.rom_model._measurement_pngv(
				state, rate, self.rom_model.params,
			))
		return float(self.rom_model._measurement(
			state, rate, self.rom_model.params,
		))

	def __call__(self, frame: pd.DataFrame, intervals: list[tuple[int, int]],
				 random_rates: np.ndarray):
		source_rate = frame["rate"].to_numpy(dtype=np.float64)
		source_soc = frame["soc"].to_numpy(dtype=np.float64)
		source_voltage = frame["voltage"].to_numpy(dtype=np.float64)
		effective_rate = source_rate.copy()
		for start, end in intervals:
			effective_rate[start:end] = random_rates[start:end]

		soc_pred = np.empty(len(frame), dtype=np.float64)
		voltage_pred = np.empty(len(frame), dtype=np.float64)
		soc_pred[0] = source_soc[0]
		voltage_pred[0] = source_voltage[0]
		state = self.initialize_rom(
			soc_pred[0], voltage_pred[0], source_rate[0],
		)
		for step in range(len(frame) - 1):
			state = self.transition(state, effective_rate[step])
			next_step = step + 1
			soc_pred[next_step] = state[0]
			voltage_pred[next_step] = self.measurement(
				state, effective_rate[next_step],
			)
		if not np.isfinite(soc_pred).all() or not np.isfinite(voltage_pred).all():
			raise RuntimeError("随机电流 ROM 输出包含非有限值")
		return soc_pred, voltage_pred, effective_rate


def main() -> None:
	args = parse_args()
	rom_checkpoint = args.rom_checkpoint or (
		ROOT / "checkpoints" / "rom" / f"{args.rom}.npz"
	)
	output_path = args.output or (
		ROOT / "results" / "condition_restorer"
		/ f"cr_{args.rom}_soc_pred_t{args.correction_interval}_infer_test.db"
	)
	rom_output_path = args.rom_output or (
		output_path.parent
		/ f"rom_{args.rom}_random_current_t{args.correction_interval}_infer_test.db"
	)
	validate_args(args, rom_checkpoint, output_path, rom_output_path)
	device = select_device(args.device)
	cr_model, cr_config, cr_checkpoint = load_cr(args.cr_checkpoint, device)
	if args.max_loss_length > int(cr_config["max_missing_len"]):
		raise ValueError(
			"--max-loss-length 超过 CR checkpoint 的 max_missing_len="
			f"{cr_config['max_missing_len']}"
		)
	soc_model, soc_checkpoint = load_model(args.soc_checkpoint, device)
	min_window = args.min_window or int(
		checkpoint_training_value(soc_checkpoint, "min_window")
	)
	max_window = args.max_window or int(
		checkpoint_training_value(soc_checkpoint, "max_window")
	)
	if min_window < int(soc_model.calibration_steps) or max_window < min_window:
		raise ValueError(
			"窗口长度必须满足 calibration_steps <= min-window <= max-window"
		)
	if max_window > LOSS_HORIZON:
		raise ValueError(f"--max-window 不能超过 {LOSS_HORIZON}")

	rom_sha256 = _checkpoint_sha256(rom_checkpoint)
	trained_rom_sha256 = cr_checkpoint.get("spm_checkpoint_sha256")
	if args.rom == "spm" and trained_rom_sha256 and trained_rom_sha256 != rom_sha256:
		raise ValueError("推理 SPM checkpoint 与 CR 训练时使用的 SPM 不一致")
	rom_model = (
		PNGVModel.load_checkpoint(rom_checkpoint)
		if args.rom == "pngv"
		else SPMModel.load_checkpoint(rom_checkpoint)
	)
	rom_only_model = (
		PNGVModel.load_checkpoint(rom_checkpoint)
		if args.rom == "pngv"
		else SPMModel.load_checkpoint(rom_checkpoint)
	)
	use_amp = device.type == "cuda" and not args.no_amp
	soc_corrector = OnlineSOCCorrector(
		soc_model, soc_checkpoint, device, min_window, max_window, use_amp,
	)
	predictor = ThreeModelPredictor(
		cr_model, rom_model, soc_corrector, args.rom, cr_config, device, use_amp,
	)
	rom_only_predictor = RandomCurrentROMPredictor(rom_only_model, args.rom)

	manifest = pd.read_csv(args.manifest, usecols=["sequence_file", "condition_code"])
	if args.limit is not None:
		manifest = manifest.iloc[:args.limit]
	output_path.parent.mkdir(parents=True, exist_ok=True)
	rom_output_path.parent.mkdir(parents=True, exist_ok=True)
	connection = sqlite3.connect(output_path)
	rom_connection = sqlite3.connect(rom_output_path)
	try:
		_initialize_database(connection)
		_initialize_database(rom_connection)
		metadata = {
			"schema_version": SCHEMA_VERSION,
			"model_name": f"condition_restorer_{args.rom}_soc_predictor",
			"checkpoint_path": str(rom_checkpoint.resolve()),
			"checkpoint_sha256": rom_sha256,
			"soc_predictor_checkpoint_path": str(args.soc_checkpoint.resolve()),
			"soc_predictor_checkpoint_sha256": _checkpoint_sha256(args.soc_checkpoint),
			"cr_checkpoint_path": str(args.cr_checkpoint.resolve()),
			"cr_checkpoint_sha256": _checkpoint_sha256(args.cr_checkpoint),
			"soc_scale": "0_to_1",
			"current_condition": "rate_with_cr_restoration_at_packet_losses",
			"inference_mode": "cr_then_rom_with_periodic_soc_predictor_correction",
			"correction_interval_steps": str(args.correction_interval),
			"min_window": str(min_window),
			"max_window": str(max_window),
			"soc0_source_normal_window": "ground_truth_at_shifted_window_start",
			"soc0_source_after_loss": "rom_recovered_soc_at_loss_end",
			"predictor_restart": "at_each_loss_end_with_correction_clock_reset",
			"rom_state_correction": "replace_soc_only_preserve_internal_dynamic_states",
			"loss_count_per_trajectory": str(args.loss_count),
			"max_loss_length": str(args.max_loss_length),
			"loss_length_distribution": "discrete_uniform_1_to_max_inclusive",
			"loss_horizon_steps_exclusive": str(LOSS_HORIZON),
			"loss_seed": str(args.seed),
			"loss_seed_scope": "sha256(global_seed:sequence_file)",
			"stored_rate": "effective_rom_input_observed_or_cr_restored",
			"cr_pre_context": "observed_soc_voltage_and_effective_rate",
			"cr_sub_context": "observed_rate_and_voltage_after_loss",
			"rate_to_current": str(soc_corrector.rate_to_current),
		}
		rom_metadata = {
			"schema_version": SCHEMA_VERSION,
			"model_name": f"{args.rom}_random_current_rom",
			"checkpoint_path": str(rom_checkpoint.resolve()),
			"checkpoint_sha256": rom_sha256,
			"soc_scale": "0_to_1",
			"current_condition": "random_rate_at_packet_losses",
			"inference_mode": "autoregressive_rom_from_initial_frame",
			"loss_count_per_trajectory": str(args.loss_count),
			"max_loss_length": str(args.max_loss_length),
			"loss_length_distribution": "discrete_uniform_1_to_max_inclusive",
			"loss_horizon_steps_exclusive": str(LOSS_HORIZON),
			"loss_seed": str(args.seed),
			"loss_seed_scope": "sha256(global_seed:sequence_file)",
			"random_rate_distribution": "uniform_per_trajectory_observed_min_max",
			"stored_rate": "observed_or_random_at_packet_losses",
		}
		_validate_run_metadata(connection, metadata)
		_validate_run_metadata(rom_connection, rom_metadata)
		completed = {
			row[0] for row in connection.execute("SELECT sequence_file FROM trajectory_status")
		}
		rom_completed = {
			row[0]
			for row in rom_connection.execute("SELECT sequence_file FROM trajectory_status")
		}
		pending = manifest[
			~manifest["sequence_file"].isin(completed & rom_completed)
		]
		print(
			f"ROM: {args.rom}; device: {device}; AMP: {use_amp}; "
			f"T: {args.correction_interval}; loss count: {args.loss_count}; "
			f"max loss length: {args.max_loss_length}"
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
			rate_rng = trajectory_rng(
				args.seed, f"{row.sequence_file}:random-current",
			)
			source_rate = frame["rate"].to_numpy(dtype=np.float64)
			random_rates = rate_rng.uniform(
				float(source_rate.min()), float(source_rate.max()), len(frame),
			)
			soc_pred, voltage_pred, effective_rate, correction_count = predictor(
				frame, intervals, args.correction_interval,
			)
			rom_soc_pred, rom_voltage_pred, rom_effective_rate = rom_only_predictor(
				frame, intervals, random_rates,
			)
			output_frame = frame.copy()
			output_frame["rate"] = effective_rate
			_write_trajectory(
				connection, row.sequence_file, row.condition_code, output_frame,
				soc_pred, voltage_pred,
			)
			rom_output_frame = frame.copy()
			rom_output_frame["rate"] = rom_effective_rate
			_write_trajectory(
				rom_connection, row.sequence_file, row.condition_code,
				rom_output_frame, rom_soc_pred, rom_voltage_pred,
			)
			if progress == 1 or progress % 25 == 0 or progress == len(pending):
				interval_text = ", ".join(f"[{start},{end})" for start, end in intervals)
				print(
					f"[{progress}/{len(pending)}] {row.sequence_file}: "
					f"loss={interval_text}; corrections={correction_count}; "
					f"random_rate_range=[{random_rates.min():.6g}, "
					f"{random_rates.max():.6g}]"
				)

		summary = {
			"cooperative": {
			"trajectory_count": connection.execute(
				"SELECT COUNT(*) FROM trajectory_status"
			).fetchone()[0],
			"prediction_count": connection.execute(
				"SELECT COUNT(*) FROM predictions"
			).fetchone()[0],
			},
			"rom_random_current": {
				"trajectory_count": rom_connection.execute(
					"SELECT COUNT(*) FROM trajectory_status"
				).fetchone()[0],
				"prediction_count": rom_connection.execute(
					"SELECT COUNT(*) FROM predictions"
				).fetchone()[0],
			},
		}
		print(f"推理完成: {json.dumps(summary, ensure_ascii=False)}")
		print(f"协作推理数据库: {output_path}")
		print(f"随机电流 ROM 数据库: {rom_output_path}")
	finally:
		connection.close()
		rom_connection.close()


if __name__ == "__main__":
	main()
