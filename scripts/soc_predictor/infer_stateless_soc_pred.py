import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "split_data"
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from models.soc_predictor.stateless_soc_pred import StatelessSoCPredictor
from scripts.rom.inference_common import run_soc_test_inference


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run causal block inference with StatelessSoCPredictor."
	)
	parser.add_argument(
		"--checkpoint", type=Path,
		default=(
			ROOT / "checkpoints" / "stateless_soc_predictor"
			/ "best.pt"
		)
	)
	parser.add_argument(
		"--output", type=Path,
		default=(
			ROOT / "results" / "stateless_soc_predictor"
			/ "stateless_soc_pred_infer_test.db"
		)
	)
	parser.add_argument("--test-dir", type=Path, default=DATA_ROOT / "test")
	parser.add_argument(
		"--manifest", type=Path,
		default=DATA_ROOT / "metadata" / "test_manifest.csv"
	)
	parser.add_argument("--min-window", type=int, default=None)
	parser.add_argument("--max-window", type=int, default=None)
	parser.add_argument(
		"--overlap", type=int, default=None,
		help="Context points shared by consecutive blocks; defaults to min-window"
	)
	parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
	parser.add_argument("--no-amp", action="store_true")
	parser.add_argument("--limit", type=int, default=None)
	return parser.parse_args()


def select_device(requested: str) -> torch.device:
	if requested == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(requested)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is unavailable")
	return device


def checkpoint_training_value(checkpoint: dict, key: str, default=None):
	value = checkpoint.get("training_config", {}).get(key, default)
	if value is None:
		raise ValueError(f"Checkpoint is missing training parameter: {key}")
	return value


def load_model(checkpoint_path: Path, device: torch.device):
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	model_config = checkpoint.get("model_config")
	if not model_config:
		raise ValueError("Checkpoint is missing model_config")
	model_class = checkpoint.get("model_class")
	if model_class not in (None, "StatelessSoCPredictor", "TrajectorySOCPredictor"):
		raise ValueError(f"Unsupported checkpoint model_class: {model_class}")
	model = StatelessSoCPredictor(**model_config)
	missing_keys, unexpected_keys = model.load_state_dict(
		checkpoint["model_state_dict"], strict=False
	)
	legacy_gate_keys = {
		"physics_gain_head.0.weight", "physics_gain_head.0.bias",
		"physics_gain_head.2.weight", "physics_gain_head.2.bias",
		"current_correction_head.weight", "current_correction_head.bias"
	}
	if unexpected_keys or set(missing_keys) - legacy_gate_keys:
		raise ValueError(
			"Checkpoint parameters do not match StatelessSoCPredictor: "
			f"missing={missing_keys}, unexpected={unexpected_keys}"
		)
	model.to(device)
	model.eval()
	return model, checkpoint


class BlockwiseStatelessSoCPredictor:
	def __init__(self, model: StatelessSoCPredictor, checkpoint: dict,
				 device: torch.device, min_window: int, max_window: int,
				 overlap: int, use_amp: bool):
		normalization = checkpoint.get("normalization")
		if not normalization:
			raise ValueError("Checkpoint is missing normalization")

		self.model = model
		self.device = device
		self.min_window = min_window
		self.max_window = max_window
		self.overlap = overlap
		self.use_amp = use_amp
		self.soc_min = float(normalization["soc_min"])
		self.soc_max = float(normalization["soc_max"])
		self.soc_range = self.soc_max - self.soc_min
		self.rate_to_current = float(normalization.get("rate_to_current", 125.0))
		if self.soc_range <= 0 or self.rate_to_current <= 0:
			raise ValueError("Checkpoint normalization parameters are invalid")

	def normalize_soc_fraction(self, soc_fraction: float) -> float:
		return (soc_fraction * 100.0 - self.soc_min) / self.soc_range

	def denormalize_soc_fraction(self, soc_normalized: np.ndarray) -> np.ndarray:
		soc_percent = soc_normalized * self.soc_range + self.soc_min
		return np.clip(soc_percent / 100.0, 0.0, 1.0)

	def _predict_block(self, voltage: np.ndarray, current: np.ndarray,
					   start_soc: float) -> np.ndarray:
		voltage_tensor = torch.from_numpy(voltage).to(
			self.device, non_blocking=True
		).unsqueeze(0)
		current_tensor = torch.from_numpy(current).to(
			self.device, non_blocking=True
		).unsqueeze(0)
		soc0_tensor = torch.tensor(
			[start_soc], dtype=torch.float32, device=self.device
		)
		lengths = torch.tensor(
			[len(voltage)], dtype=torch.long, device=self.device
		)
		with torch.inference_mode(), torch.autocast(
			device_type=self.device.type,
			dtype=torch.float16,
			enabled=self.use_amp
		):
			output = self.model(
				voltage_tensor, current_tensor, soc0_tensor, lengths
			)
		return output["soc"][0].float().cpu().numpy()

	def __call__(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
		trajectory_length = len(frame)
		if trajectory_length < self.min_window:
			raise ValueError(
				f"Trajectory length {trajectory_length} is less than "
				f"min-window={self.min_window}"
			)

		voltage = frame["voltage"].to_numpy(dtype=np.float32)
		current = (
			frame["rate"].to_numpy(dtype=np.float32) * self.rate_to_current
		)
		prediction = np.full(trajectory_length, np.nan, dtype=np.float32)
		prediction[0] = self.normalize_soc_fraction(float(frame["soc"].iloc[0]))
		first_step = self.min_window - 1
		next_step = first_step

		while next_step < trajectory_length:
			if next_step == first_step:
				window_start = 0
			else:
				window_start = max(first_step, next_step - self.overlap)
			window_end = min(window_start + self.max_window, trajectory_length)
			start_soc = prediction[window_start]
			if not np.isfinite(start_soc):
				raise RuntimeError(
					f"The SOC required at block start step={window_start} is unavailable"
				)

			block_prediction = self._predict_block(
				voltage[window_start:window_end],
				current[window_start:window_end],
				float(start_soc)
			)
			local_output_start = next_step - window_start
			prediction[next_step:window_end] = block_prediction[local_output_start:]
			next_step = window_end

		steps = np.arange(first_step, trajectory_length, dtype=np.int64)
		selected_prediction = prediction[steps]
		if not np.isfinite(selected_prediction).all():
			raise RuntimeError("SOC predictor output contains non-finite values")
		return steps, self.denormalize_soc_fraction(selected_prediction)


def validate_args(args: argparse.Namespace) -> None:
	if not args.checkpoint.exists():
		raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
	if not args.manifest.exists():
		raise FileNotFoundError(f"Test manifest not found: {args.manifest}")
	if not args.test_dir.is_dir():
		raise FileNotFoundError(f"Test directory not found: {args.test_dir}")
	if args.limit is not None and args.limit <= 0:
		raise ValueError("--limit must be positive")
	if args.output.suffix.lower() != ".db":
		raise ValueError("--output must be a .db file")


def main() -> None:
	args = parse_args()
	validate_args(args)
	device = select_device(args.device)
	model, checkpoint = load_model(args.checkpoint, device)
	min_window = args.min_window or int(
		checkpoint_training_value(checkpoint, "min_window")
	)
	max_window = args.max_window or int(
		checkpoint_training_value(checkpoint, "max_window")
	)
	overlap = args.overlap or min_window
	calibration_steps = int(model.calibration_steps)
	if min_window < calibration_steps or max_window < min_window:
		raise ValueError(
			"Window lengths must satisfy calibration-steps <= min-window <= max-window"
		)
	if overlap < min_window or overlap >= max_window:
		raise ValueError("overlap must satisfy min-window <= overlap < max-window")

	use_amp = device.type == "cuda" and not args.no_amp
	predictor = BlockwiseStatelessSoCPredictor(
		model, checkpoint, device, min_window, max_window, overlap, use_amp
	)
	print(
		f"Device: {device}; AMP: {use_amp}; window: {min_window}-{max_window}; "
		f"overlap: {overlap}; stride: {max_window - overlap}"
	)
	run_soc_test_inference(
		model_name="stateless_soc_predictor",
		checkpoint_path=args.checkpoint,
		output_path=args.output,
		test_dir=args.test_dir,
		manifest_path=args.manifest,
		predict=predictor,
		metadata={
			"inference_mode": "causal_overlapping_sequence_blocks",
			"min_window": str(min_window),
			"max_window": str(max_window),
			"calibration_steps": str(calibration_steps),
			"block_overlap": str(overlap),
			"block_stride": str(max_window - overlap),
			"rolling_soc0_source": "previous_block_prediction",
			"current_correction": str(model.use_current_correction),
			"rate_to_current": str(predictor.rate_to_current),
			"step_definition": "zero_based_source_frame",
			"soc_output": "checkpoint_inverse_normalized_then_divided_by_100"
		},
		limit=args.limit
	)


if __name__ == "__main__":
	main()
