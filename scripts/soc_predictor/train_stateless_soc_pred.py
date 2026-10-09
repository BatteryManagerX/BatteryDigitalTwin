import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "split_data"
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from models.soc_predictor.stateless_soc_pred import StatelessSoCPredictor


@dataclass(frozen=True)
class Normalization:
	soc_min: float
	soc_max: float
	voltage_mean: float
	voltage_std: float
	rate_mean: float
	rate_std: float
	rate_to_current: float = 125.0

	@property
	def soc_range(self) -> float:
		return self.soc_max - self.soc_min

	@property
	def current_mean(self) -> float:
		return self.rate_mean * self.rate_to_current

	@property
	def current_std(self) -> float:
		return self.rate_std * self.rate_to_current

	def normalize_soc(self, values):
		return (values - self.soc_min) / self.soc_range

	def denormalize_soc(self, values):
		return values * self.soc_range + self.soc_min


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train the trajectory-level physics-guided SOC predictor."
	)
	parser.add_argument("--data-dir", type=Path, default=DATA_ROOT / "train")
	parser.add_argument(
		"--manifest", type=Path,
		default=DATA_ROOT / "metadata" / "train_manifest.csv"
	)
	parser.add_argument(
		"--normalization", type=Path,
		default=DATA_ROOT / "normalization_parameters.json"
	)
	parser.add_argument(
		"--output-dir", type=Path,
		default=ROOT / "checkpoints" / "stateless_soc_predictor"
	)
	parser.add_argument("--resume", type=Path, default=None)
	parser.add_argument("--epochs", type=int, default=200)
	parser.add_argument("--batch-size", type=int, default=64,
					help="Number of trajectories loaded per batch")
	parser.add_argument("--windows-per-trajectory", type=int, default=4)
	parser.add_argument("--validation-windows-per-trajectory", type=int, default=4)
	parser.add_argument("--min-window", type=int, default=64)
	parser.add_argument("--max-window", type=int, default=2048)
	parser.add_argument("--prefix-horizon", type=int, default=2048)
	parser.add_argument("--prefix-window-fraction", type=float, default=0.75)
	parser.add_argument("--validation-fraction", type=float, default=0.2)
	parser.add_argument("--split-group-column", default="source_file")
	parser.add_argument("--validation-interval", type=int, default=10)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--grad-clip", type=float, default=1.0)
	parser.add_argument("--patience", type=int, default=100)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--seed", type=int, default=20260901)
	parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
	parser.add_argument("--no-amp", action="store_true")

	parser.add_argument("--hidden-dim", type=int, default=96)
	parser.add_argument("--latent-dim", type=int, default=48)
	parser.add_argument("--conv-layers", type=int, default=3)
	parser.add_argument("--gru-layers", type=int, default=2)
	parser.add_argument("--dropout", type=float, default=0.05)
	parser.add_argument("--calibration-steps", type=int, default=32)
	parser.add_argument("--dt-seconds", type=float, default=10.0)
	parser.add_argument("--cumulative-ah-scale", type=float, default=20.0)
	parser.add_argument("--init-capacity-ah", type=float, default=150.0)
	parser.add_argument("--capacity-log-range", type=float, default=0.0)
	parser.add_argument("--efficiency-log-range", type=float, default=0.0)
	parser.add_argument("--physics-gain-log-range", type=float, default=0.5)
	parser.add_argument(
		"--use-current-correction", action=argparse.BooleanOptionalAction,
		default=True
	)
	parser.add_argument("--current-gain-log-range", type=float, default=0.25)
	parser.add_argument("--current-bias-max-amp", type=float, default=2.0)
	parser.add_argument("--max-dynamic-residual", type=float, default=0.15)
	parser.add_argument("--min-log-variance", type=float, default=-9.0)
	parser.add_argument("--max-log-variance", type=float, default=-2.0)
	parser.add_argument("--time-scale-steps", type=float, default=2048.0)

	parser.add_argument("--endpoint-loss-weight", type=float, default=0.5)
	parser.add_argument("--physical-loss-weight", type=float, default=0.01)
	parser.add_argument("--uncertainty-loss-weight", type=float, default=0.005)
	parser.add_argument("--residual-l2-weight", type=float, default=0.001)
	parser.add_argument("--residual-smoothness-weight", type=float, default=0.01)
	parser.add_argument("--physics-gain-weight", type=float, default=0.002)
	parser.add_argument("--current-gain-weight", type=float, default=0.001)
	parser.add_argument("--current-bias-weight", type=float, default=0.001)
	parser.add_argument("--drift-loss-weight", type=float, default=0.5)
	parser.add_argument("--horizon-loss-weight", type=float, default=2.0)
	return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
	if args.epochs <= 0 or args.batch_size <= 0:
		raise ValueError("--epochs and --batch-size must be positive")
	if args.windows_per_trajectory <= 0 or args.validation_windows_per_trajectory <= 0:
		raise ValueError("window counts must be positive")
	if args.min_window < 2 or args.max_window < args.min_window:
		raise ValueError("window lengths must satisfy 2 <= min-window <= max-window")
	if args.calibration_steps < 2 or args.min_window < args.calibration_steps:
		raise ValueError("min-window must be at least calibration-steps >= 2")
	if args.prefix_horizon < args.min_window:
		raise ValueError("prefix-horizon cannot be less than min-window")
	if not 0.0 <= args.prefix_window_fraction <= 1.0:
		raise ValueError("prefix-window-fraction must be between 0 and 1")
	if not 0.0 < args.validation_fraction < 1.0:
		raise ValueError("validation-fraction must be between 0 and 1")
	if args.validation_interval <= 0:
		raise ValueError("validation-interval must be positive")
	if args.learning_rate <= 0 or args.weight_decay < 0:
		raise ValueError("learning-rate must be positive and weight-decay non-negative")
	if args.grad_clip <= 0 or args.patience <= 0:
		raise ValueError("grad-clip and patience must be positive")
	if args.physics_gain_log_range <= 0:
		raise ValueError("--physics-gain-log-range must be positive")
	if args.current_gain_log_range <= 0 or args.current_bias_max_amp <= 0:
		raise ValueError("current gain range and bias limit must be positive")
	if args.horizon_loss_weight < 1.0:
		raise ValueError("--horizon-loss-weight must be at least 1")
	if args.time_scale_steps <= 0:
		raise ValueError("time-scale-steps must be positive")
	loss_weights = (
		args.endpoint_loss_weight,
		args.physical_loss_weight,
		args.uncertainty_loss_weight,
		args.residual_l2_weight,
		args.residual_smoothness_weight,
		args.physics_gain_weight,
		args.current_gain_weight,
		args.current_bias_weight,
		args.drift_loss_weight
	)
	if any(weight < 0 for weight in loss_weights):
		raise ValueError("loss weights must be non-negative")


def load_normalization(path: Path) -> Normalization:
	with path.open("r", encoding="utf-8") as file:
		payload = json.load(file)
	fields = payload["fields"]
	normalization = Normalization(
		soc_min=float(fields["soc"]["min"]),
		soc_max=float(fields["soc"]["max"]),
		voltage_mean=float(fields["voltage"]["mean"]),
		voltage_std=float(fields["voltage"]["std_population"]),
		rate_mean=float(fields["rate"]["mean"]),
		rate_std=float(fields["rate"]["std_population"])
	)
	if normalization.soc_range <= 0:
		raise ValueError("SOC normalization range must be positive")
	if normalization.voltage_std <= 0 or normalization.rate_std <= 0:
		raise ValueError("voltage and rate standard deviations must be positive")
	return normalization


def stratified_split(manifest_path: Path, validation_fraction: float,
					 seed: int, group_column: str | None = None
					 ) -> tuple[pd.DataFrame, pd.DataFrame]:
	manifest = pd.read_csv(manifest_path)
	required = {"sequence_file", "condition_code", "records"}
	missing = required - set(manifest.columns)
	if missing:
		raise ValueError(f"Training manifest is missing: {', '.join(sorted(missing))}")
	if group_column and group_column.lower() != "none":
		if group_column not in manifest.columns:
			raise ValueError(f"Training manifest is missing group column: {group_column}")
		groups = manifest[group_column].dropna().unique()
		if len(groups) < 2:
			raise ValueError("Grouped validation split requires at least two groups")
		rng = np.random.default_rng(seed)
		validation_group_count = max(1, int(round(len(groups) * validation_fraction)))
		condition_totals = manifest["condition_code"].value_counts().sort_index()
		target_conditions = condition_totals * validation_fraction
		target_size = len(manifest) * validation_fraction
		best_groups = None
		best_score = float("inf")
		for _ in range(4096):
			candidate = rng.choice(groups, size=validation_group_count, replace=False)
			candidate_mask = manifest[group_column].isin(candidate)
			candidate_counts = manifest.loc[
				candidate_mask, "condition_code"
			].value_counts().reindex(condition_totals.index, fill_value=0)
			condition_error = np.mean(
				((candidate_counts - target_conditions) / target_conditions.clip(lower=1)) ** 2
			)
			size_error = ((candidate_mask.sum() - target_size) / target_size) ** 2
			score = float(condition_error + size_error)
			if score < best_score:
				best_score = score
				best_groups = candidate
		validation_mask = manifest[group_column].isin(best_groups)
		return (
			manifest.loc[~validation_mask].reset_index(drop=True),
			manifest.loc[validation_mask].reset_index(drop=True)
		)

	rng = np.random.default_rng(seed)
	train_indices = []
	validation_indices = []
	condition_groups = list(manifest.groupby("condition_code", sort=True))
	validation_count = max(1, int(round(
		min(len(group) for _, group in condition_groups) * validation_fraction
	)))
	for _, group in condition_groups:
		indices = group.index.to_numpy(copy=True)
		rng.shuffle(indices)
		validation_indices.extend(indices[:validation_count])
		train_indices.extend(indices[validation_count:])

	return (
		manifest.loc[train_indices].reset_index(drop=True),
		manifest.loc[validation_indices].reset_index(drop=True)
	)


class TrajectorySequenceDataset(Dataset):
	def __init__(self, manifest: pd.DataFrame, data_dir: Path,
				 normalization: Normalization, min_window: int, max_window: int,
				 windows_per_trajectory: int, prefix_horizon: int,
				 prefix_window_fraction: float, training: bool, seed: int):
		self.records = manifest[["sequence_file", "records"]].to_dict("records")
		self.data_dir = data_dir
		self.normalization = normalization
		self.min_window = min_window
		self.max_window = max_window
		self.windows_per_trajectory = windows_per_trajectory
		self.prefix_horizon = prefix_horizon
		self.prefix_window_count = int(round(
			windows_per_trajectory * prefix_window_fraction
		))
		self.training = training
		self.seed = seed
		self._random_rng = None

	def __len__(self) -> int:
		return len(self.records)

	def _read_trajectory(self, sequence_file: str):
		frame = pd.read_csv(
			self.data_dir / sequence_file,
			usecols=["soc", "voltage", "rate"],
			dtype={"soc": "float32", "voltage": "float32", "rate": "float32"}
		)
		values = frame[["soc", "voltage", "rate"]].to_numpy(dtype=np.float32)
		if len(values) < 2 or not np.isfinite(values).all():
			raise ValueError(f"Trajectory {sequence_file} is too short or non-finite")
		soc = self.normalization.normalize_soc(values[:, 0]).astype(np.float32)
		voltage = values[:, 1]
		current = values[:, 2] * self.normalization.rate_to_current
		return soc, voltage, current

	def _rng(self, index: int) -> np.random.Generator:
		if self.training:
			if self._random_rng is None:
				worker_seed = torch.initial_seed() % (2 ** 32)
				self._random_rng = np.random.default_rng(worker_seed + self.seed)
			return self._random_rng
		return np.random.default_rng(self.seed + index)

	@staticmethod
	def _sample_window_length(rng: np.random.Generator, minimum: int,
							  maximum: int, stratum_index: int,
							  stratum_count: int) -> int:
		if stratum_count <= 1:
			return int(rng.integers(minimum, maximum + 1))
		boundaries = np.rint(np.geomspace(
			minimum, maximum + 1, stratum_count + 1
		)).astype(np.int64)
		boundaries[0] = minimum
		boundaries[-1] = maximum + 1
		boundaries = np.maximum.accumulate(boundaries)
		lower = min(int(boundaries[stratum_index]), maximum)
		upper = min(max(int(boundaries[stratum_index + 1]), lower + 1), maximum + 1)
		return int(rng.integers(lower, upper))

	def __getitem__(self, index: int):
		record = self.records[index]
		soc, voltage, current = self._read_trajectory(record["sequence_file"])
		trajectory_length = len(soc)
		minimum = min(self.min_window, trajectory_length)
		maximum = min(self.max_window, trajectory_length)
		if minimum < self.min_window:
			raise ValueError(
				f"Trajectory {record['sequence_file']} is shorter than min-window"
			)

		rng = self._rng(index)
		windows = []
		for window_index in range(self.windows_per_trajectory):
			is_prefix = window_index < self.prefix_window_count
			if is_prefix:
				stratum_index = window_index
				stratum_count = self.prefix_window_count
				window_maximum = min(maximum, self.prefix_horizon)
			else:
				stratum_index = window_index - self.prefix_window_count
				stratum_count = self.windows_per_trajectory - self.prefix_window_count
				window_maximum = maximum
			window_length = self._sample_window_length(
				rng, minimum, window_maximum, stratum_index, stratum_count
			)
			start = 0 if is_prefix else int(
				rng.integers(0, trajectory_length - window_length + 1)
			)
			end = start + window_length
			windows.append((
				torch.from_numpy(voltage[start:end].copy()),
				torch.from_numpy(current[start:end].copy()),
				torch.tensor(soc[start], dtype=torch.float32),
				torch.from_numpy(soc[start:end].copy())
			))
		return windows


def collate_sequences(batch):
	windows = [window for trajectory_windows in batch for window in trajectory_windows]
	voltage, current, soc0, target = zip(*windows)
	lengths = torch.tensor([len(sequence) for sequence in voltage], dtype=torch.long)
	return (
		pad_sequence(voltage, batch_first=True),
		pad_sequence(current, batch_first=True),
		torch.stack(soc0),
		pad_sequence(target, batch_first=True),
		lengths
	)


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
	if requested == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(requested)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is unavailable")
	return device


def build_model(args: argparse.Namespace, normalization: Normalization):
	model_config = {
		"hidden_dim": args.hidden_dim,
		"latent_dim": args.latent_dim,
		"conv_layers": args.conv_layers,
		"gru_layers": args.gru_layers,
		"dropout": args.dropout,
		"calibration_steps": args.calibration_steps,
		"dt_seconds": args.dt_seconds,
		"voltage_mean": normalization.voltage_mean,
		"voltage_std": normalization.voltage_std,
		"rate_mean": normalization.current_mean,
		"rate_std": normalization.current_std,
		"cumulative_ah_scale": args.cumulative_ah_scale,
		"init_capacity_ah": args.init_capacity_ah,
		"capacity_log_range": args.capacity_log_range,
		"efficiency_log_range": args.efficiency_log_range,
		"physics_gain_log_range": args.physics_gain_log_range,
		"use_current_correction": args.use_current_correction,
		"current_gain_log_range": args.current_gain_log_range,
		"current_bias_max_amp": args.current_bias_max_amp,
		"max_dynamic_residual": args.max_dynamic_residual,
		"min_log_variance": args.min_log_variance,
		"max_log_variance": args.max_log_variance,
		"time_scale_steps": args.time_scale_steps
	}
	return StatelessSoCPredictor(**model_config), model_config


def gather_endpoints(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
	return values.gather(1, (lengths - 1).view(-1, 1)).squeeze(1)


def mean_per_sequence(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	masked_values = values * mask.to(values.dtype)
	counts = mask.sum(dim=1).clamp(min=1)
	return (masked_values.sum(dim=1) / counts).mean()


def compute_losses(output: dict, target: torch.Tensor, soc0: torch.Tensor,
				   lengths: torch.Tensor, args: argparse.Namespace):
	sequence_length = target.shape[1]
	time_index = torch.arange(sequence_length, device=target.device).view(1, -1)
	calibration_start = output["calibration_lengths"].view(-1, 1) - 1
	loss_mask = output["valid_mask"] & (time_index >= calibration_start)
	relative_horizon = time_index.to(target.dtype) / (
		lengths.view(-1, 1) - 1
	).clamp(min=1).to(target.dtype)
	horizon_weights = 1.0 + (
		args.horizon_loss_weight - 1.0
	) * relative_horizon

	prediction = output["soc"]
	sequence_loss = mean_per_sequence(
		F.smooth_l1_loss(
			prediction, target, beta=0.01, reduction="none"
		) * horizon_weights,
		loss_mask
	)
	endpoint_prediction = gather_endpoints(prediction, lengths)
	endpoint_target = gather_endpoints(target, lengths)
	endpoint_loss = F.smooth_l1_loss(
		endpoint_prediction, endpoint_target, beta=0.01
	)
	calibration_index = output["calibration_lengths"] - 1
	calibration_prediction = prediction.gather(
		1, calibration_index.view(-1, 1)
	)
	calibration_target = target.gather(1, calibration_index.view(-1, 1))
	drift_loss = mean_per_sequence(
		F.smooth_l1_loss(
			prediction - calibration_prediction,
			target - calibration_target,
			beta=0.01,
			reduction="none"
		) * horizon_weights,
		loss_mask
	)

	physical_soc = torch.clamp(
		soc0.view(-1, 1) + output["physical_contribution"], min=0.0, max=1.0
	)
	physical_loss = mean_per_sequence(
		F.smooth_l1_loss(physical_soc, target, beta=0.01, reduction="none"),
		loss_mask
	)
	error = (prediction - target).detach()
	log_variance = output["log_variance"]
	uncertainty_loss = mean_per_sequence(
		0.5 * (torch.exp(-log_variance) * error.square() + log_variance),
		loss_mask
	)

	residual = output["dynamic_residual"]
	residual_l2 = mean_per_sequence(residual.square(), loss_mask)
	physics_gain_regularization = mean_per_sequence(
		output["physics_log_gain"].square(), loss_mask
	)
	current_gain_regularization = output["current_gain_logs"].square().mean()
	current_bias_regularization = (
		output["current_bias_amp"] / args.current_bias_max_amp
	).square().mean()
	pair_mask = loss_mask[:, 1:] & loss_mask[:, :-1]
	if pair_mask.any():
		residual_smoothness = mean_per_sequence(
			(residual[:, 1:] - residual[:, :-1]).square(), pair_mask
		)
	else:
		residual_smoothness = residual_l2.new_zeros(())

	total_loss = (
		sequence_loss
		+ args.endpoint_loss_weight * endpoint_loss
		+ args.drift_loss_weight * drift_loss
		+ args.physical_loss_weight * physical_loss
		+ args.uncertainty_loss_weight * uncertainty_loss
		+ args.residual_l2_weight * residual_l2
		+ args.residual_smoothness_weight * residual_smoothness
		+ args.physics_gain_weight * physics_gain_regularization
		+ args.current_gain_weight * current_gain_regularization
		+ args.current_bias_weight * current_bias_regularization
	)
	return {
		"loss": total_loss,
		"sequence_loss": sequence_loss,
		"endpoint_loss": endpoint_loss,
		"drift_loss": drift_loss,
		"physical_loss": physical_loss,
		"uncertainty_loss": uncertainty_loss,
		"residual_l2": residual_l2,
		"residual_smoothness": residual_smoothness,
		"physics_gain_regularization": physics_gain_regularization,
		"current_gain_regularization": current_gain_regularization,
		"current_bias_regularization": current_bias_regularization
	}, loss_mask


def batch_metrics(output: dict, target: torch.Tensor, lengths: torch.Tensor,
				  loss_mask: torch.Tensor, normalization: Normalization):
	soc_scale = normalization.soc_range
	sequence_error = (output["soc"] - target) * soc_scale
	mask_float = loss_mask.to(sequence_error.dtype)
	sequence_counts = loss_mask.sum(dim=1).clamp(min=1)
	sequence_mae = (
		sequence_error.abs() * mask_float
	).sum(dim=1) / sequence_counts
	sequence_mse = (
		sequence_error.square() * mask_float
	).sum(dim=1) / sequence_counts
	endpoint_error = (
		gather_endpoints(output["soc"], lengths)
		- gather_endpoints(target, lengths)
	) * soc_scale
	return {
		"sequence_mae_sum": sequence_mae.sum().item(),
		"sequence_mse_sum": sequence_mse.sum().item(),
		"sequence_count": lengths.numel(),
		"endpoint_abs_error_sum": endpoint_error.abs().sum().item(),
		"endpoint_squared_error_sum": endpoint_error.square().sum().item(),
		"endpoint_count": lengths.numel()
	}


def move_batch(batch, device: torch.device):
	return tuple(value.to(device, non_blocking=True) for value in batch)


def evaluate(model, loader: DataLoader, device: torch.device,
			 normalization: Normalization, args: argparse.Namespace,
			 use_amp: bool = False) -> dict[str, float]:
	model.eval()
	loss_sums = {}
	batch_count = 0
	metric_sums = {
		"sequence_mae_sum": 0.0,
		"sequence_mse_sum": 0.0,
		"sequence_count": 0,
		"endpoint_abs_error_sum": 0.0,
		"endpoint_squared_error_sum": 0.0,
		"endpoint_count": 0
	}
	for batch in loader:
		voltage, current, soc0, target, lengths = move_batch(batch, device)
		with torch.no_grad(), torch.autocast(
			device_type=device.type, dtype=torch.float16, enabled=use_amp
		):
			output = model(voltage, current, soc0, lengths)
			losses, loss_mask = compute_losses(output, target, soc0, lengths, args)
		for key, value in losses.items():
			loss_sums[key] = loss_sums.get(key, 0.0) + value.detach().item()
		metrics = batch_metrics(output, target, lengths, loss_mask, normalization)
		for key, value in metrics.items():
			metric_sums[key] += value
		batch_count += 1

	result = {key: value / batch_count for key, value in loss_sums.items()}
	sequence_count = metric_sums["sequence_count"]
	endpoint_count = metric_sums["endpoint_count"]
	result.update({
		"mae_percent": metric_sums["sequence_mae_sum"] / sequence_count,
		"rmse_percent": (
			metric_sums["sequence_mse_sum"] / sequence_count
		) ** 0.5,
		"endpoint_mae_percent": (
			metric_sums["endpoint_abs_error_sum"] / endpoint_count
		),
		"endpoint_rmse_percent": (
			metric_sums["endpoint_squared_error_sum"] / endpoint_count
		) ** 0.5
	})
	return result


def train_step(model, batch, device: torch.device, normalization: Normalization,
			   args: argparse.Namespace, optimizer, scaler, use_amp: bool):
	model.train()
	voltage, current, soc0, target, lengths = move_batch(batch, device)
	optimizer.zero_grad(set_to_none=True)
	with torch.autocast(
		device_type=device.type, dtype=torch.float16, enabled=use_amp
	):
		output = model(voltage, current, soc0, lengths)
		losses, loss_mask = compute_losses(output, target, soc0, lengths, args)
	scaler.scale(losses["loss"]).backward()
	scaler.unscale_(optimizer)
	torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
	scaler.step(optimizer)
	scaler.update()

	metrics = batch_metrics(output, target, lengths, loss_mask, normalization)
	return {
		**{key: value.detach().item() for key, value in losses.items()},
		"mae_percent": (
			metrics["sequence_mae_sum"] / metrics["sequence_count"]
		),
		"rmse_percent": (
			metrics["sequence_mse_sum"] / metrics["sequence_count"]
		) ** 0.5,
		"endpoint_mae_percent": (
			metrics["endpoint_abs_error_sum"] / metrics["endpoint_count"]
		),
		"endpoint_rmse_percent": (
			metrics["endpoint_squared_error_sum"] / metrics["endpoint_count"]
		) ** 0.5
	}


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler,
					epoch: int, step_in_epoch: int, global_step: int,
					best_validation_mae: float, stale_validations: int,
					model_config: dict, normalization: Normalization,
					args: argparse.Namespace, train_files: list[str],
					validation_files: list[str]) -> None:
	payload = {
		"model_class": "StatelessSoCPredictor",
		"epoch": epoch,
		"step_in_epoch": step_in_epoch,
		"global_step": global_step,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict(),
		"scaler_state_dict": scaler.state_dict(),
		"best_validation_mae_percent": best_validation_mae,
		"stale_validations": stale_validations,
		"model_config": model_config,
		"normalization": {
			**asdict(normalization),
			"soc_formula": "(soc - soc_min) / (soc_max - soc_min)",
			"soc_inverse_formula": "soc_norm * (soc_max - soc_min) + soc_min",
			"rate_input_formula": "current_amp = rate * rate_to_current"
		},
		"training_config": vars(args),
		"train_files": train_files,
		"validation_files": validation_files
	}
	temporary_path = path.with_suffix(path.suffix + ".tmp")
	torch.save(payload, temporary_path)
	temporary_path.replace(path)


def append_history(path: Path, row: dict) -> None:
	write_header = not path.exists()
	with path.open("a", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=list(row))
		if write_header:
			writer.writeheader()
		writer.writerow(row)


def save_run_metadata(output_dir: Path, args: argparse.Namespace,
					  normalization: Normalization, model_config: dict,
					  train_manifest: pd.DataFrame,
					  validation_manifest: pd.DataFrame) -> None:
	train_manifest.to_csv(output_dir / "train_split.csv", index=False)
	validation_manifest.to_csv(output_dir / "validation_split.csv", index=False)
	config = {
		"training": {
			key: str(value) if isinstance(value, Path) else value
			for key, value in vars(args).items()
		},
		"model": model_config,
		"normalization": {
			**asdict(normalization),
			"current_mean": normalization.current_mean,
			"current_std": normalization.current_std,
			"soc_formula": "(soc - soc_min) / (soc_max - soc_min)",
			"soc_inverse_formula": "soc_norm * (soc_max - soc_min) + soc_min",
			"current_formula": "current_amp = rate * rate_to_current"
		},
		"loss": {
			"supervision": "all valid points from the calibration endpoint onward",
			"sequence_weighting": (
				"linear by relative horizon from 1 to horizon_loss_weight"
			),
			"physics_fusion": (
				"trajectory-level charge/discharge current gains and current bias"
			),
			"drift_supervision": (
				"relative SOC change from the calibration endpoint"
			),
			"validation_split": (
				f"grouped by {args.split_group_column}"
				if args.split_group_column.lower() != "none"
				else "condition-stratified trajectory split"
			),
			"selection_metric": "validation sequence MAE in SOC percentage points"
		}
	}
	with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
		json.dump(config, file, indent=2, ensure_ascii=False)


def main() -> None:
	args = parse_args()
	validate_args(args)
	seed_everything(args.seed)
	device = select_device(args.device)
	normalization = load_normalization(args.normalization)
	train_manifest, validation_manifest = stratified_split(
		args.manifest, args.validation_fraction, args.seed,
		args.split_group_column
	)

	train_dataset = TrajectorySequenceDataset(
		train_manifest, args.data_dir, normalization,
		args.min_window, args.max_window, args.windows_per_trajectory,
		args.prefix_horizon, args.prefix_window_fraction, True, args.seed
	)
	validation_dataset = TrajectorySequenceDataset(
		validation_manifest, args.data_dir, normalization,
		args.min_window, args.max_window, args.validation_windows_per_trajectory,
		args.prefix_horizon, args.prefix_window_fraction, False, args.seed + 1
	)
	loader_kwargs = {
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"collate_fn": collate_sequences,
		"pin_memory": device.type == "cuda",
		"persistent_workers": args.num_workers > 0
	}
	train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
	validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

	model, model_config = build_model(args, normalization)
	model.to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 3),
		min_lr=1e-6
	)
	use_amp = device.type == "cuda" and not args.no_amp
	scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	save_run_metadata(
		args.output_dir, args, normalization, model_config,
		train_manifest, validation_manifest
	)
	best_path = args.output_dir / "best.pt"
	last_path = args.output_dir / "last.pt"
	history_path = args.output_dir / "history.csv"
	start_epoch = 1
	global_step = 0
	best_validation_mae = float("inf")
	stale_validations = 0
	if args.resume is not None:
		checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
		if checkpoint.get("model_config") != model_config:
			raise ValueError("Resume checkpoint model_config does not match arguments")
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
		scaler.load_state_dict(checkpoint["scaler_state_dict"])
		start_epoch = int(checkpoint["epoch"]) + 1
		global_step = int(checkpoint.get("global_step", 0))
		best_validation_mae = float(checkpoint["best_validation_mae_percent"])
		stale_validations = int(checkpoint.get("stale_validations", 0))

	train_files = train_manifest["sequence_file"].tolist()
	validation_files = validation_manifest["sequence_file"].tolist()
	print(f"Device: {device}; AMP: {use_amp}")
	print(f"Train trajectories: {len(train_dataset)}; validation: {len(validation_dataset)}")
	if args.split_group_column.lower() != "none":
		print(
			f"Validation grouped by {args.split_group_column}: "
			f"{train_manifest[args.split_group_column].nunique()} train groups, "
			f"{validation_manifest[args.split_group_column].nunique()} validation groups"
		)
	condition_counts = validation_manifest["condition_code"].value_counts().sort_index()
	print(f"Validation conditions: {condition_counts.to_dict()}")
	print(f"Validate and checkpoint every {args.validation_interval} optimizer steps")
	print(
		"Targets: normalized full SOC sequence after calibration; "
		"metrics: denormalized SOC percentage points"
	)

	interval_metrics = []
	last_validation_step = global_step
	stop_training = False
	last_epoch = start_epoch
	last_step_in_epoch = 0

	def validate_and_save(epoch: int, step_in_epoch: int) -> bool:
		nonlocal best_validation_mae, stale_validations, last_validation_step
		validation_metrics = evaluate(
			model, validation_loader, device, normalization, args, use_amp
		)
		scheduler.step(validation_metrics["mae_percent"])
		train_metrics = {
			key: sum(item[key] for item in interval_metrics) / len(interval_metrics)
			for key in interval_metrics[0]
		}
		row = {
			"epoch": epoch,
			"step_in_epoch": step_in_epoch,
			"global_step": global_step,
			"learning_rate": optimizer.param_groups[0]["lr"],
			**{f"train_{key}": value for key, value in train_metrics.items()},
			**{f"validation_{key}": value for key, value in validation_metrics.items()}
		}
		append_history(history_path, row)
		print(
			f"Epoch {epoch:03d} step {step_in_epoch:04d} | global {global_step} | "
			f"train loss {train_metrics['loss']:.6f} | "
			f"val loss {validation_metrics['loss']:.6f} | "
			f"val sequence MAE {validation_metrics['mae_percent']:.4f} | "
			f"val endpoint MAE {validation_metrics['endpoint_mae_percent']:.4f} SOC points"
		)

		improved = validation_metrics["mae_percent"] < best_validation_mae
		if improved:
			best_validation_mae = validation_metrics["mae_percent"]
			stale_validations = 0
		else:
			stale_validations += 1
		save_checkpoint(
			last_path, model, optimizer, scheduler, scaler, epoch, step_in_epoch,
			global_step, best_validation_mae, stale_validations, model_config,
			normalization, args, train_files, validation_files
		)
		if improved:
			save_checkpoint(
				best_path, model, optimizer, scheduler, scaler, epoch, step_in_epoch,
				global_step, best_validation_mae, stale_validations, model_config,
				normalization, args, train_files, validation_files
			)
		last_validation_step = global_step
		interval_metrics.clear()
		return stale_validations >= args.patience

	for epoch in range(start_epoch, args.epochs + 1):
		last_epoch = epoch
		for step_in_epoch, batch in enumerate(train_loader, start=1):
			last_step_in_epoch = step_in_epoch
			interval_metrics.append(train_step(
				model, batch, device, normalization, args, optimizer, scaler, use_amp
			))
			global_step += 1
			if global_step % args.validation_interval == 0:
				stop_training = validate_and_save(epoch, step_in_epoch)
				if stop_training:
					print(
						f"Early stopping after {args.patience} validations "
						"without improvement"
					)
					break
		if stop_training:
			break

	if interval_metrics and global_step != last_validation_step:
		validate_and_save(last_epoch, last_step_in_epoch)


if __name__ == "__main__":
	main()
