import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "split_data"
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from models.condition_restorer.condition_restorer import ConditionRestorer


@dataclass(frozen=True)
class Normalization:
	soc_mean: float
	soc_std: float
	voltage_mean: float
	voltage_std: float
	rate_mean: float
	rate_std: float


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train ConditionRestorer through a frozen differentiable SPM."
	)
	parser.add_argument("--data-dir", type=Path, default=DATA_ROOT / "train")
	parser.add_argument(
		"--manifest", type=Path,
		default=DATA_ROOT / "metadata" / "train_manifest.csv",
	)
	parser.add_argument(
		"--normalization", type=Path,
		default=DATA_ROOT / "normalization_parameters.json",
	)
	parser.add_argument(
		"--spm-checkpoint", type=Path,
		default=ROOT / "checkpoints" / "rom" / "spm.npz",
	)
	parser.add_argument(
		"--output-dir", type=Path,
		default=ROOT / "checkpoints" / "condition_restorer",
	)
	parser.add_argument("--resume", type=Path, default=None)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--samples-per-trajectory", type=int, default=8)
	parser.add_argument("--validation-samples-per-trajectory", type=int, default=4)
	parser.add_argument("--pre-len", type=int, default=12)
	parser.add_argument("--sub-len", type=int, default=6)
	parser.add_argument("--max-missing-len", type=int, default=6)
	parser.add_argument("--hidden-size", type=int, default=64)
	parser.add_argument("--num-layers", type=int, default=1)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--residual-scale", type=float, default=2.0)
	parser.add_argument("--validation-fraction", type=float, default=0.2)
	parser.add_argument("--validation-interval", type=int, default=10)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--grad-clip", type=float, default=1.0)
	parser.add_argument("--patience", type=int, default=100)
	parser.add_argument("--soc-loss-weight", type=float, default=1.0)
	parser.add_argument("--charge-loss-weight", type=float, default=1.0)
	parser.add_argument(
		"--voltage-loss-weight", type=float, default=0.1,
		help="Set to 0 to train without the auxiliary voltage constraint.",
	)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--seed", type=int, default=20260901)
	parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
	parser.add_argument("--no-amp", action="store_true")
	return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
	positive = (
		args.epochs, args.batch_size, args.samples_per_trajectory,
		args.validation_samples_per_trajectory, args.pre_len, args.sub_len,
		args.max_missing_len, args.hidden_size, args.num_layers,
		args.validation_interval, args.grad_clip, args.patience,
	)
	if any(value <= 0 for value in positive):
		raise ValueError("epochs、批量、采样数、序列长度和训练间隔必须为正数")
	if args.max_missing_len >= 7:
		raise ValueError("按任务定义，--max-missing-len 必须小于 7")
	if args.pre_len < 2 or args.sub_len < 2:
		raise ValueError("ConditionRestorer 的 pre-len 和 sub-len 必须至少为 2")
	if not 0.0 < args.validation_fraction < 1.0:
		raise ValueError("--validation-fraction 必须在 (0, 1) 内")
	if not 0.0 <= args.dropout < 1.0:
		raise ValueError("--dropout 必须在 [0, 1) 内")
	if args.learning_rate <= 0 or args.weight_decay < 0:
		raise ValueError("学习率必须为正数，权重衰减不能为负数")
	loss_weights = (
		args.soc_loss_weight, args.charge_loss_weight, args.voltage_loss_weight,
	)
	if any(weight < 0 for weight in loss_weights) or sum(loss_weights) <= 0:
		raise ValueError("loss 权重不能为负，且至少一个必须大于 0")


def load_normalization(path: Path) -> Normalization:
	with path.open("r", encoding="utf-8") as file:
		fields = json.load(file)["fields"]
	normalization = Normalization(
		soc_mean=float(fields["soc"]["mean"]),
		soc_std=float(fields["soc"]["std_population"]),
		voltage_mean=float(fields["voltage"]["mean"]),
		voltage_std=float(fields["voltage"]["std_population"]),
		rate_mean=float(fields["rate"]["mean"]),
		rate_std=float(fields["rate"]["std_population"]),
	)
	if min(normalization.soc_std, normalization.voltage_std, normalization.rate_std) <= 0:
		raise ValueError("归一化标准差必须为正数")
	return normalization


def stratified_split(manifest_path: Path, validation_fraction: float,
					 seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
	manifest = pd.read_csv(manifest_path)
	required = {"sequence_file", "condition_code", "records"}
	missing = required - set(manifest.columns)
	if missing:
		raise ValueError(f"训练清单缺少字段: {', '.join(sorted(missing))}")
	rng = np.random.default_rng(seed)
	train_indices = []
	validation_indices = []
	groups = list(manifest.groupby("condition_code", sort=True))
	validation_count = max(1, int(round(
		min(len(group) for _, group in groups) * validation_fraction
	)))
	for _, group in groups:
		indices = group.index.to_numpy(copy=True)
		rng.shuffle(indices)
		validation_indices.extend(indices[:validation_count])
		train_indices.extend(indices[validation_count:])
	return (
		manifest.loc[train_indices].reset_index(drop=True),
		manifest.loc[validation_indices].reset_index(drop=True),
	)


class MissingSegmentDataset(Dataset):
	def __init__(self, manifest: pd.DataFrame, data_dir: Path, pre_len: int,
				 sub_len: int, max_missing_len: int, samples_per_trajectory: int,
				 seed: int, deterministic: bool):
		self.data_dir = data_dir
		self.pre_len = pre_len
		self.sub_len = sub_len
		self.max_missing_len = max_missing_len
		self.samples_per_trajectory = samples_per_trajectory
		self.seed = seed
		self.deterministic = deterministic
		minimum_length = pre_len + 1 + sub_len
		self.rows = [
			(str(row.sequence_file), int(row.records))
			for row in manifest.itertuples(index=False)
			if int(row.records) >= minimum_length
		]
		if not self.rows:
			raise ValueError("没有足够长的轨迹可用于模拟丢包")

	def __len__(self) -> int:
		return len(self.rows) * self.samples_per_trajectory

	def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
		trajectory_index = index // self.samples_per_trajectory
		sequence_file, manifest_length = self.rows[trajectory_index]
		if self.deterministic:
			rng = np.random.default_rng(self.seed + index)
		else:
			rng = np.random.default_rng(
				np.random.SeedSequence([self.seed, index, random.getrandbits(32)])
			)

		maximum_for_trajectory = min(
			self.max_missing_len,
			manifest_length - self.pre_len - self.sub_len,
		)
		missing_len = int(rng.integers(1, maximum_for_trajectory + 1))
		missing_start = int(rng.integers(
			self.pre_len,
			manifest_length - self.sub_len - missing_len + 1,
		))
		read_start = missing_start - self.pre_len
		read_end = missing_start + missing_len + self.sub_len
		frame = pd.read_csv(
			self.data_dir / sequence_file,
			usecols=["soc", "voltage", "rate"],
			dtype={"soc": "float32", "voltage": "float32", "rate": "float32"},
		).iloc[read_start:read_end]
		values = frame[["soc", "voltage", "rate"]].to_numpy(dtype=np.float32)
		if len(values) != read_end - read_start:
			raise ValueError(f"{sequence_file} 的实际长度与 manifest 不一致")
		if not np.isfinite(values).all():
			raise ValueError(f"{sequence_file}[{read_start}:{read_end}] 包含非有限值")

		pre = values[:self.pre_len]
		missing = values[self.pre_len:self.pre_len + missing_len]
		sub = values[self.pre_len + missing_len:]
		padded_rate = np.zeros(self.max_missing_len, dtype=np.float32)
		padded_soc = np.zeros(self.max_missing_len, dtype=np.float32)
		padded_voltage = np.zeros(self.max_missing_len, dtype=np.float32)
		padded_rate[:missing_len] = missing[:, 2]
		padded_soc[:missing_len] = values[
			self.pre_len + 1:self.pre_len + missing_len + 1, 0
		] / 100.0
		padded_voltage[:missing_len] = missing[:, 1]
		return {
			"pre_seq": torch.from_numpy(pre[:, [2, 1, 0]].copy()),
			"sub_seq": torch.from_numpy(sub[:, [2, 1]].copy()),
			"initial_soc": torch.tensor(pre[-1, 0] / 100.0, dtype=torch.float32),
			"gt_rate": torch.from_numpy(padded_rate),
			"gt_soc": torch.from_numpy(padded_soc),
			"gt_voltage": torch.from_numpy(padded_voltage),
			"missing_len": torch.tensor(missing_len, dtype=torch.long),
		}


class DifferentiableSPM(nn.Module):
	def __init__(self, checkpoint_path: Path):
		super().__init__()
		with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
			if str(checkpoint["model_type"].item()) != "SPM":
				raise ValueError("指定的 ROM checkpoint 不是 SPM")
			params = checkpoint["params"].astype(np.float32)
			dt = float(checkpoint["dt"].item())
			capacity_ah = (
				float(checkpoint["capacity_ah"].item())
				if "capacity_ah" in checkpoint.files else 1.0
			)
		if params.shape != (13,):
			raise ValueError("SPM checkpoint 参数数量应为 13")
		if not np.isfinite(capacity_ah) or capacity_ah <= 0:
			raise ValueError("SPM checkpoint 的 capacity_ah 必须为正数")
		self.register_buffer("params", torch.from_numpy(params))
		self.register_buffer("dt", torch.tensor(dt, dtype=torch.float32))
		self.capacity_ah = capacity_ah
		self.csn_max = 3.1e4
		self.csp_max = 5.1e4
		self.theta_n_0 = 0.02
		self.theta_n_100 = 0.85
		self.theta_p_0 = 0.95
		self.theta_p_100 = 0.45
		self.rg = 8.314
		self.temperature = 298.15
		self.faraday = 96485.0

	def _average_concentrations(self, soc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		bounded_soc = soc.clamp(0.0, 1.0)
		theta_n = self.theta_n_0 + bounded_soc * (self.theta_n_100 - self.theta_n_0)
		theta_p = self.theta_p_0 + bounded_soc * (self.theta_p_100 - self.theta_p_0)
		return theta_n * self.csn_max, theta_p * self.csp_max

	def _voltage(self, csn: torch.Tensor, csp: torch.Tensor,
				 current: torch.Tensor) -> torch.Tensor:
		params = self.params.to(dtype=current.dtype)
		theta_n = (csn / self.csn_max).clamp(0.0, 1.0)
		theta_p = (csp / self.csp_max).clamp(0.0, 1.0)
		un = sum(params[index] * theta_n ** index for index in range(4))
		up = sum(params[index + 4] * theta_p ** index for index in range(4))
		i0n = params[10] * (theta_n * (1.0 - theta_n)).clamp_min(1e-8).sqrt()
		i0p = params[11] * (theta_p * (1.0 - theta_p)).clamp_min(1e-8).sqrt()
		scale = 2.0 * self.rg * self.temperature / self.faraday
		eta_n = scale * torch.asinh(current / (2.0 * i0n + 1e-12))
		eta_p = scale * torch.asinh(current / (2.0 * i0p + 1e-12))
		return up - un + eta_p - eta_n - current * params[12]

	def forward(self, initial_soc: torch.Tensor, initial_rate: torch.Tensor,
				current: torch.Tensor,
				missing_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		current = current.float()
		soc = initial_soc.float()
		csn, csp = self._average_concentrations(soc)
		initial_soc_next = (
			soc - initial_rate * self.dt / (3600.0 * self.capacity_ah)
		).clamp(0.0, 1.0)
		initial_csn_avg, initial_csp_avg = self._average_concentrations(initial_soc_next)
		tau_n, tau_p = self.params[8], self.params[9]
		csn = (
			csn + self.dt / tau_n * (initial_csn_avg - csn)
			- self.dt * initial_rate * 5.0
		).clamp(1.0, self.csn_max - 1.0)
		csp = (
			csp + self.dt / tau_p * (initial_csp_avg - csp)
			+ self.dt * initial_rate * 5.0
		).clamp(1.0, self.csp_max - 1.0)
		soc = initial_soc_next
		soc_steps = []
		voltage_steps = []
		for step in range(current.size(1)):
			active = step < missing_len
			step_current = current[:, step]
			voltage_steps.append(self._voltage(csn, csp, step_current))
			next_soc = (
				soc - step_current * self.dt / (3600.0 * self.capacity_ah)
			).clamp(0.0, 1.0)
			csn_avg, csp_avg = self._average_concentrations(next_soc)
			next_csn = (
				csn + self.dt / tau_n * (csn_avg - csn)
				- self.dt * step_current * 5.0
			).clamp(1.0, self.csn_max - 1.0)
			next_csp = (
				csp + self.dt / tau_p * (csp_avg - csp)
				+ self.dt * step_current * 5.0
			).clamp(1.0, self.csp_max - 1.0)
			soc = torch.where(active, next_soc, soc)
			csn = torch.where(active, next_csn, csn)
			csp = torch.where(active, next_csp, csp)
			soc_steps.append(soc)
		return torch.stack(soc_steps, dim=1), torch.stack(voltage_steps, dim=1)


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
		raise RuntimeError("指定了 CUDA，但当前环境无法使用 CUDA")
	return device


def build_model(args: argparse.Namespace, normalization: Normalization):
	model_config = {
		"pre_len": args.pre_len,
		"sub_len": args.sub_len,
		"max_missing_len": args.max_missing_len,
		"hidden_size": args.hidden_size,
		"num_layers": args.num_layers,
		"dropout": args.dropout,
		"pre_mean": [normalization.rate_mean, normalization.voltage_mean, normalization.soc_mean],
		"pre_std": [normalization.rate_std, normalization.voltage_std, normalization.soc_std],
		"sub_mean": [normalization.rate_mean, normalization.voltage_mean],
		"sub_std": [normalization.rate_std, normalization.voltage_std],
		"rate_mean": normalization.rate_mean,
		"rate_std": normalization.rate_std,
		"residual_scale": args.residual_scale,
	}
	return ConditionRestorer(**model_config), model_config


def loss_and_metrics(pred_rate: torch.Tensor, gt_rate: torch.Tensor,
					 pred_soc: torch.Tensor, gt_soc: torch.Tensor,
					 pred_voltage: torch.Tensor, gt_voltage: torch.Tensor,
					 missing_len: torch.Tensor, normalization: Normalization,
					 spm: DifferentiableSPM, args: argparse.Namespace):
	steps = torch.arange(pred_rate.size(1), device=pred_rate.device).unsqueeze(0)
	mask = steps < missing_len.unsqueeze(1)
	valid_soc_error_points = (pred_soc - gt_soc)[mask] * 100.0
	valid_voltage_error = (pred_voltage - gt_voltage)[mask]
	soc_loss = F.smooth_l1_loss(
		valid_soc_error_points, torch.zeros_like(valid_soc_error_points), beta=0.1,
	)
	voltage_loss = F.smooth_l1_loss(
		valid_voltage_error / normalization.voltage_std,
		torch.zeros_like(valid_voltage_error), beta=0.1,
	)
	mask_float = mask.to(pred_rate.dtype)
	pred_mean_rate = (pred_rate * mask_float).sum(dim=1) / missing_len
	gt_mean_rate = (gt_rate * mask_float).sum(dim=1) / missing_len
	charge_loss = F.smooth_l1_loss(
		(pred_mean_rate - gt_mean_rate) / normalization.rate_std,
		torch.zeros_like(pred_mean_rate), beta=0.1,
	)
	loss = (
		args.soc_loss_weight * soc_loss
		+ args.charge_loss_weight * charge_loss
		+ args.voltage_loss_weight * voltage_loss
	)
	charge_error_ah = (
		(pred_rate - gt_rate) * mask_float
	).sum(dim=1) * spm.dt / 3600.0
	metrics = {
		"loss": loss.detach().item(),
		"soc_loss": soc_loss.detach().item(),
		"charge_loss": charge_loss.detach().item(),
		"voltage_loss": voltage_loss.detach().item(),
		"soc_mae_percent": valid_soc_error_points.detach().abs().mean().item(),
		"voltage_mae_v": valid_voltage_error.detach().abs().mean().item(),
		"charge_integral_mae_rate_hour": charge_error_ah.detach().abs().mean().item(),
		"mean_rate_mae": (pred_mean_rate - gt_mean_rate).detach().abs().mean().item(),
	}
	return loss, metrics, pred_rate.size(0)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
	return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_batch(model, spm: DifferentiableSPM, batch: dict[str, torch.Tensor],
			  normalization: Normalization, args: argparse.Namespace,
			  use_amp: bool):
	with torch.autocast(device_type=batch["pre_seq"].device.type,
						dtype=torch.float16, enabled=use_amp):
		pred_rate = model(batch["pre_seq"], batch["sub_seq"], batch["missing_len"])
	pred_soc, pred_voltage = spm(
		batch["initial_soc"], batch["pre_seq"][:, -1, 0].float(),
		pred_rate.float(), batch["missing_len"],
	)
	return loss_and_metrics(
		pred_rate.float(), batch["gt_rate"], pred_soc, batch["gt_soc"],
		pred_voltage, batch["gt_voltage"], batch["missing_len"],
		normalization, spm, args,
	)


def train_step(model, spm, batch, device, normalization, args, optimizer,
			   scaler, use_amp):
	model.train()
	batch = move_batch(batch, device)
	optimizer.zero_grad(set_to_none=True)
	loss, metrics, _ = run_batch(model, spm, batch, normalization, args, use_amp)
	scaler.scale(loss).backward()
	scaler.unscale_(optimizer)
	torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
	scaler.step(optimizer)
	scaler.update()
	return metrics


@torch.no_grad()
def evaluate(model, spm, loader, device, normalization, args, use_amp):
	model.eval()
	totals = {}
	count = 0
	for batch in loader:
		batch = move_batch(batch, device)
		_, metrics, sample_count = run_batch(
			model, spm, batch, normalization, args, use_amp,
		)
		for key, value in metrics.items():
			totals[key] = totals.get(key, 0.0) + value * sample_count
		count += sample_count
	if count == 0:
		raise RuntimeError("验证集没有有效的丢包帧")
	return {key: value / count for key, value in totals.items()}


def append_history(path: Path, row: dict) -> None:
	write_header = not path.exists()
	with path.open("a", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=list(row))
		if write_header:
			writer.writeheader()
		writer.writerow(row)


def sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as file:
		for block in iter(lambda: file.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler,
					epoch: int, step_in_epoch: int, global_step: int,
					best_validation_loss: float, stale_validations: int,
					model_config: dict, normalization: Normalization,
					args: argparse.Namespace, train_files: list[str],
					validation_files: list[str]) -> None:
	payload = {
		"model_name": "condition_restorer",
		"model_config": model_config,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict(),
		"scaler_state_dict": scaler.state_dict(),
		"epoch": epoch,
		"step_in_epoch": step_in_epoch,
		"global_step": global_step,
		"best_validation_loss": best_validation_loss,
		"stale_validations": stale_validations,
		"normalization": asdict(normalization),
		"input_channels": {
			"pre_seq": ["rate", "voltage", "soc_percent"],
			"sub_seq": ["rate", "voltage"],
		},
		"output_channel": "rate",
		"spm_checkpoint": str(args.spm_checkpoint.resolve()),
		"spm_checkpoint_sha256": sha256(args.spm_checkpoint),
		"training_config": {
			key: str(value) if isinstance(value, Path) else value
			for key, value in vars(args).items()
		},
		"train_files": train_files,
		"validation_files": validation_files,
	}
	temporary_path = path.with_suffix(path.suffix + ".tmp")
	torch.save(payload, temporary_path)
	temporary_path.replace(path)


def main() -> None:
	args = parse_args()
	validate_args(args)
	seed_everything(args.seed)
	if not args.spm_checkpoint.exists():
		raise FileNotFoundError(f"找不到 SPM checkpoint: {args.spm_checkpoint}")
	device = select_device(args.device)
	normalization = load_normalization(args.normalization)
	train_manifest, validation_manifest = stratified_split(
		args.manifest, args.validation_fraction, args.seed,
	)
	train_dataset = MissingSegmentDataset(
		train_manifest, args.data_dir, args.pre_len, args.sub_len,
		args.max_missing_len, args.samples_per_trajectory, args.seed, False,
	)
	validation_dataset = MissingSegmentDataset(
		validation_manifest, args.data_dir, args.pre_len, args.sub_len,
		args.max_missing_len, args.validation_samples_per_trajectory,
		args.seed + 1_000_000, True,
	)
	loader_kwargs = {
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"pin_memory": device.type == "cuda",
		"persistent_workers": args.num_workers > 0,
	}
	train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
	validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
	model, model_config = build_model(args, normalization)
	model.to(device)
	spm = DifferentiableSPM(args.spm_checkpoint).to(device)
	spm.eval()
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="min", factor=0.5,
		patience=max(2, args.patience // 3), min_lr=1e-6,
	)
	use_amp = device.type == "cuda" and not args.no_amp
	scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	best_path = args.output_dir / "cr_best.pt"
	last_path = args.output_dir / "cr_last.pt"
	history_path = args.output_dir / "cr_history.csv"
	start_epoch = 1
	global_step = 0
	best_validation_loss = float("inf")
	stale_validations = 0
	train_files = train_manifest["sequence_file"].tolist()
	validation_files = validation_manifest["sequence_file"].tolist()
	if args.resume is not None:
		checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
		if checkpoint.get("model_name") != "condition_restorer":
			raise ValueError("恢复权重不是 ConditionRestorer")
		if checkpoint.get("model_config") != model_config:
			raise ValueError("恢复权重的模型配置与当前配置不一致")
		if checkpoint.get("normalization") != asdict(normalization):
			raise ValueError("恢复权重的归一化参数与当前配置不一致")
		if checkpoint.get("spm_checkpoint_sha256") != sha256(args.spm_checkpoint):
			raise ValueError("恢复权重使用的 SPM checkpoint 与当前文件不一致")
		if checkpoint.get("train_files") != train_files or checkpoint.get("validation_files") != validation_files:
			raise ValueError("恢复权重的数据划分与当前配置不一致")
		saved_training_config = checkpoint.get("training_config", {})
		for key in ("soc_loss_weight", "charge_loss_weight", "voltage_loss_weight"):
			if saved_training_config.get(key) != getattr(args, key):
				raise ValueError(f"恢复权重的 {key} 与当前配置不一致")
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
		scaler.load_state_dict(checkpoint["scaler_state_dict"])
		start_epoch = int(checkpoint["epoch"]) + 1
		global_step = int(checkpoint.get("global_step", 0))
		best_validation_loss = float(checkpoint["best_validation_loss"])
		stale_validations = int(checkpoint.get("stale_validations", 0))

	print(f"Device: {device}; AMP: {use_amp}; SPM dt: {spm.dt.item():g} s")
	print(f"Train samples/epoch: {len(train_dataset)}; validation samples: {len(validation_dataset)}")
	print(f"Missing length: 1-{args.max_missing_len}; validate every {args.validation_interval} steps")
	print(
		"Loss weights: "
		f"SOC={args.soc_loss_weight:g}, charge={args.charge_loss_weight:g}, "
		f"voltage={args.voltage_loss_weight:g}"
	)

	interval_metrics = []
	last_validation_step = global_step
	stop_training = False
	last_epoch = start_epoch
	last_step_in_epoch = 0

	def validate_and_save(epoch: int, step_in_epoch: int) -> bool:
		nonlocal best_validation_loss, stale_validations, last_validation_step
		validation_metrics = evaluate(
			model, spm, validation_loader, device, normalization, args, use_amp,
		)
		scheduler.step(validation_metrics["loss"])
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
			**{f"validation_{key}": value for key, value in validation_metrics.items()},
		}
		append_history(history_path, row)
		print(
			f"Epoch {epoch:03d} step {step_in_epoch:04d} | global {global_step} | "
			f"train {train_metrics['loss']:.6f} | val {validation_metrics['loss']:.6f} | "
			f"charge MAE {validation_metrics['charge_integral_mae_rate_hour']:.8f} rate*h | "
			f"SOC MAE {validation_metrics['soc_mae_percent']:.4f} points | "
			f"voltage MAE {validation_metrics['voltage_mae_v']:.6f} V"
		)
		improved = validation_metrics["loss"] < best_validation_loss
		if improved:
			best_validation_loss = validation_metrics["loss"]
			stale_validations = 0
		else:
			stale_validations += 1
		save_checkpoint(
			last_path, model, optimizer, scheduler, scaler, epoch, step_in_epoch,
			global_step, best_validation_loss, stale_validations, model_config,
			normalization, args, train_files, validation_files,
		)
		if improved:
			save_checkpoint(
				best_path, model, optimizer, scheduler, scaler, epoch, step_in_epoch,
				global_step, best_validation_loss, stale_validations, model_config,
				normalization, args, train_files, validation_files,
			)
		interval_metrics.clear()
		last_validation_step = global_step
		return stale_validations >= args.patience

	for epoch in range(start_epoch, args.epochs + 1):
		last_epoch = epoch
		for step_in_epoch, batch in enumerate(train_loader, start=1):
			last_step_in_epoch = step_in_epoch
			metrics = train_step(
				model, spm, batch, device, normalization, args,
				optimizer, scaler, use_amp,
			)
			interval_metrics.append(metrics)
			global_step += 1
			if global_step % args.validation_interval == 0:
				if validate_and_save(epoch, step_in_epoch):
					stop_training = True
					break
		if stop_training:
			print(f"Early stopping after {stale_validations} validations without improvement")
			break

	if interval_metrics and global_step != last_validation_step:
		validate_and_save(last_epoch, last_step_in_epoch)
	print(f"Best checkpoint: {best_path}")
	print(f"Latest checkpoint: {last_path}")


if __name__ == "__main__":
	main()
