import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from models.rom.pngv import PNGVModel
from scripts.rom.calibration_common import (
	calibrate_pngv, estimate_equivalent_capacity, load_trajectories,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Fit PNGV parameters from data/split_data/train.")
	parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "split_data" / "train")
	parser.add_argument(
		"--manifest", type=Path,
		default=ROOT / "data" / "split_data" / "metadata" / "train_manifest.csv"
	)
	parser.add_argument("--sequences-per-condition", type=int, default=10)
	parser.add_argument("--max-points-per-sequence", type=int, default=500)
	parser.add_argument("--max-nfev", type=int, default=200)
	parser.add_argument("--seed", type=int, default=20260831)
	parser.add_argument("--save-on-failure", action="store_true")
	parser.add_argument(
		"--output",
		type=Path,
		default=ROOT / "checkpoints" / "rom" / "pngv.npz"
	)
	args = parser.parse_args()
	if args.sequences_per_condition <= 0:
		parser.error("--sequences-per-condition 必须为正整数")
	if args.max_points_per_sequence < 2:
		parser.error("--max-points-per-sequence 必须至少为 2")
	if args.max_nfev <= 0:
		parser.error("--max-nfev 必须为正整数")
	return args


def main() -> None:
	args = parse_args()
	trajectories, sample_interval = load_trajectories(
		args.data_dir,
		args.manifest,
		args.sequences_per_condition,
		args.max_points_per_sequence,
		args.seed
	)
	equivalent_capacity = estimate_equivalent_capacity(trajectories)

	print(f"加载轨迹数: {len(trajectories)}")
	print(f"训练采样点数: {sum(len(frame) for frame in trajectories)}")
	print(f"采样间隔中位数: {sample_interval:.3f} s")
	print(f"等效容量估计: {equivalent_capacity:.6f}")

	result = calibrate_pngv(trajectories, equivalent_capacity, sample_interval, args.max_nfev)
	if not result["success"] and not args.save_on_failure:
		raise RuntimeError(
			f"PNGV 优化未收敛，未保存权重: {result['message']}。"
			"可增大 --max-nfev，或使用 --save-on-failure 强制保存。"
		)
	model = PNGVModel(dt=sample_interval)
	model.params = np.asarray(result["params"], dtype=float)
	metadata = {
		"mse": result["mse"],
		"success": result["success"],
		"message": result["message"],
		"nfev": result["nfev"],
		"trajectory_count": len(trajectories),
		"point_count": sum(len(frame) for frame in trajectories),
		"seed": args.seed,
		"train_dir": str(args.data_dir.resolve()),
		"train_manifest": str(args.manifest.resolve()),
		"soc_input_scale": "0_to_1",
		"current_input": "rate"
	}
	model.save_checkpoint(args.output, metadata=metadata)

	print(f"训练完成，MSE: {result['mse']:.8f}")
	print(f"优化状态: {result['message']}")
	print(f"PNGV 权重已保存至: {args.output}")


if __name__ == "__main__":
	main()
