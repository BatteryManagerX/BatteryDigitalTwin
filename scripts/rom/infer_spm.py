import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from models.rom.spm import SPMModel
from scripts.rom.inference_common import run_test_inference


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run SPM inference on data/split_data/test.")
	parser.add_argument("--test-dir", type=Path, default=ROOT / "data" / "split_data" / "test")
	parser.add_argument(
		"--manifest", type=Path,
		default=ROOT / "data" / "split_data" / "metadata" / "test_manifest.csv"
	)
	parser.add_argument(
		"--checkpoint", type=Path,
		default=ROOT / "checkpoints" / "rom" / "spm.npz"
	)
	parser.add_argument(
		"--output", type=Path,
		default=None,
		help="结果数据库（默认: results/rom/spm_infer_t<T>_test.db）"
	)
	parser.add_argument(
		"-T", type=int, default=100, metavar="STEPS",
		help="每隔 T 步使用 GT SOC 重置 ROM 状态（默认: 100）"
	)
	parser.add_argument("--limit", type=int, default=None)
	args = parser.parse_args()
	if args.T <= 0:
		parser.error("-T 必须为正整数")
	if args.limit is not None and args.limit <= 0:
		parser.error("--limit 必须为正整数")
	return args


def main() -> None:
	args = parse_args()
	output_path = args.output or (
		ROOT / "results" / "rom" / f"spm_infer_t{args.T}_test.db"
	)
	model = SPMModel.load_checkpoint(args.checkpoint)

	def predict(initial_soc, initial_voltage, current, soc_ground_truth):
		result = model.predict_sequence(
			initial_soc,
			initial_voltage,
			current,
			soc_ground_truth=soc_ground_truth,
			teacher_forcing_interval=args.T
		)
		return result["soc"], result["voltage"]

	run_test_inference(
		model_name=model.model_type,
		checkpoint_path=args.checkpoint,
		output_path=output_path,
		test_dir=args.test_dir,
		manifest_path=args.manifest,
		predict=predict,
		metadata={
			"inference_mode": "periodic_soc_teacher_forcing",
			"teacher_forcing_interval": str(args.T)
		},
		limit=args.limit,
		pass_soc_ground_truth=True
	)


if __name__ == "__main__":
	main()
