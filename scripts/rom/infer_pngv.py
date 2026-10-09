import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from models.rom.pngv import PNGVModel
from scripts.rom.inference_common import run_test_inference


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run PNGV autoregressive inference on data/split_data/test.")
	parser.add_argument("--test-dir", type=Path, default=ROOT / "data" / "split_data" / "test")
	parser.add_argument(
		"--manifest", type=Path,
		default=ROOT / "data" / "split_data" / "metadata" / "test_manifest.csv"
	)
	parser.add_argument(
		"--checkpoint", type=Path,
		default=ROOT / "checkpoints" / "rom" / "pngv.npz"
	)
	parser.add_argument(
		"--output", type=Path,
		default=ROOT / "results" / "rom" / "pngv_infer_test.db"
	)
	parser.add_argument("--limit", type=int, default=None)
	args = parser.parse_args()
	if args.limit is not None and args.limit <= 0:
		parser.error("--limit 必须为正整数")
	return args


def main() -> None:
	args = parse_args()
	model = PNGVModel.load_checkpoint(args.checkpoint)
	run_test_inference(
		model_name=model.model_type,
		checkpoint_path=args.checkpoint,
		output_path=args.output,
		test_dir=args.test_dir,
		manifest_path=args.manifest,
		predict=model.predict_sequence,
		limit=args.limit
	)


if __name__ == "__main__":
	main()
