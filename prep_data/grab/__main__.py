"""GRAB dataset to WebDataset converter (seq tars only)."""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prep_data.converter import SceneConverterBase
from prep_data.grab.processor import TEST_OBJ, process_sequence_file, sequence_file_meta

class GrabConverter(SceneConverterBase):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        mode: str = "train",
        max_sample_per_shard: int = 500,
    ):
        # Output goes to output_dir / {mode}_seq_tars
        self.mode = mode
        seq_tars_dir = Path(output_dir) / f"{mode}_seq_tars"
        super().__init__(
            dataset_name="grab",
            input_dir=input_dir,
            output_dir=str(seq_tars_dir),
            max_sample_per_shard=max_sample_per_shard,
        )

    def convert_dataset(self, dataset_dir: Path = None):
        data_root = self.input_dir if dataset_dir is None else Path(dataset_dir)
        txt2motion = {}
        data_root_str = str(data_root)

        for idx in range(10):
            sbj = f"s{idx + 1}"
            sbj_dir = data_root / "data" / sbj
            if not sbj_dir.is_dir():
                continue

            for fn in tqdm(sorted(os.listdir(sbj_dir)), desc=sbj):
                meta = sequence_file_meta(data_root_str, sbj, fn)
                if meta is None:
                    continue
                obj_name, intent, _ = meta
                if self.mode == "train" and obj_name in TEST_OBJ or self.mode == "test" and obj_name not in TEST_OBJ:
                    continue
                txt = f"{intent} the {obj_name}"
                if self.mode == "test" and txt2motion.get(txt, 0) >= 4:
                    continue

                samples = process_sequence_file(data_root_str, sbj, fn, self.mode)
                for sample in samples:
                    if self.mode == "test":
                        if txt2motion.get(txt, 0) >= 4:
                            break
                        txt2motion[txt] = txt2motion.get(txt, 0) + 1
                    self.shard_manager.add_sample_from_dict(sample)


def main():
    parser = argparse.ArgumentParser(description="GRAB Dataset to WebDataset Converter (seq tars only)")
    parser.add_argument("--input_dir", required=True, help="Path to raw GRAB dataset (contains data/ subdir)")
    parser.add_argument("--output_dir", required=True, help="Output root (seq tars written to <output_dir>/<mode>_seq_tars/)")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Split to convert",
    )
    parser.add_argument(
        "--max_sample_per_shard",
        type=int,
        default=500,
        help="Max samples per seq tar shard",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    converter = GrabConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        max_sample_per_shard=args.max_sample_per_shard,
    )

    print("Starting GRAB conversion (seq tars only)...")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {converter.output_dir}")
    print(f"  Mode:   {args.mode}")

    converter.convert_dataset(input_path)
    converter.finalize()


if __name__ == "__main__":
    main()
