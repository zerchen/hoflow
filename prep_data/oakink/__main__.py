import argparse
import os
import sys
from pathlib import Path

import numpy as np

from ..converter import SceneConverterBase

OBJ_TO_ACTION = {
    "binoculars": ["see", "offhand"],
    "bowl": ["drink", "pass"],
    "camera": ["takepicture", "browse"],
    "can": ["inspect", "offhand"],
    "drill": ["pass", "lift"],
    "eyeglasses": ["clean", "wear"],
    "flashlight": ["on", "offhand"],
    "fryingpan": ["cook", "pass"],
    "gamecontroller": ["play", "lift"],
    "hammers": ["use", "pass"],
    "headphones": ["use", "offhand"],
    "knife": ["chop", "peel"],
    "lightbulb": ["pass", "screw"],
    "mouse": ["use", "offhand"],
    "mug": ["drink", "toast"],
    "phone": ["call", "pass"],
    "teapot": ["pour", "lift"],
    "toothbrush": ["brush", "lift"],
    "wineglass": ["drink", "toast"],
    "waterbottle": ["open", "shake", "drink", "pour"],
    "screwdriver": ["pass", "lift"],
    "donut": ["eat", "lift"],
}


def _action_for_obj(obj_name: str, seq_path: str) -> str:
    """Resolve action for object: from path if present, else first in OBJ_TO_ACTION."""
    base = obj_name.split("_")[0].lower()
    actions = OBJ_TO_ACTION.get(base, ["manipulate"])
    path_lower = Path(seq_path).as_posix().lower()
    for a in actions:
        if a in path_lower:
            return a
    return actions[0]


class OakInkConverter(SceneConverterBase):
    """Convert OakInk motion .npy files to WebDataset tar with meta only (no pose data)."""

    def __init__(self, input_dir: str, output_dir: str, max_sample_per_shard: int = 100, verbose: bool = False):
        super().__init__(dataset_name="oakink", input_dir=input_dir, output_dir=output_dir, max_sample_per_shard=max_sample_per_shard)
        self.verbose = verbose
        self.stats = {
            "total_processed": 0,
            "skipped_samples": 0,
        }

    def _iter_samples(self, dataset_dir: Path):
        """Yield metadata for motion files, followed by mesh/action pairs."""
        dataset_dir = Path(dataset_dir)
        npy_files = sorted(dataset_dir.rglob("*.npy"))
        for seq_file in npy_files:
            self.stats["total_processed"] += 1
            try:
                rel = seq_file.relative_to(dataset_dir)
                seq_id = str(rel.with_suffix("")).replace(os.sep, "-")
                data = np.load(seq_file, allow_pickle=True).item()
            except Exception:
                self.stats["skipped_samples"] += 1
                continue
            if "right_hand" not in data:
                self.stats["skipped_samples"] += 1
                continue
            obj_name = next((k for k in data if k != "right_hand"), None)
            if obj_name is None:
                self.stats["skipped_samples"] += 1
                continue
            rh_trans = data["right_hand"].get("trans")
            if rh_trans is None or len(rh_trans) == 0:
                self.stats["skipped_samples"] += 1
                continue
            seq_len = len(rh_trans)
            base_obj = obj_name.split("_")[0].lower()
            action = _action_for_obj(obj_name, str(seq_file))
            text = f"{action} the {base_obj}"
            yield {
                "id": seq_id,
                "obj_name": obj_name,
                "text": text,
                "seq_len": seq_len,
            }

        mesh_dir = dataset_dir / "oakink_aligned"
        for ply_file in sorted(mesh_dir.glob("*.ply")):
            obj_full_name = ply_file.stem
            obj_category = obj_full_name.split("_")[0]
            if obj_category in OBJ_TO_ACTION:
                for action in OBJ_TO_ACTION[obj_category]:
                    text = f"{action} the {obj_category}"
                    yield {
                        "id": f"{obj_full_name}-{action}",
                        "obj_name": obj_full_name,
                        "text": text,
                    }

    def convert_dataset(self, dataset_dir: Path):
        for sample in self._iter_samples(dataset_dir):
            if self.verbose:
                print(f"\nProcessing {sample['id']} ✓")
            self.shard_manager.add_sample_from_dict(sample)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("CONVERSION SUMMARY")
        print("=" * 60)
        print(f"\n{self.stats['total_processed']} processed, {self.shard_manager.sample_count} written, {self.stats['skipped_samples']} skipped")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="OakInk: build WebDataset tar with meta only (no pose data).")
    parser.add_argument("--input_dir", required=True, help="Root directory containing .npy motion files")
    parser.add_argument("--output_dir", required=True, help="Directory for output .tar shards")
    parser.add_argument("--max_sample_per_shard", type=int, default=2000, help="Max samples per shard")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample progress")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    converter = OakInkConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_sample_per_shard=args.max_sample_per_shard,
        verbose=args.verbose,
    )

    print("OakInk conversion (meta only)")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")

    converter.convert_dataset(input_path)
    converter.finalize()
    converter.print_summary()


if __name__ == "__main__":
    main()
