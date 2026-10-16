import json
import time
from pathlib import Path
from .shard_manager import ShardManager


class SceneConverterBase:
    """Convert data to WebDataset format."""
    def __init__(self, dataset_name: str, input_dir: str, output_dir: str, max_sample_per_shard: int = 1):
        self.dataset_name = dataset_name
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.shard_manager = ShardManager(dataset_name=dataset_name, output_dir=output_dir, max_sample_per_shard=max_sample_per_shard)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close all shard writers."""
        self.shard_manager.close()

    def convert_dataset(self, dataset_dir: Path):
        """Main entry to convert dataset."""
        raise NotImplementedError("Subclasses must implement convert_dataset method.")

    def finalize(self):
        """Close writers and record statistics for this conversion run."""
        self.close()

        # Create dataset info
        total_samples = self.shard_manager.sample_count
        total_shards = self.shard_manager.shard_count
        dataset_name = self.dataset_name

        info = {
            "dataset": dataset_name,
            "total_samples": total_samples,
            "total_shards": total_shards,
            "conversion_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        info_path = self.output_dir / f"info_{dataset_name}.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        print(f"\n{'='*60}")
        print(f"CONVERSION COMPLETE: {dataset_name} Dataset")
        print(f"{'='*60}")
        print(f"Total samples: {total_samples}")
        print(f"Total shards: {total_shards}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")
