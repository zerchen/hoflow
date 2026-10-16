"""Write motion arrays and metadata to WebDataset tar shards."""

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


class ShardManager:
    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
        max_sample_per_shard: int = 1000,
    ):
        if max_sample_per_shard < 1:
            raise ValueError("max_sample_per_shard must be positive")
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.max_sample_per_shard = max_sample_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Counts describe this run; shard_index may be advanced when resuming.
        self.shard_index = 0
        self.shard_count = 0
        self.sample_count = 0
        self.current_sample_counts = 0
        self.sample_id_counter = 0
        self.shard_writer: Optional[tarfile.TarFile] = None
        self.shard_path: Optional[Path] = None

    def close(self):
        """Close the current shard writer."""
        self._close_shard()
        print(f"Closed. Total samples: {self.sample_count}")

    def _close_shard(self):
        if self.shard_writer is not None:
            self.shard_writer.close()
            self.shard_writer = None
            print(f"Closed shard {self.shard_path}")

    def get_or_create_shard_writer(self):
        """Open a shard lazily and rotate when its sample limit is reached."""
        if self.shard_writer is not None and self.current_sample_counts >= self.max_sample_per_shard:
            self._close_shard()

        if self.shard_writer is None:
            if self.shard_path is not None:
                self.shard_index += 1
            shard_name = f"{self.dataset_name}-{self.shard_index:06d}.tar"
            self.shard_path = self.output_dir / shard_name
            print(f"Creating new shard: {shard_name}")
            self.shard_writer = tarfile.open(self.shard_path, "w")
            self.current_sample_counts = 0
            self.shard_count += 1

        return self.shard_writer

    def add_sample(
        self,
        sample_id: Optional[Union[int, str]] = None,
        nested_tar_files: Optional[Dict[str, np.ndarray]] = None,
        flattened_tar_files: Optional[Dict[str, np.ndarray]] = None,
        **additional_metadata,
    ):
        """Write a sequence, individual frames, or a metadata-only sample.

        The existing payload keys are retained for dataset processors. Both
        array modes store .npy members directly in the shard.
        """
        if nested_tar_files is not None and flattened_tar_files is not None:
            raise ValueError("Choose either sequence arrays or frame arrays")

        arrays = nested_tar_files if nested_tar_files is not None else flattened_tar_files
        if arrays is not None:
            for name, values in arrays.items():
                if not isinstance(values, np.ndarray):
                    raise TypeError(f"{name} must be a NumPy array, got {type(values)}")

        if flattened_tar_files is not None:
            seq_len = additional_metadata["seq_len"]
            if seq_len < 0:
                raise ValueError("seq_len must be non-negative")
            for name, values in flattened_tar_files.items():
                if values.ndim == 0 or len(values) != seq_len:
                    raise ValueError(f"{name} must have seq_len={seq_len} frames")
            for frame_idx in range(seq_len):
                self._write_sample(
                    sample_id,
                    {name: values[frame_idx] for name, values in flattened_tar_files.items()},
                    {"frame_idx": frame_idx, **additional_metadata},
                )
        else:
            self._write_sample(sample_id, nested_tar_files or {}, additional_metadata)

    def _write_sample(self, sample_id, arrays, metadata):
        shard_writer = self.get_or_create_shard_writer()
        item_id = f"{self.dataset_name}_{self.sample_id_counter:012d}"
        meta = {
            "id": item_id,
            "source": self.dataset_name,
            "sample_id": sample_id,
            **metadata,
        }
        self._add_to_tar(shard_writer, f"{item_id}.meta.json", json.dumps(meta).encode("utf-8"))
        for name, values in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, values)
            self._add_to_tar(shard_writer, f"{item_id}.{name}.npy", buffer.getvalue())

        self.sample_id_counter += 1
        self.current_sample_counts += 1
        self.sample_count += 1

    @staticmethod
    def _add_to_tar(shard_writer: tarfile.TarFile, name: str, data: bytes):
        tarinfo = tarfile.TarInfo(name=name)
        tarinfo.size = len(data)
        shard_writer.addfile(tarinfo, io.BytesIO(data))

    def add_sample_from_dict(self, data_dict: Dict[str, Any]):
        self.add_sample(
            sample_id=data_dict.get("id"),
            nested_tar_files=data_dict.get("nested_tar_files"),
            flattened_tar_files=data_dict.get("flattened_tar_files"),
            **{
                key: value
                for key, value in data_dict.items()
                if key not in {"id", "nested_tar_files", "flattened_tar_files"}
            },
        )
