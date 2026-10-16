"""CPU-only regression checks for preprocessing output formats."""

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prep_data.converter import SceneConverterBase
from prep_data.oakink.__main__ import OakInkConverter
from prep_data.shard_manager import ShardManager


def read_samples(output_dir):
    samples = []
    for path in sorted(Path(output_dir).glob('*.tar')):
        with tarfile.open(path) as shard:
            for member in shard:
                if member.name.endswith('.meta.json'):
                    samples.append(json.load(shard.extractfile(member)))
    return samples


class PreprocessingTests(unittest.TestCase):
    def test_sequence_frames_and_metadata_keep_output_format(self):
        with tempfile.TemporaryDirectory() as root:
            manager = ShardManager('motion', root, max_sample_per_shard=2)
            poses = np.arange(12, dtype=np.float32).reshape(3, 4)
            manager.add_sample_from_dict({
                'id': 'sequence', 'seq_len': 3,
                'nested_tar_files': {'rh_poses': poses},
            })
            manager.add_sample_from_dict({
                'id': 'frames', 'seq_len': 3,
                'flattened_tar_files': {'rh_poses': poses},
            })
            manager.add_sample_from_dict({'id': 'mesh', 'text': 'lift the mug'})
            manager.close()

            self.assertEqual((manager.sample_count, manager.shard_count), (5, 3))
            metadata = read_samples(root)
            self.assertEqual([sample['sample_id'] for sample in metadata],
                             ['sequence', 'frames', 'frames', 'frames', 'mesh'])
            self.assertEqual([sample['frame_idx'] for sample in metadata[1:4]], [0, 1, 2])
            self.assertEqual(len({sample['id'] for sample in metadata}), 5)
            payloads = []
            for path in sorted(Path(root).glob('*.tar')):
                with tarfile.open(path) as shard:
                    meta_names = [name for name in shard.getnames() if name.endswith('.meta.json')]
                    self.assertLessEqual(len(meta_names), 2)
                    for member in shard:
                        if member.name.endswith('.rh_poses.npy'):
                            payloads.append(np.load(io.BytesIO(shard.extractfile(member).read())))
            for actual, expected in zip(payloads, [poses, *poses]):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(len(payloads), 4)

    def test_finalize_counts_current_run_including_empty_and_resumed_runs(self):
        for count, start_index in [(0, 0), (5, 0), (3, 8), (0, 8)]:
            with self.subTest(count=count, start_index=start_index), tempfile.TemporaryDirectory() as root:
                converter = SceneConverterBase('motion', root, root, max_sample_per_shard=2)
                converter.shard_manager.shard_index = start_index
                for index in range(count):
                    converter.shard_manager.add_sample(sample_id=index)
                converter.finalize()
                info = json.loads((Path(root) / 'info_motion.json').read_text())
                self.assertEqual(info['total_samples'], count)
                self.assertEqual(info['total_shards'], (count + 1) // 2)
                self.assertEqual(len(list(Path(root).glob('*.tar'))), (count + 1) // 2)

    def test_empty_frames_and_invalid_arrays_do_not_create_shards(self):
        with tempfile.TemporaryDirectory() as root:
            manager = ShardManager('motion', root)
            manager.add_sample(flattened_tar_files={'poses': np.empty((0, 4))}, seq_len=0)
            with self.assertRaises(ValueError):
                manager.add_sample(flattened_tar_files={'poses': np.zeros((2, 4))}, seq_len=3)
            with self.assertRaises(ValueError):
                manager.add_sample(nested_tar_files={}, flattened_tar_files={})
            with self.assertRaises(TypeError):
                manager.add_sample(nested_tar_files={'poses': ['unused.png']})
            manager.close()
            self.assertEqual(list(Path(root).glob('*.tar')), [])

    def test_oakink_motion_and_mesh_samples(self):
        with tempfile.TemporaryDirectory() as root:
            input_dir = Path(root) / 'input'
            mesh_dir = input_dir / 'oakink_aligned'
            mesh_dir.mkdir(parents=True)
            np.save(input_dir / 'mug_lift.npy', {
                'right_hand': {'trans': np.zeros((3, 3))}, 'mug_001': {},
            })
            (mesh_dir / 'mug_002.ply').touch()
            output_dir = Path(root) / 'output'
            with OakInkConverter(str(input_dir), str(output_dir)) as converter:
                converter.convert_dataset(input_dir)
                converter.finalize()
            samples = read_samples(output_dir)
            self.assertEqual(len(samples), 3)
            self.assertEqual(samples[0]['seq_len'], 3)
            self.assertEqual([sample['text'] for sample in samples[1:]],
                             ['drink the mug', 'toast the mug'])
            self.assertTrue(all('seq_len' not in sample for sample in samples[1:]))


if __name__ == '__main__':
    unittest.main()
