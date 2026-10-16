"""CPU regression checks for raw sequences, shard discovery, and latent export."""

import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from precompute import export_latents, load_vae_weights
from datasets.shards import find_shards
from prep_data.dexycb.processor import read_sequence, process_sequence
from prep_data.shard_manager import ShardManager


class ReleasePipelineTests(unittest.TestCase):
    def test_shard_discovery_accepts_custom_sizes_and_missing_directories(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'latent_tars'
            self.assertEqual(find_shards(root, 'latent_tars', 'grab'), [])
            folder.mkdir()
            for name in ('grab-000003.tar', 'grab-000000.tar', 'other-000000.tar', 'grab-invalid.tar'):
                (folder / name).touch()
            self.assertEqual([Path(p).name for p in find_shards(root, 'latent_tars', 'grab')],
                             ['grab-000000.tar', 'grab-000003.tar'])

    def test_raw_dexycb_split_invalid_frames_and_rigid_transforms(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            sequence = root / '20200709-subject-01' / 'sequence'
            sequence.mkdir(parents=True)
            meta = {'ycb_grasp_ind': 0, 'ycb_ids': [14], 'mano_sides': ['left'], 'mano_calib': ['subject']}
            (sequence / 'meta.yml').write_text(yaml.safe_dump(meta))
            calib = root / 'calibration' / 'mano_subject'
            calib.mkdir(parents=True)
            (calib / 'mano.yml').write_text(yaml.safe_dump({'betas': [0.0] * 10}))
            mesh = root / 'models' / '025_mug'
            mesh.mkdir(parents=True)
            (mesh / 'textured_simple.obj').touch()
            hand = np.zeros((3, 1, 51), dtype=np.float32)
            hand[:2, 0, 48:] = [1., 2., 3.]
            obj = np.zeros((3, 1, 7), dtype=np.float32)
            obj[:, 0, :4] = [0., 0., np.sqrt(0.5), np.sqrt(0.5)]
            obj[:, 0, 4:] = [4., 5., 6.]
            np.savez(sequence / 'pose.npz', pose_m=hand, pose_y=obj)
            self.assertIsNone(read_sequence(root, sequence / 'pose.npz', 'train'))
            data = read_sequence(root, sequence / 'pose.npz', 'test')
            self.assertEqual(data['hand_side'], 'left')
            self.assertEqual(data['hand'].shape, (2, 51))
            np.testing.assert_allclose(data['obj_transf'][0, :3, :3],
                                       [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
            np.testing.assert_array_equal(data['obj_transf'][0, :3, 3], [4, 5, 6])

    def test_dexycb_centering_and_tar_schema(self):
        data = {
            'id': 'subject-sequence', 'obj_name': '025_mug', 'hand_side': 'left',
            'hand': np.ones((2, 51), dtype=np.float32), 'betas': np.zeros(10),
            'obj_transf': np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy(),
        }
        data['obj_transf'][:, :3, 3] = [[1, 2, 3], [2, 3, 4]]
        fitted = (
            np.zeros((2, 48), dtype=np.float32),
            np.full((2, 3), 4., dtype=np.float32),
            np.full((2, 778, 3), 6., dtype=np.float32),
            np.full((2, 21, 3), 5., dtype=np.float32),
            np.broadcast_to(np.eye(3), (2, 16, 3, 3)).copy(),
        )
        with patch('prep_data.grab.processor.solve_hand', return_value=fitted) as fit:
            sample = process_sequence(data)
        self.assertFalse(fit.call_args.kwargs['flat_hand_mean'])
        arrays = sample['nested_tar_files']
        np.testing.assert_array_equal(arrays['h_trans'][0], [3, 2, 1])
        np.testing.assert_array_equal(arrays['obj_transf'][0, :3, 3], [0, 0, 0])
        np.testing.assert_array_equal(arrays['h_transf'][0, 0, :3, 3], [1, 1, 1])
        with tempfile.TemporaryDirectory() as root:
            writer = ShardManager('dexycb', root)
            writer.add_sample_from_dict(sample)
            writer.close()
            with tarfile.open(Path(root) / 'dexycb-000000.tar') as shard:
                names = shard.getnames()
                meta = json.load(shard.extractfile(names[0]))
                self.assertEqual(meta['text'], 'grasp the mug')
                self.assertEqual(meta['hand_side'], 'left')
                self.assertEqual(len(names), 7)

    def test_latent_export_loads_into_generator_reader(self):
        if importlib.util.find_spec('webdataset') is None or importlib.util.find_spec('kornia') is None:
            self.skipTest('Install project dependencies for WebDataset integration')
        import webdataset as wds
        from datasets.dexycb.dexycb import dexycb

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def precompute(self, inputs):
                assert not self.training and not torch.is_grad_enabled()
                return {'lh_obj_mean': torch.ones(32, 24), 'lh_mean': torch.full((32, 24), 2.)}

        encoder = Encoder()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            mesh = root / '025_mug' / 'textured_simple.obj'
            mesh.parent.mkdir()
            mesh.touch()
            samples = [({'init_obj_rotmat': torch.eye(3)[None], 'text': ['grasp the mug']},
                        {'id': ['subject-sequence'], 'obj_id': [str(mesh)]})]
            writer = ShardManager('dexycb', str(root))
            export_latents(encoder, samples, writer, torch.device('cpu'))
            writer.close()
            sample, = list(wds.WebDataset(str(root / 'dexycb-000000.tar'), shardshuffle=False).decode())
            reader = dexycb.__new__(dexycb)
            reader.cfg = SimpleNamespace(window_size=96)
            reader.object_pcds = {'025_mug': np.zeros((1024, 3), dtype=np.float32)}
            inputs, targets, metas = reader.make_latent_sample(sample)
            self.assertEqual(inputs['text'], 'grasp the mug')
            self.assertEqual(metas['id'], 'subject-sequence')
            self.assertEqual(targets['rh_mean'].shape, (32, 24))
            torch.testing.assert_close(targets['rh_mean'], targets['lh_mean'])
            torch.testing.assert_close(targets['rh_obj_mean'], torch.ones(32, 24))

    def test_vae_checkpoint_prefixes_are_loaded_strictly(self):
        for prefix in ('', 'module.', 'module._orig_mod.'):
            model = torch.nn.Linear(2, 1)
            state = {prefix + key: torch.ones_like(value) for key, value in model.state_dict().items()}
            load_vae_weights(model, {'network': state})
            self.assertTrue(all(torch.all(value == 1) for value in model.state_dict().values()))
        with self.assertRaises(RuntimeError):
            load_vae_weights(model, {'network': {}})


if __name__ == '__main__':
    unittest.main()
