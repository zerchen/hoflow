"""Convert raw DexYCB sequence poses to WebDataset shards."""

import argparse
from pathlib import Path

from tqdm import tqdm

from prep_data.converter import SceneConverterBase
from .processor import process_sequence, read_sequence


class DexYCBConverter(SceneConverterBase):
    def __init__(self, input_dir, output_dir, mode='train', max_sample_per_shard=1000):
        self.mode = mode
        destination = Path(output_dir) / f'{mode}_seq_tars'
        if any(destination.glob('*.tar')):
            raise FileExistsError(f'Shards already exist in {destination}; choose an empty output directory')
        super().__init__('dexycb', input_dir, str(destination), max_sample_per_shard)

    def convert_dataset(self, dataset_dir=None):
        data_root = self.input_dir if dataset_dir is None else Path(dataset_dir)
        sequences = sorted(data_root.glob('*subject*/*/pose.npz'))
        if not sequences:
            raise FileNotFoundError(f'No <subject>/<sequence>/pose.npz files under {data_root}')
        for pose_path in tqdm(sequences, desc=f'DexYCB {self.mode}'):
            data = read_sequence(data_root, pose_path, self.mode)
            if data is not None:
                self.shard_manager.add_sample_from_dict(process_sequence(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', required=True, help='Raw DexYCB root, including calibration/ and models/')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--max_sample_per_shard', type=int, default=1000)
    args = parser.parse_args()
    if not Path(args.input_dir).is_dir():
        parser.error(f'Input directory does not exist: {args.input_dir}')
    with DexYCBConverter(**vars(args)) as converter:
        converter.convert_dataset()
        converter.finalize()


if __name__ == '__main__':
    main()
