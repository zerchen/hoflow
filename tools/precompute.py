#!/usr/bin/env python3
"""Export sequence VAE latents on one GPU without starting or resuming training."""

import argparse
import importlib
import json
import os
from pathlib import Path

import _init_paths
from _init_paths import add_task_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', '-e', required=True)
    parser.add_argument('--ckpt', required=True, help='Trained inter_vae checkpoint')
    parser.add_argument('--gpu', default='0', help='One GPU index')
    parser.add_argument('--window_size', type=int, default=160)
    parser.add_argument('--max_sample_per_shard', type=int, default=1000)
    parser.add_argument('--output_dir', help='Default: datasets/<dataset>/data/latent_tars')
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error('--gpu must select exactly one GPU')
    if args.window_size <= 0 or args.window_size % 4:
        parser.error('--window_size must be a positive multiple of 4')
    if args.max_sample_per_shard < 1:
        parser.error('--max_sample_per_shard must be positive')
    if not Path(args.ckpt).is_file():
        parser.error(f'Checkpoint not found: {args.ckpt}')
    return args


def load_vae_weights(model, checkpoint):
    """Accept plain, DDP, and compiled VAE state dictionaries."""
    weights = {
        key.removeprefix('module.').removeprefix('_orig_mod.'): value
        for key, value in checkpoint['network'].items()
    }
    model.load_state_dict(weights)


def export_latents(model, samples, writer, device):
    """Encode a batch-size-one iterator, retaining conditioning metadata."""
    import torch
    from tqdm import tqdm

    model.eval()
    with torch.no_grad():
        for inputs, metas in tqdm(samples, desc='Exporting latents'):
            inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            latent = model.precompute(inputs)
            if not latent:
                continue
            arrays = {key: value.detach().cpu().numpy() for key, value in latent.items()}
            arrays['init_obj_rotmat'] = inputs['init_obj_rotmat'][0].cpu().numpy()
            sample_id = str(metas['id'][0])
            writer.add_sample(
                sample_id=sample_id,
                nested_tar_files=arrays,
                id=sample_id,
                obj_id=str(Path(metas['obj_id'][0]).resolve()),
                text=inputs['text'][0],
            )


def main():
    args = parse_args()
    args.cfg = str(Path(args.cfg).resolve())
    args.ckpt = str(Path(args.ckpt).resolve())
    if args.output_dir:
        args.output_dir = str(Path(args.output_dir).resolve())
    # Models use repository-relative MANO paths from tools/.
    os.chdir(Path(__file__).resolve().parent)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    task = add_task_path(args.cfg)
    if task != 'inter_vae':
        raise ValueError('Use a inter_vae experiment for latent export')

    import torch
    import webdataset as wds
    from torch.utils.data import default_collate
    from functools import partial
    from config import cfg
    from net import get_model
    from prep_data.shard_manager import ShardManager

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    dataset_name = cfg.trainset_3d
    if dataset_name not in ('grab', 'graspxl', 'dexycb'):
        raise ValueError('Export one dataset at a time: grab, graspxl, or dexycb')
    cfg.window_size = cfg.test_window_size = args.window_size
    cfg.use_3d_aug = False
    cfg.freeze()
    data_root = Path(_init_paths.root_path).resolve() / 'datasets' / dataset_name / 'data'
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_root / 'latent_tars'
    if any(output_dir.glob('*.tar')):
        raise FileExistsError(f'Latent shards already exist in {output_dir}; choose an empty --output_dir')

    dataset_cls = getattr(importlib.import_module(f'datasets.{dataset_name}.{dataset_name}'), dataset_name)
    dataset = dataset_cls(cfg, cfg.trainset_3d_split)
    if not dataset.train_shard_path:
        raise FileNotFoundError(f'No sequence training shards found under {data_root}')
    stream = wds.WebDataset(dataset.train_shard_path, shardshuffle=False).decode().map(
        partial(dataset.make_seq_sample, mode='test')
    )
    samples = wds.WebLoader(stream, batch_size=1, num_workers=0, collate_fn=default_collate)
    model = get_model(cfg, is_train=False).cuda()
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    load_vae_weights(model, checkpoint)
    del checkpoint

    writer = ShardManager(dataset_name, str(output_dir), args.max_sample_per_shard)
    try:
        export_latents(model, samples, writer, torch.device('cuda'))
    finally:
        writer.close()
    info = {
        'dataset': dataset_name,
        'vae_checkpoint': str(Path(args.ckpt).resolve()),
        'window_size': args.window_size,
        'total_samples': writer.sample_count,
        'total_shards': writer.shard_count,
    }
    (output_dir / f'info_{dataset_name}.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    main()
