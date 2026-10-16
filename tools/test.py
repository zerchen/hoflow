#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :test.py
#@Date :2022/05/03 16:40:33
#@Author :zerui chen
#@Contact :zerui.chen@inria.fr


import os
import argparse
from tqdm import tqdm
import torch
from loguru import logger
import _init_paths
from _init_paths import add_task_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-e', required=True, type=str)
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--test_epoch', default=0, type=int)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    # argument parse and create log
    args = parse_args()
    add_task_path(args.cfg)
    from config import cfg, update_config
    update_config(cfg, args, mode='test')
    from base import Tester
    if args.test_epoch == 0:
        args.test_epoch = cfg.num_iters - 1

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    logger.info('Distributed Process %d, Total %d.' % (local_rank, world_size))

    tester = Tester(local_rank, args.test_epoch)
    tester._make_batch_generator()
    tester._make_model(local_rank)
    with torch.no_grad():
        for itr, (inputs, metas) in tqdm(enumerate(tester.batch_generator)):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = inputs[k].cuda(non_blocking=True)

            for k, v in metas.items():
                if isinstance(v, torch.Tensor):
                    metas[k] = metas[k].cuda(non_blocking=True)

            # forward
            tester.model(inputs, targets=None, metas=metas, mode='test')


if __name__ == "__main__":
    main()
