#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :train.py
#@Date :2022/05/03 16:40:33
#@Author :zerui chen
#@Contact :zerui.chen@inria.fr

import os
import argparse
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
import _init_paths
from _init_paths import add_task_path


def parse_args():
    parser = argparse.ArgumentParser(description='Train a HO-Flow VAE or motion generator with torchrun.')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging.')
    parser.add_argument('--cfg', '-e', required=True, type=str)
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--local_rank', default=0, type=int)
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
    from base import Trainer, Tester
    from utils.dist_utils import reduce_tensor
    update_config(cfg, args)

    cudnn.benchmark = True
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    logger.info('Distributed Process %d, Total %d.' % (local_rank, world_size))

    if rank == 0 and args.wandb:
        import wandb

        wandb.init(project=cfg.task, name=os.path.basename(cfg.output_dir), dir=cfg.log_dir, resume="allow")

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model(local_rank)

    if rank == 0:
        writer_dict = {'train_global_steps': trainer.start_iters}

    for _ in range(trainer.start_iters):
        trainer.scheduler.step()

    trainer.read_timer.tic()
    for itr, (inputs, targets, metas) in zip(range(trainer.start_iters, cfg.num_iters), trainer.batch_generator):
        if itr >= cfg.num_iters:
            break

        trainer.read_timer.toc()
        trainer.gpu_timer.tic()

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = inputs[k].cuda(non_blocking=True)

        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                targets[k] = targets[k].cuda(non_blocking=True)

        for k, v in metas.items():
            if isinstance(v, torch.Tensor):
                metas[k] = metas[k].cuda(non_blocking=True)

        # forward
        trainer.optimizer.zero_grad()
        loss, _ = trainer.model(inputs, targets, metas, 'train')
        all_loss = sum(loss[k] for k in loss)

        if torch.isnan(all_loss) or torch.isinf(all_loss):
            logger.error(f"NaN/Inf detected in loss at iteration {itr}. Skipping backward pass.")
            trainer.save_model({'iters': itr, 'network': trainer.model.state_dict()}, itr)
            continue

        # backward
        trainer.scaler.scale(all_loss).backward()

        trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)

        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()

        # update EMA model
        if hasattr(cfg, 'use_ema') and cfg.use_ema:
            trainer.model.module.update_ema(cfg.ema_decay)

        torch.cuda.synchronize()

        trainer.gpu_timer.toc()
        
        allocated_memory_mb = torch.cuda.memory_reserved() / 1024 ** 2
        screen = ['itr %d/%d:' % (itr, cfg.num_iters), 'lr: %g' % (trainer.get_lr()), 'speed: (%.2fs r%.2f)s/itr' % (trainer.gpu_timer.average_time, trainer.read_timer.average_time), 'gpu_mem: %.1fMB' % allocated_memory_mb]

        record_dict = {}
        for k, v in loss.items():
            record_dict[k] = reduce_tensor(v.detach().float(), world_size) * 1000.
        screen += ['%s: %.3f' % ('loss_' + k, v) for k, v in record_dict.items()]

        if rank == 0:
            global_steps = writer_dict['train_global_steps']
            if itr % 10 == 0:
                trainer.logger.info(' '.join(screen))
                log_dict = {}
                for k, v in record_dict.items():
                    log_dict['loss_' + k] = v
                log_dict['lr'] = trainer.get_lr()
                log_dict['global_steps'] = global_steps
                if args.wandb:
                    wandb.log(log_dict, step=global_steps)
                writer_dict['train_global_steps'] = global_steps + 10

        trainer.read_timer.tic()
        trainer.scheduler.step()
        if rank == 0 and itr > 0 and (itr % cfg.model_save_freq == 0 or itr == cfg.num_iters - 1):
            trainer.save_model({'iters': itr, 'network': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict(), 'scaler': trainer.scaler.state_dict()}, itr)
        
    torch.cuda.empty_cache()
    tester = Tester(local_rank, cfg.num_iters - 1)
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

            tester.model(inputs, targets=None, metas=metas, mode='test')
    
    if rank == 0 and args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
