#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :config.py
#@Date :2025/08/13 17:09:06
#@Author :Zerui Chen
#@Contact :zeruicasia@gmail.com

import os
import os.path as osp
import sys
from yacs.config import CfgNode as CN
from loguru import logger
from contextlib import redirect_stdout


cfg = CN()

cfg.task = 'inter_vae'
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '..', '..')
cfg.data_dir = osp.join(cfg.root_dir, 'datasets')
cfg.output_dir = '.'
cfg.model_dir = './model_dump'
cfg.vis_dir = './vis'
cfg.log_dir = './log'
cfg.result_dir = './result'
cfg.ckpt = 'pretrained_model.ckpt'

## dataset
cfg.trainset_3d = 'grab'
cfg.trainset_3d_split = 's0_7k'
cfg.trainset_3d_sampling_weights = ()
cfg.testset = 'grab'
cfg.testset_split = 's0_1k'
cfg.window_size = 64

## model setting
cfg.latent_dim = 32
cfg.hidden_dim = 512
cfg.pointnet_dim = 96
cfg.n_layers = 2
cfg.stride = 2
cfg.n_depth = 3

## training config
cfg.warmup_iters = 500
cfg.num_iters = 50000
cfg.weight_decay = 1e-4
cfg.lr = 2e-4
cfg.end_lr = 2e-5
cfg.kl_weight = 0.0001
cfg.lr_dec_style = 'cosine'
cfg.train_batch_size = 8

## testing config
cfg.test_batch_size = 1
cfg.test_window_size = 160

## others
cfg.num_threads = 8
cfg.gpu_ids = 0
cfg.num_gpus = 1
cfg.checkpoint = 'model.pth.tar'
cfg.model_save_freq = 20000
cfg.use_3d_aug = True
cfg.use_compile = False
cfg.compile_mode = 'reduce-overhead'

def update_config(cfg, args, mode='train'):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.gpu_ids = args.gpu_ids
    cfg.num_gpus = len(cfg.gpu_ids.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logger.info('>>> Using GPU: {}'.format(cfg.gpu_ids))

    if mode == 'train':
        exp_info = [cfg.task, cfg.trainset_3d, 'pdim' + str(int(cfg.pointnet_dim)), 'ldim' + str(int(cfg.latent_dim)), 'nl' + str(int(cfg.n_layers)), 'ndepth' + str(cfg.n_depth), 'ws' + str(cfg.window_size), 'niter' + str(cfg.num_iters), 'b' + str(cfg.num_gpus * cfg.train_batch_size), '3daug' + str(int(cfg.use_3d_aug)), 'lr' + str(cfg.lr)]

        cfg.output_dir = osp.join(cfg.root_dir, 'outputs', cfg.task, '_'.join(exp_info))
        cfg.model_dir = osp.join(cfg.output_dir, 'model_dump')
        cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
        cfg.log_dir = osp.join(cfg.output_dir, 'log')
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.testset]))

        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.vis_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.result_dir, exist_ok=True)
        cfg.freeze()
        with open(osp.join(cfg.output_dir, 'exp.yaml'), 'w') as f:
            with redirect_stdout(f): print(cfg.dump())
    else:
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.testset]))
        cfg.log_dir = osp.join(cfg.output_dir, 'test_log')
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        cfg.freeze()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
sys.path.insert(0, osp.join(cfg.root_dir, 'common', 'networks'))
from utils.dir_utils import add_pypath
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
