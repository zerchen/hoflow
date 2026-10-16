#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :config.py
#@Date :2026/01/08 17:09:06
#@Author :Zerui Chen
#@Contact :zeruicasia@gmail.com

import os
import os.path as osp
import sys
from yacs.config import CfgNode as CN
from loguru import logger
from contextlib import redirect_stdout


cfg = CN()

cfg.task = 'autoreg_flow'
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
cfg.window_size = 160

## model setting
cfg.arch = "MARDM-SiT-XL"
cfg.mar_output_dim = 128
cfg.mar_num_layers = 1
cfg.mar_dropout = 0.0
cfg.mar_cond_drop_prob = 0.2

## ae setting
cfg.latent_dim = 32
cfg.hidden_dim = 512
cfg.pointnet_dim = 96
cfg.n_layers = 2
cfg.stride = 2
cfg.n_depth = 3
cfg.ae_ckpt = "outputs"

## training config
cfg.warmup_iters = 2000
cfg.num_iters = 300000
cfg.weight_decay = 1e-5
cfg.lr = 2e-4
cfg.end_lr = 2e-5
cfg.lr_dec_style = 'cosine'
cfg.train_batch_size = 32

## ema config
cfg.use_ema = True
cfg.ema_decay = 0.9999

## testing config
cfg.test_batch_size = 1
cfg.test_window_size = 160
cfg.time_steps = 32
cfg.temperature = 1.0
cfg.cfg_weight = 1.5
cfg.hard_pseudo_reorder = False

## others
cfg.num_threads = 8
cfg.gpu_ids = 0
cfg.num_gpus = 1
cfg.checkpoint = 'model.pth.tar'
cfg.model_save_freq = 50000

def update_config(cfg, args, mode='train'):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.gpu_ids = args.gpu_ids
    cfg.num_gpus = len(cfg.gpu_ids.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logger.info('>>> Using GPU: {}'.format(cfg.gpu_ids))

    if mode == 'train':
        exp_info = [cfg.task, cfg.arch, 'nmar' + str(cfg.mar_num_layers), 'dr' + str(cfg.mar_dropout), 'cdp' + str(cfg.mar_cond_drop_prob), cfg.trainset_3d, cfg.trainset_3d_split, 'nitr' + str(cfg.num_iters), 'b' + str(cfg.num_gpus * cfg.train_batch_size), 'lr' + str(cfg.lr)]

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
