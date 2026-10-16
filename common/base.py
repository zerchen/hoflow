import os
import os.path as osp
import math
import time
import glob
import abc
import importlib
from loguru import logger
from tqdm import tqdm
import numpy as np
import webdataset as wd
import torch
import torch.optim
import torchvision.transforms as transforms
from webdataset.shardlists import split_by_node
from datasets.webdataset_loader import WebDatasetBuilder
from torch.utils.data import DataLoader, default_collate
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, MultiStepLR
from fvcore.nn.parameter_count import parameter_count
from utils.timer import Timer
from config import cfg
from net import get_model
from functools import partial


class Base(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, log_name ='logs.txt'):
        self.cur_epoch = 0
        # timer
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        # logger
        self.logger = logger
        self.logger.add(osp.join(cfg.log_dir, log_name))

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.999)), weight_decay=cfg.get("weight_decay", 0.0))

        return optimizer

    def get_scheduler(self, optimizer):
        if cfg.warmup_iters > 0:
            warmup_scheduler = LinearLR(optimizer, total_iters=cfg.warmup_iters)
            if cfg.lr_dec_style == 'cosine':
                decay_scheduler = CosineAnnealingLR(optimizer, T_max=(cfg.num_iters - cfg.warmup_iters), eta_min=cfg.get("end_lr", 1e-5))
            elif cfg.lr_dec_style == 'step':
                decay_scheduler = MultiStepLR(optimizer, milestones=cfg.lr_dec_iters, gamma=cfg.lr_dec_factor)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[cfg.warmup_iters])
        else:
            if cfg.lr_dec_style == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_iters, eta_min=cfg.get("end_lr", 1e-5))
            elif cfg.lr_dec_style == 'step':
                scheduler = MultiStepLR(optimizer, milestones=cfg.lr_dec_iters, gamma=cfg.lr_dec_factor)

        return scheduler

    def save_model(self, state, iters):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(iters)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer, scaler):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
        if len(model_file_list) == 0:
            if os.path.exists(cfg.ckpt):
                ckpt = torch.load(cfg.ckpt, map_location=torch.device('cpu'))
                model.load_state_dict(ckpt['network'])
                start_iters = 0
                self.logger.info('Load checkpoint from {}'.format(cfg.ckpt))
                
                return start_iters, model, optimizer, scaler
            else:
                start_iters = 0
                self.logger.info('Start training from scratch')
                return start_iters, model, optimizer, scaler
        else:
            cur_iters = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
            ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_iters) + '.pth.tar')
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu')) 
            start_iters = ckpt['iters'] + 1
            model.load_state_dict(ckpt['network'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scaler.load_state_dict(ckpt['scaler'])
            
            self.logger.info('Continue training and load checkpoint from {}'.format(ckpt_path))
            return start_iters, model, optimizer, scaler

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def _make_precompute_generator(self):
        # data load and construct batch generator for precomputation
        self.logger.info("Creating dataset for precomputation")

        exec(f'from datasets.{cfg.trainset_3d}.{cfg.trainset_3d} import {cfg.trainset_3d}')
        trainset3d_class = eval(cfg.trainset_3d)(cfg, cfg.trainset_3d_split)

        precompute_db = wd.WebDataset(trainset3d_class.train_shard_path).decode().map(partial(trainset3d_class.make_sample, mode='train'))
        self.precompute_loader = wd.WebLoader(precompute_db, batch_size=1, num_workers=1, collate_fn=default_collate, prefetch_factor=4, persistent_workers=True)
        self.precompute_generator = self.precompute_loader
        self.logger.info(f"Finish building precompute generator")
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset for training")

        shuffle_size = 20 * cfg.train_batch_size
        trainset_db_info = WebDatasetBuilder(cfg)
        trainset_db = wd.WebDataset(trainset_db_info.train_shard_paths, nodesplitter=split_by_node, resampled=True).shuffle(shuffle_size).decode().map(partial(trainset_db_info.make_sample, stage='train'))
        self.trainset_loader = wd.WebLoader(trainset_db, batch_size=cfg.train_batch_size, num_workers=cfg.num_threads, collate_fn=default_collate, prefetch_factor=4, persistent_workers=True, pin_memory=True)
        self.batch_generator = self.trainset_loader
        self.tot_itrs = cfg.num_iters
        self.logger.info(f"Total training iterations: {self.tot_itrs}, num_workers: {cfg.num_threads}")
    
    def _get_model_info(self, model):
        num_params = parameter_count(model)
        self.logger.info(f"Model prameters: {num_params['']/1e6}M")

    def _make_model(self, local_rank):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model(cfg, True)
        model = model.cuda()
        self._get_model_info(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        use_compile = getattr(cfg, 'use_compile', False)
        compile_mode = getattr(cfg, 'compile_mode', 'reduce-overhead')
        if use_compile:
            self.logger.info(f"Compiling model with mode='{compile_mode}'...")
            model = torch.compile(model, mode=compile_mode)

        optimizer = self.get_optimizer(model)
        scaler = torch.amp.GradScaler()
        scheduler = self.get_scheduler(optimizer)
        model = NativeDDP(model, device_ids=[local_rank], output_device=local_rank)
        model.train()

        start_iters, model, optimizer, scaler = self.load_model(model, optimizer, scaler)

        self.start_iters = start_iters
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler


class Tester(Base):
    def __init__(self, local_rank, test_epoch):
        self.local_rank = local_rank
        self.test_epoch = test_epoch
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        self.logger.info(f"Creating dataset for testing")

        testset_db_info = WebDatasetBuilder(cfg)
        testset_db = wd.WebDataset(testset_db_info.test_shard_paths, nodesplitter=split_by_node, empty_check=False).decode().map(partial(testset_db_info.make_sample, stage='test'))
        self.testset_loader = wd.WebLoader(testset_db, batch_size=cfg.test_batch_size, num_workers=1, collate_fn=default_collate, prefetch_factor=4, persistent_workers=True)
        self.batch_generator = self.testset_loader
    
    def _make_model(self, local_rank):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(cfg, is_train=False)
        model = model.cuda()
        model = NativeDDP(model, device_ids=[local_rank], output_device=local_rank)
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['network'])
        model.eval()
        self.model = model
