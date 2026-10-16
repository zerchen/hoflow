#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :_init_paths.py
#@Date        :2022/04/07 23:37:58
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
from pathlib import Path


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

root_path = osp.join(this_dir, '..')
lib_path = osp.join(this_dir, '..', 'common')
networks_path = osp.join(this_dir, '..', 'common', 'networks')
add_path(root_path)
add_path(lib_path)
add_path(networks_path)

SUPPORTED_TASKS = ('inter_vae', 'autoreg_flow')


def add_task_path(config_file, supported_tasks=SUPPORTED_TASKS):
    """Resolve a task from an experiment YAML or a saved exp.yaml."""
    import yaml

    config_path = Path(config_file).expanduser().resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    task = config.get('task')
    if task is None and config_path.parent.name == 'experiments':
        task = config_path.parent.parent.name
    if task not in supported_tasks:
        supported = ', '.join(supported_tasks)
        raise ValueError(f"Unsupported task {task!r} in {config_path}; expected {supported}.")

    task_path = Path(root_path) / 'playground' / task
    for filename in ('config.py', 'net.py'):
        if not (task_path / filename).is_file():
            raise FileNotFoundError(task_path / filename)
    add_path(str(task_path))
    return task
