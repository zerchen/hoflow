import sys
import os
import random
import torch
from torch import distributed as dist
from copy import deepcopy
from webdataset.pytorch import IterableDataset
import webdataset.utils as utils
from webdataset.shardlists import expand_source


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of URLs with resampling."""

    def __init__(self, urls, nshards=sys.maxsize, seed=0, worker_seed=None, deterministic=False, max_urls=int(1e6), empty_check=True):
        """Initialize the ResampledShards.

        Args:
            urls: A list of URLs as a Python list or brace notation string.
            nshards (int): The number of shards to yield. Defaults to sys.maxsize.
            seed (int): The seed for random number generation.
            worker_seed (Callable or None): A function to generate worker-specific seeds.
            deterministic (bool): Whether to use deterministic sampling.
            max_urls (int): Maximum number of URLs to consider.
            empty_check (bool): Whether to check for empty URL list.

        Raises:
            ValueError: If empty_check is True and no shards are found.
        """
        super().__init__()
        self.urls = expand_source(urls, max_urls)
        if empty_check:
            if len(self.urls) == 0:
                raise ValueError("empty_check=True, but no shards found in ResampledShards")
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.worker_seed = utils.pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.seed = seed
        self.epoch = -1

    def __iter__(self):
        """Return an iterator over the shards.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        self.epoch += 1
        seed = utils.make_seed(self.seed)
        if os.environ.get("WDS_SHOW_SEED", "0") == "1":
            print(f"# ResampledShards seed {seed}")
        
        self.rng = random.Random(seed)
        urls = deepcopy(self.urls)
        self.rng.shuffle(urls)

        if len(urls) == 0:
            raise ValueError("empty_check=True, but no shards found in ResampledShards")
        for _ in range(self.nshards):
            self.rng.shuffle(urls)

            for url in urls:
                yield dict(url=url)