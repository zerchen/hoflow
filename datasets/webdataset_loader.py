import os
import torch
import numpy as np


class WebDatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task = cfg.task
        self.dataset_classes = {}
        self.train_shard_paths, self.test_shard_paths = [], []
        self.train_dataset_multipliers = {}
        self.train_dataset_weights = None

        train_dataset_names = cfg.trainset_3d.split('-')
        train_dataset_splits = cfg.trainset_3d_split.split('-')
        assert len(train_dataset_names) == len(train_dataset_splits), \
            "trainset_3d and trainset_3d_split should have same number of datasets"

        train_shard_paths_by_dataset = {}
        for dataset_name, data_split in zip(train_dataset_names, train_dataset_splits):
            exec(f'from datasets.{dataset_name}.{dataset_name} import {dataset_name}')
            self.dataset_classes[dataset_name] = eval(dataset_name)(cfg, data_split)
            train_shard_paths_by_dataset[dataset_name] = self.dataset_classes[dataset_name].train_shard_path

        self.train_dataset_weights = self._parse_trainset_sampling_weights(train_dataset_names)
        if self.train_dataset_weights is None:
            for dataset_name in train_dataset_names:
                self.train_shard_paths.extend(train_shard_paths_by_dataset[dataset_name])
                self.train_dataset_multipliers[dataset_name] = 1
        else:
            self.train_shard_paths = self._build_weighted_train_shard_paths(
                train_dataset_names, train_shard_paths_by_dataset, self.train_dataset_weights
            )

        test_dataset_names = cfg.testset.split('-')
        test_dataset_splits = cfg.testset_split.split('-')
        assert len(test_dataset_names) == len(test_dataset_splits), \
            "testset and testset_split should have same number of datasets"
        for dataset_name, data_split in zip(test_dataset_names, test_dataset_splits):
            exec(f'from datasets.{dataset_name}.{dataset_name} import {dataset_name}')
            self.dataset_classes[dataset_name] = eval(dataset_name)(cfg, data_split)
            self.test_shard_paths.extend(self.dataset_classes[dataset_name].test_shard_path)

    def _parse_trainset_sampling_weights(self, train_dataset_names):
        weights_cfg = getattr(self.cfg, "trainset_3d_sampling_weights", "")
        if weights_cfg is None:
            return None
        if not isinstance(weights_cfg, str) and len(weights_cfg) == 0:
            return None

        if isinstance(weights_cfg, str):
            weights_cfg = weights_cfg.strip()
            if weights_cfg == "":
                return None

            if ":" in weights_cfg:
                # Format: "grab:0.4,graspxl:0.4,dexycb:0.2"
                parsed = {}
                for part in weights_cfg.split(","):
                    token = part.strip()
                    if token == "":
                        continue
                    if ":" not in token:
                        raise ValueError(
                            "Invalid trainset_3d_sampling_weights format. "
                            "Use either 'w1-w2-...' or 'name:w,name:w'."
                        )
                    name, value = token.split(":", 1)
                    parsed[name.strip()] = float(value.strip())
                weights = [parsed.get(name, 0.0) for name in train_dataset_names]
            else:
                # Format aligned with trainset_3d order: "0.4-0.4-0.2"
                tokens = [x.strip() for x in weights_cfg.split("-") if x.strip() != ""]
                if len(tokens) != len(train_dataset_names):
                    raise ValueError(
                        f"Expected {len(train_dataset_names)} weights in trainset_3d_sampling_weights, "
                        f"got {len(tokens)}"
                    )
                weights = [float(x) for x in tokens]
        else:
            weights = [float(x) for x in weights_cfg]
            if len(weights) != len(train_dataset_names):
                raise ValueError(
                    f"Expected {len(train_dataset_names)} weights in trainset_3d_sampling_weights, "
                    f"got {len(weights)}"
                )

        weights = np.asarray(weights, dtype=np.float64)
        if np.any(weights < 0):
            raise ValueError("trainset_3d_sampling_weights should be non-negative")
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("trainset_3d_sampling_weights sum should be > 0")
        return {name: float(w / weight_sum) for name, w in zip(train_dataset_names, weights)}

    def _build_weighted_train_shard_paths(self, train_dataset_names, train_shard_paths_by_dataset, train_dataset_weights):
        # webdataset(resampled=True) samples from shard URLs, so we realize dataset-level
        # probabilities by repeating each dataset's shard list with an integer multiplier.
        ratios = []
        for dataset_name in train_dataset_names:
            n_shards = len(train_shard_paths_by_dataset[dataset_name])
            if n_shards <= 0:
                raise ValueError(f"Dataset {dataset_name} has no train shards")
            ratios.append(train_dataset_weights[dataset_name] / float(n_shards))

        min_ratio = min(r for r in ratios if r > 0)
        weighted_train_shard_paths = []
        for dataset_name in train_dataset_names:
            shard_paths = train_shard_paths_by_dataset[dataset_name]
            ratio = train_dataset_weights[dataset_name] / float(len(shard_paths))
            multiplier = max(1, int(round(ratio / min_ratio)))
            self.train_dataset_multipliers[dataset_name] = multiplier
            weighted_train_shard_paths.extend(shard_paths * multiplier)

        return weighted_train_shard_paths

    def make_sample(self, sample, stage):
        if self.task in ["motion_ae", "motion_vae"]:
            return self.make_frame_sample(sample, stage)
        elif self.task in ["seq_motion_ae", "inter_vae"]:
            return self.make_seq_sample(sample, stage)
        elif self.task in ["motion_ar", "autoreg_flow", "seq_motion_diffusion"]:
            if stage == "train":
                return self.make_latent_sample(sample, stage)
            else:
                return self.make_seq_sample(sample, stage)

    def make_seq_sample(self, sample, stage):
        dataset_name = sample["meta.json"]["source"]
        return self.dataset_classes[dataset_name].make_seq_sample(sample, stage)

    def make_latent_sample(self, sample, stage):
        dataset_name = sample["meta.json"]["source"]
        return self.dataset_classes[dataset_name].make_latent_sample(sample, stage)

    def make_frame_sample(self, sample, stage):
        dataset_name = sample["meta.json"]["source"]
        return self.dataset_classes[dataset_name].make_frame_sample(sample, stage)
