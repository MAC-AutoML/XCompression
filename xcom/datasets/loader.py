# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

from xcom.core.config import cfg
from xcom.datasets.cifar10 import xcom_Cifar10
from xcom.datasets.imagenet import xcom_ImageFolder
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler


# Supported datasets
_DATASETS = {"cifar10": xcom_Cifar10, "imagenet": xcom_ImageFolder}


def construct_loader(dataset_name, split_list, batch_size):
    if cfg.DATA_LOADER.MEMORY_DATA:
        _DATA_DIR = "/userhome/temp_data"
    else:
        _DATA_DIR = "/gdata"
    # Relative data paths to default data directory
    _PATHS = {"cifar10": "cifar10/cifar-10-batches-py",
              "imagenet": "ImageNet2012"}
    # Constructs the data loader for the given dataset
    assert dataset_name in _DATASETS and dataset_name in _PATHS, "Dataset '{}' not supported".format(
        dataset_name)
    # Retrieve the data path for the dataset
    data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name])
    print("reading data from {}".format(data_path))
    # Construct the dataset
    loader = _DATASETS[dataset_name](
        data_path, split_list, backend=cfg.DATA_LOADER.BACKEND, batch_size=batch_size)
    return loader


# def shuffle(loader, cur_epoch):
#     err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
#     assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
#     # RandomSampler handles shuffling automatically
#     if isinstance(loader.sampler, DistributedSampler):
#         # DistributedSampler shuffles data based on epoch
#         loader.sampler.set_epoch(cur_epoch)
