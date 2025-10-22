# Modified from SAM2 (https://github.com/facebookresearch/sam2)
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser

import torch

from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from omegaconf import OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers
from training.utils.debug_utils import *

os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def main(args) -> None:
    add_pythonpath_to_sys_path()

    cfg = compose(config_name=args.config)
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "syntra_logs", args.config
        )
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    
    makedir(cfg.launcher.experiment_log_dir)
    with open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved) # a copy of cfg 

    with open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))
    
    # Priotrize cmd line args
    cfg.dataset.train_split = (
        args.data_train_split if args.data_train_split is not None else cfg.dataset.train_split
    )
    cfg.dataset.root = (
        args.data_root if args.data_root is not None else cfg.dataset.root
    )
    cfg.scratch.num_epochs = (
        args.max_epochs if args.batch_size is not None else cfg.scratch.num_epochs
    )
    cfg.scratch.num_train_workers = (
        args.num_workers if args.num_workers is not None else cfg.scratch.num_train_workers
    )
    
    main_port = random.randint(
            cfg.launcher.port_range[0], cfg.launcher.port_range[1]
        )
    single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=1)


if __name__=="__main__":
    initialize_config_module(config_module="syntra", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, 
                        help="config file path")
    parser.add_argument("-bs", "--batch_size", type=int, default=None, 
                        help="batch size for training")
    parser.add_argument("-dr", "--data-root", type=str, default=None, 
                        help="the root directory that contains all datasets")
    parser.add_argument("-e", "--max-epochs", type=int, default=None,
                        help="max training epochs")
    parser.add_argument("-nw", "--num-workers", type=int, default=None,
                        help="number of workers for data loading")
    parser.add_argument("-split", "--data_train_split", type=int, default=None, 
                        help="a json file that list all data samples for training")
    parser.add_argument("-dset", "--dataset", type=str, default=None, 
                        help="dataset names in a row, split by comma, if not set, all datasets will be used")
    args = parser.parse_args()
    register_omegaconf_resolvers()
    main(args)