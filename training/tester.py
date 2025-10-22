import copy
import gc
import json
import logging
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Iterable

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from training.optimizer import construct_optimizer

from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
    exclude_frozen_params_byside_dora,
)

from training.utils.logger import Logger, setup_logging
from training.optimizer import construct_optimizer
from training.utils.data_utils import BatchedSrcTgtDatapoint
from training.utils.distributed import all_reduce_max, barrier, get_rank
from training.utils.eval import GlobalEvaluator

from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    print_model_summary,
)


from training.trainer import (
    CORE_LOSS_KEY, unwrap_ddp_if_wrapped,
    LoggingConf, CheckpointConf, OptimConf
)


class Tester:
    EPSILON = 1e-8
    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any] = None,
        optim: Dict[str, Any] = None,
        evaluator: Dict[str, Any] = None,
        logging: Dict[str, Any] = None,
        checkpoint: Dict[str, Any] = None,
        seed_value: int = 123,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ):
        self._setup_timers()
        
        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters
        self.loss_conf = loss
        self.eval_conf = evaluator

        self._setup_device()
        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )
        set_seeds(seed_value, 1, self.distributed_rank)

        self._setup_components()  # Except Optimizer everything is setup here.
        self._move_to_device()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        self.load_checkpoint()
        

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TEST], 0)
    
    def _setup_device(self):
        self.local_rank = 0
        self.distributed_rank = 0
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.local_rank)
    
    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )

        self.model.to(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def _auto_parse_dataset_cfg(self, phase):
        auto_parse = self.data_conf.get(phase).get("auto_parse_datasets", None)
        info_file_override = self.data_conf.get("meta", {}).get(f"{phase}_split", None)

        if auto_parse is not None:
            if info_file_override is not None:
                logging.info(f"Overriding {phase} data info_file to {info_file_override}.")
            
            data_cfg = self.data_conf.get(phase).datasets
            root_folder = data_cfg[0].dataset.datasets[0].syntra_dataset.root_folder
            if auto_parse == 'all':
                dataset_names = [x for x in os.listdir(root_folder) if g_pathmgr.isdir(os.path.join(root_folder, x))]
            elif isinstance(auto_parse, Iterable):
                dataset_names = auto_parse
            else:
                raise ValueError(f"Unknown auto_parse_datasets value {auto_parse}, should be 'all' or a list of dataset names.")
            multipliers = self.data_conf.get(phase).get("multipliers", [1.0] * len(dataset_names))
            logging.info(f"Auto parsing datasets from {root_folder}, found {dataset_names}.")
            
            datasets_cfg = []
            for i, dn in enumerate(dataset_names):
                cur_dataset_cfg = copy.deepcopy(data_cfg[0].dataset.datasets[0])
                cur_dataset_cfg.syntra_dataset.root_folder = os.path.join(root_folder, dn)
                cur_dataset_cfg.multiplier = multipliers[i] 
                datasets_cfg.append(cur_dataset_cfg)
                if info_file_override is not None:
                    cur_dataset_cfg.syntra_dataset.data_info_file = info_file_override
            data_cfg[0].dataset.datasets = datasets_cfg

    def _setup_components(self):
        assert Phase.TEST in self.data_conf, f"Test phase {Phase.TEST} not found in data config."

        self._auto_parse_dataset_cfg(Phase.TEST)

        logging.info("Setting up teset components: Model, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TEST: 0}

        self.logger = open(os.path.join(self.logging_conf.log_dir, "test_log.txt"), "a") if self.distributed_rank == 0 else None
        makedir(os.path.join(self.logging_conf.log_dir, "test_results"))
        self.evaluator = instantiate(self.eval_conf, _convert_="all")
        self.evaluator.set_logger(self.logger)
        self.model = instantiate(self.model_conf, _convert_="all")

        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el  # wrap_base_loss(el)
                for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)

        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        logging.info("Finished setting up test components: Model, meters etc.")
    
    def _setup_dataloaders(self):
        self.test_dataset = instantiate(self.data_conf.get(Phase.TEST, None))
    
    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        assert os.path.exists(ckpt_path), \
            f"The checkpoint folder {self.checkpoint_conf.save_dir} does not contain a checkpoint to load from!"
        if self.checkpoint_conf.initialize_after_preemption or self.model.dora_rank > 0:
            self._call_model_initializer()
        self._load_resuming_checkpoint(ckpt_path)
    
    def _init_model_state(self):
        # Checking that parameters that won't be saved are indeed frozen
        # We do this check here before even saving the model to catch errors
        # are early as possible and not at the end of the first epoch
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )

        # Checking that parameters that won't be saved are initialized from
        # within the model definition, unless `initialize_after_preemption`
        # is explicitly set to `True`. If not, this is a bug, and after
        # preemption, the `skip_saving_parameters` will have random values
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self):
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

            if self.model.dora_rank > 0:
                self.model.dora_adapt()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            dora_adapted = self.model.dora_rank > 0,
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )

    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _step(
        self,
        batch: BatchedSrcTgtDatapoint,
        model: nn.Module,
        phase: str,
    ):

        outputs = model(batch)
        targets = batch.tgt_mask_batch
        batch_size = len(batch.img_batch)

        self.evaluator.add_samples(batch, outputs['pred_masks_high_res'])

        key = batch.dict_key  # key for dataset
        loss = self.loss[key](outputs, targets)
        loss_str = f"Losses/{phase}_{key}_loss"

        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages=outputs,
                        find_metadatas=batch.metadata,
                    )

        return ret_tuple

    def run(self):
        dataloader = self.test_dataset.get_loader(epoch=self.epoch)
        outs = self.test_epoch(dataloader, phase=Phase.TEST)
        del dataloader
        gc.collect()
        self.logger.close()

    def test_epoch(self, test_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(test_loader)

        curr_phases = [phase]
        curr_models = [self.model]

        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Test Epoch: [{}]".format(self.epoch),
        )

        end = time.time()

        for data_iter, batch in enumerate(test_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            batch = batch.to(self.device, non_blocking=True)

            # compute output
            with torch.no_grad():
                with torch.amp.autocast("cuda",
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        loss_dict, batch_size, extra_losses = self._step(
                            batch,
                            model,
                            phase,
                        )

                        assert len(loss_dict) == 1
                        loss_key, loss = loss_dict.popitem()

                        loss_mts[loss_key].update(loss.item(), batch_size)

                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            extra_loss_mts[k].update(v.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            if data_iter % 10 == 0:
                progress.display(data_iter)
                # dist.barrier()

        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        for model in curr_models:
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_end()

        iou, prec, rec, acc = self.evaluator.eval()
        out_dict = {"IoU": iou, "Precision": prec, "Recall": rec, "Accuracy": acc}
        logging.info(f"Test IoU: {iou}, Precision: {prec}, Recall: {rec}, Accuracy: {acc}")
        return out_dict
    
    def _return_core_loss(self, loss, loss_str, step):
        core_loss = loss.pop(CORE_LOSS_KEY)
        return core_loss

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()
    
    def _log_sync_data_times(self, phase, data_times):
        data_times = all_reduce_max(torch.tensor(data_times)).tolist()
        steps = range(self.steps[phase] - len(data_times), self.steps[phase])
        for step, data_time in zip(steps, data_times):
            if step % self.logging_conf.log_scalar_frequency == 0:
                self.logger.log(
                    os.path.join("Step_Stats", phase, "Data Time Synced"),
                    data_time,
                    step,
                )

