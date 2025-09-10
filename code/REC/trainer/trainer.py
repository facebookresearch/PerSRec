# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.
import gc
import os
import sys
import tempfile
import time as t
from logging import getLogger, INFO
from time import time

import deepspeed
import numpy as np
import torch
import torch.optim as optim

from REC.config import get_file_name
from REC.data.dataset import BatchTextDataset
from REC.data.dataset.collate_fn import customize_rmpad_collate
from REC.evaluator import Collector, Evaluator
from REC.utils import (
    _PATH_MANAGER as PathManager,
    calculate_valid_score,
    dict2str,
    early_stopping,
    ensure_dir,
    get_tensorboard,
    set_color,
    WandbLogger,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from REC.utils.lr_scheduler import *
import REC.data.comm as comm

from typing import Dict, List, Optional, Union

from lightning.fabric.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy, DeepSpeedStrategy

class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)

        self.wandblogger = WandbLogger(config)

        self.optim_args = config['optim_args']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']

        self.rank = torch.distributed.get_rank()
        self.checkpoint_dir = config['checkpoint_dir']
        if self.checkpoint_dir.startswith('manifold://'):
            # create a local dir to save checkpoints
            self.manifold_dir = self.checkpoint_dir
            if comm.get_local_rank() == 0:
                # only rank 0 will be effective
                checkpoint_dir = tempfile.mkdtemp()
            else:
                checkpoint_dir = None
            shared_checkpoint_dir = comm.all_gather(checkpoint_dir, comm._LOCAL_PROCESS_GROUP)
            self.logger.info(f"broadcast {checkpoint_dir=}/{shared_checkpoint_dir=} for {comm.get_local_rank()=}")
            self.checkpoint_dir = [x for x in shared_checkpoint_dir if x is not None][0]
        else:
            self.manifold_dir = None
        if self.rank == 0:
            self.tensorboard = get_tensorboard(self.logger, os.path.join(self.checkpoint_dir, "log_tensorboard"))
            ensure_dir(self.checkpoint_dir)
            if self.manifold_dir is not None:
                ensure_dir(self.manifold_dir)

        self.use_text = config['use_text']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.best_epoch = None
        self.train_loss_dict = dict()
        self.update_interval = config['update_interval'] if config['update_interval'] else 20
        self.scheduler_config = config['scheduler_args']

        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_feature = None
        self.tot_item_num = None
        self._setup_fabric()

    def setup_model(self, model: torch.nn.Module) -> None:
        # set up model for train or inference
        self.model = model

        # set up learnable parameter
        self.logger.info("set up learnable parameter")
        if self.config['freeze_prefix'] or self.config['freeze_ad']:
            freeze_prefix = self.config['freeze_prefix'] if self.config['freeze_prefix'] else []
            if self.config['freeze_ad']:
                freeze_prefix.extend(['item_llm', 'item_emb_tokens'])
            if not self.config['ft_item']:
                freeze_prefix.extend(['item_embedding'])

            self._freeze_params(freeze_prefix)
        for n, p in self.model.named_parameters():
            self.logger.info(f"{n} {p.size()=} {p.requires_grad=} {p.device=}")

        # set up optimizer
        self.logger.info("set up optimizer")
        self.optimizer = self._build_optimizer()

    def _freeze_params(self, freeze_prefix):
        for name, param in self.model.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.logger.info(f"freeze_params: {name}")
                    param.requires_grad = False

    def _build_scheduler(self, warmup_steps=None, tot_steps=None):
        if self.scheduler_config['type'] == 'cosine':
            self.logger.info(f"Use consine scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        elif self.scheduler_config['type'] == 'liner':
            self.logger.info(f"Use linear scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        else:
            self.logger.info(f"Use constant scheduler")
            return get_constant_schedule(self.optimizer)

    def _build_optimizer(self):
        if len(self.optim_args) == 4:
            params = self.model.named_parameters()
            modal_params = []
            recsys_params = []
            modal_decay_params = []
            recsys_decay_params = []
            decay_check_name = self.config['decay_check_name']
            for index, (name, param) in enumerate(params):
                if param.requires_grad:
                    if 'visual_encoder' in name:
                        modal_params.append(param)
                    else:
                        recsys_params.append(param)
                    if decay_check_name:
                        if decay_check_name in name:
                            modal_decay_params.append(param)
                        else:
                            recsys_decay_params.append(param)
            if decay_check_name:
                optimizer = optim.AdamW([
                    {'params': modal_decay_params, 'lr': self.optim_args['modal_lr'], 'weight_decay': self.optim_args['modal_decay']},
                    {'params': recsys_decay_params, 'lr': self.optim_args['rec_lr'], 'weight_decay': self.optim_args['rec_decay']}
                ])
                optim_output = set_color(f'recsys_decay_params_len: {len(recsys_decay_params)}  modal_params_decay_len: {len(modal_decay_params)}', 'blue')
                self.logger.info(optim_output)
            else:
                optimizer = optim.AdamW([
                    {'params': modal_params, 'lr': self.optim_args['modal_lr'], 'weight_decay': self.optim_args['modal_decay']},
                    {'params': recsys_params, 'lr': self.optim_args['rec_lr'], 'weight_decay': self.optim_args['rec_decay']}
                ])
                optim_output = set_color(f'recsys_lr_params_len: {len(recsys_params)}  modal_lr_params_len: {len(modal_params)}', 'blue')
                self.logger.info(optim_output)
        elif self.config['lr_mult_prefix'] and self.config['lr_mult_rate']:
            normal_params_dict = {
                "params": [],
                "lr": self.optim_args['learning_rate'],
                "weight_decay": self.optim_args['weight_decay']
            }
            high_lr_params_dict = {
                "params": [],
                "lr": self.optim_args['learning_rate'] * self.config['lr_mult_rate'],
                "weight_decay": self.optim_args['weight_decay']
            }
            self.logger.info(f'Use higher lr rate {self.config["lr_mult_rate"]} x {self.optim_args["learning_rate"]} for prefix {self.config["lr_mult_prefix"]}')

            for n, p in self.model.named_parameters():
                if any(n.startswith(x) for x in self.config['lr_mult_prefix']):
                    self.logger.info(f"high lr param: {n} {self.optim_args['learning_rate'] * self.config['lr_mult_rate']}")
                    high_lr_params_dict["params"].append(p)
                else:
                    normal_params_dict["params"].append(p)
            optimizer = optim.AdamW([normal_params_dict, high_lr_params_dict])
        elif self.config['strategy'] == 'deepspeed':
            # need newer deepspeed version
            params = self.model.parameters()
            optimizer = deepspeed.ops.adam.fused_adam.FusedAdam(params, lr=self.optim_args['learning_rate'], weight_decay=self.optim_args['weight_decay'])
            # optimizer = deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam(params, lr=self.optim_args['learning_rate'], weight_decay=self.optim_args['weight_decay'])
            self.logger.info("using DeepSpeedCPUAdam due to deepspeed strategy")
        else:
            params = self.model.parameters()
            optimizer = optim.AdamW(params, lr=self.optim_args['learning_rate'], weight_decay=self.optim_args['weight_decay'])
            self.logger.info("AdamW")
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, show_progress=False):
        self.model.train()
        total_loss = 0
        if self.rank == 0:
            pbar = tqdm(
                total=len(train_data),
                miniters=self.update_interval,
                desc=set_color(f"Train [{epoch_idx:>3}/{self.epochs:>3}]", 'pink'),
                file=sys.stdout
            )
        bwd_time = t.time()
        for batch_idx, data in enumerate(train_data):
            start_time = bwd_time
            self.optimizer.zero_grad()
            data = self.to_device(data)
            data_time = t.time()
            losses = self.model(data)
            del data
            fwd_time = t.time()
            if self.config['loss'] and self.config['loss'] == 'nce':
                model_out = losses
                losses = model_out.pop('loss')
            self._check_nan(losses)
            total_loss = total_loss + losses.item()
            self.lite.backward(losses)
            grad_norm = self.optimizer.step()
            bwd_time = t.time()
            if self.scheduler_config:
                self.lr_scheduler.step()
            if show_progress and self.rank == 0 and batch_idx % self.update_interval == 0:
                msg = f"loss: {losses:.4f} data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f} bwd: {bwd_time-fwd_time:.3f}"
                if self.scheduler_config:
                    msg = f"lr: {self.lr_scheduler.get_lr()[0]:.7f} " + msg
                if self.config['loss'] and self.config['loss'] == 'nce':
                    for k, v in model_out.items():
                        msg += f" {k}: {v:.3f}"
                if grad_norm:
                    msg = msg + f" grad_norm: {grad_norm.sum():.4f}"
                pbar.set_postfix_str(msg, refresh=False)
                pbar.update(self.update_interval)
                self.logger.info("\n" + "-"*50)
            del losses
            if self.config['debug'] and batch_idx >= 10:
                break
        gc.collect()
        return total_loss

    def _valid_epoch(self, valid_data, epoch_idx: int, show_progress=False, item_text: Optional[List[Dict[str, str]]]=None):
        torch.distributed.barrier()
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress, item_text=item_text)
        del valid_data
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        gc.collect()
        torch.distributed.barrier()
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            "best_epoch": self.best_epoch,
        }
        # this only happens on rank 0
        saved_model_name = '{}-{}.pth'.format(self.config['model'], epoch)
        self.lite.save(os.path.join(self.checkpoint_dir, saved_model_name), state=state)
        del state
        if self.rank == 0:
            self.logger.info(f"Checkpoint saved to {os.path.join(self.checkpoint_dir, saved_model_name)}")
        if self.rank == 0 and self.manifold_dir is not None:
            # upload to manifold
            PathManager.copy_from_local(
                os.path.join(self.checkpoint_dir, saved_model_name),
                os.path.join(self.manifold_dir, saved_model_name),
                overwrite=True,
            )
            self.logger.info(f"Checkpoint uploaded to {os.path.join(self.manifold_dir, saved_model_name)}")
        # synchronize after saving checkpoint
        # make sure barrier is called by all GPUs!
        del saved_model_name
        gc.collect()
        torch.distributed.barrier()

    def _resume_from_checkpoint(self, checkpoint_path: Union[str, bool]):
        if isinstance(checkpoint_path, bool):
            # find the latest checkpoint
            checkpoint_dir = self.manifold_dir if self.manifold_dir is not None else self.checkpoint_dir
            names = PathManager.ls(checkpoint_dir)
            names = [name for name in names if name.endswith('.pth')]
            if len(names) < 1:
                self.logger.info(f"no checkpoint found in {checkpoint_dir}")
                return
            names = sorted(names, key=lambda x: int(x.split('-')[-1].split('.')[0]))
            checkpoint_path = os.path.join(checkpoint_dir, names[-1])
            self.logger.info(f"find the latest checkpoint {checkpoint_path} from {checkpoint_dir}")
            # TODO: this should only happen on rank 0
            if self.manifold_dir is not None:
                self.logger.info(f"downloading checkpoint from {checkpoint_path}")
                checkpoint_path = PathManager.get_local_path(checkpoint_path, recursive=True)
            torch.distributed.barrier()
        self.logger.info(f"auto resume is enabled, will try to load from {checkpoint_path}")
        # TODO: support dataloader. Now it only resumes from an epoch.
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
        }
        state = self.lite.load(checkpoint_path, state)
        self.start_epoch = state['epoch']
        self.best_epoch = state.get('best_epoch', self.start_epoch)
        self.cur_step = state['cur_step']
        self.best_valid_score = state['best_valid_score']
        torch.set_rng_state(state['rng_state'])
        torch.cuda.set_rng_state(state['cuda_rng_state'])
        # the model and optimizer should be already loaded inplace
        torch.distributed.barrier()
        del state
        gc.collect()

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        del epoch_idx, s_time, e_time, losses, des
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)
        del epoch_idx, losses, tag

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'learning_rate': self.config['learning_rate'],
            'weight_decay': self.config['weight_decay'],
            'train_batch_size': self.config['train_batch_size']
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values() for parameter in parameters
        }.union({'model', 'dataset', 'config_files', 'device'})
        # other model-specific hparam
        hparam_dict.update({
            para: val
            for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            k = k.replace('@', '_')
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})
        del hparam_dict, best_valid_result, unrecorded_parameter

    def to_device(self, data):
        device = self.device
        if isinstance(data, tuple) or isinstance(data, list):
            tdata = ()
            for d in data:
                d = d.to(device)
                tdata += (d,)
            return tdata
        elif isinstance(data, dict):
            for k, v in data.items():
                data[k] = v.to(device)
            return data
        else:
            return data.to(device)

    def _setup_fabric(self) -> None:
        world_size, local_world_size = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
        nnodes = world_size // local_world_size
        assert nnodes == int(os.environ.get('GROUP_WORLD_SIZE', 1)), f"inconsistent {nnodes=} != GROUP_WORLD_SIZE={os.environ.get('GROUP_WORLD_SIZE', 1)}"
        precision = self.config['precision'] if self.config['precision'] else '32'
        if self.config['strategy'] == 'deepspeed':
            self.logger.info(f"Use deepspeed strategy")
            self.strategy = DeepSpeedStrategy(stage=self.config["stage"], precision=precision)
            self.lite = Fabric(accelerator='gpu', strategy=self.strategy, precision=precision, num_nodes=nnodes, loggers=self.logger)
        else:
            self.logger.info(f"Use DDP strategy")
            self.strategy = DDPStrategy(find_unused_parameters=True)
            self.lite = Fabric(accelerator='gpu', strategy=self.strategy, precision=precision, num_nodes=nnodes, loggers=self.logger)
        # The launch() method should only be used if you intend to specify accelerator, devices, and so on in the code (programmatically).
        # If you are launching with the Lightning CLI, fabric run ..., remove launch() from your code.
        if not bool(int(os.environ.get("LT_CLI_USED", "0"))):
            self.logger.info(f"detected it is not launched from lightning CLI, will need fabric.launch")
            self.lite.launch()
        else:
            self.logger.info(f"detected it is launched from lightning CLI, will by pass fabric.launch")

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.scheduler_config:
            # it must happen before self.lite.setup
            warmup_rate = self.scheduler_config.get('warmup', 0.001)
            tot_steps = len(train_data) * self.epochs
            warmup_steps = tot_steps * warmup_rate
            self.lr_scheduler = self._build_scheduler(warmup_steps=warmup_steps, tot_steps=tot_steps)

        # set up fabric
        self.logger.info("set up fabric with model and optimizer")
        self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)

        # load checkpoint must happen after setup.
        if self.config['auto_resume']:
            self._resume_from_checkpoint(self.config["auto_resume"])

        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            if self.config['need_training'] == None or self.config['need_training']:
                train_data.sampler.set_epoch(epoch_idx)
                training_start_time = time()
                train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
                self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
                if verbose:
                    self.logger.info(train_loss_output)
                if self.rank == 0:
                    self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
                self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx}, head='train')
                del train_loss

            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, epoch_idx, show_progress=show_progress, item_text=train_data.dataset.env)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                    (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if self.rank == 0:
                    self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                    for name, value in valid_result.items():
                        self.tensorboard.add_scalar(name.replace('@', '_'), value, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result
                    self.best_epoch  = epoch_idx

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    self.best_epoch = epoch_idx - self.cur_step * self.eval_step
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                        (self.best_epoch)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1
                del valid_score, valid_result, valid_start_time, valid_end_time

        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def _full_sort_batch_eval(self, batched_data):
        user, time_seq, history_index, positive_u, positive_i, user_id = batched_data
        del batched_data
        interaction = self.to_device(user)
        batch_size = interaction.shape[0]
        time_seq = self.to_device(time_seq)
        user_id = self.to_device(user_id)
        self.logger.info(f"{self.tot_item_num=}, {self.item_feature.shape=}, {interaction.shape=}")
        if self.config['model'] == 'HLLM':
            if self.config['stage'] == 3:
                scores = self.model.module.predict(interaction, time_seq, self.item_feature, user_id)
            else:
                scores = self.model((interaction, time_seq, self.item_feature, user_id), mode='predict')
        else:
            scores = self.model.module.predict(interaction, time_seq, self.item_feature, user_id)
        del user, time_seq, interaction
        # there could be some mismatch, e.g., text dataset contains more items than interaction dataset
        # scores = scores.view(-1, self.tot_item_num)
        scores = scores.view(batch_size, -1)
        scores[:, 0] = -np.inf
        if self.config.get("suppress_history", True) and history_index is not None:
            scores[history_index] = -np.inf
        del history_index
        return scores, positive_u, positive_i

    @torch.no_grad()
    def compute_item_feature(self, config, item_text: Optional[List[Dict[str, str]]]=None):
        if self.use_text:
            num_workers = config.get('num_workers', 14)
            item_data = BatchTextDataset(config, item_text)
            item_batch_size = config['MAX_ITEM_LIST_LENGTH'] * config['train_batch_size']
            item_loader = DataLoader(item_data, batch_size=item_batch_size, num_workers=num_workers, shuffle=False,
                pin_memory=True, collate_fn=customize_rmpad_collate)
            del item_data
            self.logger.info(f"Inference item_data with {item_batch_size = } {len(item_loader) = }")
            self.item_feature = []
            with torch.no_grad():
                for _, items in tqdm(enumerate(item_loader), total=len(item_loader)):
                    items = self.to_device(items)
                    items = self.model(items, mode='compute_item')
                    self.item_feature.append(items)
                if isinstance(items, tuple):
                    self.item_feature = torch.cat([x[0] for x in self.item_feature]), torch.cat([x[1] for x in self.item_feature])
                else:
                    self.item_feature = torch.cat(self.item_feature)
                if self.config['stage'] == 3:
                    self.item_feature = self.item_feature.bfloat16()
            del item_loader, item_batch_size
        else:
            with torch.no_grad():
                self.item_feature = self.model.module.compute_item_all()

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        result = concat.sum() / num_total_examples
        del tensor, output_tensors, concat
        return result

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False, init_model=False, item_text: Optional[List[Dict[str, str]]]=None):
        if not eval_data:
            return
        if init_model:
            # set up fabric
            self.logger.info("set up fabric with model and optimizer")
            self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)

        if load_best_model or model_file is not None:
            checkpoint_file = model_file
            if checkpoint_file is None and self.best_epoch is not None:
                saved_model_name = '{}-{}.pth'.format(self.config['model'], self.best_epoch)
                checkpoint_file = os.path.join(self.checkpoint_dir, saved_model_name)
                if not os.path.exists(checkpoint_file):
                    # in case this job is resumed from previous one, try from manifold
                    self.logger.info(f"Checkpoint {checkpoint_file} not found, will try to load from manifold")
                    checkpoint_file = PathManager.get_local_path(os.path.join(self.manifold_dir, saved_model_name), recursive=True)
                self.logger.info(f"Loading best model from {checkpoint_file} with {self.best_epoch}")
            assert checkpoint_file is not None, "checkpoint_file is None"
            if checkpoint_file.startswith("manifold:"):
                # recursive=True is only allowed for manifold :(
                checkpoint_file = PathManager.get_local_path(checkpoint_file, recursive=True)
            state = {"model": self.model}

            self.lite.load(get_file_name(checkpoint_file), state)
            del state
            gc.collect()
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        with torch.no_grad():
            self.model.eval()
            eval_func = self._full_sort_batch_eval

            self.tot_item_num = eval_data.dataset.interaction_data.item_num
            self.compute_item_feature(self.config, item_text=item_text)
            iter_data = (
                tqdm(
                    eval_data,
                    total=len(eval_data),
                    ncols=150,
                    desc=set_color(f"Evaluate   ", 'pink'),
                    file=sys.stdout
                ) if show_progress and self.rank == 0 else eval_data
            )
            num_total_examples = len(eval_data.sampler.dataset)
            del eval_data
            fwd_time = t.time()
            total_fwd_time = 0
            for _, batched_data in enumerate(iter_data):
                start_time = fwd_time
                data_time = t.time()
                scores, positive_u, positive_i = eval_func(batched_data)
                del batched_data
                fwd_time = t.time()
                total_fwd_time += fwd_time - data_time

                if show_progress and self.rank == 0:
                    iter_data.set_postfix_str(f"data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f}", refresh=False)
                self.eval_collector.eval_batch_collect(scores, positive_u, positive_i)
                del scores, positive_u, positive_i
            print(f"evaluate completed in {total_fwd_time:.3f} seconds")
            del iter_data
            struct = self.eval_collector.get_data_struct()
            result = self.evaluator.evaluate(struct)

            metric_decimal_place = 5 if self.config['metric_decimal_place'] == None else self.config['metric_decimal_place']
            for k, v in result.items():
                result_cpu = self.distributed_concat(torch.tensor([v]).to(self.device), num_total_examples).cpu()
                result[k] = round(result_cpu.item(), metric_decimal_place)
            self.wandblogger.log_eval_metrics(result, head='eval')

            return result
