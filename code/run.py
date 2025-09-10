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
import argparse
import datetime
import gc
import json
import os
import tempfile
import uuid

import logging

import torch
from REC.data import *
import torch.distributed as dist
from REC.config import Config

from REC.data.dataload import InteractionData
from REC.trainer import Trainer
from REC.utils import get_model, init_logger, init_seed, set_color
import REC.data.comm as comm
import torch.distributed as dist


def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except Exception:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s


def run_loop(local_rank, config_file=None, saved=True, extra_args=[]):

    # configurations initialization
    config = Config(config_file_list=config_file)
    # turn on auto resume by default. You could disable it from command line as well.
    config["auto_resume"] = True

    device = torch.device("cuda", local_rank)
    config['device'] = device
    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                else:
                    config[key] = value
            except Exception as e:
                raise ValueError(f"{key} {value} invalid due to {e}")
    # override the output path
    if not config['val_only']:
        mast_job_id = config['model']
        if "MAST_HPC_JOB_NAME" in os.environ:
            try:
                mast_job_id = os.environ["MAST_HPC_JOB_NAME"]
            except Exception as e:
                print(f"failed to get mast job id: {e}")
        config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], mast_job_id)
    else:
        print("val_only is True, will not override the output path")
    init_seed(config['seed'], config['reproducibility'])
    print(f"config is {config}")

    # logger initialization
    # init_logger(config)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"{config=}")
    if not config.get("suppress_history", True):
        logger.info("You configure to not suppress seen items from history and it will be different from the paper.")

    # load item text information
    if 'text_path' in config:
        if os.path.isfile(os.path.join(config['text_path'], config['dataset'] + '.csv')):
            config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.csv')
        elif os.path.isfile(os.path.join(config['text_path'], config['dataset'] + '.parquet')):
            config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.parquet')
        else:
            raise ValueError(f'File {os.path.join(config["text_path"], config["dataset"])} not exist.')
        logger.info(f"Update text_path to {config['text_path']}")

    # get data
    logger.info("loading data, please be patient...")
    interaction_data = InteractionData(config)
    train_loader, valid_loader, test_loader = bulid_dataloader(config, interaction_data)
    logger.info(f"data loaded as {len(train_loader) = }")
    logger.info(f"{interaction_data=}")
    item_num = interaction_data.item_num
    user_num = interaction_data.user_num
    del interaction_data
    gc.collect()

    # get model
    trainer = Trainer(config)
    logger.info(f"creating {config['model']} with {item_num=}, {user_num=}")
    if False: # config['strategy'] == 'deepspeed':
        # not yet working
        logger.info(f"USe efficient model initilization with deepspeed")
        with trainer.strategy.module_init_context():
            model = get_model(config['model'])(config, item_num, user_num)
    else:
        # intialize the model on device to reduce CPU memory
        with torch.cuda.device(device):
            model = get_model(config['model'])(config, item_num, user_num).to(device)
    logger.info(f"{model=}")
    trainer.setup_model(model)
    del model
    gc.collect()

    world_size = torch.distributed.get_world_size()
    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    # synchronize before training begins
    torch.distributed.barrier()

    if config['val_only']:
        del valid_loader
        gc.collect()
        test_result = trainer.evaluate(
            test_loader,
            load_best_model=True,
            show_progress=config['show_progress'],
            init_model=True,
            item_text=train_loader.dataset.env,
            model_file=os.path.join(config['checkpoint_dir']),
        )
        del train_loader
        gc.collect()
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
    else:
        # training process
        best_valid_score, best_valid_result = trainer.fit(
            train_loader, valid_loader, saved=saved, show_progress=config['show_progress']
        )
        logger.info(f'Trianing Ended' + set_color('best valid ', 'yellow') + f': {best_valid_result}')
        del valid_loader
        gc.collect()

        # model evaluation
        test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=config['show_progress'], item_text=train_loader.dataset.env)
        del train_loader
        gc.collect()

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str)
    args, extra_args = parser.parse_known_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp("matplotlib")
    os.environ['TRANSFORMERS_CACHE'] = tempfile.mkdtemp("huggingface")
    # optimize the memory: reduce the fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF") + ",expandable_segments:True"
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # disable lightning.fabric.launch
    os.environ['LT_CLI_USED'] = "1"
    # https://www.deepspeed.ai/tutorials/advanced-install/#cuda-version-mismatch
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"
    os.environ["LOGLEVEL"] = "INFO"
    config_file = args.config_file
    print("env", os.environ)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=30))

    # Setup the local process group (which contains ranks within the same machine)
    machine_rank = int(os.environ.get('GROUP_RANK', 0))
    num_machines = int(os.environ.get('GROUP_WORLD_SIZE', 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    assert comm._LOCAL_PROCESS_GROUP is None
    # initialize each group on every rank, but only assign appropriate group
    for i in range(num_machines):
        ranks_on_machine = list(range(local_world_size * i, local_world_size * i + local_world_size))
        print(f"create local group for node {i} with ranks {ranks_on_machine}")
        pg = dist.new_group(ranks_on_machine)
        if i == machine_rank:
            print(f"set local group {pg} for node {i} with ranks {ranks_on_machine}")
            comm._LOCAL_PROCESS_GROUP = pg
    comm.synchronize()

    run_loop(local_rank=local_rank, config_file=config_file, extra_args=extra_args)


if __name__ == '__main__':
    main()
