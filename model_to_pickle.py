# Copyright (c) Facebook, Inc. and its affiliates.
import warnings
warnings.filterwarnings('ignore') # never print matching warnings
import logging
import os
from collections import OrderedDict
import torch
import uniperceiver.utils.comm as comm
from uniperceiver.config import get_cfg, CfgNode
from uniperceiver.engine import DefaultTrainer, default_argument_parser, default_setup, launch, build_engine, add_moe_arguments

#!TODO re-implement hooks
from uniperceiver.engine import hooks
from uniperceiver.modeling import add_config
from uniperceiver.utils.env import init_distributed_mode, check_dist_portfile
try:
    import deepspeed
    DEEPSPEED_INSTALLED = True
except:
    DEEPSPEED_INSTALLED = False

import copy


def add_data_prefix(cfg):
    # TODO: more flexible method
    # data_dir = os.getenv("DATA_PATH", None)
    data_dir = '/home/hhiromasa/code/Uni-Perceiver/data' #CHANGED
    mapping_list = [
        [cfg.DATALOADER, 'FEATS_FOLDER', ['DATALOADER',]],
        [cfg.DATALOADER, 'ANNO_FOLDER', ['DATALOADER', ]],
        [cfg.DATALOADER, 'CLASS_NAME_FILE', ['DATALOADER', ]],
        [cfg.INFERENCE, 'VOCAB', ['INFERENCE', ]],
        [cfg.INFERENCE, 'VAL_ANNFILE', ['INFERENCE', ]],
        [cfg.INFERENCE, 'TEST_ANNFILE', ['INFERENCE',]],
        [cfg.MODEL, 'WEIGHTS', ['MODEL',]],
    ]
    whitelist = ["BERT", "CLIP", "CLIP_CAPTION"]
    if data_dir:
        for node, attr ,_ in mapping_list:
            if node[attr] != '' and not node[attr].startswith('.') and not node[attr].startswith('/') and not node[attr].startswith('work_dirs') and not node[attr].startswith('cluster') and not node[attr].startswith('s3://') and node[attr] not in whitelist:
                setattr(node, attr, os.path.join(data_dir, node[attr]))
    for task in cfg.TASKS:
        for _, item, key_list in mapping_list:
            config_tmp = task
            for key in key_list:
                if key in config_tmp:
                    config_tmp = config_tmp[key]
            if item in config_tmp and  config_tmp[item] != '' and not config_tmp[item].startswith('.') and not config_tmp[item].startswith('/') and not config_tmp[item].startswith('work_dirs') and not config_tmp[item].startswith('cluster') and not config_tmp[item].startswith('s3://') and config_tmp[item] not in whitelist:
                config_tmp[item] =  os.path.join(data_dir, config_tmp[item])

    mapping_list = [
        ['', 'FILE_PATH', ['SHARED_TARGETS_CFG',]],
    ]
    if cfg.SHARED_TARGETS is None:
        cfg.SHARED_TARGETS = []
    for share_targets in cfg.SHARED_TARGETS:
        for _, item, key_list in mapping_list:
            config_tmp = share_targets
            for key in key_list:
                config_tmp = config_tmp[key]
            if item in config_tmp and config_tmp[item] != '' and not config_tmp[item].startswith('.') and not config_tmp[item].startswith(
                    '/') and not config_tmp[item].startswith('work_dirs') and not config_tmp[item].startswith(
                        'cluster') and not config_tmp[item].startswith('s3://') and config_tmp[item] not in whitelist:
                config_tmp[item] = os.path.join(data_dir, config_tmp[item])



def add_default_setting_for_multitask_config(cfg):
    # merge some default config in (CfgNode) uniperceiver/config/defaults.py to each task config (dict)

    tasks_config_temp = cfg.TASKS
    num_tasks = len(tasks_config_temp)
    cfg.pop('TASKS', None)

    cfg.TASKS = [copy.deepcopy(cfg) for _ in range(num_tasks)]

    for i, task_config in enumerate(tasks_config_temp):
        cfg.TASKS[i].merge_from_other_cfg(CfgNode(task_config))
        cfg.TASKS[i] = cfg.TASKS[i].to_dict_object()
        pass


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    tmp_cfg = cfg.load_from_file_tmp(args.config_file)
    add_config(cfg, tmp_cfg)

    cfg.merge_from_file(args.config_file)
    add_data_prefix(cfg)

    cfg.merge_from_list(args.opts)
    #
    add_default_setting_for_multitask_config(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    trainer = build_engine(cfg)
    trainer.resume_or_load(resume=args.resume)
    import pickle 
    filename = 'pt_in1k'
    with open(f'/home/hhiromasa/code/Uni-Perceiver/asset/{filename}.pkl', 'wb') as f:
        pickle.dump(trainer.model, f)
    # trainer.cast_layers()


def get_args_parser():
    parser = default_argument_parser()
    if DEEPSPEED_INSTALLED:
        parser = deepspeed.add_config_arguments(parser)
    parser = add_moe_arguments(parser)

    parser.add_argument('--init_method', default='slurm', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument("--eval-ema", action="store_true", help="perform evaluation using ema")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args_parser()
    print("Command Line Args:", args)
    if args.init_method == 'slurm':
        # slurm init
        check_dist_portfile()
        init_distributed_mode(args)
        main(args)
    elif args.init_method == 'pytorch':
        main(args)
    else:
        # follow 'd2' use default `mp.spawn` to init dist training
        print('using \'mp.spawn\' for dist init! ')
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )