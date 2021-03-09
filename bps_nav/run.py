#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random

import numpy as np
import torch

from bps_nav.config.default import get_config
from bps_nav.rl.ddppo.algo import DDPPOTrainer
from bps_nav.rl.ppo.ppo_trainer import PPOTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, resume_from=None, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    if run_type == "train":
        config = get_config(exp_config, opts)
    elif run_type == "eval":
        from habitat import get_config as get_config_habitat
        config = get_config(exp_config, opts, get_config_habitat)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    trainer = DDPPOTrainer(config, resume_from)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
