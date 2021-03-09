import os
import random
import socket

import attr
import numpy as np
import submitit

from habitat import logger
from bps_nav.common.baseline_registry import baseline_registry
from bps_nav.config.default import get_config


@attr.s
class EvaluatorArgs:
    opts = attr.ib()
    exp_config = attr.ib(
        default="bps_nav/config/pointnav/ddppo_pointnav.yaml"
    )


class Evaluator:
    def __call__(self, args: EvaluatorArgs, prev_ckpt_ind=-1, num_frames=0):
        logger.info(
            "CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"])
        )
        logger.info("Hostname: {}".format(socket.gethostname()))

        config = get_config(args.exp_config, args.opts)

        random.seed(config.TASK_CONFIG.SEED)
        np.random.seed(config.TASK_CONFIG.SEED)

        trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
        assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
        self.trainer = trainer_init(config)
        self.trainer.prev_ckpt_ind = prev_ckpt_ind
        self.trainer.num_frames = num_frames

        self.trainer.eval()

    def checkpoint(self, args, prev_ckpt_ind=-1, num_frames=0):
        return submitit.helpers.DelayedSubmission(
            Evaluator(), args, self.trainer.prev_ckpt_ind, self.trainer.num_frames,
        )
