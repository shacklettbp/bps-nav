#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
import copy
import multiprocessing
import psutil
import socket
import warnings
from collections import OrderedDict, defaultdict, deque

import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
import torch.nn.functional as F
import psutil

#  import v4r_example
from gym import spaces
from gym.spaces import Dict as SpaceDict
from torch.optim.lr_scheduler import LambdaLR

from bps_nav.common.env_utils import construct_envs
from bps_nav.common.rollout_storage import DoubleBufferedRolloutStorage
from bps_nav.common.tensorboard_utils import TensorboardWriter
from bps_nav.common.utils import Timing, batch_obs, linear_decay
from bps_nav.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from bps_nav.rl.ddppo.algo.ddppo import DDPPO
from bps_nav.common.tree_utils import (
    tree_select,
    tree_copy_in_place,
)
from bps_nav.rl.ppo.ppo_trainer import PPOTrainer
from bps_nav.rl.ddppo.policy.resnet import Dropblock
import socket
from bps_nav.common.logger import logger
from bps_nav.rl.ddppo.policy import ResNetPolicy

try:
    import psutil
except ImportError:
    psutil = None

warnings.filterwarnings("ignore", torch.optim.lr_scheduler.SAVE_STATE_WARNING)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

BURN_IN_UPDATES = 50

BPS_BENCHMARK = os.environ.get("BPS_BENCHMARK", "0") != "0"

if BPS_BENCHMARK:
    logger.warn("In benchmark mode")


def set_cpus(local_rank, world_size):
    local_size = min(world_size, 8)

    curr_process = psutil.Process()
    total_cpus = curr_process.cpu_affinity()
    total_cpu_count = len(total_cpus)

    # Assuming things where already set
    if total_cpu_count > multiprocessing.cpu_count() / world_size:

        orig_cpus = total_cpus
        total_cpus = []
        for i in range(total_cpu_count // 2):
            total_cpus.append(orig_cpus[i])
            total_cpus.append(orig_cpus[i + total_cpu_count // 2])

        ptr = 0
        local_cpu_count = 0
        local_cpus = []
        CORE_GROUPING = min(
            local_size,
            4 if total_cpu_count / 2 >= 20 else (2 if total_cpu_count / 2 >= 10 else 1),
        )
        CORE_GROUPING = 1
        core_dist_size = max(local_size // CORE_GROUPING, 1)
        core_dist_rank = local_rank // CORE_GROUPING

        for r in range(core_dist_rank + 1):
            ptr += local_cpu_count
            local_cpu_count = total_cpu_count // core_dist_size + (
                1 if r < (total_cpu_count % core_dist_size) else 0
            )

        local_cpus += total_cpus[ptr : ptr + local_cpu_count]
        pop_inds = [
            ((local_rank + offset + 1) % CORE_GROUPING)
            for offset in range(CORE_GROUPING - 1)
        ]
        for ind in sorted(pop_inds, reverse=True):
            local_cpus.pop(ind)

        if BPS_BENCHMARK and world_size == 1:
            local_cpus = total_cpus[0:12]

        curr_process.cpu_affinity(local_cpus)

    logger.info(
        "Rank {} uses cpus {}".format(local_rank, sorted(curr_process.cpu_affinity()))
    )


class DDPPOTrainer(PPOTrainer):
    # DD-PPO cuts rollouts short to mitigate the straggler effect
    # This, in theory, can cause some rollouts to be very short.
    # All rollouts contributed equally to the loss/model-update,
    # thus very short rollouts can be problematic.  This threshold
    # limits the how short a short rollout can be as a fraction of the
    # max rollout length
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config=None, resume_from=None):
        self.resume_from = resume_from
        interrupted_state = load_interrupted_state(resume_from=self.resume_from)
        if interrupted_state is not None:
            config = interrupted_state["config"]

        super().__init__(config)

    def _setup_actor_critic_agent(self, ppo_cfg) -> None:
        r"""Sets up actor critic and agent for DD-PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        if hasattr(self.config.RL.DDPPO, 'use_avg_pool'):
            use_avg_pool = self.config.RL.DDPPO.use_avg_pool
        else:
            use_avg_pool = False

        self.actor_critic = ResNetPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=self.config.RL.DDPPO.rnn_type,
            num_recurrent_layers=self.config.RL.DDPPO.num_recurrent_layers,
            backbone=self.config.RL.DDPPO.backbone,
            resnet_baseplanes=self.config.RL.DDPPO.resnet_baseplanes,
            use_avg_pool=use_avg_pool,
        )
        self.actor_critic.to(self.device)

        if self.config.RL.DDPPO.pretrained_encoder or self.config.RL.DDPPO.pretrained:
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.ac.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.ac.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            self.actor_critic.ac.critic.layer_init()

        self.agent = DDPPO(actor_critic=self.actor_critic, ppo_cfg=ppo_cfg)
        self.agent.to(self.device)

    def _update_policy(self):
        pass

    def _n_buffered_sampling(
        self,
        rollouts,
        current_episode_reward,
        running_episode_stats,
        buffer_ranges,
        real_steps,
        num_rollouts_done_store,
    ):
        count_steps_delta = 0
        sim_step_reses = [None for _ in range(len(rollouts))]
        actions = [None for _ in range(len(rollouts))]
        is_double_buffered = len(rollouts) > 1

        for idx in range(len(rollouts)):
            actions[idx] = self._inference(rollouts, idx)

            if is_double_buffered and idx == 0:
                self._start_simulation(actions[idx], idx)

        for step in range(real_steps):
            is_last_step = (step + 1) == real_steps
            if (
                (step + 1) >= max(real_steps * self.SHORT_ROLLOUT_THRESHOLD, 1)
            ) and int(num_rollouts_done_store.get("num_done")) >= (
                self.config.RL.DDPPO.sync_frac * self.world_size
            ):
                is_last_step = True

            for idx in range(len(rollouts)):
                if is_double_buffered:
                    sim_step_reses[idx] = self._wait_simulation(idx)

                    if len(rollouts) > 1:
                        other_idx = (idx + 1) % len(rollouts)
                        if not is_last_step or other_idx > idx:
                            self._start_simulation(actions[other_idx], other_idx)

                    self._render(idx)
                elif True:
                    self._start_simulation(actions[idx], idx)
                    sim_step_reses[idx] = self._wait_simulation(idx)
                    self._render(idx)
                else:
                    sim_step_reses[idx] = self._step_simulation(actions[idx], idx)

                self._update_stats(
                    rollouts,
                    current_episode_reward,
                    running_episode_stats,
                    sim_step_reses[idx],
                    buffer_ranges[idx],
                    idx,
                )

                count_steps_delta += self._sync_renderer_and_insert(
                    rollouts, sim_step_reses[idx], idx
                )

                if not is_last_step:
                    actions[idx] = self._inference(rollouts, idx)

            if is_last_step:
                break

        return count_steps_delta

    def _warmup(self, rollouts):
        model_state = {k: v.clone() for k, v in self.agent.state_dict().items()}
        optim_state = self.agent.optimizer.state.copy()
        self.agent.eval()

        for _ in range(20):
            self._inference(rollouts, 0)
            # Do a cache empty as sometimes cudnn searching
            # doesn't do that
            torch.cuda.empty_cache()

        t_inference_start = time.time()
        n_infers = 200
        for _ in range(n_infers):
            self._inference(rollouts, 0)

        if self.world_rank == 0:
            logger.info(
                "Inference time: {:.3f} ms".format(
                    (time.time() - t_inference_start) / n_infers * 1000
                )
            )
            logger.info(
                "PyTorch CUDA Memory Cache Size: {:.3f} GB".format(
                    torch.cuda.memory_reserved(self.device) / 1e9
                )
            )

        self.agent.train()
        for _ in range(10):
            self._update_agent(rollouts, warmup=True)
            # Do a cache empty as sometimes cudnn searching
            # doesn't do that
            torch.cuda.empty_cache()

        t_learning_start = time.time()
        n_learns = 15
        for _ in range(n_learns):
            self._update_agent(rollouts, warmup=True)

        if self.world_rank == 0:
            logger.info(
                "Learning time: {:.3f} ms".format(
                    (time.time() - t_learning_start) / n_learns * 1000
                )
            )
            logger.info(self.timing)
            logger.info(
                "PyTorch CUDA Memory Cache Size: {:.3f} GB".format(
                    torch.cuda.memory_reserved(self.device) / 1e9
                )
            )

        self.agent.load_state_dict(model_state)
        self.agent.optimizer.state = optim_state
        self.agent.ada_scale.zero_grad()

        self.timing = Timing()

    def train(self) -> None:
        r"""Main method for DD-PPO.

        Returns:
            None
        """
        import apex

        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )
        # add_signal_handlers()
        self.timing = Timing()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore("rollout_tracker", tcp_store)
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        set_cpus(self.local_rank, self.world_size)

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += self.world_rank * self.config.SIM_BATCH_SIZE
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        double_buffered = False
        self._num_worker_groups = self.config.NUM_PARALLEL_SCENES

        self._depth = self.config.DEPTH
        self._color = self.config.COLOR

        if self.config.TASK.lower() == "pointnav":
            self.observation_space = SpaceDict(
                {
                    "pointgoal_with_gps_compass": spaces.Box(
                        low=0.0, high=1.0, shape=(2,), dtype=np.float32
                    )
                }
            )
        else:
            self.observation_space = SpaceDict({})

        self.action_space = spaces.Discrete(4)

        if self._color:
            self.observation_space = SpaceDict(
                {
                    "rgb": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=(3, *self.config.RESOLUTION),
                        dtype=np.uint8,
                    ),
                    **self.observation_space.spaces,
                }
            )

        if self._depth:
            self.observation_space = SpaceDict(
                {
                    "depth": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=(1, *self.config.RESOLUTION),
                        dtype=np.float32,
                    ),
                    **self.observation_space.spaces,
                }
            )

        ppo_cfg = self.config.RL.PPO
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER) and self.world_rank == 0:
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        self.count_steps = 0
        burn_steps = 0
        burn_time = 0
        count_checkpoints = 0
        prev_time = 0
        self.update = 0

        LR_SCALE = (
            max(
                np.sqrt(
                    ppo_cfg.num_steps
                    * self.config.SIM_BATCH_SIZE
                    * ppo_cfg.num_accumulate_steps
                    / ppo_cfg.num_mini_batch
                    * self.world_size
                    / (128 * 2)
                ),
                1.0,
            )
            if (self.config.RL.DDPPO.scale_lr and not self.config.RL.PPO.ada_scale)
            else 1.0
        )

        def cosine_decay(x):
            if x < 1:
                return (np.cos(x * np.pi) + 1.0) / 2.0
            else:
                return 0.0

        def warmup_fn(x):
            return LR_SCALE * (0.5 + 0.5 * x)

        def decay_fn(x):
            return LR_SCALE * (DECAY_TARGET + (1 - DECAY_TARGET) * cosine_decay(x))

        DECAY_TARGET = (
            0.01 / LR_SCALE
            if self.config.RL.PPO.ada_scale or True
            else (0.25 / LR_SCALE if self.config.RL.DDPPO.scale_lr else 1.0)
        )
        DECAY_PERCENT = 1.0 if self.config.RL.PPO.ada_scale or True else 0.5
        WARMUP_PERCENT = (
            0.01
            if (self.config.RL.DDPPO.scale_lr and not self.config.RL.PPO.ada_scale)
            else 0.0
        )

        def lr_fn():
            x = self.percent_done()
            if x < WARMUP_PERCENT:
                return warmup_fn(x / WARMUP_PERCENT)
            else:
                return decay_fn((x - WARMUP_PERCENT) / DECAY_PERCENT)

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer, lr_lambda=lambda x: lr_fn()
        )

        interrupted_state = load_interrupted_state(resume_from=self.resume_from)
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])

        self.agent.init_amp(self.config.SIM_BATCH_SIZE)
        self.actor_critic.init_trt(self.config.SIM_BATCH_SIZE)
        self.actor_critic.script_net()
        self.agent.init_distributed(find_unused_params=False)

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )

        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            self.observation_space = SpaceDict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **self.observation_space,
                }
            )
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        nenvs = self.config.SIM_BATCH_SIZE
        rollouts = DoubleBufferedRolloutStorage(
            ppo_cfg.num_steps,
            nenvs,
            self.observation_space,
            self.action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.num_recurrent_layers,
            use_data_aug=ppo_cfg.use_data_aug,
            aug_type=ppo_cfg.aug_type,
            double_buffered=double_buffered,
            vtrace=ppo_cfg.vtrace,
        )
        rollouts.to(self.device)
        rollouts.to_fp16()

        self._warmup(rollouts)

        (
            self.envs,
            self._observations,
            self._rewards,
            self._masks,
            self._rollout_infos,
            self._syncs,
        ) = construct_envs(
            self.config,
            num_worker_groups=self.config.NUM_PARALLEL_SCENES,
            double_buffered=double_buffered,
        )

        def _setup_render_and_populate_initial_frame():
            for idx in range(2 if double_buffered else 1):
                self.envs.reset(idx)

                batch = self._observations[idx]
                self._syncs[idx].wait()

                tree_copy_in_place(
                    tree_select(0, rollouts[idx].storage_buffers["observations"]),
                    batch,
                )

        _setup_render_and_populate_initial_frame()

        current_episode_reward = torch.zeros(nenvs, 1)
        running_episode_stats = dict(
            count=torch.zeros(nenvs, 1,), reward=torch.zeros(nenvs, 1,),
        )

        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )
        time_per_frame_window = deque(maxlen=ppo_cfg.reward_window_size)

        buffer_ranges = []
        for i in range(2 if double_buffered else 1):
            start_ind = buffer_ranges[-1].stop if i > 0 else 0
            buffer_ranges.append(
                slice(
                    start_ind,
                    start_ind
                    + self.config.SIM_BATCH_SIZE // (2 if double_buffered else 1),
                )
            )

        if interrupted_state is not None:
            requeue_stats = interrupted_state["requeue_stats"]

            self.count_steps = requeue_stats["count_steps"]
            self.update = requeue_stats["start_update"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]
            burn_steps = requeue_stats["burn_steps"]
            burn_time = requeue_stats["burn_time"]

            self.agent.ada_scale.load_state_dict(interrupted_state["ada_scale_state"])

            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            if "amp_state" in interrupted_state:
                apex.amp.load_state_dict(interrupted_state["amp_state"])

            if "grad_scaler_state" in interrupted_state:
                self.agent.grad_scaler.load_state_dict(
                    interrupted_state["grad_scaler_state"]
                )

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR,
                flush_secs=self.flush_secs,
                purge_step=int(self.count_steps),
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            distrib.barrier()
            t_start = time.time()
            while not self.is_done():
                t_rollout_start = time.time()
                if self.update == BURN_IN_UPDATES:
                    burn_time = t_rollout_start - t_start
                    burn_steps = self.count_steps

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        self.percent_done(), final_decay=ppo_cfg.decay_factor,
                    )

                if (
                    not BPS_BENCHMARK
                    and (REQUEUE.is_set() or ((self.update + 1) % 100) == 0)
                    and self.world_rank == 0
                ):
                    requeue_stats = dict(
                        count_steps=self.count_steps,
                        count_checkpoints=count_checkpoints,
                        start_update=self.update,
                        prev_time=(time.time() - t_start) + prev_time,
                        burn_time=burn_time,
                        burn_steps=burn_steps,
                    )

                    def _cast(param):
                        if "Half" in param.type():
                            param = param.to(dtype=torch.float32)

                        return param

                    save_interrupted_state(
                        dict(
                            state_dict={
                                k: _cast(v) for k, v in self.agent.state_dict().items()
                            },
                            ada_scale_state=self.agent.ada_scale.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                            grad_scaler_state=self.agent.grad_scaler.state_dict(),
                        )
                    )

                if EXIT.is_set():
                    self._observations = None
                    self._rewards = None
                    self._masks = None
                    self._rollout_infos = None
                    self._syncs = None

                    del self.envs
                    self.envs = None

                    requeue_job()
                    return

                self.agent.eval()

                count_steps_delta = self._n_buffered_sampling(
                    rollouts,
                    current_episode_reward,
                    running_episode_stats,
                    buffer_ranges,
                    ppo_cfg.num_steps,
                    num_rollouts_done_store,
                )

                num_rollouts_done_store.add("num_done", 1)

                if not rollouts.vtrace:
                    self._compute_returns(ppo_cfg, rollouts)

                (value_loss, action_loss, dist_entropy) = self._update_agent(rollouts)

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                lr_scheduler.step()

                with self.timing.add_time("Logging"):
                    stats_ordering = list(sorted(running_episode_stats.keys()))
                    stats = torch.stack(
                        [running_episode_stats[k] for k in stats_ordering], 0,
                    ).to(device=self.device)
                    distrib.all_reduce(stats)
                    stats = stats.to(device="cpu")

                    for i, k in enumerate(stats_ordering):
                        window_episode_stats[k].append(stats[i])

                    stats = torch.tensor(
                        [
                            value_loss,
                            action_loss,
                            count_steps_delta,
                            *self.envs.swap_stats,
                        ],
                        device=self.device,
                    )
                    distrib.all_reduce(stats)
                    stats = stats.to(device="cpu")
                    count_steps_delta = int(stats[2].item())
                    self.count_steps += count_steps_delta

                    time_per_frame_window.append(
                        (time.time() - t_rollout_start) / count_steps_delta
                    )

                    if self.world_rank == 0:
                        losses = [
                            stats[0].item() / self.world_size,
                            stats[1].item() / self.world_size,
                        ]
                        deltas = {
                            k: (
                                (v[-1] - v[0]).sum().item()
                                if len(v) > 1
                                else v[0].sum().item()
                            )
                            for k, v in window_episode_stats.items()
                        }
                        deltas["count"] = max(deltas["count"], 1.0)

                        writer.add_scalar(
                            "reward",
                            deltas["reward"] / deltas["count"],
                            self.count_steps,
                        )

                        # Check to see if there are any metrics
                        # that haven't been logged yet
                        metrics = {
                            k: v / deltas["count"]
                            for k, v in deltas.items()
                            if k not in {"reward", "count"}
                        }
                        if len(metrics) > 0:
                            writer.add_scalars("metrics", metrics, self.count_steps)

                        writer.add_scalars(
                            "losses",
                            {k: l for l, k in zip(losses, ["value", "policy"])},
                            self.count_steps,
                        )

                        optim = self.agent.optimizer
                        writer.add_scalar(
                            "optimizer/base_lr",
                            optim.param_groups[-1]["lr"],
                            self.count_steps,
                        )
                        if "gain" in optim.param_groups[-1]:
                            for idx, group in enumerate(optim.param_groups):
                                writer.add_scalar(
                                    f"optimizer/lr_{idx}",
                                    group["lr"] * group["gain"],
                                    self.count_steps,
                                )
                                writer.add_scalar(
                                    f"optimizer/gain_{idx}",
                                    group["gain"],
                                    self.count_steps,
                                )

                        # log stats
                        if (
                            self.update > 0
                            and self.update % self.config.LOG_INTERVAL == 0
                        ):
                            logger.info(
                                "update: {}\twindow fps: {:.3f}\ttotal fps: {:.3f}\tframes: {}".format(
                                    self.update,
                                    1.0
                                    / (
                                        sum(time_per_frame_window)
                                        / len(time_per_frame_window)
                                    ),
                                    (self.count_steps - burn_steps)
                                    / ((time.time() - t_start) + prev_time - burn_time),
                                    self.count_steps,
                                )
                            )

                            logger.info(
                                "swap percent: {:.3f}\tscenes in use: {:.3f}\tenvs per scene: {:.3f}".format(
                                    stats[3].item() / self.world_size,
                                    stats[4].item() / self.world_size,
                                    stats[5].item() / self.world_size,
                                )
                            )

                            logger.info(
                                "Average window size: {}  {}".format(
                                    len(window_episode_stats["count"]),
                                    "  ".join(
                                        "{}: {:.3f}".format(k, v / deltas["count"])
                                        for k, v in deltas.items()
                                        if k != "count"
                                    ),
                                )
                            )

                            logger.info(self.timing)
                            # self.envs.print_renderer_stats()

                        # checkpoint model
                        if self.should_checkpoint():
                            self.save_checkpoint(
                                f"ckpt.{count_checkpoints}.pth",
                                dict(
                                    step=self.count_steps,
                                    wall_clock_time=(
                                        (time.time() - t_start) + prev_time
                                    ),
                                ),
                            )
                            count_checkpoints += 1

                self.update += 1

            self.save_checkpoint(
                "ckpt.done.pth",
                dict(
                    step=self.count_steps,
                    wall_clock_time=((time.time() - t_start) + prev_time),
                ),
            )
            self._observations = None
            self._rewards = None
            self._masks = None
            self._rollout_infos = None
            self._syncs = None
            del self.envs
            self.envs = None
