#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import OrderedDict, defaultdict, deque
import copy

import numpy as np
import torch
import torch.distributed as distrib
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger, make_dataset
from bps_nav.common.baseline_registry import baseline_registry
from bps_nav.common.env_utils import construct_envs
from bps_nav.common.environments import get_env_class
from bps_nav.common.rollout_storage import DoubleBufferedRolloutStorage
from bps_nav.common.tensorboard_utils import TensorboardWriter
from bps_nav.common.utils import Timing, batch_obs, linear_decay
from bps_nav.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    init_distrib,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from bps_nav.rl.ddppo.algo.ddppo import DDPPO
from bps_nav.rl.ddppo.algo.ddppo_trainer import DDPPOTrainer


def run_learner_worker(
    device_id_base,
    local_rank,
    world_rank,
    world_size,
    config,
    actor_critic,
    actor_critic_version,
    task_queue: mp.SimpleQueue,
    learner_output_queue: mp.SimpleQueue,
    policy_lock: mp.Semaphore,
):
    timing = Timing()
    tcp_store = init_distrib(
        world_rank, world_size, config.RL.DDPPO.distrib_backend, port_offset=1
    )
    world_rank = distrib.get_rank()
    world_size = distrib.get_world_size()

    if torch.cuda.is_available():
        device = torch.device("cuda", device_id_base + local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    actor_critic.to(device)
    actor_critic.share_memory()

    ppo_cfg = config.RL.PPO

    agent = DDPPO(
        actor_critic=actor_critic,
        clip_param=ppo_cfg.clip_param,
        ppo_epoch=ppo_cfg.ppo_epoch,
        num_mini_batch=ppo_cfg.num_mini_batch,
        value_loss_coef=ppo_cfg.value_loss_coef,
        entropy_coef=ppo_cfg.entropy_coef,
        lr=ppo_cfg.lr,
        eps=ppo_cfg.eps,
        max_grad_norm=ppo_cfg.max_grad_norm,
        use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        vtrace=True,
    )
    agent.to(device)
    agent.init_amp()
    agent.init_distributed(find_unused_params=False)
    agent.train()

    shared_ac_state = {
        k: v.share_memory_() for k, v in actor_critic.state_dict().items()
    }

    if local_rank == 0:
        learner_output_queue.put(shared_ac_state)
    else:
        learner_output_queue.put(None)

    task_queue.get()

    while True:
        with timing.add_time("Rollout-Wait"):
            rollouts = task_queue.get()
        if rollouts is None:
            break

        with timing.add_time("PPO"):
            if local_rank == 0:
                value_loss, action_loss, dist_entropy = agent.update(
                    rollouts, timing, actor_critic_version, shared_ac_state, policy_lock
                )
            else:
                value_loss, action_loss, dist_entropy = agent.update(rollouts, timing)

        learner_output_queue.put(
            dict(
                value_loss=value_loss,
                action_loss=action_loss,
                dist_entropy=dist_entropy,
                timing=timing,
            )
        )


@baseline_registry.register_trainer(name="addppo")
class ADDPPOTrainer(DDPPOTrainer):
    def _update_policy(self):
        if self._curr_ac_version != int(self._actor_critic_version):
            with self._policy_lock:
                self._curr_ac_version = int(self._actor_critic_version)
                self.actor_critic.load_state_dict(self._shared_ac_state)

    def train(self) -> None:
        r"""Main method for DD-PPO.

        Returns:
            None
        """
        LEARNERS_PER = 3
        self.local_rank, tcp_store = init_distrib_slurm("gloo", port_offset=0)
        add_signal_handlers()
        self.timing = Timing()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore("rollout_tracker", tcp_store)
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += self.world_rank * self.config.NUM_PROCESSES
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
        self._num_worker_groups = self.config.NUM_PARALLEL_SCENES

        self._depth = self.config.DEPTH
        self._color = self.config.COLOR

        self.observation_space = SpaceDict(
            {
                "pointgoal_with_gps_compass": spaces.Box(
                    low=0.0, high=1.0, shape=(2,), dtype=np.float32
                )
            }
        )
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
        self._actor_critic_version = torch.zeros((1,))
        self._actor_critic_version.share_memory_()

        _spawn_ctx = mp.get_context("spawn")
        self._learner_task_queues = []
        self._learner_output_queues = []
        self._learners = []
        self._policy_lock = _spawn_ctx.Semaphore()

        for _ in range(LEARNERS_PER):
            self._learner_task_queues.append(_spawn_ctx.SimpleQueue())
            self._learner_output_queues.append(_spawn_ctx.SimpleQueue())

            self._learners.append(
                _spawn_ctx.Process(
                    target=run_learner_worker,
                    args=(
                        self.config,
                        copy.deepcopy(self.actor_critic),
                        self._actor_critic_version,
                        self._learner_task_queues[-1],
                        self._learner_output_queues[-1],
                        self._policy_lock,
                    ),
                )
            )
            self._learners[-1].start()

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.actor_critic.parameters()
                        if param.requires_grad
                    )
                )
            )

        nenvs = self.config.NUM_PROCESSES
        all_rollouts = []
        for _ in range(2):
            rollouts = DoubleBufferedRolloutStorage(
                ppo_cfg.num_steps,
                nenvs,
                self.observation_space,
                self.envs[0].action_spaces[0],
                ppo_cfg.hidden_size,
                num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
                use_data_aug=ppo_cfg.use_data_aug,
                aug_type=ppo_cfg.aug_type,
                double_buffered=double_buffered,
                vtrace=True,
            )
            rollouts.to(self.device)
            rollouts.to_fp16()
            rollouts.share_memory()
            all_rollouts.append(rollouts)

        self._current_rollout_idx = 0
        rollouts = all_rollouts[self._current_rollout_idx]

        def _setup_render_and_populate_initial_frame():
            for idx in range(2 if double_buffered else 1):
                self.envs.reset(idx)

                batch = self._observations[idx]
                self._syncs[idx].wait()

                for sensor in rollouts[idx].observations:
                    rollouts[idx].observations[sensor][0].copy_(batch[sensor])

        _setup_render_and_populate_initial_frame()

        current_episode_reward = torch.zeros(nenvs, 1)
        running_episode_stats = dict(
            count=torch.zeros(nenvs, 1,), reward=torch.zeros(nenvs, 1,),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )
        time_per_frame_window = deque(maxlen=ppo_cfg.reward_window_size)

        count_steps = 0
        count_checkpoints = 0
        prev_time = 0
        update = 0

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(interrupted_state["optim_state"])
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        next_swap_group_idx = 0

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR,
                flush_secs=self.flush_secs,
                purge_step=int(count_steps),
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            self._shared_ac_state = self._learner_output_queue.get()
            self.actor_critic.to(
                device=self.device,
                dtype=next(iter(self._shared_ac_state.values())).dtype,
            )
            self._curr_ac_version = int(self._actor_critic_version)
            self.actor_critic.load_state_dict(self._shared_ac_state)

            distrib.barrier()

            self._learner_task_queue.put("Go")
            t_start = time.time()
            while not self.is_done(update, count_steps):
                t_rollout_start = time.time()
                if self.envs.can_swap_scene():
                    self.envs.trigger_swap_scene(next_swap_group_idx)
                    next_swap_group_idx = (
                        next_swap_group_idx + 1
                    ) % self._num_worker_groups

                if REQUEUE.is_set() and self.world_rank == 0:
                    requeue_stats = dict(
                        count_steps=self.count_steps,
                        count_checkpoints=count_checkpoints,
                        start_update=self.update,
                        prev_time=(time.time() - t_start) + prev_time,
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
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            amp_state=apex.amp.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
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

                self.actor_critic.eval()
                step_ramp = self.config.RL.DDPPO.step_ramp
                step_ramp_start = self.config.RL.DDPPO.step_ramp_start
                real_steps = min(
                    int(
                        step_ramp_start
                        + max(ppo_cfg.num_steps - step_ramp_start, 0)
                        * (update + 1)
                        / step_ramp
                    ),
                    ppo_cfg.num_steps,
                )

                count_steps_delta = self._n_buffered_sampling(
                    rollouts,
                    current_episode_reward,
                    running_episode_stats,
                    buffer_ranges,
                    real_steps,
                    num_rollouts_done_store,
                )

                num_rollouts_done_store.add("num_done", 1)

                with self.timing.add_time("Learner-Wait"):
                    learner_res = self._learner_output_queue.get()

                distrib.barrier()
                self._learner_task_queue.put(rollouts)

                prev_rollouts = rollouts
                self._current_rollout_idx = (self._current_rollout_idx + 1) % len(
                    all_rollouts
                )
                rollouts = all_rollouts[self._current_rollout_idx]
                rollouts.after_update(prev_rollouts)

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                if learner_res is not None:
                    value_loss = learner_res["value_loss"]
                    action_loss = learner_res["action_loss"]
                else:
                    continue

                with self.timing.add_time("Logging"):
                    stats_ordering = list(sorted(running_episode_stats.keys()))
                    stats = torch.stack(
                        [running_episode_stats[k] for k in stats_ordering], 0,
                    ).to(device="cpu")
                    distrib.all_reduce(stats)

                    for i, k in enumerate(stats_ordering):
                        window_episode_stats[k].append(
                            stats[i].clone().to(device="cpu")
                        )

                    stats = torch.tensor(
                        [value_loss, action_loss, count_steps_delta], device="cpu",
                    )
                    distrib.all_reduce(stats)
                    stats = stats.to(device="cpu")
                    count_steps += int(stats[2].item())

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
                            "reward", deltas["reward"] / deltas["count"], count_steps,
                        )

                        # Check to see if there are any metrics
                        # that haven't been logged yet
                        metrics = {
                            k: v / deltas["count"]
                            for k, v in deltas.items()
                            if k not in {"reward", "count"}
                        }
                        if len(metrics) > 0:
                            writer.add_scalars("metrics", metrics, count_steps)

                        writer.add_scalars(
                            "losses",
                            {k: l for l, k in zip(losses, ["value", "policy"])},
                            count_steps,
                        )

                        # log stats
                        if update > 0 and update % self.config.LOG_INTERVAL == 0:
                            logger.info(
                                "update: {}\tfps: {:.3f}\tframes: {}".format(
                                    update,
                                    count_steps / ((time.time() - t_start) + prev_time),
                                    count_steps,
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
                            logger.info(learner_res["timing"])

                        # checkpoint model
                        if False and self.should_checkpoint(update, count_steps):
                            self.save_checkpoint(
                                f"ckpt.{count_checkpoints}.pth", dict(step=count_steps),
                            )
                            count_checkpoints += 1

                update += 1

            self.save_checkpoint("ckpt.done.pth", dict(step=count_steps))
            for e in self.envs:
                e.close()

            del self._renderer
            self._renderer = None

            self.dataset_groups = None
            self._episode_manager.shutdown()
            self._episode_manager.join()
            self._episode_manager = None
