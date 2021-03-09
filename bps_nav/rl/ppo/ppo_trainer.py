#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from bps_nav.common.base_trainer import BaseRLTrainer
from bps_nav.common.env_utils import construct_envs, construct_envs_habitat
from bps_nav.common.rollout_storage import RolloutStorage
from bps_nav.common.tensorboard_utils import TensorboardWriter
from bps_nav.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from bps_nav.common.logger import logger
from bps_nav.rl.ppo.ppo import PPO
from bps_nav.common.tree_utils import (
    tree_append_in_place,
    tree_clone_shallow,
    tree_map,
    tree_select,
    tree_clone_structure,
    tree_copy_in_place,
)

from bps_nav.rl.ddppo.policy import ResNetPolicy
from gym import spaces
from gym.spaces import Dict as SpaceDict

@torch.jit.script
def so3_to_matrix(q, m):
    m[..., 0, 0] = 1.0 - 2.0 * (q[..., 2] ** 2 + q[..., 3] ** 2)
    m[..., 0, 1] = 2.0 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
    m[..., 0, 2] = 2.0 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
    m[..., 1, 0] = 2.0 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
    m[..., 1, 1] = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 3] ** 2)
    m[..., 1, 2] = 2.0 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
    m[..., 2, 0] = 2.0 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
    m[..., 2, 1] = 2.0 * (q[..., 2] * q[..., 3] + q[..., 1] * q[..., 0])
    m[..., 2, 2] = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)


@torch.jit.script
def se3_to_4x4(se3_states):
    n = se3_states.size(0)

    mat = torch.zeros((n, 4, 4), dtype=torch.float32, device=se3_states.device)
    mat[:, 3, 3] = 1

    so3 = se3_states[:, 0:4]
    so3_to_matrix(so3, mat[:, 0:3, 0:3])

    mat[:, 0:3, 3] = se3_states[:, 4:]

    return mat


class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None, resume_from=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        #  if config is not None:
        #  logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = ResNetPolicy(
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=self.config.RL.DDPPO.rnn_type,
            num_recurrent_layers=self.config.RL.DDPPO.num_recurrent_layers,
            backbone=self.config.RL.DDPPO.backbone,
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """

        def _cast(param):
            if "Half" in param.type():
                param = param.to(dtype=torch.float32)

            return param

        checkpoint = {
            "state_dict": {k: _cast(v) for k, v in self.agent.state_dict().items()},
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict[str, Any]) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(v).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif v is None:
                result[k] = None
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _inference(self, rollouts, idx):
        with torch.no_grad(), self.timing.add_time("Rollout-Step"):
            with self.timing.add_time("Inference"):
                step_input = tree_select(
                    rollouts[idx].step, rollouts[idx].storage_buffers
                )

                (
                    values,
                    dist_result,
                    recurrent_hidden_states,
                ) = self.actor_critic.act_fast(
                    step_input["observations"],
                    step_input["recurrent_hidden_states"],
                    step_input["prev_actions"],
                    step_input["masks"],
                )

            with self.timing.add_time("Rollouts-Insert"):
                rollouts[idx].insert(
                    recurrent_hidden_states=recurrent_hidden_states,
                    action_log_probs=dist_result["action_log_probs"],
                    value_preds=values,
                    actions=dist_result["actions"],
                    non_blocking=False,
                )

            with self.timing.add_time("Inference"):
                cpu_actions = dist_result["actions"].squeeze(-1).to(device="cpu")

        return cpu_actions

    def _step_simulation(self, cpu_actions, idx):
        with self.timing.add_time("Rollout-Step"), self.timing.add_time(
            "Habitat-Step-Start"
        ):
            self.envs.step(idx, cpu_actions.numpy())

            obs = self._observations[idx]
            rewards = self._rewards[idx]
            masks = self._masks[idx]
            infos = self._rollout_infos[idx]

            return obs, rewards, masks, infos

    def _start_simulation(self, cpu_actions, idx):
        with self.timing.add_time("Rollout-Step"), self.timing.add_time(
            "Habitat-Step-Start"
        ):
            self.envs.step_start(idx, cpu_actions.numpy())

    def _wait_simulation(self, idx):
        with self.timing.add_time("Rollout-Step"), self.timing.add_time(
            "Habitat-Step-Wait"
        ):
            self.envs.step_end(idx)

            obs = self._observations[idx]
            rewards = self._rewards[idx]
            masks = self._masks[idx]
            infos = self._rollout_infos[idx]

            return obs, rewards, masks, infos

    def _render(self, idx):
        with self.timing.add_time("Rollout-Step"), self.timing.add_time(
            "Renderer-Start"
        ):
            self.envs.render(idx)

    def _sync_renderer_and_insert(self, rollouts, sim_step_res, idx):
        with self.timing.add_time("Rollout-Step"):
            batch, rewards, masks, infos = sim_step_res
            with self.timing.add_time("Renderer-Wait"):
                self._syncs[idx].wait()
                torch.cuda.current_stream().synchronize()

            with self.timing.add_time("Rollouts-Insert"):
                rollouts[idx].insert(
                    batch, rewards=rewards, masks=masks, non_blocking=False
                )

            rollouts[idx].advance()

            return masks.size(0)

    def _update_stats(
        self,
        rollouts,
        current_episode_reward,
        running_episode_stats,
        sim_step_res,
        stats_inds,
        idx,
    ):
        with self.timing.add_time("Rollout-Step"):
            batch, rewards, masks, infos = sim_step_res

            with self.timing.add_time("Update-Stats"):
                dones = masks == 0

                def _masked(v):
                    return torch.where(dones, v, v.new_zeros(()))

                current_episode_reward[stats_inds] += rewards
                running_episode_stats["reward"][stats_inds] += _masked(
                    current_episode_reward[stats_inds]
                )
                running_episode_stats["count"][stats_inds] += dones.type_as(
                    running_episode_stats["count"]
                )
                for k, v in infos.items():
                    if k not in running_episode_stats:
                        running_episode_stats[k] = torch.zeros_like(
                            running_episode_stats["count"]
                        )

                    running_episode_stats[k][stats_inds] += _masked(v)

                current_episode_reward[stats_inds].masked_fill_(dones, 0)

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        with self.timing.add_time("Rollout-Step"):
            with torch.no_grad(), self.timing.add_time("Inference"):
                with torch.no_grad():
                    step_observation = {
                        k: v[rollouts.step] for k, v in rollouts.observations.items()
                    }

                    (
                        values,
                        dist_result,
                        recurrent_hidden_states,
                    ) = self.actor_critic.act(
                        step_observation,
                        rollouts.recurrent_hidden_states[rollouts.step],
                        rollouts.prev_actions[rollouts.step],
                        rollouts.masks[rollouts.step],
                    )

                    cpu_actions = actions.squeeze(-1).to(device="cpu")

            with self.timing.add_time("Habitat-Step-Start"):
                self.envs.async_step(cpu_actions)

            with self.timing.add_time("Habitat-Step-Wait"):
                batch, rewards, masks, infos = self.envs.wait_step()

            with self.timing.add_time("Renderer-Render"):
                sync = self._draw_batch(batch)

            with self.timing.add_time("Update-Stats"):
                current_episode_reward += rewards
                running_episode_stats["reward"] += (1 - masks) * current_episode_reward
                running_episode_stats["count"] += 1 - masks
                for k, v in infos.items():
                    if k not in running_episode_stats:
                        running_episode_stats[k] = torch.zeros_like(
                            running_episode_stats["count"]
                        )

                    running_episode_stats[k] += (1 - masks) * v

                current_episode_reward *= masks

            with self.timing.add_time("Rollouts-Insert"):
                rollouts.insert(
                    rewards=rewards, masks=masks,
                )

            with self.timing.add_time("Renderer-Wait"):
                batch = self._fill_batch_result(batch, sync)

            with self.timing.add_time("Rollouts-Insert"):
                rollouts.insert(batch)

            rollouts.advance()

        return self.envs.num_envs

    @staticmethod
    def _update_agent_internal_fn(
        rollouts, agent, actor_critic, _static_encoder, timing, warmup=False
    ):
        actor_critic.train()
        if _static_encoder:
            _encoder.eval()

        with timing.add_time("PPO"):
            value_loss, action_loss, dist_entropy = agent.update(
                rollouts, timing, warmup=warmup
            )

        rollouts.after_update()

        return (value_loss, action_loss, dist_entropy)

    def _compute_returns(self, ppo_cfg, rollouts):
        with self.timing.add_time("Learning"), torch.no_grad(), self.timing.add_time(
            "Inference"
        ):
            for idx in range(len(rollouts)):
                last_input = tree_select(
                    rollouts[idx].step, rollouts[idx].storage_buffers
                )

                next_value = self.actor_critic.get_value(
                    last_input["observations"],
                    last_input["recurrent_hidden_states"],
                    last_input["prev_actions"],
                    last_input["masks"],
                )

                with self.timing.add_time("Compute-Returns"):
                    rollouts[idx].compute_returns(
                        next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
                    )

    def _update_agent(self, rollouts, warmup=False):
        with self.timing.add_time("Learning"):
            losses = self._update_agent_internal_fn(
                rollouts,
                self.agent,
                self.actor_critic,
                self._static_encoder,
                self.timing,
                warmup=warmup,
            )

            if self.actor_critic.trt_enabled():
                with self.timing.add_time("TRT Refit"):
                    with self.timing.add_time("TRT Weights"):
                        weights = self.actor_critic.get_trt_weights()
                    with self.timing.add_time("TRT Update"):
                        self.actor_critic.update_trt_weights(weights)

            return losses

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """

        from habitat_baselines.common.environments import get_env_class

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        #  logger.info(f"env config: {config}")
        self.envs = construct_envs_habitat(config, get_env_class(config.ENV_NAME))
        self.observation_space = SpaceDict(
            {
                "pointgoal_with_gps_compass": spaces.Box(
                    low=0.0, high=1.0, shape=(2,), dtype=np.float32
                )
            }
        )

        if self.config.COLOR:
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

        if self.config.DEPTH:
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

        self.action_space = self.envs.action_spaces[0]
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        self.actor_critic.script_net()
        self.actor_critic.eval()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_PROCESSES,
            self.actor_critic.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.bool
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        evals_per_ep = 5
        count_per_ep = defaultdict(lambda: 0)

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (_, dist_result, test_recurrent_hidden_states) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                actions = dist_result["actions"]

                prev_actions.copy_(actions)
                actions = actions.to("cpu")

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[False] if done else [True] for done in dones],
                dtype=torch.bool,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                next_count_key = (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                )

                if count_per_ep[next_count_key] == evals_per_ep:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    count_key = (
                        current_episodes[i].scene_id,
                        current_episodes[i].episode_id,
                    )
                    count_per_ep[count_key] = count_per_ep[count_key] + 1

                    ep_key = (count_key, count_per_ep[count_key])
                    stats_episodes[ep_key] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        self.envs.close()
        pbar.close()
        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            values = [
                v[stat_key] for v in stats_episodes.values() if v[stat_key] is not None
            ]
            if len(values) > 0:
                aggregated_stats[stat_key] = sum(values) / len(values)
            else:
                aggregated_stats[stat_key] = 0

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward", {"average reward": aggregated_stats["reward"]}, step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.num_frames = step_id

        time.sleep(30)
