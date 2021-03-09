#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections import defaultdict, OrderedDict

#  import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from bps_nav.common.tree_utils import (
    tree_append_in_place,
    tree_clone_shallow,
    tree_map,
    tree_map_in_place,
    tree_select,
    tree_clone_structure,
    tree_copy_in_place,
    tree_indexed_copy_in_place,
    tree_multi_map,
)


class RolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        use_data_aug=False,
        aug_type="crop",
        vtrace=False,
    ):
        self.vtrace = vtrace
        self.storage_buffers = {}

        observations = {}

        for sensor in observation_space.spaces:
            shape = observation_space.spaces[sensor].shape
            observations[sensor] = torch.from_numpy(
                np.zeros(
                    (num_steps + 1, num_envs, *shape),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.storage_buffers["observations"] = observations

        self.aug_transforms = nn.Sequential()
        if use_data_aug:
            if aug_type == "crop":
                self.aug_transforms = nn.Sequential(
                    nn.ReplicationPad2d(7),
                    K.RandomCrop((128, 128), same_on_batch=True),
                )
            elif aug_type == "affine":
                self.aug_transforms = nn.Sequential(
                    K.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),
                        scale=(0.1, 0.1),
                        same_on_batch=True,
                    )
                )

        self.storage_buffers["recurrent_hidden_states"] = torch.zeros(
            num_steps + 1, num_envs, num_recurrent_layers, recurrent_hidden_state_size,
        )

        self.storage_buffers["rewards"] = torch.zeros(num_steps + 1, num_envs, 1)
        self.storage_buffers["value_preds"] = torch.zeros(num_steps + 1, num_envs, 1)
        self.storage_buffers["returns"] = torch.zeros(num_steps + 1, num_envs, 1)

        self.storage_buffers["action_log_probs"] = torch.zeros(
            num_steps + 1, num_envs, 1
        )
        action_shape = 1

        self.storage_buffers["actions"] = torch.zeros(
            num_steps + 1, num_envs, action_shape
        ).long()
        self.storage_buffers["prev_actions"] = torch.zeros(
            num_steps + 1, num_envs, action_shape
        ).long()

        self.storage_buffers["masks"] = torch.zeros(
            num_steps + 1, num_envs, 1, dtype=torch.bool
        )
        # Valids is a thing that so we can pad for DD-PPO so that cuDNN benchmark works
        self.storage_buffers["valids"] = torch.zeros(
            num_steps + 1, num_envs, 1, dtype=torch.bool
        )

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.device = device
        tree_map_in_place(lambda v: v.to(self.device), self.storage_buffers)

    def to_fp16(self):
        def _obs_to_fp16(v):
            if v.dtype == torch.float32:
                return v.to(dtype=torch.float16)

            return v

        tree_map_in_place(_obs_to_fp16, self.storage_buffers["observations"])

        self.storage_buffers["recurrent_hidden_states"] = self.storage_buffers[
            "recurrent_hidden_states"
        ].to(dtype=torch.float16)

    def share_memory(self):
        tree_map_in_place(lambda v: v.share_memory_(), self.storage_buffers)

    def insert(
        self,
        observations=None,
        recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        masks=None,
        non_blocking=False,
    ):
        next_step_insert_tree = OrderedDict()
        current_step_insert_tree = OrderedDict()
        if observations is not None:
            next_step_insert_tree["observations"] = observations

        if recurrent_hidden_states is not None:
            next_step_insert_tree["recurrent_hidden_states"] = recurrent_hidden_states

        if masks is not None:
            next_step_insert_tree["masks"] = masks

        if actions is not None:
            current_step_insert_tree["actions"] = actions
            next_step_insert_tree["prev_actions"] = actions

        if action_log_probs is not None:
            current_step_insert_tree["action_log_probs"] = action_log_probs

        if value_preds is not None:
            current_step_insert_tree["value_preds"] = value_preds

        if rewards is not None:
            current_step_insert_tree["rewards"] = rewards

        if len(next_step_insert_tree) > 0:
            tree_indexed_copy_in_place(
                self.storage_buffers,
                next_step_insert_tree,
                target_index=self.step + 1,
                non_blocking=non_blocking,
            )

        if len(current_step_insert_tree) > 0:
            tree_indexed_copy_in_place(
                self.storage_buffers,
                current_step_insert_tree,
                target_index=self.step,
                non_blocking=non_blocking,
            )

    def advance(self):
        self.storage_buffers["valids"][self.step] = True
        self.step = self.step + 1

    def after_update(self, prev_store=None):
        if prev_store is None:
            prev_store = self

        copy_step = prev_store.step

        tree_indexed_copy_in_place(
            self.storage_buffers,
            prev_store.storage_buffers,
            target_index=0,
            source_index=copy_step,
            non_blocking=True,
        )

        self.storage_buffers["masks"][1:].fill_(False)
        self.storage_buffers["valids"].fill_(False)
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.storage_buffers["value_preds"][self.step] = next_value

            value_preds = self.storage_buffers["value_preds"]
            rewards = self.storage_buffers["rewards"]
            returns = self.storage_buffers["returns"]
            masks = self.storage_buffers["masks"]
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    rewards[step]
                    + gamma * value_preds[step + 1] * masks[step + 1]
                    - value_preds[step]
                )
                gae = delta + gamma * tau * gae * masks[step + 1]
                returns[step] = gae + value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = advantages.size(1)
        nbuffers = len(self.buffers)

        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )

        for mb_inds in torch.randperm(num_processes).chunk(num_mini_batch):
            mb = tree_clone_structure(list, self.storage_buffers)
            mb["advantages"] = []

            for ind in mb_inds:
                mb = tree_append_in_place(
                    mb, tree_select((slice(self.step), ind), self.storage_buffers)
                )
                mb["advantages"].append(advantages[: self.step, ind])
                mb["recurrent_hidden_states"][-1] = mb["recurrent_hidden_states"][-1][
                    0:1
                ]

            mb = tree_map(
                lambda v: torch.flatten(v, 0, 1),
                tree_map(
                    lambda v: v.to(device=self.device),
                    tree_map(lambda v: torch.stack(v, 1), mb),
                ),
            )

            yield mb


class DoubleBufferedRolloutStorage:
    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        use_data_aug=False,
        aug_type="crop",
        double_buffered=True,
        vtrace=False,
    ):
        self.vtrace = vtrace

        nbuffers = 2 if double_buffered else 1
        self.buffers = [
            RolloutStorage(
                num_steps,
                num_envs // nbuffers,
                observation_space,
                action_space,
                recurrent_hidden_state_size,
                num_recurrent_layers,
                use_data_aug,
                aug_type,
                vtrace,
            )
            for _ in range(nbuffers)
        ]

    def to(self, device):
        for buf in self.buffers:
            buf.to(device)

    def to_fp16(self):
        for buf in self.buffers:
            buf.to_fp16()

    def share_memory(self):
        for buf in self.buffers:
            buf.share_memory()

    def after_update(self, prev_store=None):
        if prev_store is None:
            prev_store = self

        for i, buf in enumerate(self.buffers):
            buf.after_update(prev_store[i])

    def __getitem__(self, idx):
        return self.buffers[idx]

    def __len__(self):
        return len(self.buffers)

    def recurrent_generator(self, advantages, num_mini_batch, timing, device=None):
        num_processes = advantages.size(1)
        nbuffers = len(self.buffers)

        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )

        if device is None:
            device = self.buffers[0].rewards.device

        self.step = self[0].step
        for idx in range(nbuffers):
            assert self.step == self[idx].step

        self.step = self[0].num_steps

        if self.vtrace:
            self.step += 1

        for mb_inds in torch.randperm(num_processes).chunk(num_mini_batch):
            with timing.add_time("Generate-Mini-Batch"):
                adv_inds = []
                inds_for_buffer = [[] for _ in range(nbuffers)]

                for ind in mb_inds:
                    adv_inds.append(ind)

                    buffer_ind = ind // (num_processes // nbuffers)
                    env_ind = ind % (num_processes // nbuffers)

                    inds_for_buffer[buffer_ind].append(env_ind)

                mbs = []
                for i in range(nbuffers):
                    mbs.append(
                        tree_select(
                            (slice(self.step), inds_for_buffer[i]),
                            self[i].storage_buffers,
                        )
                    )
                if nbuffers == 1:
                    mb = mbs[0]
                else:
                    mb = tree_multi_map(
                        lambda *tensors: torch.cat(tensors, 1), mbs[0], *mbs[1:]
                    )

                mb["advantages"] = advantages[: self.step, adv_inds]
                mb["recurrent_hidden_states"] = mb["recurrent_hidden_states"][0:1]

                mb = tree_map(
                    lambda v: torch.flatten(v.contiguous(), 0, 1),
                    tree_map(lambda v: v.to(device=device), mb),
                )

            yield mb
