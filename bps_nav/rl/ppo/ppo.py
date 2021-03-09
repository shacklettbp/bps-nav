#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Dict, Any, Callable, Union
from collections import defaultdict
import types


import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bps_nav.common.running_mean_and_var import RunningMeanAndVar
from bps_nav.rl.ppo.lamb import Lamb
from bps_nav.rl.ppo.fp16_adascale import FP16AdaScale
from bps_nav.rl.ppo.policy import Policy
from bps_nav.common.rollout_storage import DoubleBufferedRolloutStorage
from bps_nav.common.tree_utils import (
    tree_append_in_place,
    tree_clone_shallow,
    tree_map,
    tree_select,
    tree_clone_structure,
    tree_copy_in_place,
)


EPS_PPO = 1e-5


@torch.no_grad()
def vtrace(rewards_batch, value_preds, masks_batch, ratios, gamma, tau, rho, c):
    T, N, _ = rewards_batch.size()
    value_preds = value_preds.view(T + 1, N, 1)
    masks_batch = masks_batch.view(T + 1, N, 1)
    ratios = ratios.view(T + 1, N, 1)

    vs = value_preds.clone()

    gamma_masks = gamma * masks_batch[1:]

    clipped_rho = torch.min(
        ratios[0:T], torch.tensor(rho, device=ratios.device, dtype=ratios.dtype)
    )
    deltas = clipped_rho * (
        rewards_batch + gamma_masks * value_preds[1:] - value_preds[0:T]
    )

    c_masks = (
        tau
        * gamma_masks
        * torch.min(
            ratios[0:T], torch.tensor(c, device=ratios.device, dtype=ratios.dtype)
        )
    )

    for step in reversed(range(T)):
        vs[step] = (
            value_preds[step]
            + deltas[step]
            + c_masks[step] * (vs[step + 1] - value_preds[step + 1])
        )

    advantages = clipped_rho * (rewards_batch + gamma_masks * vs[1:] - value_preds[0:T])

    return advantages, vs[0:T]


def compute_ppo_loss(ratio, adv_targ, valids_batch=None, clip_param=None):
    surr1 = ratio * adv_targ
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_targ
    action_loss = torch.min(surr1, surr2)
    if valids_batch is not None:
        action_loss = -torch.masked_select(action_loss, valids_batch).mean()
    else:
        action_loss = -action_loss.mean()

    return action_loss


def compute_value_loss(values, return_batch, valids_batch=None):
    value_loss = 0.5 * (return_batch - values).pow(2)

    if valids_batch is not None:
        value_loss = torch.masked_select(value_loss, valids_batch).mean()
    else:
        value_loss = value_loss.mean()

    return value_loss


class PPO(nn.Module):
    def __init__(
        self, actor_critic: Policy, ppo_cfg,
    ):

        super().__init__()
        self.ppo_cfg = ppo_cfg

        self.actor_critic = actor_critic
        self.vtrace = ppo_cfg.vtrace
        self.tau = ppo_cfg.tau
        self.gamma = ppo_cfg.gamma

        self.clip_param = ppo_cfg.clip_param
        self.ppo_epoch = ppo_cfg.ppo_epoch
        self.num_mini_batch = ppo_cfg.num_mini_batch

        self.value_loss_coef = ppo_cfg.value_loss_coef
        self.entropy_coef = ppo_cfg.entropy_coef

        self.max_grad_norm = ppo_cfg.max_grad_norm

        adam_param_names = [
            "bias",
            "gamma",
            "beta",
            "LayerNorm",
            "GroupNorm",
            "fixup",
        ]

        adam_params = [
            p
            for name, p in actor_critic.named_parameters()
            if p.requires_grad and any(an in name for an in adam_param_names)
        ]

        lamb_params = [
            p
            for name, p in actor_critic.named_parameters()
            if p.requires_grad and not any(an in name for an in adam_param_names)
        ]

        assert len(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters()))
        ) == (len(adam_params) + len(lamb_params))

        self.optimizer = Lamb(
            [
                # min_trust == 1.0 makes this Adam,
                dict(params=adam_params, min_trust=1.0),
                dict(params=lamb_params),
            ],
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            weight_decay=ppo_cfg.weight_decay,
            min_trust=ppo_cfg.lamb_min_trust if ppo_cfg.lamb else 1.0,
        )

        self.use_normalized_advantage = ppo_cfg.use_normalized_advantage

        #  self.register_buffer("adv_mean_biased", torch.tensor(0.0, device=self.device))

        #  self.register_buffer("adv_mean_unbias", torch.tensor(0.0, device=self.device))
    @property
    def device(self):
        return next(self.actor_critic.parameters()).device

    def init_amp(self, num_envs: int):
        self.actor_critic.ac.to(dtype=torch.float16)

        self.optimizer.load_state_dict(self.optimizer.state_dict())

        self.ada_scale = FP16AdaScale(
            self.optimizer,
            scale=int(
                self.ppo_cfg.num_steps
                * num_envs
                * torch.distributed.get_world_size()
                * self.ppo_cfg.num_accumulate_steps
                / self.ppo_cfg.num_mini_batch
                / (128 * 8)
            ),
            num_accumulate_steps=self.ppo_cfg.num_accumulate_steps,
            enabled=self.ppo_cfg.ada_scale,
        )
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.ada_scale.loss_scale = self.grad_scaler.get_scale()

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts: DoubleBufferedRolloutStorage):
        advantages = []
        for idx in range(len(rollouts.buffers)):
            advantages.append(
                rollouts[idx].returns[:-1] - rollouts[idx].value_preds[:-1]
            )

        advantages = torch.cat(advantages, 1)
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts: DoubleBufferedRolloutStorage, timing, warmup=False):
        advantages = self.get_advantages(rollouts)
        device = next(self.actor_critic.parameters()).device

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages,
                self.num_mini_batch * self.ada_scale.num_accumulate_steps,
                timing,
                device,
            )

            for mb in data_generator:
                # Reshape to do in a single forward pass for all steps
                eval_result = self.actor_critic.evaluate_actions(
                    mb["observations"],
                    mb["recurrent_hidden_states"],
                    mb["prev_actions"],
                    mb["masks"],
                    mb["actions"],
                )

                eval_result["ratio"] = torch.exp(
                    eval_result["action_log_probs"].to(dtype=torch.float32)
                    - mb["action_log_probs"].to(dtype=torch.float32)
                )
                if self.vtrace:
                    adv_targ, return_batch = vtrace(
                        rewards_batch,
                        values,
                        masks_batch,
                        ratio,
                        gamma=self.gamma,
                        tau=self.tau,
                        rho=1.0,
                        c=1.0,
                    )
                    T, N, _ = adv_targ.size()
                    valids_batch = valids_batch.view(T, N, 1)

                    eval_result = tree_map(
                        lambda v: v.view(T + 1, N, 1)[0:T], eval_result
                    )

                value_loss = compute_value_loss(
                    eval_result["value"], mb["returns"], mb["valids"],
                )

                action_loss = compute_ppo_loss(
                    eval_result["ratio"],
                    mb["advantages"],
                    mb["valids"],
                    self.clip_param,
                )

                dist_entropy = torch.masked_select(
                    eval_result["entropy"], mb["valids"]
                ).mean()

                total_loss = (
                    self.value_loss_coef * value_loss
                    + action_loss
                    - self.entropy_coef * dist_entropy
                )

                self.optim_step_and_backward(total_loss, warmup=warmup)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch * self.ada_scale.num_accumulate_steps

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def optim_step_and_backward(self, loss, warmup=False):
        self.before_backward(
            loss, self.ada_scale.accum_step == (self.ada_scale.num_accumulate_steps - 1)
        )
        self.grad_scaler.scale(loss).backward()
        self.ada_scale.inc_accumulate()

        if not self.ada_scale.accum_step == self.ada_scale.num_accumulate_steps:
            return

        if warmup:
            for param in self.ada_scale.parameters():
                if param.grad is not None:
                    param.grad.data.zero_()

        self.after_backward(loss)

        self.before_step()
        self.grad_scaler.step(self.ada_scale)
        self.after_step()

    def before_backward(self, loss, will_step_optim=False):
        pass

    def after_backward(self, loss):
        self.grad_scaler.unscale_(self.ada_scale)

    def before_step(self):
        if self.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.ada_scale.parameters(), self.max_grad_norm)

    def after_step(self):
        self.grad_scaler.update()
        self.ada_scale.zero_grad()
        self.ada_scale.loss_scale = self.grad_scaler.get_scale()
