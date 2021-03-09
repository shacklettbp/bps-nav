#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Tuple, Optional
import abc
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from bps_nav.common.utils import CategoricalNet
from bps_nav.common.running_mean_and_var import RunningMeanAndVar
from bps_nav.rl.models.rnn_state_encoder import build_rnn_state_encoder
from bps_nav.rl.models.simple_cnn import SimpleCNN


@torch.jit.script
def _process_depth(
    observations: Dict[str, torch.Tensor], n_discrete_depth: int = 10
) -> Dict[str, torch.Tensor]:
    if "depth" in observations:
        depth_observations = observations["depth"]
        if depth_observations.shape[1] != 1:
            depth_observations = depth_observations.permute(0, 3, 1, 2)

        depth_observations.clamp_(0.0, 10.0).mul_(1.0 / 10.0)

        if n_discrete_depth > 0:
            dd_range = torch.arange(
                0,
                1 + 1.0 / n_discrete_depth,
                1.0 / n_discrete_depth,
                device=depth_observations.device,
                dtype=depth_observations.dtype,
            )
            dd_range = dd_range.view(n_discrete_depth + 1, 1, 1)

            discrete_depth = (depth_observations >= dd_range[:-1]) & (
                depth_observations < dd_range[1:]
            )

            # This acts as torch.cat((depth_observations, discrete_depth), 1)
            # but works with the different dtypes, avoiding a
            # discrete_depth.type_as(depth_observations)
            depth_observations = depth_observations.repeat(
                1, n_discrete_depth + 1, 1, 1
            )
            depth_observations[:, 1:].copy_(discrete_depth)

        observations["depth"] = depth_observations

    return observations


class SNIBottleneck(nn.Module):
    active: bool
    __constants__ = ["active"]

    def __init__(self, input_size, output_size, active=False):
        super().__init__()
        self.active: bool = active

        if active:
            self.output_size = output_size
            self.bottleneck = nn.Sequential(nn.Linear(input_size, 2 * output_size))
        else:
            self.output_size = input_size
            self.bottleneck = nn.Sequential()

    def forward(
        self, x
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.active:
            return x, None, None
        else:
            x = self.bottleneck(x)
            mu, sigma = torch.chunk(x, 2, x.dim() - 1)

            if self.training:
                sigma = F.softplus(sigma)
                sample = torch.addcmul(mu, sigma, torch.randn_like(sigma), value=1.0)

                # This is KL with standard normal for only
                # the parts that influence the gradient!
                kl = torch.addcmul(-torch.log(sigma), mu, mu, value=0.5)
                kl = torch.addcmul(kl, sigma, sigma, value=0.5)
            else:
                sample = None
                kl = None

            return mu, sample, kl


class ScriptableAC(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net

        self.action_distribution = CategoricalNet(self.net.output_size, dim_actions)
        self.critic = CriticHead(self.net.output_size)

    def post_net(
        self, features, rnn_hidden_states, deterministic: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        logits = self.action_distribution(features)
        value = self.critic(features)["value"]

        dist_result = self.action_distribution.dist.act(
            logits, sample=not deterministic
        )

        return (
            value,
            dist_result,
            rnn_hidden_states,
        )

    @torch.jit.export
    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        return self.post_net(features, rnn_hidden_states, deterministic)

    @torch.jit.export
    def act_post_visual(
        self,
        visual_out,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        features, rnn_hidden_states = self.net.rnn_forward(
            visual_out,
            observations["pointgoal_with_gps_compass"],
            rnn_hidden_states,
            prev_actions,
            masks,
        )

        return self.post_net(features, rnn_hidden_states, deterministic)

    @torch.jit.export
    def get_value(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)["value"]

    @torch.jit.export
    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
    ) -> Dict[str, torch.Tensor]:
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        result: Dict[str, torch.Tensor] = {}

        logits = self.action_distribution(features)

        result.update(self.action_distribution.dist.evaluate_actions(logits, action))
        result.update(self.critic(features))

        return result


class Policy(nn.Module):
    def __init__(self, net, observation_space, dim_actions):
        super().__init__()
        self.dim_actions = dim_actions

        self.num_recurrent_layers = net.num_recurrent_layers
        self.is_blind = net.is_blind

        self.ac = ScriptableAC(net, self.dim_actions)
        self.accelerated_net = None
        self.accel_out = None

        if "rgb" in observation_space.spaces:
            self.running_mean_and_var = RunningMeanAndVar(
                observation_space.spaces["rgb"].shape[0]
                + (
                    observation_space.spaces["depth"].shape[0]
                    if "depth" in observation_space.spaces
                    else 0
                ),
                initial_count=1e4,
            )
        else:
            self.running_mean_and_var = None

    def script_net(self):
        self.ac = torch.jit.script(self.ac)

    def init_trt(self):
        raise NotImplementedError

    def update_trt_weights(self):
        raise NotImplementedError

    def trt_enabled(self):
        return self.accelerated_net != None

    def forward(self, *x):
        raise NotImplementedError

    def _preprocess_obs(self, observations):
        dtype = next(self.parameters()).dtype
        observations = {
            k: v.to(dtype=dtype, copy=True) for k, v in observations.items()
        }

        observations = _process_depth(observations)

        if "rgb" in observations:
            rgb = observations["rgb"]
            if rgb.shape[1] != 3:
                rgb = rgb.permute(0, 3, 1, 2)

            rgb.mul_(1.0 / 255.0)
            x = [rgb]
            if "depth" in observations:
                x.append(observations["depth"])

            x = self.running_mean_and_var(torch.cat(x, 1))

            observations["rgb"] = x[:, 0:3]
            if "depth" in observations:
                observations["depth"] = x[:, 3:]

        return observations

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False,
    ):
        observations = self._preprocess_obs(observations)
        return self.ac.act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )

    def act_fast(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False,
    ):
        observations = self._preprocess_obs(observations)
        if self.accelerated_net == None:
            return self.ac.act(
                observations, rnn_hidden_states, prev_actions, masks, deterministic
            )
        else:
            if "rgb" in observations:
                trt_input = observations["rgb"]
            elif "depth" in observations:
                trt_input = observations["depth"]
            else:
                assert False

            self.accelerated_net.infer(
                trt_input.data_ptr(), torch.cuda.current_stream().cuda_stream
            )
            return self.ac.act_post_visual(
                self.accel_out,
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                deterministic,
            )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        observations = self._preprocess_obs(observations)
        return self.ac.get_value(observations, rnn_hidden_states, prev_actions, masks)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action,
    ):
        observations = self._preprocess_obs(observations)
        return self.ac.evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action
        )


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

        self.layer_init()

    def layer_init(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

            if isinstance(m, nn.Linear):
                m.weight.data *= 0.1 / torch.norm(
                    m.weight.data, p=2, dim=1, keepdim=True
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        return {"value": self.fc(x)}

class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass
