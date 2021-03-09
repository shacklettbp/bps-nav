#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Type, Union
import os.path as osp
import multiprocessing
import torch.distributed as distrib
import socket
from collections import OrderedDict

import numpy as np
import torch

try:
    import psutil
except ImportError:
    psutil = None


class Sync:
    def __init__(self, envs, idx):
        self._envs = envs
        self._idx = idx

    def wait(self):
        self._envs.wait_for_frame(self._idx)

def construct_envs(
    config, num_worker_groups: int = 4, double_buffered: bool = True,
):
    import bps_sim 
    import bps_pytorch

    batch_size = config.SIM_BATCH_SIZE
    num_workers = -1

    episodes_folder = osp.join(
        osp.dirname(
            config.TASK_CONFIG.DATASET.DATA_PATH.format(
                split=config.TASK_CONFIG.DATASET.SPLIT
            )
        ),
        "content",
    )

    should_set_affinity = True
    task = config.TASK.lower()
    assert task in {"pointnav", "flee", "exploration"}

    if task == "pointnav":
        envs_class = bps_sim.PointNavRolloutGenerator
    elif task == "flee":
        envs_class = bps_sim.FleeRolloutGenerator
    else:
        envs_class = bps_sim.ExplorationRolloutGenerator

    envs = envs_class(
        episodes_folder,
        "data/scene_datasets",
        batch_size,
        num_worker_groups,
        num_workers,
        config.SIMULATOR_GPU_ID,
        config.RESOLUTION,
        config.COLOR,
        config.DEPTH,
        double_buffered,
        config.TASK_CONFIG.SEED,
        should_set_affinity,
    )

    observations = []
    rewards = []
    masks = []
    infos = []
    syncs = []

    for i in range(2 if double_buffered else 1):
        observation = OrderedDict()

        if config.COLOR:
            observation["rgb"] = bps_pytorch.make_color_tensor(
                envs.rgba(i),
                config.SIMULATOR_GPU_ID,
                batch_size // (2 if double_buffered else 1),
                config.RESOLUTION,
            )[..., 0:3].permute(0, 3, 1, 2)

        if config.DEPTH:
            observation["depth"] = bps_pytorch.make_depth_tensor(
                envs.depth(i),
                config.SIMULATOR_GPU_ID,
                batch_size // (2 if double_buffered else 1),
                config.RESOLUTION,
            ).unsqueeze(1)

        if task == "pointnav":
            observation["pointgoal_with_gps_compass"] = torch.from_numpy(
                envs.get_polars(i)
            )

        packed_info = envs.get_infos(i)
        info = {
            k: torch.from_numpy(packed_info[k]).view(-1, 1)
            for k in packed_info.dtype.names
        }

        observations.append(observation)
        rewards.append(torch.from_numpy(envs.get_rewards(i)).view(-1, 1))
        masks.append(torch.from_numpy(envs.get_masks(i)).view(-1, 1))
        infos.append(info)
        syncs.append(Sync(envs, i))

    return (envs, observations, rewards, masks, infos, syncs)


def construct_envs_habitat(
    config,
    env_class,
    workers_ignore_signals: bool = False,
):
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_processes as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    import habitat
    from habitat import make_dataset
    from habitat_baselines.utils.env_utils import make_env_fn

    num_processes = config.NUM_PROCESSES
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_processes:
            raise RuntimeError(
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )

        random.shuffle(scenes)

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = config.SIMULATOR_GPU_ID

        sensors = []
        if config.DEPTH:
            sensors += ["DEPTH_SENSOR"]

        if config.COLOR:
            sensors += ["RGB_SENSOR"]

        task_config.SIMULATOR.AGENT_0.SENSORS = sensors

        task_config.SIMULATOR.RGB_SENSOR.HEIGHT = config.RESOLUTION[1]
        task_config.SIMULATOR.RGB_SENSOR.WIDTH = config.RESOLUTION[0]

        task_config.SIMULATOR.DEPTH_SENSOR.HEIGHT = config.RESOLUTION[1]
        task_config.SIMULATOR.DEPTH_SENSOR.WIDTH = config.RESOLUTION[0]
        task_config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        task_config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 20.0

        proc_config.freeze()
        configs.append(proc_config)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(tuple(zip(configs, env_classes))),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
