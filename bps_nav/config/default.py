#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from typing import List, Optional, Union
from bps_nav.config import Config

import numpy as np

CN = Config

MODULE_DIR = osp.dirname(osp.dirname(osp.dirname(__file__)))

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
# task config can be a list of configs like "A.yaml,B.yaml"
_C.BASE_TASK_CONFIG_PATH = "bps_nav/configs/tasks/pointnav.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "NavRLEnv"
_C.POLICY_NAME = "ResNetPolicy"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = -1
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SIM_BATCH_SIZE = 128
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = -1
_C.TOTAL_NUM_STEPS = 1e4
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.NUM_CHECKPOINTS = 100
_C.RESOLUTION = [128, 128]
_C.COLOR = False
_C.DEPTH = True
_C.NUM_PARALLEL_SCENES = 4
_C.TASK = "PointNav"
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.REWARD_MEASURE = "distance_to_goal"
_C.RL.SUCCESS_MEASURE = "spl"
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.decay_factor = 0.33
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 7e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.use_normalized_advantage = True
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.use_data_aug = False
_C.RL.PPO.aug_type = "crop"
_C.RL.PPO.vtrace = False
_C.RL.PPO.lamb = True
_C.RL.PPO.lamb_min_trust = 0.01
_C.RL.PPO.weight_decay = 1e-4
_C.RL.PPO.ada_scale = False
_C.RL.PPO.num_accumulate_steps = 1
# -----------------------------------------------------------------------------
# DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
# -----------------------------------------------------------------------------
_C.RL.DDPPO = CN()
_C.RL.DDPPO.sync_frac = 0.6
_C.RL.DDPPO.distrib_backend = "GLOO"
_C.RL.DDPPO.rnn_type = "GRU"
_C.RL.DDPPO.num_recurrent_layers = 1
_C.RL.DDPPO.backbone = "resnet18"
_C.RL.DDPPO.pretrained_weights = "data/ddppo-models/gibson-2plus-resnet50.pth"
# Loads pretrained weights
_C.RL.DDPPO.pretrained = False
# Loads just the visual encoder backbone weights
_C.RL.DDPPO.pretrained_encoder = False
# Whether or not the visual encoder backbone will be trained
_C.RL.DDPPO.train_encoder = True
# Whether or not to reset the critic linear layer
_C.RL.DDPPO.reset_critic = True
_C.RL.DDPPO.scale_lr = True
_C.RL.DDPPO.use_avg_pool = False
_C.RL.DDPPO.use_batch_norm = False
_C.RL.DDPPO.resnet_baseplanes = 32
_C.RL.DDPPO.step_ramp = 5000
_C.RL.DDPPO.step_ramp_start = 2

def get_task_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """

    config = CN()

    config.SEED = 100
    config.ENVIRONMENT = CN()
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
    config.ENVIRONMENT.MAX_EPISODE_SECONDS = 10000000
    config.ENVIRONMENT.ITERATOR_OPTIONS = CN()
    config.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = True
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
    config.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = True
    config.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = int(1e4)
    config.ENVIRONMENT.ITERATOR_OPTIONS.STEP_REPETITION_RANGE = 0.2
    config.TASK = CN()
    config.TASK.TYPE = "Nav-v0"
    config.TASK.SUCCESS_DISTANCE = 0.2
    config.TASK.SENSORS = []
    config.TASK.MEASUREMENTS = []
    config.TASK.GOAL_SENSOR_UUID = "pointgoal"
    config.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    ACTIONS = CN()
    ACTIONS.STOP = CN()
    ACTIONS.STOP.TYPE = "StopAction"
    ACTIONS.MOVE_FORWARD = CN()
    ACTIONS.MOVE_FORWARD.TYPE = "MoveForwardAction"
    ACTIONS.TURN_LEFT = CN()
    ACTIONS.TURN_LEFT.TYPE = "TurnLeftAction"
    ACTIONS.TURN_RIGHT = CN()
    ACTIONS.TURN_RIGHT.TYPE = "TurnRightAction"
    ACTIONS.LOOK_UP = CN()
    ACTIONS.LOOK_UP.TYPE = "LookUpAction"
    ACTIONS.LOOK_DOWN = CN()
    ACTIONS.LOOK_DOWN.TYPE = "LookDownAction"
    ACTIONS.TELEPORT = CN()
    ACTIONS.TELEPORT.TYPE = "TeleportAction"
    
    config.TASK.ACTIONS = ACTIONS
    config.TASK.POINTGOAL_SENSOR = CN()
    config.TASK.POINTGOAL_SENSOR.TYPE = "PointGoalSensor"
    config.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "POLAR"
    config.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 2
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR = config.TASK.POINTGOAL_SENSOR.clone()
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.TYPE = (
        "PointGoalWithGPSCompassSensor"
    )

    config.TASK.SUCCESS = CN()
    config.TASK.SUCCESS.TYPE = "Success"
    config.TASK.SUCCESS.SUCCESS_DISTANCE = 0.2
    config.TASK.SPL = CN()
    config.TASK.SPL.TYPE = "SPL"

    config.SIMULATOR = CN()
    config.SIMULATOR.TYPE = "Sim-v0"
    config.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # in metres
    config.SIMULATOR.DEFAULT_AGENT_ID = 0

    SIMULATOR_SENSOR = CN()
    SIMULATOR_SENSOR.HEIGHT = 480
    SIMULATOR_SENSOR.WIDTH = 640
    SIMULATOR_SENSOR.HFOV = 90  # horizontal field of view in degrees
    SIMULATOR_SENSOR.POSITION = [0, 1.25, 0]
    SIMULATOR_SENSOR.ORIENTATION = [0.0, 0.0, 0.0]  # Euler's angles

    config.SIMULATOR.RGB_SENSOR = SIMULATOR_SENSOR.clone()
    config.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
    config.SIMULATOR.DEPTH_SENSOR = SIMULATOR_SENSOR.clone()
    config.SIMULATOR.DEPTH_SENSOR.TYPE = "HabitatSimDepthSensor"
    config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10.0
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = True

    config.SIMULATOR.AGENT_0 = CN()
    config.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.SIMULATOR.AGENT_0.RADIUS = 0.1
    config.SIMULATOR.AGENT_0.MASS = 32.0
    config.SIMULATOR.AGENT_0.LINEAR_ACCELERATION = 20.0
    config.SIMULATOR.AGENT_0.ANGULAR_ACCELERATION = 4 * 3.14
    config.SIMULATOR.AGENT_0.LINEAR_FRICTION = 0.5
    config.SIMULATOR.AGENT_0.ANGULAR_FRICTION = 1.0
    config.SIMULATOR.AGENT_0.COEFFICIENT_OF_RESTITUTION = 0.0
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    config.SIMULATOR.AGENT_0.IS_SET_START_STATE = False
    config.SIMULATOR.AGENT_0.START_POSITION = [0, 0, 0]
    config.SIMULATOR.AGENT_0.START_ROTATION = [0, 0, 0, 1]
    config.SIMULATOR.AGENTS = ["AGENT_0"]
    config.SIMULATOR.HABITAT_SIM_V0 = CN()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = False
    config.DATASET = CN()
    config.DATASET.TYPE = "PointNav-v1"
    config.DATASET.SPLIT = "train"
    config.DATASET.SCENES_DIR = "data/scene_datasets"
    config.DATASET.CONTENT_SCENES = ["*"]
    config.DATASET.DATA_PATH = (
        "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config

def get_config(
    config_paths: Optional[Union[List[str], str]] = None, opts: Optional[list] = None,
    get_task_config_override = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            if not osp.exists(config_path):
                config_path = osp.join(MODULE_DIR, config_path)
            config.merge_from_file(config_path)

    if opts:
        for k, v in zip(opts[0::2], opts[1::2]):
            if k == "BASE_TASK_CONFIG_PATH":
                config.BASE_TASK_CONFIG_PATH = v

    if get_task_config_override != None:
        config.TASK_CONFIG = get_task_config_override(config.BASE_TASK_CONFIG_PATH)
    else:
        config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    config.freeze()
    return config
