#!/bin/bash
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

RUN_NAME="depth"


set -x
python bps_nav/run.py \
    --exp-config bps_nav/config/pointnav/ddppo_pointnav.yaml \
    --run-type eval \
    TENSORBOARD_DIR "tb/${RUN_NAME}" \
    CHECKPOINT_FOLDER "data/checkpoints/${RUN_NAME}" \
    EVAL_CKPT_PATH_DIR "data/checkpoints/${RUN_NAME}" \
    NUM_PROCESSES 6 \
    RL.DDPPO.backbone se_resnet9_fixup \
    RL.DDPPO.resnet_baseplanes 64 \
    RESOLUTION "[64, 64]" \
    COLOR False \
    DEPTH True
