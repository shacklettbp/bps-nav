#!/bin/bash
set -e

export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

RUN_NAME="rgb"

set -x
python bps_nav/run.py \
    --exp-config bps_nav/config/pointnav/ddppo_pointnav.yaml \
    --run-type train \
    TASK_CONFIG.DATASET.SPLIT "train-2plus" \
    NUM_PARALLEL_SCENES 4 \
    TENSORBOARD_DIR "tb/${RUN_NAME}" \
    CHECKPOINT_FOLDER "data/checkpoints/${RUN_NAME}" \
    EVAL_CKPT_PATH_DIR "data/checkpoints/${RUN_NAME}" \
    LOG_INTERVAL 50 \
    SIM_BATCH_SIZE 128 \
    RL.DDPPO.step_ramp 1 \
    RL.PPO.num_mini_batch 2 \
    RL.PPO.ppo_epoch 1 \
    RL.PPO.num_steps 32 \
    RL.PPO.max_grad_norm 1.0 \
    TOTAL_NUM_STEPS 2.5e9 \
    RL.DDPPO.backbone se_resnet9_fixup \
    RL.PPO.weight_decay 1e-2 \
    RL.PPO.lamb True \
    RL.DDPPO.scale_lr True \
    RL.PPO.lamb_min_trust 0.01 \
    RL.PPO.lr 5e-4  \
    RL.PPO.vtrace False \
    COLOR True \
    DEPTH False \
    RESOLUTION "[64, 64]"
