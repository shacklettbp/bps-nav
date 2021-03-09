#/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 1 \
    bps_nav/run.py \
    --exp-config bps_nav/config/pointnav/ddppo_pointnav.yaml \
    --run-type train
