baselines
==============================
### Installation

The `bps_nav` sub-package is NOT included upon installation by default. To install `bps_nav`, use the following command instead:
```bash
pip install -r requirements.txt
python setup.py develop --all
```
This will also install additional requirements for each sub-module in `bps_nav/`, which are specified in `requirements.txt` files located in the sub-module directory.


### Reinforcement Learning (RL)

**Proximal Policy Optimization (PPO)**

**paper**: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

**code**: majority of the PPO implementation is taken from
[pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr).

**dependencies**: pytorch 1.0, for installing refer to [pytorch.org](https://pytorch.org/)

For training on sample data please follow steps in the repository README. You should download the sample [test scene data](http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip), extract it under the main repo (`habitat-api/`, extraction will create a data folder at `habitat-api/data`) and run the below training command.

**train**:
```bash
python -u bps_nav/run.py --exp-config bps_nav/config/pointnav/ppo_pointnav_example.yaml --run-type train
```

**test**:
```bash
python -u bps_nav/run.py --exp-config bps_nav/config/pointnav/ppo_pointnav_example.yaml --run-type eval
```

We also provide trained RGB, RGBD, Blind PPO models.
To use them download pre-trained pytorch models from [link](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/bps_nav_v1.zip) and unzip and specify model path [here](agents/ppo_agents.py#L132).

The `bps_nav/config/pointnav/ppo_pointnav.yaml` config has better hyperparamters for large scale training and loads the [Gibson PointGoal Navigation Dataset](/README.md#task-datasets) instead of the test scenes.
Change the field `task_config` in `bps_nav/config/pointnav/ppo_pointnav.yaml` to `configs/tasks/pointnav_mp3d.yaml` for training on [MatterPort3D PointGoal Navigation Dataset](/README.md#task-datasets).

### Classic

**SLAM based**

- [Handcrafted agent baseline](slambased/README.md) adopted from the paper
"Benchmarking Classic and Learned Navigation in Complex 3D Environments".
### Additional Utilities

**Episode iterator options**:
Coming very soon

**Tensorboard and video generation support**

Enable tensorboard by changing `tensorboard_dir` field in `bps_nav/config/pointnav/ppo_pointnav.yaml`.

Enable video generation for `eval` mode by changing `video_option`: `tensorboard,disk` (for displaying on tensorboard and for saving videos on disk, respectively)

Generated navigation episode recordings should look like this on tensorboard:
<p align="center">
  <img src="../res/img/tensorboard_video_demo.gif"  height="500">
</p>
