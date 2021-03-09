Batch Processing Simulator: High-Performance Batch Simulation for 3D PointGoal Navigation
=================================================

This repository is the reference implementation for the PointGoal navigation batch simulator and training infrastructure described in _Large Batch Simulation for Deep Reinforcement Learning_.

The bps-nav simulator is not intended to be a full-featured 3D simulator for reinforcement learning; the batch simulator primarily supports training PointGoal navigation agents on the Gibson and Matterport3D datasets (support for the simple "flee" and "explore" tasks is also provided).

bps-nav depends on the [bps3D batch renderer](https://github.com/shacklettbp/bps3D). Unlike the tightly scoped simulator, bps3D is designed to be dataset and task agnostic, and can be used standalone in other projects for state-of-the-art 3D rendering performance.

Dependencies
------------

* CMake
* C++-17 Compiler
* NVIDIA GPU
* CUDA 10.1 or higher
* Python 3.8
* PyTorch 1.6 or 1.8
* NVIDIA driver with Vulkan 1.1 support (440.33 or later confirmed to work)
* Vulkan headers and loader (described below)
* (Optional) [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) for running evaluation

The easiest way to obtain the Vulkan dependencies is by installing the latest version of the official Vulkan SDK, available at <https://vulkan.lunarg.com/sdk/home>. Detailed instructions for installing the SDK are [here](https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html), specifically the "System Requirements" and "Set up the runtime environment" sections. Be sure that the Vulkan SDK is enabled in your current shell before building.

Building
--------

To use bps-nav, you must install 3 components: the C++ simulator, the simulator's pytorch interface and the bps-nav training code. These are all packaged as separate python modules, and can be installed as follows:

```bash
git clone --recursive https://github.com/shacklettbp/bps-nav.git
cd bps-nav
cd simulator/python/bps_sim
pip install -e . # Build simulator
cd ../bps_pytorch
pip install -e . # Build pytorch integration
cd ../../
pip install -e . # Install main bps_nav module
mkdir data # Create data directory for datasets, model checkpoints etc.
```

Preprocessing Gibson and Matterport3D Datasets
----------------------------------------------

The underlying bps3D renderer requires 3D scenes to be converted into a custom format before rendering. The following instructions describe specifically how to convert the Gibson and Matterport3D datasets for use with bps-nav, for more generic instructions, refer to the [bps3D repository](https://github.com/shacklettbp/bps3D).

### Prerequisites

First, download the Gibson and (optionally) Matterport3D datasets. Follow the instructions at <https://github.com/facebookresearch/habitat-lab#data> to set-up both the scene datasets and Pointgoal navigation task datasets in the `bps-nav/data` subdirectory that you created in the prior section (be sure to follow the same filesystem structure as specified by Habitat-Lab).

 Once completed, you should have the Gibson `.glb` and `.navmesh` files saved in `data/scene_datasets/gibson/` and the MP3D assets stored in `data/scene_datasets/mp3d`. Additionally, the Pointgoal navigation episode JSON files should be saved in `data/datasets/pointnav/gibson/v1`.


### Conversion

This repository includes a simple script to preprocess all the Gibson and Matterport3D `*.glb` files and produce `*.bps` files alongside them in the `data/scene_datasets` directory.

```bash
cd bps-nav
./tools/preprocess_datasets.sh 
```

The `preprocess_datasets.sh` script also extracts RGB textures for the Gibson dataset into `bps-nav/textures`, which must be compressed, as described in the following section.

### Texture Compression

In addition to scene preprocessing, bps3D also requires textures to be block compressed to reduce GPU memory usage. This requires the `toktx` tool from the [KTX-Software repository](https://github.com/KhronosGroup/KTX-Software), which can be built as follows:

```bash
git clone --recursive https://github.com/KhronosGroup/KTX-Software.git
cd KTX-Software
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make toktx
```

`toktx` can then be found in `KTX-Software/build/tools/toktx/toktx`, and can be used as follows to convert `src.jpg` into `dst.ktx2`:
```bash
toktx --uastc --uastc_rdo_q 0.9 --uastc_rdo_d 1024 --zcmp 20 --genmipmap dst.ktx2 src.jpg
```

All of the Gibson textures extracted to the `textures` directory in the prior section must be individually converted to KTX2 using the above command. Due to the large size of the textures, this process is extremely compute intensive, and should be distributed across multiple cores or machines if possible.

Compressing the Matterport3D textures requires an additional step of fetching the original source assets, as the Habitat versions of the Matterport3D assets use precompressed textures with a different (lower quality) compression scheme than required by bps3D. Use the `download-mp.py` script (available after signing the Matterport3D license agreement) with `--type matterport_mesh` to fetch the `jpg` texture files.

For both datasets, the compressed `*.ktx2` files should be copied into the same directory as their corresponding `*bps` file under `data/scene_datasets`.

Usage
-----
`train_rgb.sh` and `train_depth.sh` are example scripts that show how to train RGB camera and depth sensor agents, respectively. The parameters in these scripts can be changed to increase agent resolution or DNN capacity. Model checkpoints are stored in `data/checkpoints/`. The `eval_rgb.sh` and `eval_depth.sh` scripts perform evaluation on the saved checkpoints. Note that Habitat-Lab and Habitat-Sim must be installed before running evaluation.

Citation
--------

```
@article{shacklett21bps,
    title   = {Large Batch Simulation for Deep Reinforcement Learning},
    author  = {Brennan Shacklett and Erik Wijmans and Aleksei Petrenko and Manolis Savva and Dhruv Batra and Vladlen Koltun and Kayvon Fatahalian},
    journal = {International Conference On Learning Representations (ICLR)},
    year    = {2021}
}
```
