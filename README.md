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
* PyTorch 1.6 (PyTorch 1.7 contains [a bug](https://github.com/pytorch/pytorch/issues/47138) in the JIT compiler than prevents the training code from working)
* NVIDIA driver with Vulkan 1.1 support (440.33 or later confirmed to work)
* Vulkan headers and loader (described below)

The easiest way to obtain the Vulkan dependencies is by installing the latest version of the official Vulkan SDK, available at <https://vulkan.lunarg.com/sdk/home>. Detailed instructions for installing the SDK are [here](https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html), specifically the "System Requirements" and "Set up the runtime environment" sections. Be sure that the Vulkan SDK is enabled in your current shell before building.

Building
--------

bps-nav 


Preprocessing Gibson and Matterport3D Datasets
----------------------------------------------

The underlying bps3D renderer requires 3D scenes to be converted into a custom format before rendering. The following instructions describe specifically how to convert the Gibson and Matterport3D datasets for use with bps-nav, for more generic instructions, refer to the [bps3D repository](https://github.com/shacklettbp/bps3D).

### Prerequisites


### Conversion


### Texture Compression

In addition to scene preprocessing, bps3D also requires textures to be block compressed to reduce GPU memory usage. 

Texture Compression
-------------------

bps3D relies on texture compression to reduce memory footprint. All textures must be stored in the [KTX2](https://github.khronos.org/KTX-Specification/) container, with UASTC compression. Use the `toktx` tool from the [KTX-Software repository](https://github.com/KhronosGroup/KTX-Software), which has built-in support for common source image formats.

The following is an example command that compresses a JPEG source texture, with rate distortion optimization to reduce on-disk file size:
```bash
toktx --uastc --uastc_rdo_q 0.9 --uastc_rdo_d 1024 --zcmp 20 --genmipmap dst.ktx2 src.jpg
```

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
