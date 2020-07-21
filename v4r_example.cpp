#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/gtc/type_ptr.hpp>
#include <v4r.hpp>

#include <array>
#include <fstream>
#include <vector>

using namespace std;
using namespace v4r;

namespace py = pybind11;

// Create a tensor that references this memory
//
at::Tensor convertToTensor(void *dev_ptr, int dev_id, uint32_t batch_size,
                           const std::vector<uint32_t> &resolution) {
    array<int64_t, 4> sizes{{batch_size, resolution[0], resolution[1], 4}};

    // This would need to be more precise for multi gpu machines
    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

class PyTorchSync {
  public:
    PyTorchSync(RenderSync &&sync) : sync_(move(sync)) {}

    void wait() {
        // Get the current CUDA stream from pytorch and force it to wait
        // on the renderer to finish
        cudaStream_t cuda_strm = at::cuda::getCurrentCUDAStream().stream();
        sync_.gpuWait(cuda_strm);
    }

  private:
    RenderSync sync_;
};

struct V4RRenderGroup {
    V4RRenderGroup(CommandStream &strm, std::shared_ptr<Scene> &scene,
                   uint32_t batch_size, int gpu_id,
                   const std::vector<uint32_t> &resolution)
        : cmd_strm_{strm}, envs{}, color_batch_{convertToTensor(
                                       cmd_strm_.getColorDevPtr(), gpu_id,
                                       batch_size, resolution)} {
        for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            envs.emplace_back(
                move(cmd_strm_.makeEnvironment(scene, 90, 0.01, 1000)));
        }
    }

    PyTorchSync render(const std::vector<at::Tensor> &cameraPoses) {
        AT_ASSERT(cameraPoses.size() == envs.size(),
                  "Must have the same number of camera poses as batch_size");
        {
            int i = 0;
            for (auto &env : envs) {
                AT_ASSERT(!cameraPoses[i].is_cuda(), "Must be on CPU");
                AT_ASSERT(cameraPoses[i].scalar_type() == at::ScalarType::Float,
                          "Must be float32");
                glm::mat4 pose = glm::transpose(
                    (glm::make_mat4(cameraPoses[i++].data_ptr<float>())));
                env.setCameraView(pose);
            }
        }

        auto sync = cmd_strm_.render(envs);

        return PyTorchSync(move(sync));
    }

  private:
    CommandStream &cmd_strm_;
    vector<Environment> envs;

  public:
    at::Tensor color_batch_;
};

class DoubleBuffered {
  public:
    DoubleBuffered(const std::vector<std::string> &scene_paths,
                   uint32_t batch_size, int gpu_id,
                   const std::vector<uint32_t> &resolution)
        : renderer_(
              {gpu_id, // gpuID
               1,      // numLoaders
               1,      // numStreams
               batch_size,
               resolution[1],
               resolution[0],
               glm::mat4(1, 0, 0, 0, 0, -1.19209e-07, -1, 0, 0, 1, -1.19209e-07,
                         0, 0, 0, 0, 1) // Habitat coordinate txfm matrix
               ,
               {RenderFeatures::MeshColor::Texture,
                RenderFeatures::Pipeline::Unlit, RenderFeatures::Outputs::Color,
                RenderFeatures::Options::CpuSynchronization |
                    RenderFeatures::Options::DoubleBuffered}}),
          loader_(renderer_.makeLoader()),
          cmd_strm_{renderer_.makeCommandStream()}, loaded_scenes_() {
        for (auto &scene_path : scene_paths) {
            loaded_scenes_.emplace_back(loader_.loadScene(scene_path));
        }

        for (int i = 0; i < 2; ++i) {
            renderGroups_.emplace_back(new V4RRenderGroup(
                cmd_strm_, loaded_scenes_[i], batch_size, gpu_id, resolution));
        }
    }

    ~DoubleBuffered() = default;

    at::Tensor getColorTensor(const uint32_t groupIdx) const {
        return renderGroups_[groupIdx]->color_batch_;
    }

    PyTorchSync render(const std::vector<at::Tensor> &cameraPoses,
                       const uint32_t groupIdx) {
        return std::move(renderGroups_[groupIdx]->render(cameraPoses));
    }

  private:
    BatchRenderer renderer_;
    AssetLoader loader_;
    CommandStream cmd_strm_;
    vector<shared_ptr<Scene>> loaded_scenes_;

    vector<std::unique_ptr<V4RRenderGroup>> renderGroups_;
};

class SingleBuffered {
  public:
    SingleBuffered(const std::vector<std::string> &scene_paths,
                   uint32_t batch_size, int gpu_id,
                   const std::vector<uint32_t> &resolution)
        : renderer_(
              {gpu_id, // gpuID
               1,      // numLoaders
               1,      // numStreams
               batch_size * scene_paths.size(),
               resolution[1],
               resolution[0],
               glm::mat4(1, 0, 0, 0, 0, -1.19209e-07, -1, 0, 0, 1, -1.19209e-07,
                         0, 0, 0, 0, 1) // Habitat coordinate txfm matrix
               ,
               {
                   RenderFeatures::MeshColor::Texture,
                   RenderFeatures::Pipeline::Unlit,
                   RenderFeatures::Outputs::Color,
                   {},
               }}),
          loader_(renderer_.makeLoader()),
          cmd_strm_{renderer_.makeCommandStream()},
          loaded_scenes_(), envs{}, color_batch_{convertToTensor(
                                        cmd_strm_.getColorDevPtr(), gpu_id,
                                        batch_size * scene_paths.size(),
                                        resolution)},
          batch_size_{batch_size} {
        for (auto &scene_path : scene_paths) {
            loaded_scenes_.emplace_back(loader_.loadScene(scene_path));
        }

        for (uint32_t sceneIdx = 0; sceneIdx < loaded_scenes_.size();
             ++sceneIdx) {
            for (uint32_t i = 0; i < batch_size_; ++i) {
                envs.emplace_back(move(cmd_strm_.makeEnvironment(
                    loaded_scenes_[sceneIdx], 90, 0.01, 1000)));
            }
        }
    }

    ~SingleBuffered() = default;

    void loadNewScene(const std::string &scenePath) {
        loaded_scenes_.emplace_back(loader_.loadScene(scenePath));
    }

    void swapScene(const uint32_t dropIdx) {
        for (uint32_t i = 0; i < batch_size_; ++i) {
            envs[i + dropIdx * batch_size_] = move(cmd_strm_.makeEnvironment(
                loaded_scenes_[loaded_scenes_.size() - 1], 90, 0.01, 1000));
        }

        std::swap(loaded_scenes_[dropIdx],
                  loaded_scenes_[loaded_scenes_.size() - 1]);
        loaded_scenes_.pop_back();
    }

    void swapScenes(const std::vector<std::string> &scene_paths) {
        envs.clear();
        loaded_scenes_.clear();

        for (auto &scene_path : scene_paths) {
            loaded_scenes_.emplace_back(loader_.loadScene(scene_path));
        }

        for (uint32_t sceneIdx = 0; sceneIdx < scene_paths.size(); ++sceneIdx) {
            for (uint32_t i = 0; i < batch_size_; ++i) {
                envs.emplace_back(move(cmd_strm_.makeEnvironment(
                    loaded_scenes_[sceneIdx], 90, 0.01, 1000)));
            }
        }
    }

    at::Tensor getColorTensor() { return color_batch_; }

    PyTorchSync render(const std::vector<at::Tensor> &cameraPoses) {
        AT_ASSERT(cameraPoses.size() == envs.size(),
                  "Must have the same number of camera poses as batch_size");
        {
            int i = 0;
            for (auto &env : envs) {
                AT_ASSERT(!cameraPoses[i].is_cuda(), "Must be on CPU");
                AT_ASSERT(cameraPoses[i].scalar_type() == at::ScalarType::Float,
                          "Must be float32");
                glm::mat4 pose = glm::transpose(
                    (glm::make_mat4(cameraPoses[i++].data_ptr<float>())));
                env.setCameraView(pose);
            }
        }

        auto sync = cmd_strm_.render(envs);

        return PyTorchSync(move(sync));
    }

  private:
    BatchRenderer renderer_;
    AssetLoader loader_;
    CommandStream cmd_strm_;
    vector<shared_ptr<Scene>> loaded_scenes_;
    vector<Environment> envs;

    at::Tensor color_batch_;
    uint32_t batch_size_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<PyTorchSync>(m, "PyTorchSync").def("wait", &PyTorchSync::wait);

    py::class_<DoubleBuffered>(m, "DoubleBuffered")
        .def(py::init<const std::vector<string> &, int, int,
                      const std::vector<uint32_t> &>())
        .def("render", &DoubleBuffered::render)
        .def("rgba", &DoubleBuffered::getColorTensor);

    py::class_<SingleBuffered>(m, "SingleBuffered")
        .def(py::init<const std::vector<string> &, int, int,
                      const std::vector<uint32_t> &>())
        .def("render", &SingleBuffered::render)
        .def("rgba", &SingleBuffered::getColorTensor)
        .def("swap_scenes", &SingleBuffered::swapScenes)
        .def("load_new_scene", &SingleBuffered::loadNewScene)
        .def("swap_scene", &SingleBuffered::swapScene);
}
