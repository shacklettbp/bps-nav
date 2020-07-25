#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/gtc/type_ptr.hpp>
#include <v4r.hpp>

#include <array>
#include <condition_variable>
#include <fstream>
#include <future>
#include <queue>
#include <utility>
#include <vector>

using namespace std;
using namespace v4r;

namespace py = pybind11;

BatchRenderer makeRenderer(int32_t gpu_id, uint32_t renderer_batch_size,
                           const std::vector<uint32_t> &resolution, bool color,
                           bool depth) {
    RenderFeatures feats;
    if (color) {
        feats.colorSrc = RenderFeatures::MeshColor::Texture;
    } else {
        feats.colorSrc = RenderFeatures::MeshColor::None;
    }

    feats.pipeline = RenderFeatures::Pipeline::Unlit;

    if (color) {
        feats.outputs |= RenderFeatures::Outputs::Color;
    }

    if (depth) {
        feats.outputs |= RenderFeatures::Outputs::Depth;
    }

    return BatchRenderer(
        {gpu_id, // gpuID
         1,      // numLoaders
         1,      // numStreams
         renderer_batch_size, resolution[1], resolution[0],
         glm::mat4(1, 0, 0, 0, 0, -1.19209e-07, -1, 0, 0, 1, -1.19209e-07, 0, 0,
                   0, 0, 1) // Habitat coordinate txfm matrix
         ,
         feats});
}

// Create a tensor that references this memory
//
at::Tensor convertToTensorColor(void *dev_ptr, int dev_id, uint32_t batch_size,
                                const std::vector<uint32_t> &resolution) {
    array<int64_t, 4> sizes{{batch_size, resolution[0], resolution[1], 4}};

    // This would need to be more precise for multi gpu machines
    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

// Create a tensor that references this memory
//
at::Tensor convertToTensorDepth(void *dev_ptr, int dev_id, uint32_t batch_size,
                                const std::vector<uint32_t> &resolution) {
    array<int64_t, 3> sizes{{batch_size, resolution[0], resolution[1]}};

    // This would need to be more precise for multi gpu machines
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

template <typename R> bool isReady(const std::future<R> &f) {
    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
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
        : cmd_strm_{strm}, envs{}, color_batch_{convertToTensorColor(
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
                         0, 0, 0, 0,
                         1) // Habitat coordinate txfm matrix
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
                   const std::vector<uint32_t> &resolution, bool color,
                   bool depth)
        : renderer_(makeRenderer(
              gpu_id, batch_size * static_cast<uint32_t>(scene_paths.size()),
              resolution, color, depth)),
          loader_(renderer_.makeLoader()),
          cmd_strm_{renderer_.makeCommandStream()},
          loaded_scenes_(), envs{}, batch_size_{batch_size}, loader_mutex_(),
          loader_cv_(), loader_exit_(false), loader_requests_(),
          loader_thread_([&]() { loaderLoop(); }) {
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

        if (color) {
            color_batch_ = convertToTensorColor(
                cmd_strm_.getColorDevPtr(), gpu_id,
                batch_size * scene_paths.size(), resolution);
        }

        if (depth) {
            depth_batch_ = convertToTensorDepth(
                cmd_strm_.getDepthDevPtr(), gpu_id,
                batch_size * scene_paths.size(), resolution);
        }
    }

    ~SingleBuffered() {
        {
            lock_guard<mutex> cv_lock(loader_mutex_);
            loader_exit_ = true;
        }
        loader_cv_.notify_one();
        loader_thread_.join();
    }

    bool isLoadingNextScene() { return newSceneFuture_.valid(); }

    bool nextSceneReady() {
        return isLoadingNextScene() && isReady(newSceneFuture_);
    }

    void loadNewScene(const std::string &scenePath) {
        AT_ASSERT(!isLoadingNextScene(), "Already loading the next scene!");
        newSceneFuture_ = asyncLoadScene(scenePath);
    }

    void swapScene(const uint32_t dropIdx) {
        newSceneFuture_.wait();
        auto nextScene = newSceneFuture_.get();
        for (uint32_t i = 0; i < batch_size_; ++i) {
            envs[i + dropIdx * batch_size_] =
                move(cmd_strm_.makeEnvironment(nextScene, 90, 0.01, 1000));
        }

        loaded_scenes_[dropIdx] = nextScene;
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
    at::Tensor getDepthTensor() { return depth_batch_; }

    PyTorchSync render(at::Tensor cameraPoses) {
        AT_ASSERT(static_cast<std::size_t>(cameraPoses.size(0)) == envs.size(),
                  "Must have the same number of camera poses as batch_size");
        AT_ASSERT(!cameraPoses.is_cuda(), "Must be on CPU");

        {
            auto posesAccessor = cameraPoses.accessor<float, 3>();
            for (uint32_t envIdx = 0; envIdx < envs.size(); ++envIdx) {
                glm::mat4 pose;
                for (int j = 0; j < 4; ++j)
                    for (int i = 0; i < 4; ++i)
                        pose[i][j] = posesAccessor[envIdx][j][i];

                envs[envIdx].setCameraView(pose);
            }
        }

        auto sync = cmd_strm_.render(envs);

        return PyTorchSync(move(sync));
    }

  private:
    future<shared_ptr<Scene>> asyncLoadScene(const string &scene_path) {
        future<shared_ptr<Scene>> loader_future;

        {
            lock_guard<mutex> wait_lock(loader_mutex_);

            promise<shared_ptr<Scene>> loader_promise;
            loader_future = loader_promise.get_future();

            loader_requests_.emplace(scene_path, move(loader_promise));
        }
        loader_cv_.notify_one();

        return loader_future;
    }

    void loaderLoop() {
        while (true) {
            string scene_path;
            promise<shared_ptr<Scene>> loader_promise;
            {
                unique_lock<mutex> wait_lock(loader_mutex_);
                while (loader_requests_.size() == 0) {
                    loader_cv_.wait(wait_lock);
                    if (loader_exit_) {
                        return;
                    }
                }

                scene_path = move(loader_requests_.front().first);
                loader_promise = move(loader_requests_.front().second);
                loader_requests_.pop();
            }

            auto scene = loader_.loadScene(scene_path);
            loader_promise.set_value(move(scene));
        }
    };

    BatchRenderer renderer_;
    AssetLoader loader_;
    CommandStream cmd_strm_;
    vector<shared_ptr<Scene>> loaded_scenes_;
    vector<Environment> envs;

    at::Tensor color_batch_;
    at::Tensor depth_batch_;
    uint32_t batch_size_;

    mutex loader_mutex_;
    condition_variable loader_cv_;
    bool loader_exit_;
    queue<pair<string, promise<shared_ptr<Scene>>>> loader_requests_;
    thread loader_thread_;

    std::future<shared_ptr<Scene>> newSceneFuture_;
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
                      const std::vector<uint32_t> &, bool, bool>())
        .def("render", &SingleBuffered::render)
        .def("rgba", &SingleBuffered::getColorTensor)
        .def("depth", &SingleBuffered::getDepthTensor)
        .def("swap_scenes", &SingleBuffered::swapScenes)
        .def("load_new_scene", &SingleBuffered::loadNewScene)
        .def("swap_scene", &SingleBuffered::swapScene)
        .def("next_scene_ready", &SingleBuffered::nextSceneReady)
        .def("is_loading_next_scene", &SingleBuffered::isLoadingNextScene);
}
