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
                           bool depth, bool doubleBuffered) {
    RenderOptions options {};
    if (doubleBuffered) {
        options |= RenderOptions::DoubleBuffered;
    }

    auto make = [&](auto features) {
        return BatchRenderer(
            {gpu_id, // gpuID
             1,      // numLoaders
             1,      // numStreams
             renderer_batch_size, resolution[1], resolution[0],
             glm::mat4(1, 0, 0, 0, 0, -1.19209e-07, -1, 0, 0, 1, -1.19209e-07, 0, 0,
                       0, 0, 1) // Habitat coordinate txfm matrix
             ,
             }, features);
    };

    if (color && depth) {
        return make(RenderFeatures<Unlit<RenderOutputs::Color | RenderOutputs::Depth,
                                   DataSource::Texture>> { options });
    } else if (color) {
        return make(RenderFeatures<Unlit<RenderOutputs::Color,
                                   DataSource::Texture>> { options });
    } else {
        return make(RenderFeatures<Unlit<RenderOutputs::Depth,
                                   DataSource::None>> { options });
    }

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

struct BackgroundSceneLoader {
    explicit BackgroundSceneLoader(AssetLoader &&loader)
        : loader_{std::move(loader)}, loader_mutex_{}, loader_cv_{},
          loader_exit_{false}, loader_requests_{}, loader_thread_{[&]() {
              loaderLoop();
          }} {};

    ~BackgroundSceneLoader() {
        {
            lock_guard<mutex> cv_lock(loader_mutex_);
            loader_exit_ = true;
        }
        loader_cv_.notify_one();
        loader_thread_.join();
    }

    std::shared_ptr<Scene> loadScene(const std::string &scene_path) {
        auto fut = asyncLoadScene(scene_path);
        fut.wait();
        return fut.get();
    }

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

  private:
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

    AssetLoader loader_;

    mutex loader_mutex_;
    condition_variable loader_cv_;
    bool loader_exit_;
    queue<pair<string, promise<shared_ptr<Scene>>>> loader_requests_;
    thread loader_thread_;
};

struct V4RRenderGroup {
    V4RRenderGroup(CommandStream &strm, std::vector<Environment> &&envs,
                   uint32_t batch_size_per_scene, at::Tensor color,
                   at::Tensor depth)

        : cmd_strm_{strm}, envs_{std::move(envs)},
          batch_size_per_scene_{batch_size_per_scene}, color_batch_{color},
          depth_batch_{depth} {}

    PyTorchSync render(at::Tensor cameraPoses) {
        AT_ASSERT(static_cast<std::size_t>(cameraPoses.size(0)) == envs_.size(),
                  "Must have the same number of camera poses as "
                  "envs per group");
        AT_ASSERT(!cameraPoses.is_cuda(), "Must be on CPU");

        {
            auto posesAccessor = cameraPoses.accessor<float, 3>();
            for (uint32_t envIdx = 0; envIdx < envs_.size(); ++envIdx) {
                glm::mat4 pose;
                for (int j = 0; j < 4; ++j)
                    for (int i = 0; i < 4; ++i)
                        pose[i][j] = posesAccessor[envIdx][j][i];

                envs_[envIdx].setCameraView(pose);
            }
        }

        auto sync = cmd_strm_.render(envs_);

        return PyTorchSync(move(sync));
    }

    void swapScene(const uint32_t envIdx, std::shared_ptr<Scene> &nextScene) {
        envs_[envIdx] = cmd_strm_.makeEnvironment(nextScene, 90, 0.01, 1000);
    }

  private:
    CommandStream &cmd_strm_;
    vector<Environment> envs_;
    const uint32_t batch_size_per_scene_;

  public:
    at::Tensor color_batch_, depth_batch_;
};

class DoubleBuffered {
  public:
    DoubleBuffered(BatchRenderer &&renderer, uint32_t numGroups,
                   const std::vector<std::string> &scene_paths,
                   uint32_t batch_size_per_scene, int gpu_id,
                   const std::vector<uint32_t> &resolution, bool color,
                   bool depth)
        : renderer_{std::move(renderer)}, loader_{renderer_.makeLoader()},
          cmd_strm_{renderer_.makeCommandStream()},
          numScenes_{scene_paths.size()},
          batch_size_per_scene_{batch_size_per_scene},
          envsPerGroup_{numScenes_ * batch_size_per_scene_ / numGroups},
          renderGroups_{} {

        for (uint32_t i = 0, sceneIdx = 0; i < numGroups; ++i) {
            std::vector<Environment> envs;
            for (uint32_t __i = 0; __i < scene_paths.size() / numGroups;
                 ++__i, ++sceneIdx) {

                auto scene = loader_.loadScene(scene_paths[sceneIdx]);
                for (uint32_t __j = 0; __j < batch_size_per_scene; ++__j)
                    envs.emplace_back(
                        move(cmd_strm_.makeEnvironment(scene, 90, 0.01, 1000)));
            }

            at::Tensor color_batch, depth_batch;

            if (color) {
                color_batch =
                    convertToTensorColor(cmd_strm_.getColorDevPtr(i), gpu_id,
                                         envs.size(), resolution);
            }

            if (depth) {
                depth_batch =
                    convertToTensorDepth(cmd_strm_.getDepthDevPtr(i), gpu_id,
                                         envs.size(), resolution);
            }

            renderGroups_.emplace_back(new V4RRenderGroup(
                cmd_strm_, std::move(envs), batch_size_per_scene, color_batch,
                depth_batch));
        }
    }

    DoubleBuffered(const std::vector<std::string> &scene_paths,
                   uint32_t batch_size_per_scene, int gpu_id,
                   const std::vector<uint32_t> &resolution, bool color,
                   bool depth)
        : DoubleBuffered{
              makeRenderer(gpu_id,
                           batch_size_per_scene *
                               static_cast<uint32_t>(scene_paths.size()) / 2,
                           resolution, color, depth, /*doubleBuffered=*/true),
              2,
              scene_paths,
              batch_size_per_scene,
              gpu_id,
              resolution,
              color,
              depth} {};

    ~DoubleBuffered() = default;

    at::Tensor getColorTensor(const uint32_t groupIdx) const {
        return renderGroups_[groupIdx]->color_batch_;
    }

    at::Tensor getDepthTensor(const uint32_t groupIdx) const {
        return renderGroups_[groupIdx]->depth_batch_;
    }

    PyTorchSync render(at::Tensor cameraPoses, const uint32_t groupIdx) {
        return std::move(renderGroups_[groupIdx]->render(cameraPoses));
    }

    bool isLoadingNextScene() {
        return nextScene_ != nullptr || newSceneFuture_.valid();
    }

    bool nextSceneLoaded() {
        return newSceneFuture_.valid() && isReady(newSceneFuture_);
    }

    void loadNewScene(const std::string &scenePath) {
        AT_ASSERT(!isLoadingNextScene(), "Already loading the next scene!");
        newSceneFuture_ = loader_.asyncLoadScene(scenePath);
    }

    bool hasNextScene() { return nextScene_ != nullptr; }

    void acquireNextScene() {
        newSceneFuture_.wait();
        nextScene_ = newSceneFuture_.get();
    }

    void doneSwappingScene() { nextScene_ = nullptr; }

    void swapScene(const uint32_t envIdx, const uint32_t groupIdx) {
        AT_ASSERT(hasNextScene(), "Next scene is nullptr");

        renderGroups_[groupIdx]->swapScene(envIdx, nextScene_);
    }

  private:
    BatchRenderer renderer_;
    BackgroundSceneLoader loader_;
    CommandStream cmd_strm_;
    const uint32_t numScenes_, batch_size_per_scene_, envsPerGroup_;

    vector<std::unique_ptr<V4RRenderGroup>> renderGroups_;
    std::future<shared_ptr<Scene>> newSceneFuture_;
    std::shared_ptr<Scene> nextScene_ = nullptr;
};

class SingleBuffered : public DoubleBuffered {
  public:
    SingleBuffered(const std::vector<std::string> &scene_paths,
                   uint32_t batch_size_per_scene, int gpu_id,
                   const std::vector<uint32_t> &resolution, bool color,
                   bool depth)
        : DoubleBuffered{
              makeRenderer(gpu_id,
                           batch_size_per_scene *
                               static_cast<uint32_t>(scene_paths.size()),
                           resolution, color, depth, /*doubleBuffered=*/false),
              1,
              scene_paths,
              batch_size_per_scene,
              gpu_id,
              resolution,
              color,
              depth} {};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<PyTorchSync>(m, "PyTorchSync").def("wait", &PyTorchSync::wait);

    py::class_<DoubleBuffered>(m, "DoubleBuffered")
        .def(py::init<const std::vector<string> &, int, int,
                      const std::vector<uint32_t> &, bool, bool>())
        .def("render", &DoubleBuffered::render)
        .def("rgba", &DoubleBuffered::getColorTensor)
        .def("depth", &DoubleBuffered::getDepthTensor)
        .def("done_swapping_scenes", &DoubleBuffered::doneSwappingScene)
        .def("acquire_next_scene", &DoubleBuffered::acquireNextScene)
        .def("load_new_scene", &DoubleBuffered::loadNewScene)
        .def("swap_scene", &DoubleBuffered::swapScene)
        .def("next_scene_loaded", &DoubleBuffered::nextSceneLoaded)
        .def("has_next_scene", &DoubleBuffered::hasNextScene)
        .def("is_loading_next_scene", &DoubleBuffered::isLoadingNextScene);

    py::class_<SingleBuffered, DoubleBuffered>(m, "SingleBuffered")
        .def(py::init<const std::vector<string> &, int, int,
                      const std::vector<uint32_t> &, bool, bool>());
}
