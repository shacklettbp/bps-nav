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
    V4RRenderGroup(Unlit::CommandStream &&strm, std::shared_ptr<Scene> &scene,
                   uint32_t batch_size, int gpu_id)
        : cmd_strm_{std::move(strm)}, envs{}, color_batch_{convertToTensor(
                                                  cmd_strm_.getColorDevPtr(),
                                                  gpu_id, batch_size)} {
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
    Unlit::CommandStream cmd_strm_;
    vector<Environment> envs;

    at::Tensor convertToTensor(void *dev_ptr, int dev_id, uint32_t batch_size) {
        std::cout << batch_size << ", " << dev_id << std::endl;
        std::cout << dev_ptr << std::endl;
        array<int64_t, 4> sizes{{batch_size, 256, 256, 4}};

        // This would need to be more precise for multi gpu machines
        auto options = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .device(torch::kCUDA, (short)dev_id);

        std::cout << "Before from_blob" << std::endl;
        auto t = torch::from_blob(dev_ptr, sizes, options);
        std::cout << "End from_blob" << std::endl;

        return t;
    }

  public:
    at::Tensor color_batch_;
};

class V4RExample {
  public:
    V4RExample(const std::vector<std::string> &scene_paths, uint32_t batch_size,
               int gpu_id)
        : renderer_({
              gpu_id, // gpuID
              1,      // numLoaders
              2,      // numStreams
              batch_size,
              256, // imgWidth,
              256, // imgHeight
              glm::mat4(1, 0, 0, 0, 0, -1.19209e-07, -1, 0, 0, 1, -1.19209e-07,
                        0, 0, 0, 0, 1) // Habitat coordinate txfm matrix
          }),
          loader_(renderer_.makeLoader()), loaded_scenes_() {
        for (auto &scene_path : scene_paths) {
            loaded_scenes_.emplace_back(loader_.loadScene(scene_path));
        }

        for (int i = 0; i < 2; ++i) {
            renderGroups_.emplace_back(
                new V4RRenderGroup(std::move(renderer_.makeCommandStream()),
                                   loaded_scenes_[i], batch_size, gpu_id));
        }
    }

    ~V4RExample() = default;

    at::Tensor getColorTensor(const uint32_t groupIdx) const {
        return renderGroups_[groupIdx]->color_batch_;
    }

    PyTorchSync render(const std::vector<at::Tensor> &cameraPoses,
                       const uint32_t groupIdx) {
        return std::move(renderGroups_[groupIdx]->render(cameraPoses));
    }

  private:
    Unlit::BatchRenderer renderer_;
    Unlit::AssetLoader loader_;
    vector<shared_ptr<Scene>> loaded_scenes_;

    vector<std::unique_ptr<V4RRenderGroup>> renderGroups_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<V4RExample>(m, "V4RExample")
        .def(py::init<const std::vector<string> &, int, int>())
        .def("render", &V4RExample::render)
        .def("rgba", &V4RExample::getColorTensor);

    py::class_<PyTorchSync>(m, "PyTorchSync").def("wait", &PyTorchSync::wait);
}
