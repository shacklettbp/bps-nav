#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
namespace py = pybind11;

// Create a tensor that references this memory
//
at::Tensor convertToTensorColor(void *dev_ptr, int dev_id, uint32_t batch_size,
                                const array<uint32_t, 2> &resolution)
{
    array<int64_t, 4> sizes{{batch_size, resolution[0], resolution[1], 4}};

    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

// Create a tensor that references this memory
//
at::Tensor convertToTensorDepth(void *dev_ptr, int dev_id, uint32_t batch_size,
                                const array<uint32_t, 2> &resolution)
{
    array<int64_t, 3> sizes{{batch_size, resolution[0], resolution[1]}};

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

class PyTorchSync {
public:
    PyTorchSync(cudaExternalSemaphore_t sema)
        : sema_(sema)
    {}

    void wait() 
    {
        // Get the current CUDA stream from pytorch and force it to wait
        // on the renderer to finish
        cudaStream_t cuda_strm = at::cuda::getCurrentCUDAStream().stream();
        cudaExternalSemaphoreWaitParams params {};
        cudaError_t res =
            cudaWaitExternalSemaphoresAsync(&sema_, &params, 1, cuda_strm);
        if (res != cudaSuccess) {
            cerr << "PyTorchSync: failed to wait on external semaphore" << endl;
            abort();
        }
    }

private:
    cudaExternalSemaphore_t sema_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<PyTorchSync>(m, "PyTorchSync").def("wait", &PyTorchSync::wait);
    m.def("make_color_tensor", &convertToTensorColor);
    m.def("make_depth_tensor", &convertToTensorDepth);
}
