#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
namespace py = pybind11;

// Create a tensor that references this memory
//
at::Tensor convertToTensorColor(const py::capsule &ptr_capsule,
                                int dev_id,
                                uint32_t batch_size,
                                const array<uint32_t, 2> &resolution)
{
    uint8_t *dev_ptr(ptr_capsule);

    array<int64_t, 4> sizes {{batch_size, resolution[0], resolution[1], 4}};

    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

// Create a tensor that references this memory
//
at::Tensor convertToTensorDepth(const py::capsule &ptr_capsule,
                                int dev_id,
                                uint32_t batch_size,
                                const array<uint32_t, 2> &resolution)
{
    float *dev_ptr(ptr_capsule);

    array<int64_t, 3> sizes {{batch_size, resolution[0], resolution[1]}};

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

// TensorRT helpers
at::Tensor convertToTensorFCOut(const py::capsule &ptr_capsule,
                                int dev_id,
                                uint32_t batch_size,
                                uint32_t num_features)
{ 
    __half *dev_ptr(ptr_capsule);

    array<int64_t, 2> sizes {{batch_size, num_features}};

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::kCUDA, (short)dev_id);

    return torch::from_blob(dev_ptr, sizes, options);
}

py::capsule tensorToCapsule(const at::Tensor &tensor)
{
    return py::capsule(tensor.data_ptr());
}

class PyTorchSync {
public:
    PyTorchSync(const py::capsule &cap) : sema_(cudaExternalSemaphore_t(cap))
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
            cerr << "PyTorchSync: failed to wait on external semaphore"
                 << endl;
            abort();
        }
    }

private:
    cudaExternalSemaphore_t sema_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<PyTorchSync>(m, "PyTorchSync")
        .def(py::init<const py::capsule &>())
        .def("wait", &PyTorchSync::wait);
    m.def("make_color_tensor", &convertToTensorColor);
    m.def("make_depth_tensor", &convertToTensorDepth);
    m.def("make_fcout_tensor", &convertToTensorFCOut);
    m.def("tensor_to_capsule", &tensorToCapsule);
}
