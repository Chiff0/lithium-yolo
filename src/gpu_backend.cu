#include "gpu_backend.hpp"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cfloat>



namespace lithium 
{
    void GPUBackend::conv(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {

    }

    void GPUBackend::maxpool(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {

    }

    void GPUBackend::upsample(const Tensor& in, Tensor& out, int stride)
    {

    }

    void GPUBackend::concat(const std::vector<Tensor>& ins, Tensor& out)
    {
    }

    void GPUBackend::download(const Tensor& device, float* host)
    {
    }

    void GPUBackend::yolo(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {

    }
}
