#include "gpu_backend.hpp"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cfloat>
#include <cmath>


#ifdef LITHIUM_FP16
using work_t = __half;
#else
using work_t = float;
#endif


#ifdef LITHIUM_FP16
static void* upload_as_half(const float* source, std::size_t count)
{
    std::vector<__half> staging(count);
    for (std::size_t i{0}; i < count; ++i)
    {
        staging[i] = __float2half(source[i]);
    }

    void* device{nullptr};
    const cudaError_t allocated{cudaMalloc(&device, count * sizeof(__half))};
    if (allocated != cudaSuccess)
    {
        std::fprintf(stderr, "half weights alloc (%zu elements): %s\n",
                     count, cudaGetErrorString(allocated));
        std::abort();
    }

    const cudaError_t copied{cudaMemcpy(device, staging.data(),
                                        count * sizeof(__half),
                                        cudaMemcpyHostToDevice)};
    if (copied != cudaSuccess)
    {
        std::fprintf(stderr, "half weights upload: %s\n", cudaGetErrorString(copied));
        std::abort();
    }
    return device;
}
#endif




static void gemm(
                 const work_t* in, lithium::Tensor& out, const work_t* weights, 
                 int patches, int patch_len, int filters, cublasContext* handle
                )
{ 
    float alpha{1.0f};
    float beta{};


#ifdef LITHIUM_FP16
    auto status = cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, patches, filters, patch_len, &alpha, in, CUDA_R_16F, patch_len, weights, CUDA_R_16F, patch_len, &beta, out.data, CUDA_R_32F, patches);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "cublasSgemmEx: %s\n", cublasGetStatusString(status));
        std::abort();
    }
    
#else
    auto status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, patches, filters, patch_len, &alpha, in, patch_len, weights, patch_len, &beta, out.data, patches);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "cublasSgemm: %s\n", cublasGetStatusString(status));
        std::abort();
    }
#endif
}



__global__ void im2col(const float* in, work_t* out,
                       int out_len, int window,
                       int stride, int pad, int patch_len, int hi, int wi, int wo)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= out_len) { return; }

    int out_idx{idx / patch_len}; // not the im2col out kernel but the out kernel we would use for naive convolution

    int channel{(idx % patch_len) / (window * window)};

    int pos{idx % (window * window)};
    int ox{out_idx % wo}, oy{out_idx / wo};

    int ix{pos % window + ox * stride - pad}, iy{pos / window + oy * stride - pad};

    out[idx] = (ix < 0 || iy < 0 || ix >= wi || iy >= hi)
        ? 0.0f
        : in[(channel * hi + iy) * wi + ix];

    return;
}

__global__ void upsample(float* in, float* out, int n, int co, int ho, int wo, 
                         int ci, int hi, int wi, int stride)
{
    int idx_out = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx_out >= n) { return; }
    
    int y = ((idx_out / wo) % ho) / stride;
    int x = (idx_out % wo) / stride;
    int c = idx_out / (ho * wo);

    int idx_in = wi * (c * hi + y) + x;

    out[idx_out] = in[idx_in];
    
    return;
}

__global__ void maxpool(float* in, float* out, int window, int stride, int len_out, int pad,
                        int co, int ho, int wo, int ci, int hi, int wi)
{
    int idx_out = threadIdx.x + blockDim.x * blockIdx.x; 

    if (idx_out >= len_out) { return; }
    float max{-FLT_MAX};

    int y = ((idx_out / wo) % ho) * stride - pad / 2;
    int x = (idx_out % wo) * stride - pad / 2;
    int c = idx_out / (ho * wo);
    float curr{};
    int idx_in{};

    for (int i{0}; i < window; ++i)
    {
        for (int j{0}; j < window; ++j)
        {   
            idx_in = wi * (c * hi + y + j) + x + i; 
            curr   = (x + i < 0 || y + j < 0 || x + i >= wi || y + j >= hi) 
                     ? -FLT_MAX
                     : in[idx_in];
            
            if (max < curr)
            {
                max = curr;
            }
        }
    }
    out[idx_out] = max;
    return;
}


__global__ void apply_sigmoid(float* arr, int n, int slice_dim, int slices)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x; 
    if (idx >= n) { return; }

    int anchor_offset{slice_dim * slices};
    int slice{(idx % anchor_offset) / slice_dim};

    if (slice != 2 && slice != 3)
    {
        arr[idx] = 1.0f / (1.0f + exp(-arr[idx]));
    }

    return;

}



static void concat(const std::vector<lithium::Tensor>& ins, lithium::Tensor& out)
{
    int offset{0};
    for (auto& tensor : ins)
    {
        int len{static_cast<int>(tensor.count())};
        cudaMemcpyAsync(out.data + offset, tensor.data, sizeof(float) * len, cudaMemcpyDefault);
        offset += len;
    }
    return;
}






__global__ void batchnorm(float* out, int n, int hw, 
                          const float* scale, const float* bias, 
                          const float* mean /* >:( */, const float* variance)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= n) { return; }

    int i = idx / hw;

    out[idx] = (out[idx] - mean[i]) / sqrtf(variance[i] + 1e-5f);
    out[idx] *= scale[i];
    out[idx] += bias[i];

    return;
}


__global__ void add_bias(float* out, int n, int hw, const float* bias)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= n) { return; }

    int i = idx / hw;

    out[idx] += bias[i];
}



__global__ void apply_leaky(float* out, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= n) { return; }

    out[idx] = (out[idx] > 0) ? out[idx] : out[idx] * 0.1f;
}



static int ceil_div(int len, int threads)
{
    return (len + threads - 1) / threads;
}


namespace lithium 
{
    void GPUBackend::conv(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        check_handle();
        
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};
        const int window{layer.spec.size};
        const int patch_len{in.c * window * window};
        const int cols{out.h * out.w * patch_len};
        reserve_workspace(static_cast<std::size_t>(cols));


        work_t* work{static_cast<work_t*>(workspace)};

        ::im2col<<<ceil_div(cols, threads), threads>>>(
            in.data, work, cols, window,
            layer.spec.stride, layer.spec.pad, patch_len,
            in.h, in.w, out.w);

        gemm(work, out, static_cast<const work_t*>(weights_for(layer)),
             out.h * out.w, patch_len, out.c, handle);
        
        if (layer.spec.batch_norm)
        {
            batchnorm<<<blocks, threads>>>(out.data, len, out.h * out.w,
                layer.weights.scales.data(), layer.weights.biases.data(),
                layer.weights.rolling_mean.data(), layer.weights.rolling_variance.data());
            }
            else
            {
                add_bias<<<blocks, threads>>>(out.data, len, out.h * out.w, layer.weights.biases.data());
            }
            switch(layer.spec.activation)
            {
                case lithium::Activation::Leaky:
                apply_leaky<<<blocks, threads>>>(out.data, len);
                break;
                case lithium::Activation::Linear:
                break;
        }
    }
    
    void GPUBackend::maxpool(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};

        ::maxpool<<<blocks, threads>>>(in.data, out.data, layer.spec.size, layer.spec.stride, len,
                                       layer.spec.pad,
                                       out.c, out.h, out.w, in.c, in.h, in.w);
    }
        
    void GPUBackend::upsample(const Tensor& in, Tensor& out, int stride)
    {
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};
        ::upsample<<<blocks, threads>>>(in.data, out.data, len, 
            out.c, out.h, out.w, 
            in.c, in.h, in.w, 
            stride);
            
        }
        
    void GPUBackend::concat(const std::vector<Tensor>& ins, Tensor& out)
    {
        ::concat(ins, out);
    }
    
    void GPUBackend::download(const Tensor& device, float* host)
    {
        std::copy(device.data, device.data + device.count(), host);
    }
            
    void GPUBackend::yolo(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        cudaMemcpyAsync(out.data, in.data, sizeof(float) * static_cast<int>(in.count()), cudaMemcpyDefault);
        int hw{out.h * out.w};
        
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};
        int entries{5 + layer.spec.yolo_classes};
        apply_sigmoid<<<blocks, threads>>>(out.data, len, hw, entries);
        
    }

    void GPUBackend::sync()
    {
        const cudaError_t status{cudaDeviceSynchronize()};
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(status));
        }
    }

    const void* GPUBackend::weights_for(const NetworkLayer& layer)
    {
#ifndef LITHIUM_FP16
        return layer.weights.weights.data();
#else

        const float* key{layer.weights.weights.data()};
        const std::size_t count{layer.weights.weights.size()};

        for (const HalfWeights& entry : half_weights)
        {
            if (entry.key != key)
            {
                continue;
            }
            if (entry.count != count)
            {
                std::fprintf(stderr,
                             "half weight cache: %p was %zu elements, now %zu\n",
                             static_cast<const void*>(key), entry.count, count);
                std::abort();
            }
            return entry.data;
        }
        sync();
        void* converted{upload_as_half(key, count)};
        half_weights.push_back({key, converted, count});
        return converted;
#endif
    }

    void GPUBackend::reserve_workspace(std::size_t floats)
    {
        if (floats <= workspace_floats)
        {
            return;
        }
        if (workspace != nullptr)
        {
            cudaFree(workspace);
        }
        
        const cudaError_t status{cudaMallocManaged(&workspace, floats * sizeof(work_t))};
        
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "im2col workspace: %s\n", cudaGetErrorString(status));
            std::abort();
        }
        workspace_floats = floats;
    }

    GPUBackend::~GPUBackend()
    {
        for (const HalfWeights& entry : half_weights)
        {
            if (entry.data != nullptr)
            {
                cudaFree(entry.data);
            }
        }
        half_weights.clear();

        if (workspace != nullptr)
        {
            cudaFree(workspace);
        }
        if (handle != nullptr)
        {
            cublasDestroy(handle);
        }
    }


    void GPUBackend::check_handle()
    {
        if (handle != nullptr)
        {
            return;
        }
        const cublasStatus_t status{cublasCreate(&handle)};
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "cublasCreate: %s\n", cublasGetStatusString(status));
            std::abort();
        }

#ifdef LITHIUM_TF32
        const cublasStatus_t math{cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)};
        if (math != CUBLAS_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "cublasSetMathMode: %s\n", cublasGetStatusString(math));
            std::abort();
        }
        std::fprintf(stderr, "cublas math: TF32 (10-bit mantissa)\n");
#elif defined(LITHIUM_FP16)
        std::fprintf(stderr, "cublas math: FP16 storage, FP32 accumulate "
                             "(10-bit mantissa, 5-bit exponent)\n");
#else
        std::fprintf(stderr, "cublas math: FP32\n");
#endif
    }
}
