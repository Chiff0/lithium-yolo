#include "gpu_backend.hpp"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cfloat>
#include <cmath>




static void gemm(
                 const float* in, lithium::Tensor& out, const float* weights, 
                 int patches, int patch_len, int filters, cublasContext* handle
                )
{ 
    float alpha{1.0f};
    float beta{};
    auto status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, patches, filters, patch_len, &alpha, in, patch_len, weights, patch_len, &beta, out.data, patches);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "cublasSgemm: %s\n", cublasGetStatusString(status));
        std::abort();
    }

}



__global__ void im2col(const float* in, float* out, int out_len, int window,
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

        ::im2col<<<ceil_div(cols, threads), threads>>>(
            in.data, workspace, cols, window,
            layer.spec.stride, layer.spec.pad, patch_len,
            in.h, in.w, out.w);

        gemm(workspace, out, layer.weights.weights.data(),
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
        const cudaError_t status{cudaMallocManaged(&workspace, floats * sizeof(float))};
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "im2col workspace: %s\n", cudaGetErrorString(status));
            std::abort();
        }
        workspace_floats = floats;
    }

    GPUBackend::~GPUBackend()
    {
        if (workspace != nullptr)
        {
            cudaFree(workspace);
        }
        if (handle != nullptr)
        {
            cublasDestroy(handle);
        }
    }

    // Created on first conv rather than in a constructor: the harnesses construct a
    // GPUBackend even when running --cpu, and this way that costs nothing.
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
    }
}
