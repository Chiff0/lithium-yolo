#include "gpu_backend.hpp"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cfloat>
#include <cmath>


using act_t = lithium::act_t;

#ifdef LITHIUM_HALF
static constexpr cudaDataType act_type{CUDA_R_16F};

__device__ inline float ld(act_t v) { return __half2float(v); }
__device__ inline act_t st(float v) { return __float2half(v); }
#else
static constexpr cudaDataType act_type{CUDA_R_32F};

__device__ inline float ld(act_t v) { return v; }
__device__ inline act_t st(float v) { return v; }
#endif


#ifdef LITHIUM_HALF
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
                 const act_t* in, lithium::Tensor& out, const act_t* weights, 
                 int patches, int patch_len, int filters, cublasContext* handle
                )
{ 
    float alpha{1.0f};
    float beta{};

    auto status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, filters, patches, patch_len, &alpha, weights, act_type, patch_len, in, act_type, patch_len, &beta, out.data, act_type, filters, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "cublasGemmEx: %s\n", cublasGetStatusString(status));
        std::abort();
    }
}



__global__ void im2col(const act_t* in, act_t* out,
                       int out_len, int window,
                       int stride, int pad, int patch_len, int ci, int hi, int wi, int wo)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= out_len) { return; }

    int out_idx{idx / patch_len}; // not the im2col out kernel but the out kernel we would use for naive convolution

    int channel{(idx % patch_len) % ci};

    int pos{(idx % patch_len) / ci};
    int ox{out_idx % wo}, oy{out_idx / wo};

    int ix{pos % window + ox * stride - pad}, iy{pos / window + oy * stride - pad};

    out[idx] = (ix < 0 || iy < 0 || ix >= wi || iy >= hi)
        ? st(0.0f)
        : in[(iy * wi + ix) * ci + channel];

    return;
}

__global__ void upsample(const act_t* in, act_t* out, int n, int co, int ho, int wo, 
                         int ci, int hi, int wi, int stride)
{
    int idx_out = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx_out >= n) { return; }
    
    int c = idx_out % co;
    int x = ((idx_out / co) % wo) / stride;
    int y = (idx_out / (co * wo)) / stride;

    int idx_in = (y * wi + x) * ci + c;

    out[idx_out] = in[idx_in];
    
    return;
}

__global__ void maxpool(const act_t* in, act_t* out, int window, int stride, int len_out, int pad,
                        int co, int ho, int wo, int ci, int hi, int wi)
{
    int idx_out = threadIdx.x + blockDim.x * blockIdx.x; 

    if (idx_out >= len_out) { return; }
    float max{-FLT_MAX};

    int c = idx_out % co;
    int x = ((idx_out / co) % wo) * stride - pad / 2;
    int y = (idx_out / (co * wo)) * stride - pad / 2;
    float curr{};
    int idx_in{};

    for (int i{0}; i < window; ++i)
    {
        for (int j{0}; j < window; ++j)
        {   
            idx_in = ((y + j) * wi + x + i) * ci + c;
            curr   = (x + i < 0 || y + j < 0 || x + i >= wi || y + j >= hi) 
                     ? -FLT_MAX
                     : ld(in[idx_in]);
            
            if (max < curr)
            {
                max = curr;
            }
        }
    }
    out[idx_out] = st(max);
    return;
}


__global__ void apply_sigmoid(act_t* arr, int n, int channels, int slices, int used)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x; 
    if (idx >= n) { return; }
    
    int channel{idx % channels};
    int slice{channel % slices};
    
    if (channel < used && slice != 2 && slice != 3)
    {
        arr[idx] = st(1.0f / (1.0f + expf(-ld(arr[idx]))));
    }
    
    return;
    
}



__global__ void concat_slice(const act_t* in, act_t* out, int n, int cin, int cout, int offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) { return; }
    
    out[(idx / cin) * cout + offset + idx % cin] = in[idx];
}






__global__ void batchnorm(act_t* out, int n, int channels, 
                          const float* scale, const float* bias, 
                          const float* mean /* >:( */, const float* variance)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= n) { return; }
    
    int i = idx % channels;
    
    out[idx] = st((ld(out[idx]) - mean[i]) / sqrtf(variance[i] + 1e-5f) * scale[i] + bias[i]);
    
    return;
}


__global__ void bias_activate(act_t* out, int n, int channels, const float* bias, bool leaky)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= n) { return; }
    
    float v{ld(out[idx]) + bias[idx % channels]};
    if (leaky && v < 0.0f)
    {
        v *= 0.1f;
    }
    out[idx] = st(v);
}

__global__ void bias_activate_maxpool(const act_t* in, act_t* out, int window, int stride, int len_out, int pad,
                                      int co, int ho, int wo, int ci, int hi, int wi, const float* bias, bool leaky)
{
    int idx_out = threadIdx.x + blockDim.x * blockIdx.x;
    
    
    if (idx_out >= len_out) { return; }
    

    // maxpool
    float max{-FLT_MAX};
    
    int c = idx_out % co;
    int x = ((idx_out / co) % wo) * stride - pad / 2;
    int y = (idx_out / (co * wo)) * stride - pad / 2;
    float curr{};
    int idx_in{};
    
    for (int i{0}; i < window; ++i)
    {
        for (int j{0}; j < window; ++j)
        {   
            idx_in = ((y + j) * wi + x + i) * ci + c;
            curr   = (x + i < 0 || y + j < 0 || x + i >= wi || y + j >= hi) 
                    ? -FLT_MAX
                    : ld(in[idx_in]);
            
            if (max < curr)
            {
                max = curr;
            }
        }
    }
    float v{max + bias[c]};
    if (leaky && v < 0.0f)
    {
        v *= 0.1f;
    }
    out[idx_out] = st(v);

    return;
            
}
        
        

__global__ void apply_leaky(act_t* out, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= n) { return; }

    float v{ld(out[idx])};
    out[idx] = st((v > 0.0f) ? v : v * 0.1f);
}



static int ceil_div(int len, int threads)
{
    return (len + threads - 1) / threads;
}


namespace lithium 
{
    void GPUBackend::convolve(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        check_handle();
        
        int threads{256};
        const int window{layer.spec.size};
        const int patch_len{in.c * window * window};
        const int cols{out.h * out.w * patch_len};
        reserve_workspace(static_cast<std::size_t>(cols));


        act_t* work{static_cast<act_t*>(workspace)};

        ::im2col<<<ceil_div(cols, threads), threads>>>(
            in.data, work, cols, window,
            layer.spec.stride, layer.spec.pad, patch_len,
            in.c, in.h, in.w, out.w);

        gemm(work, out, static_cast<const act_t*>(weights_for(layer)),
             out.h * out.w, patch_len, out.c, handle);
    }

    void GPUBackend::conv(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        convolve(layer, in, out);

        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};
        const bool leaky{layer.spec.activation == lithium::Activation::Leaky};

        if (layer.spec.batch_norm)
        {
            batchnorm<<<blocks, threads>>>(out.data, len, out.c,
                layer.weights.scales.data(), layer.weights.biases.data(),
                layer.weights.rolling_mean.data(), layer.weights.rolling_variance.data());
            if (leaky)
            {
                apply_leaky<<<blocks, threads>>>(out.data, len);
            }
            return;
        }
        bias_activate<<<blocks, threads>>>(out.data, len, out.c, layer.weights.biases.data(), leaky);
    }

    void GPUBackend::conv_maxpool(const NetworkLayer& conv, const NetworkLayer& pool,
                                  const Tensor& in, Tensor& conv_out, Tensor& pool_out)
    {
        if (conv.spec.batch_norm)
        {
            GPUBackend::conv(conv, in, conv_out);
            GPUBackend::maxpool(pool, conv_out, pool_out);
            return;
        }

        convolve(conv, in, conv_out);

        int len{static_cast<int>(pool_out.count())};
        int threads{256};
        bias_activate_maxpool<<<ceil_div(len, threads), threads>>>(
            conv_out.data, pool_out.data, pool.spec.size, pool.spec.stride, len, pool.spec.pad,
            pool_out.c, pool_out.h, pool_out.w, conv_out.c, conv_out.h, conv_out.w,
            conv.weights.biases.data(), conv.spec.activation == lithium::Activation::Leaky);
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
        int threads{256};
        int offset{0};
        for (const Tensor& tensor : ins)
        {
            int len{static_cast<int>(tensor.count())};
            concat_slice<<<ceil_div(len, threads), threads>>>(tensor.data, out.data, len,
                                                              tensor.c, out.c, offset);
            offset += tensor.c;
        }
    }
    
    void GPUBackend::download(const Tensor& device, float* host)
    {
        std::transform(device.data, device.data + device.count(), host, lithium::to_float);
    }
            
    void GPUBackend::yolo(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        cudaMemcpyAsync(out.data, in.data, sizeof(act_t) * in.count(), cudaMemcpyDefault);
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};
        int entries{5 + layer.spec.yolo_classes};
        int used{static_cast<int>(layer.spec.yolo_mask.size()) * entries};
        apply_sigmoid<<<blocks, threads>>>(out.data, len, out.c, entries, used);
        
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
#ifndef LITHIUM_HALF
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
        
        const cudaError_t status{cudaMallocManaged(&workspace, floats * sizeof(act_t))};
        
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

#ifdef LITHIUM_HALF
        std::fprintf(stderr, "activations: FP16, FP32 accumulate\n");
#else
        std::fprintf(stderr, "activations: FP32\n");
#endif
    }
}
