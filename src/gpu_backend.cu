#include "gpu_backend.hpp"
#include <cublas_v2.h>
#include <cstdio>
#include <algorithm>
#include <cfloat>
#include <cmath>



static std::vector<float> im2col(
                                const lithium::Tensor& input, const lithium::Tensor& out, 
                                int pad, int stride, int h, int w
                                )
{
    std::vector<float> arr{};
    int nx{out.w},
        ny{out.h};
    arr.resize(static_cast<std::size_t>(h * w) * input.c * nx * ny);

    std::size_t counter{0};
    for (int j{0}; j < ny; ++j)
    {
        for (int k{0}; k < nx; ++k)
        {
            for (int i{0}; i < input.c; ++i)
            {
                for (int pos{0}; pos < h * w; ++pos)
                {
                    int x{pos % w + k * stride - pad},
                        y{pos / w + j * stride - pad};

                    arr[counter++] = (x < 0 || y < 0 || x >= input.w || y >= input.h)
                        ? 0.0f
                        : input.data[input.index(i, x, y)];
                }
            }
        }
    }
    return arr;
}

static void gemm(
    const float* in, lithium::Tensor& out, const float* weights, 
    int patch_len
)
{
    const int patches{out.h * out.w};
    for (int f{0}; f < out.c; ++f)
    {
        for (int p{0}; p < patches; ++p)
        {
            float sum{0.0f};
            for (int i{0}; i < patch_len; ++i)
            {
                sum += weights[f * patch_len + i] * in[p * patch_len + i];
            }
            out.data[f * patches + p] = sum;
        }
    }
    return;
}



















// CUDA FUNCITONS

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
        cudaMemcpy(out.data + offset, tensor.data, sizeof(float) * len, cudaMemcpyDefault);
        offset += len;
    }
    // no need for sync cause memcpy blocks
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


//MAIN INTERFACE
namespace lithium 
{
    void GPUBackend::conv(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {

        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};
        auto cols{im2col(
            in, out, 
            layer.spec.pad, layer.spec.stride, 
            layer.spec.size, layer.spec.size
        )};
        gemm(cols.data(), out, layer.weights.weights.data(), 
        in.c * layer.spec.size * layer.spec.size);
        
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
        sync();
    }
    
    void GPUBackend::maxpool(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};

        ::maxpool<<<blocks, threads>>>(in.data, out.data, layer.spec.size, layer.spec.stride, len,
                                       layer.spec.pad,
                                       out.c, out.h, out.w, in.c, in.h, in.w);
        sync();
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
            
            sync();
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
        cudaMemcpy(out.data, in.data, sizeof(float) * static_cast<int>(in.count()), cudaMemcpyDefault);
        int hw{out.h * out.w};
        
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};
        int entries{5 + layer.spec.yolo_classes};
        apply_sigmoid<<<blocks, threads>>>(out.data, len, hw, entries);
        sync();
        
    }

    void GPUBackend::sync()
    {
        const cudaError_t status{cudaDeviceSynchronize()};
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(status));
        }
    }
}
