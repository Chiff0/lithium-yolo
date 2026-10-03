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

static void batch_normalize(lithium::Tensor& out, 
                            const lithium::LayerWeights& weights
)
{
    std::size_t size_per_channel{static_cast<std::size_t>(out.h) * out.w};
    for (auto i{0}; i < out.c; ++i)
    {
        auto dev = static_cast<float>(std::sqrt(static_cast<double>(weights.rolling_variance[i] + 1e-5)));
        auto offset{size_per_channel * i};
        auto scale{weights.scales[i]}, bias{weights.biases[i]}, mean{weights.rolling_mean[i]};
        for(auto j{offset}; j < offset + size_per_channel; ++j)
        {
            // I do all 3 there, why not 
            out.data[j] = (out.data[j] - mean) / dev; //normalize
            out.data[j] *= scale; // scale
            out.data[j] += bias; // add bias
            
        }
    }
}




static void add_bias(lithium::Tensor& out, 
                    const lithium::LayerWeights& weights
)
{
    std::size_t size_per_channel{static_cast<std::size_t>(out.h) * out.w};
    for (auto i{0}; i < out.c; ++i)
    {
        auto offset{size_per_channel * i};
        auto bias{weights.biases[i]};
        for (auto j{offset}; j < offset + size_per_channel; ++j)
        {
            out.data[j] += bias;
        }
    }
}







static float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

static void apply_sigmoid(float* arr, std::size_t start, int len)
{
    for (int i{0}; i < len; ++i)
    {
        arr[start + i] = sigmoid(arr[start + i]);
    }
}

static void concat(const std::vector<lithium::Tensor>& ins, lithium::Tensor& out)
{
    int offset{0};
    for (auto& tensor : ins)
    {
        int len{static_cast<int>(tensor.count())};
        std::copy(tensor.data, tensor.data + len, out.data + offset);
        offset += len;
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
    
    
    out[idx_out] = in[idx_in];
    
    return;
}

__global__ void maxpool(float* in, float* out, int window, int stride, int len_out, 
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

































// UTIL FUNCTIONS

static float leaky(float x)
{
    return (x > 0) ? x : x * 0.1f; 
}

[[maybe_unused]] static float linear(float x)
{
    return x;
}

static void apply_activation(lithium::Tensor& out, lithium::Activation activation)
{
    switch(activation)
    {
        case lithium::Activation::Leaky:
        for (std::size_t i{0}; i < out.count(); ++i)
        {
            out.data[i] = leaky(out.data[i]);
        }
        break;
        case lithium::Activation::Linear:
        break;
    }
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
        auto cols{im2col(
            in, out, 
            layer.spec.pad, layer.spec.stride, 
            layer.spec.size, layer.spec.size
        )};
        gemm(cols.data(), out, layer.weights.weights.data(), 
        in.c * layer.spec.size * layer.spec.size);
        if (layer.spec.batch_norm)
        {
            batch_normalize(out, layer.weights);
        }
        else
        {
            add_bias(out, layer.weights);
        }
        apply_activation(out, layer.spec.activation);
    }
    
    void GPUBackend::maxpool(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};

        ::maxpool<<<blocks, threads>>>(in.data, out.data, layer.spec.size, layer.spec.stride, len, 
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
        std::copy(in.data, in.data + in.count(), out.data);
        int hw{out.h * out.w};
        int entries{5 + layer.spec.yolo_classes};

        for (std::size_t i{0}; i < layer.spec.yolo_mask.size(); ++i)
        {
            int base{static_cast<int>(i) * entries * hw};

            apply_sigmoid(out.data, static_cast<std::size_t>(base), 2 * hw);
            apply_sigmoid(out.data, static_cast<std::size_t>(base + 4 * hw), (entries - 4) * hw);
        }
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
