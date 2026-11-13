#include "gpu_backend.hpp"
#ifdef LITHIUM_CUTLASS
#include "cutlass_conv.hpp"
#endif
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


static void* upload(const float* source, std::size_t count, bool half)
{
    std::vector<__half> staging{};
    const void* from{source};
    std::size_t bytes{count * sizeof(float)};
    if (half)
    {
        staging.resize(count);
        for (std::size_t i{0}; i < count; ++i)
        {
            staging[i] = __float2half(source[i]);
        }
        from = staging.data();
        bytes = count * sizeof(__half);
    }

    void* device{nullptr};
    const cudaError_t allocated{cudaMalloc(&device, bytes)};
    if (allocated != cudaSuccess)
    {
        std::fprintf(stderr, "weights alloc (%zu elements): %s\n",
                     count, cudaGetErrorString(allocated));
        std::abort();
    }

    const cudaError_t copied{cudaMemcpy(device, from, bytes, cudaMemcpyHostToDevice)};
    if (copied != cudaSuccess)
    {
        std::fprintf(stderr, "weights upload: %s\n", cudaGetErrorString(copied));
        std::abort();
    }
    return device;
}




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


__constant__ float layer_0_w[16 * 3 * 3 * 3];
__constant__ float layer_0_b[16];

__global__ void layer_0_conv(const act_t* in, act_t* out, int H, int W)
{
    /*
    Magic numbers:
    C = 3 (RGB)
    16 filters in the first layer
    3x3 conv kernel, stride 1, pad 1
    */
    constexpr int C{3};

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= H * W)
        return;

    int x{idx % W}, y{idx / W};

    float acc[16]{};

    #pragma unroll
    for (int k{}; k < 16; ++k)
        acc[k] = layer_0_b[k];
    

    #pragma unroll
    for (int r{}; r < 3; ++r)
    {
        int iy{y + r - 1};

        if (iy < 0 || iy >= H) 
            continue;

        #pragma unroll
        for (int s{}; s < 3; ++s)
        {
            int ix{x + s - 1};

            if (ix < 0 || ix >= W) 
                continue;
            
            #pragma unroll
            for (int c{}; c < C; ++c)
            {
                float t{ld(in[(iy * W + ix) * C + c])};
                #pragma unroll
                for (int k{}; k < 16; ++k)
                    acc[k] += t * layer_0_w[((k * 3 + r) * 3 + s) * C + c];
            }
        }
    }

    #pragma unroll
    for (int k{}; k < 16; ++k)
        out[(y * W + x) * 16 + k] = st(acc[k] < 0.0f ? 0.1f * acc[k] : acc[k]);
}

__global__ void layer_0_conv_maxpool(const act_t* in, act_t* out, int H, int W)
{
    /*
    Magic numbers:
    C = 3 (RGB)
    16 filters in the first layer
    3x3 conv kernel, stride 1, pad 1
    2x2 maxpool, stride 2, so the output is H/2 x W/2
    H, W are the conv (input) dimensions
    */
    constexpr int C{3};

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= (H / 2) * (W / 2))
        return;

    int px{idx % (W / 2)}, py{idx / (W / 2)};
    int x{px * 2}, y{py * 2}; // top left conv pixel of the 2x2 pool window

    float acc1[16]{}, acc2[16]{}, acc3[16]{}, acc4[16]{};

    #pragma unroll
    for (int r{}; r < 3; ++r)
    {
        int iy1{y + r - 1};
        int iy2{iy1 + 1}; // window of the conv pixel one row down

        bool y1ok{iy1 >= 0 && iy1 < H}, y2ok{iy2 >= 0 && iy2 < H};

        #pragma unroll
        for (int s{}; s < 3; ++s)
        {
            int ix1{x + s - 1};
            int ix2{ix1 + 1}; // window of the conv pixel one column right

            bool x1ok{ix1 >= 0 && ix1 < W}, x2ok{ix2 >= 0 && ix2 < W};

            #pragma unroll
            for (int c{}; c < C; ++c)
            {
                float t1{(y1ok && x1ok) ? ld(in[(iy1 * W + ix1) * C + c]) : 0.0f};
                float t2{(y1ok && x2ok) ? ld(in[(iy1 * W + ix2) * C + c]) : 0.0f};
                float t3{(y2ok && x1ok) ? ld(in[(iy2 * W + ix1) * C + c]) : 0.0f};
                float t4{(y2ok && x2ok) ? ld(in[(iy2 * W + ix2) * C + c]) : 0.0f};

                #pragma unroll
                for (int k{}; k < 16; ++k)
                {
                    float wk{layer_0_w[((k * 3 + r) * 3 + s) * C + c]};
                    acc1[k] += t1 * wk;
                    acc2[k] += t2 * wk;
                    acc3[k] += t3 * wk;
                    acc4[k] += t4 * wk;
                }
            }
        }
    }

    // bias is the same for all 4, so it can be added after the max
    #pragma unroll
    for (int k{}; k < 16; ++k)
    {
        float hi{acc1[k]};
        hi = fmaxf(hi, acc2[k]);
        hi = fmaxf(hi, acc3[k]);
        hi = fmaxf(hi, acc4[k]);
        hi += layer_0_b[k];
        out[(py * (W / 2) + px) * 16 + k] = st(hi < 0.0f ? 0.1f * hi : hi);
    }
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


__global__ void apply_sigmoid(const act_t* in, act_t* out, int n, int channels, int slices, int used)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x; 
    if (idx >= n) { return; }
    
    int channel{idx % channels};
    int slice{channel % slices};
    
    if (channel < used && slice != 2 && slice != 3)
    {
        out[idx] = st(1.0f / (1.0f + expf(-ld(in[idx]))));
        return;
    }
    out[idx] = in[idx];
    
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

static bool is_layer_0(const lithium::NetworkLayer& layer, const lithium::Tensor& in, const lithium::Tensor& out)
{
    return !layer.spec.batch_norm && in.c == 3 && out.c == 16 && layer.spec.size == 3
        && layer.spec.stride == 1 && layer.spec.pad == 1 && in.h == out.h && in.w == out.w
        && layer.spec.activation == lithium::Activation::Leaky;
}


namespace lithium 
{
    void GPUBackend::convolve(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        check_handle();

#ifdef LITHIUM_CUTLASS
        if (cutlass == nullptr)
        {
            cutlass = cutlass_create();
        }
        if (cutlass_conv(cutlass, layer, in, out, weights_for(layer), stream))
        {
            return;
        }
#endif
        
        int threads{256};
        const int window{layer.spec.size};
        const int patch_len{in.c * window * window};
        const int cols{out.h * out.w * patch_len};
        reserve_workspace(static_cast<std::size_t>(cols));


        act_t* work{static_cast<act_t*>(workspace)};

        ::im2col<<<ceil_div(cols, threads), threads, 0, stream>>>(
            in.data, work, cols, window,
            layer.spec.stride, layer.spec.pad, patch_len,
            in.c, in.h, in.w, out.w);

        gemm(work, out, static_cast<const act_t*>(weights_for(layer)),
             out.h * out.w, patch_len, out.c, handle);
    }

    void GPUBackend::conv(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        if (is_layer_0(layer, in, out))
        {
            int threads{256};
            upload_layer_0(layer);
            layer_0_conv<<<ceil_div(out.h * out.w, threads), threads, 0, stream>>>(in.data, out.data, out.h, out.w);
            return;
        }

        convolve(layer, in, out);

        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};
        const bool leaky{layer.spec.activation == lithium::Activation::Leaky};

        if (layer.spec.batch_norm)
        {
            batchnorm<<<blocks, threads, 0, stream>>>(out.data, len, out.c,
                layer.weights.scales.data(), layer.weights.biases.data(),
                layer.weights.rolling_mean.data(), layer.weights.rolling_variance.data());
            if (leaky)
            {
                apply_leaky<<<blocks, threads, 0, stream>>>(out.data, len);
            }
            return;
        }
        bias_activate<<<blocks, threads, 0, stream>>>(out.data, len, out.c, bias_for(layer), leaky);
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

        if (is_layer_0(conv, in, conv_out) && pool.spec.size == 2 && pool.spec.stride == 2
            && conv_out.h % 2 == 0 && conv_out.w % 2 == 0)
        {
            int threads{256};
            upload_layer_0(conv);
            layer_0_conv_maxpool<<<ceil_div(pool_out.h * pool_out.w, threads), threads, 0, stream>>>(
                in.data, pool_out.data, conv_out.h, conv_out.w);
            return;
        }

        convolve(conv, in, conv_out);

        int len{static_cast<int>(pool_out.count())};
        int threads{256};
        bias_activate_maxpool<<<ceil_div(len, threads), threads, 0, stream>>>(
            conv_out.data, pool_out.data, pool.spec.size, pool.spec.stride, len, pool.spec.pad,
            pool_out.c, pool_out.h, pool_out.w, conv_out.c, conv_out.h, conv_out.w,
            bias_for(conv), conv.spec.activation == lithium::Activation::Leaky);
    }


    
    void GPUBackend::maxpool(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};

        ::maxpool<<<blocks, threads, 0, stream>>>(in.data, out.data, layer.spec.size, layer.spec.stride, len,
                                       layer.spec.pad,
                                       out.c, out.h, out.w, in.c, in.h, in.w);
    }
        
    void GPUBackend::upsample(const Tensor& in, Tensor& out, int stride)
    {
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};
        ::upsample<<<blocks, threads, 0, stream>>>(in.data, out.data, len, 
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
            concat_slice<<<ceil_div(len, threads), threads, 0, stream>>>(tensor.data, out.data, len,
                                                              tensor.c, out.c, offset);
            offset += tensor.c;
        }
    }
    
    void GPUBackend::download(const Tensor& device, float* host)
    {
        std::vector<act_t> staging(device.count());
        const cudaError_t status{cudaMemcpy(staging.data(), device.data, staging.size() * sizeof(act_t),
                                            cudaMemcpyDefault)};
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "download: %s\n", cudaGetErrorString(status));
        }
        std::transform(staging.begin(), staging.end(), host, lithium::to_float);
    }

    Tensor GPUBackend::stage(const Tensor& input)
    {
        if (input_from != input.data)
        {
            cudaPointerAttributes attributes{};
            input_on_device = cudaPointerGetAttributes(&attributes, input.data) == cudaSuccess
                && (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged);
            cudaGetLastError();
            input_from = input.data;
        }
        if (input_on_device)
        {
            return input;
        }

        const std::size_t bytes{input.count() * sizeof(act_t)};
        if (bytes > input_bytes)
        {
            sync();
            if (input_device != nullptr)
            {
                cudaFree(input_device);
            }
            const cudaError_t status{cudaMalloc(&input_device, bytes)};
            if (status != cudaSuccess)
            {
                std::fprintf(stderr, "input buffer: %s\n", cudaGetErrorString(status));
                std::abort();
            }
            input_bytes = bytes;
        }
        cudaMemcpyAsync(input_device, input.data, bytes, cudaMemcpyHostToDevice, stream);

        Tensor staged{input};
        staged.data = static_cast<act_t*>(input_device);
        return staged;
    }
            
    void GPUBackend::yolo(const NetworkLayer& layer, const Tensor& in, Tensor& out)
    {
        int len{static_cast<int>(out.count())};
        int threads{256};
        int blocks{ceil_div(len, threads)};
        int entries{5 + layer.spec.yolo_classes};
        int used{static_cast<int>(layer.spec.yolo_mask.size()) * entries};
        apply_sigmoid<<<blocks, threads, 0, stream>>>(in.data, out.data, len, out.c, entries, used);
        
    }

    void GPUBackend::sync()
    {
        const cudaError_t status{cudaDeviceSynchronize()};
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(status));
        }
    }

    void GPUBackend::upload_layer_0(const NetworkLayer& layer)
    {
        const float* src{layer.weights.weights.data()};
        if (layer_0_uploaded == src)
        {
            return;
        }
        sync();
        cudaError_t status{cudaMemcpyToSymbol(layer_0_w, src, sizeof(layer_0_w), 0, cudaMemcpyDefault)};
        if (status == cudaSuccess)
        {
            status = cudaMemcpyToSymbol(layer_0_b, layer.weights.biases.data(), sizeof(layer_0_b), 0, cudaMemcpyDefault);
        }
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(status));
            return;
        }
        layer_0_uploaded = src;
    }

    const void* GPUBackend::device_copy(const float* key, std::size_t count, bool half)
    {
        for (const DeviceCopy& entry : device_copies)
        {
            if (entry.key != key)
            {
                continue;
            }
            if (entry.count != count)
            {
                std::fprintf(stderr,
                             "device weight cache: %p was %zu elements, now %zu\n",
                             static_cast<const void*>(key), entry.count, count);
                std::abort();
            }
            return entry.data;
        }
        sync();
        void* copy{upload(key, count, half)};
        device_copies.push_back({key, copy, count});
        return copy;
    }

    const void* GPUBackend::weights_for(const NetworkLayer& layer)
    {
#ifdef LITHIUM_HALF
        constexpr bool half{true};
#else
        constexpr bool half{false};
#endif
        return device_copy(layer.weights.weights.data(), layer.weights.weights.size(), half);
    }

    const float* GPUBackend::bias_for(const NetworkLayer& layer)
    {
        return static_cast<const float*>(device_copy(layer.weights.biases.data(), layer.weights.biases.size(), false));
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
        
        const cudaError_t status{cudaMalloc(&workspace, floats * sizeof(act_t))};
        
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "im2col workspace: %s\n", cudaGetErrorString(status));
            std::abort();
        }
        workspace_floats = floats;
    }

    GPUBackend::GPUBackend()
    {
        const cudaError_t status{cudaStreamCreate(&stream)};
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "cuda stream: %s\n", cudaGetErrorString(status));
            std::abort();
        }
    }

    void GPUBackend::run([[maybe_unused]] const PassKey& key, const std::function<void()>& pass)
    {
#ifndef LITHIUM_GRAPHS
        pass();
#else
        Graph* graph{nullptr};
        for (Graph& existing : graphs)
        {
            if (existing.key == key)
            {
                graph = &existing;
                break;
            }
        }
        if (graph == nullptr)
        {
            graph = &graphs.emplace_back(Graph{key});
        }

        if (graph->exec == nullptr)
        {
            if (graph->failed || graph->passes++ == 0)
            {
                pass();
                return;
            }

            cudaGraph_t captured{nullptr};
            cudaError_t status{cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)};
            if (status == cudaSuccess)
            {
                pass();
                status = cudaStreamEndCapture(stream, &captured);
            }
            if (status == cudaSuccess)
            {
                status = cudaGraphInstantiate(&graph->exec, captured, 0);
            }
            std::size_t nodes{0};
            if (captured != nullptr)
            {
                cudaGraphGetNodes(captured, nullptr, &nodes);
                cudaGraphDestroy(captured);
            }
            if (status != cudaSuccess)
            {
                std::fprintf(stderr, "cuda graph: %s, running without one\n", cudaGetErrorString(status));
                cudaGetLastError();
                graph->exec = nullptr;
                graph->failed = true;
                pass();
                return;
            }
            std::fprintf(stderr, "cuda graph: %zu nodes\n", nodes);
        }

        const cudaError_t status{cudaGraphLaunch(graph->exec, stream)};
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "cuda graph launch: %s\n", cudaGetErrorString(status));
        }
#endif
    }

    GPUBackend::~GPUBackend()
    {
        sync();
        for (const Graph& graph : graphs)
        {
            if (graph.exec != nullptr)
            {
                cudaGraphExecDestroy(graph.exec);
            }
        }
        for (const DeviceCopy& entry : device_copies)
        {
            if (entry.data != nullptr)
            {
                cudaFree(entry.data);
            }
        }
        device_copies.clear();
        if (input_device != nullptr)
        {
            cudaFree(input_device);
        }

        if (workspace != nullptr)
        {
            cudaFree(workspace);
        }
        if (handle != nullptr)
        {
            cublasDestroy(handle);
        }
#ifdef LITHIUM_CUTLASS
        cutlass_destroy(cutlass);
#endif
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
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
        cublasSetStream(handle, stream);

#ifdef LITHIUM_HALF
        std::fprintf(stderr, "activations: FP16, FP32 accumulate\n");
#else
        std::fprintf(stderr, "activations: FP32\n");
#endif
    }
}
