// TensorRT runner with the same per-frame scope as the lithium and Darknet
// runners: upload the input from pinned host memory, run the engine, apply the
// YOLO activation (the ONNX export ends at the raw head convolutions), and
// download the head outputs to pinned host memory. Built with nvcc (-x cu).
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>

namespace
{
    struct Logger : nvinfer1::ILogger
    {
        void log(Severity s, const char* msg) noexcept override
        {
            if (s <= Severity::kERROR) { std::fprintf(stderr, "[trt] %s\n", msg); }
        }
    };

    // NCHW head output; per anchor 85 entries, sigmoid on all but w and h,
    // the same rule as darknet's yolo layer and lithium's apply_sigmoid
    __global__ void yolo_activate(float* out, int n, int plane, int entries, int used)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= n) { return; }
        int channel{idx / plane};
        int slice{channel % entries};
        if (channel < used && slice != 2 && slice != 3)
        {
            out[idx] = 1.0f / (1.0f + expf(-out[idx]));
        }
    }

    struct Head
    {
        float* device{nullptr};
        float* host{nullptr};
        int n{0};
        int channels{0};
        int plane{0};
    };
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::puts("usage: fps_runner_trt <engine.plan> <input.bin> <fps|0>");
        return 2;
    }
    const double fps{std::atof(argv[3])};

    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::puts("plan"); return 1; }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    Logger logger{};
    auto* runtime{nvinfer1::createInferRuntime(logger)};
    auto* engine{runtime->deserializeCudaEngine(blob.data(), blob.size())};
    if (engine == nullptr) { std::puts("deserialize"); return 1; }
    auto* ctx{engine->createExecutionContext()};

    cudaStream_t stream{};
    cudaStreamCreate(&stream);

    float* input_device{nullptr};
    float* input_host{nullptr};
    std::size_t input_count{0};
    std::vector<Head> heads{};
    for (int i{0}; i < engine->getNbIOTensors(); ++i)
    {
        const char* name{engine->getIOTensorName(i)};
        if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
        {
            std::printf("tensor %s is not float\n", name);
            return 1;
        }
        const nvinfer1::Dims d{engine->getTensorShape(name)};
        std::size_t n{1};
        for (int k{0}; k < d.nbDims; ++k) { n *= static_cast<std::size_t>(d.d[k]); }
        float* device{nullptr};
        float* host{nullptr};
        if (cudaMalloc(&device, n * sizeof(float)) != cudaSuccess
            || cudaMallocHost(&host, n * sizeof(float)) != cudaSuccess)
        {
            std::puts("alloc");
            return 1;
        }
        ctx->setTensorAddress(name, device);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            input_device = device;
            input_host = host;
            input_count = n;
        }
        else
        {
            heads.push_back({device, host, static_cast<int>(n), static_cast<int>(d.d[1]),
                             static_cast<int>(d.d[2] * d.d[3])});
        }
    }
    if (input_device == nullptr) { std::puts("no input tensor"); return 1; }

    std::ifstream in(argv[2], std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(input_host), static_cast<std::streamsize>(input_count * sizeof(float))))
    {
        std::printf("input: %s does not hold %zu floats\n", argv[2], input_count);
        return 1;
    }

    auto frame = [&] {
        cudaMemcpyAsync(input_device, input_host, input_count * sizeof(float), cudaMemcpyHostToDevice, stream);
        ctx->enqueueV3(stream);
        for (const Head& head : heads)
        {
            constexpr int threads{256};
            yolo_activate<<<(head.n + threads - 1) / threads, threads, 0, stream>>>(
                head.device, head.n, head.plane, 85, 3 * 85);
            cudaMemcpyAsync(head.host, head.device, head.n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
    };

    for (int i{0}; i < 20; ++i) { frame(); }

    const auto period = (fps > 0.0)
        ? std::chrono::duration<double>(1.0 / fps)
        : std::chrono::duration<double>(0.0);
    auto next = std::chrono::steady_clock::now();
    auto window = std::chrono::steady_clock::now();
    long frames{0};
    double busy{0.0};

    while (true)
    {
        const auto t0 = std::chrono::steady_clock::now();
        frame();
        const auto t1 = std::chrono::steady_clock::now();
        busy += std::chrono::duration<double>(t1 - t0).count();
        ++frames;

        if (fps > 0.0)
        {
            next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            if (next > t1) { std::this_thread::sleep_until(next); }
            else           { next = t1; }
        }

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - window).count();
        if (elapsed >= 1.0)
        {
            std::printf("{\"fps\": %.2f, \"ms\": %.3f, \"duty\": %.4f}\n",
                        frames / elapsed, 1000.0 * busy / frames, busy / elapsed);
            std::fflush(stdout);
            frames = 0; busy = 0.0; window = now;
        }
    }
    return 0;
}
