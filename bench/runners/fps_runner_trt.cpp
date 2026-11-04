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
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::puts("usage: fps_runner_trt <engine.plan> <fps|0>");
        return 2;
    }
    const double fps{std::atof(argv[2])};

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
    std::vector<void*> owned{};
    for (int i{0}; i < engine->getNbIOTensors(); ++i)
    {
        const char* name{engine->getIOTensorName(i)};
        const nvinfer1::Dims d{engine->getTensorShape(name)};
        std::size_t n{1};
        for (int k{0}; k < d.nbDims; ++k) { n *= static_cast<std::size_t>(d.d[k]); }
        void* p{nullptr};
        if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) { std::puts("alloc"); return 1; }
        cudaMemset(p, 0, n * sizeof(float));
        ctx->setTensorAddress(name, p);
        owned.push_back(p);
    }

    for (int i{0}; i < 20; ++i) { ctx->enqueueV3(stream); }
    cudaStreamSynchronize(stream);

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
        ctx->enqueueV3(stream);
        cudaStreamSynchronize(stream);
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
