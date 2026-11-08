#include "cpu_backend.hpp"
#include "network.hpp"
#include "preprocess.hpp"

#ifdef LITHIUM_CUDA
#include "allocator.hpp"
#include "gpu_backend.hpp"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::puts("usage: fps_runner <cfg> <weights> <image> <fps|0>");
        return 2;
    }
    const double fps{std::atof(argv[4])};

#ifdef LITHIUM_CUDA
    lithium::cuda_resource managed{};
    std::pmr::memory_resource* resource{&managed};
#else
    std::pmr::memory_resource* resource{std::pmr::get_default_resource()};
#endif

    auto cfg{lithium::parse_cfg(argv[1])};
    if (!cfg) { std::puts("cfg"); return 1; }
    auto weights{lithium::parse_weights(argv[2], *cfg, resource)};
    if (!weights) { std::puts("weights"); return 1; }
    auto network{lithium::build_network(*cfg, std::move(*weights))};
    if (!network) { std::puts("network"); return 1; }

    lithium::Tensor input{};
    input.h = cfg->net.height;
    input.w = cfg->net.width;
    input.c = cfg->net.channels;
    std::pmr::vector<lithium::act_t> pixels(input.count(), network->resource);
    input.data = pixels.data();
    if (!lithium::preprocess(argv[3], input)) { std::puts("preprocess"); return 1; }

    lithium::CPUBackend cpu{};
#ifdef LITHIUM_CUDA
    lithium::GPUBackend gpu{};
    lithium::Backend* backend{&gpu};
#else
    lithium::Backend* backend{&cpu};
#endif

    for (int i{0}; i < 20; ++i) { forward(*network, *backend, input); }
    backend->sync();

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
        forward(*network, *backend, input);
        backend->sync();
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
