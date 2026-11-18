// One fps runner for every step of the optimisation ladder. Each step is an old
// commit built as it was; this file is added to that tree as an extra target, so
// only the setup below differs between API eras (LADDER_API, set by build.sh):
//
//   1  build_network(cfg, weights), preprocess(path, out), managed memory
//   2  preprocess(path, out, channels) with network->image_channels
//   3  device memory for activations, pinned input, build_network(..., activations)
//
// The measured frame is the same for every step and matches the Darknet and
// TensorRT runners: the input is uploaded from pinned host memory, forward()
// runs, the YOLO head outputs are downloaded to pinned host memory, then sync.
// Eras 1 and 2 had no upload of their own (the input lived in managed memory),
// so the runner copies the pinned input into it every frame.
//
//   ladder_runner <cfg> <weights> <image> <fps|0>
#include "cpu_backend.hpp"
#include "network.hpp"
#include "preprocess.hpp"

#ifdef LITHIUM_CUDA
#include "allocator.hpp"
#include "gpu_backend.hpp"

#include <cuda_runtime_api.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory_resource>
#include <thread>
#include <type_traits>
#include <vector>

#ifndef LADDER_API
#error "LADDER_API must be 1, 2 or 3"
#endif

using element = std::remove_pointer_t<decltype(lithium::Tensor{}.data)>;

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::puts("usage: ladder_runner <cfg> <weights> <image> <fps|0>");
        return 2;
    }
    const double fps{std::atof(argv[4])};

    std::pmr::memory_resource* weights_resource{std::pmr::get_default_resource()};
#ifdef LITHIUM_CUDA
#if LADDER_API >= 3
    lithium::device_resource device{};
#else
    lithium::cuda_resource managed{};
    weights_resource = &managed;
#endif
#endif

    auto cfg{lithium::parse_cfg(argv[1])};
    if (!cfg) { std::puts("cfg"); return 1; }
    auto weights{lithium::parse_weights(argv[2], *cfg, weights_resource)};
    if (!weights) { std::puts("weights"); return 1; }
#if LADDER_API >= 3 && defined(LITHIUM_CUDA)
    auto network{lithium::build_network(*cfg, std::move(*weights), &device)};
#else
    auto network{lithium::build_network(*cfg, std::move(*weights))};
#endif
    if (!network) { std::puts("network"); return 1; }

    lithium::Tensor input{};
    input.h = cfg->net.height;
    input.w = cfg->net.width;
    input.c = network->net.channels;
    const std::size_t bytes{input.count() * sizeof(element)};

    // the engine's own input buffer: managed memory before era 3, pinned after
    std::pmr::vector<element> engine_input(input.count(), network->resource);
    input.data = engine_input.data();
#if LADDER_API >= 2
    if (!lithium::preprocess(argv[3], input, network->image_channels)) { std::puts("preprocess"); return 1; }
#else
    if (!lithium::preprocess(argv[3], input)) { std::puts("preprocess"); return 1; }
#endif

    lithium::CPUBackend cpu{};
    lithium::Backend* backend{&cpu};
#ifdef LITHIUM_CUDA
    lithium::GPUBackend gpu{};
    backend = &gpu;

    element* pinned_input{nullptr};
    if (cudaMallocHost(reinterpret_cast<void**>(&pinned_input), bytes) != cudaSuccess) { std::puts("pinned"); return 1; }
    std::memcpy(pinned_input, input.data, bytes);
#if LADDER_API >= 3
    input.data = pinned_input;
#endif

    std::vector<const lithium::Tensor*> heads{};
    std::vector<element*> results{};
    for (std::size_t i{0}; i < network->layers.size(); ++i)
    {
        if (network->layers[i].spec.type == lithium::LayerSpec::LayerType::Yolo)
        {
            heads.push_back(&network->outputs[i]);
            element* out{nullptr};
            if (cudaMallocHost(reinterpret_cast<void**>(&out), network->outputs[i].count() * sizeof(element)) != cudaSuccess)
            {
                std::puts("pinned");
                return 1;
            }
            results.push_back(out);
        }
    }
#endif

    auto frame = [&] {
#if defined(LITHIUM_CUDA) && LADDER_API < 3
        cudaMemcpyAsync(engine_input.data(), pinned_input, bytes, cudaMemcpyDefault, 0);
#endif
        forward(*network, *backend, input);
#ifdef LITHIUM_CUDA
        for (std::size_t k{0}; k < heads.size(); ++k)
        {
            cudaMemcpyAsync(results[k], heads[k]->data, heads[k]->count() * sizeof(element),
                            cudaMemcpyDefault, 0);
        }
#endif
        backend->sync();
    };

    const auto warm{std::chrono::steady_clock::now()};
    for (int i{0}; i < 20 && std::chrono::steady_clock::now() - warm < std::chrono::seconds(3); ++i) { frame(); }

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
