#include "cpu_backend.hpp"
#include "network.hpp"
#include "preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>


static double percentile(const std::vector<double>& sorted, double p)
{
    if (sorted.empty())
    {
        return 0.0;
    }
    const auto rank = static_cast<std::size_t>((static_cast<double>(sorted.size() - 1) * p) / 100.0);
    return sorted[rank];
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::puts("usage: bench <cfg> <weights> <image> [iterations] [warmup]");
        return 2;
    }

    const int iterations{(argc > 4) ? std::atoi(argv[4]) : 200};
    const int warmup{(argc > 5) ? std::atoi(argv[5]) : 20};

    auto cfg{lithium::parse_cfg(argv[1])};
    if (!cfg)
    {
        std::puts("failed to parse cfg");
        return 1;
    }

    auto weights{lithium::parse_weights(argv[2], *cfg)};
    if (!weights)
    {
        std::puts("failed to parse weights");
        return 1;
    }

    auto network{lithium::build_network(*cfg, std::move(*weights))};
    if (!network)
    {
        std::puts("failed to build network");
        return 1;
    }

    lithium::Tensor input{};
    input.h = cfg->net.height;
    input.w = cfg->net.width;
    input.c = cfg->net.channels;
    std::pmr::vector<float> pixels(
        static_cast<std::size_t>(input.c) * input.h * input.w, network->resource);
    input.data = pixels.data();

    auto loaded{lithium::preprocess(argv[3], input)};
    if (!loaded)
    {
        std::puts("failed to preprocess image");
        return 1;
    }

    lithium::CPUBackend backend{};

    for (int i{0}; i < warmup; ++i)
    {
        forward(*network, backend, input);
    }

    std::vector<double> ms{};
    ms.reserve(static_cast<std::size_t>(iterations));
    const auto wall_start{std::chrono::steady_clock::now()};
    for (int i{0}; i < iterations; ++i)
    {
        const auto t0{std::chrono::steady_clock::now()};
        forward(*network, backend, input);
        const auto t1{std::chrono::steady_clock::now()};
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    const auto wall_end{std::chrono::steady_clock::now()};

    double sum{0.0};
    for (double v : ms)
    {
        sum += v;
    }
    const double mean{sum / static_cast<double>(ms.size())};
    const double wall{std::chrono::duration<double>(wall_end - wall_start).count()};
    std::sort(ms.begin(), ms.end());

    std::printf("\nforward() over %d iterations, %d warmup, %dx%d input\n\n",
                iterations, warmup, input.w, input.h);
    std::printf("  mean    %8.2f ms\n", mean);
    std::printf("  min     %8.2f ms\n", ms.front());
    std::printf("  p50     %8.2f ms\n", percentile(ms, 50.0));
    std::printf("  p90     %8.2f ms\n", percentile(ms, 90.0));
    std::printf("  p99     %8.2f ms\n", percentile(ms, 99.0));
    std::printf("  p99.9   %8.2f ms\n", percentile(ms, 99.9));
    std::printf("  max     %8.2f ms\n", ms.back());
    std::printf("\n  throughput   %6.2f fps\n", static_cast<double>(iterations) / wall);
    std::printf("  jitter p99/p50  %5.2fx\n\n",
                percentile(ms, 99.0) / percentile(ms, 50.0));

    return 0;
}
