#define DARKNET_INCLUDE_ORIGINAL_API
#include "darknet.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::puts("usage: fps_runner_darknet <cfg> <weights> <input.bin> <fps|0>");
        return 2;
    }
    const double fps{std::atof(argv[4])};

    DarknetNetworkPtr net = load_network_custom(argv[1], argv[2], 0, 1);
    if (!net) { std::puts("load"); return 1; }

    std::FILE* f = std::fopen(argv[3], "rb");
    if (!f) { std::puts("input"); return 1; }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<float> input(bytes / sizeof(float));
    if (std::fread(input.data(), 1, bytes, f) != static_cast<size_t>(bytes)) { std::puts("read"); return 1; }
    std::fclose(f);

    for (int i = 0; i < 20; ++i) { network_predict_ptr(net, input.data()); }

    const auto period = (fps > 0.0)
        ? std::chrono::duration<double>(1.0 / fps)
        : std::chrono::duration<double>(0.0);
    auto next = std::chrono::steady_clock::now();
    auto window = std::chrono::steady_clock::now();
    long frames = 0;
    double busy = 0.0;

    while (true)
    {
        const auto t0 = std::chrono::steady_clock::now();
        network_predict_ptr(net, input.data());
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
