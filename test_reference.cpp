#include "cpu_backend.hpp"
#include "network.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>


// A layer passes when its largest absolute error, measured against the largest
// magnitude in the reference tensor, is below this. Per-element relative error is
// the wrong metric here: most activations sit near zero after leaky, so a 1e-6
// absolute difference looks catastrophic next to a 1e-7 reference value.
static constexpr float tolerance{1e-5f};


static std::vector<float> load(const std::string& path)
{
    std::ifstream file{path, std::ios::binary};
    if (!file)
    {
        return {};
    }
    file.seekg(0, std::ios::end);
    const auto floats = static_cast<std::size_t>(file.tellg()) / sizeof(float);
    file.seekg(0, std::ios::beg);

    std::vector<float> data(floats);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(floats * sizeof(float)));
    return file ? data : std::vector<float>{};
}

static const char* type_name(lithium::LayerSpec::LayerType type)
{
    using LayerType = lithium::LayerSpec::LayerType;
    switch (type)
    {
        case LayerType::Conv:     return "conv";
        case LayerType::Maxpool:  return "maxpool";
        case LayerType::Route:    return "route";
        case LayerType::Upsample: return "upsample";
        case LayerType::Yolo:     return "yolo";
    }
    return "?";
}

int main(int argc, char** argv)
{
    using namespace lithium;

    const std::string root{(argc > 1) ? argv[1] : "reference"};
    const std::string variant{(argc > 2) ? argv[2] : "dog-letterbox"};
    const std::string dumps{root + "/" + variant};

    const auto cfg = parse_cfg(root + "/yolov3-tiny-letterbox.cfg");
    if (!cfg)
    {
        std::printf("parse_cfg failed: %s/yolov3-tiny-letterbox.cfg\n", root.c_str());
        return 2;
    }

    auto parsed = parse_weights(root + "/yolov3-tiny.weights", *cfg);
    if (!parsed)
    {
        std::printf("parse_weights failed: %s/yolov3-tiny.weights\n", root.c_str());
        return 2;
    }

    auto network = build_network(*cfg, std::move(*parsed));
    if (!network)
    {
        std::puts("build_network failed");
        return 2;
    }

    auto pixels = load(dumps + "/input_c3_h416_w416.bin");
    const auto expected = static_cast<std::size_t>(network->net.channels)
                        * network->net.height * network->net.width;
    if (pixels.size() != expected)
    {
        std::printf("input: wanted %zu floats, %s/input_c3_h416_w416.bin has %zu\n",
                    expected, dumps.c_str(), pixels.size());
        return 2;
    }

    Tensor input{pixels.data(), network->net.channels,
                 network->net.height, network->net.width};

    CPUBackend backend;
    forward(*network, backend, input);

    std::printf("%s\n\n", dumps.c_str());
    std::puts("layer  type          shape       max|diff|      scale   diff/scale");

    int failures{0};
    int missing{0};
    for (std::size_t i{0}; i < network->outputs.size(); ++i)
    {
        const Tensor& out = network->outputs[i];
        const auto name = std::format("{}/layer_{:02}_c{}_h{}_w{}.bin",
                                      dumps, i, out.c, out.h, out.w);

        if (!std::filesystem::exists(name))
        {
            ++missing;
            std::printf("%4zu  %-9s %4dx%3dx%-4d  no reference dump for this shape\n",
                        i, type_name(network->layers[i].spec.type), out.c, out.h, out.w);
            continue;
        }

        const auto want = load(name);
        if (want.size() != out.count())
        {
            ++failures;
            std::printf("%4zu  %-9s  reference has %zu floats, layer has %zu\n",
                        i, type_name(network->layers[i].spec.type), want.size(), out.count());
            continue;
        }

        float worst{0.0f};
        float scale{0.0f};
        for (std::size_t k{0}; k < want.size(); ++k)
        {
            worst = std::fmax(worst, std::fabs(want[k] - out.data[k]));
            scale = std::fmax(scale, std::fabs(want[k]));
        }
        const float relative{worst / std::fmax(scale, 1e-12f)};
        const bool ok{relative < tolerance};
        if (!ok)
        {
            ++failures;
        }

        std::printf("%4zu  %-9s %4dx%3dx%-4d %10.3g %10.3f   %9.2e  %s\n",
                    i, type_name(network->layers[i].spec.type), out.c, out.h, out.w,
                    worst, scale, relative, ok ? "" : "  <-- FAIL");
    }

    std::printf("\n%d of %zu layers match darknet", 
                static_cast<int>(network->outputs.size()) - failures - missing,
                network->outputs.size());
    if (missing != 0)
    {
        std::printf(", %d had no reference dump", missing);
    }
    std::printf("\n");
    return (failures == 0 && missing == 0) ? 0 : 1;
}
