#include "cpu_backend.hpp"
#include "network.hpp"

#ifdef LITHIUM_CUDA
#include "allocator.hpp"
#include "gpu_backend.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
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

static std::vector<float> to_nhwc(const std::vector<float>& nchw, int c, int h, int w)
{
    std::vector<float> nhwc(nchw.size());
    for (int ch{0}; ch < c; ++ch)
    {
        for (int y{0}; y < h; ++y)
        {
            for (int x{0}; x < w; ++x)
            {
                nhwc[(static_cast<std::size_t>(y) * w + x) * c + ch] =
                    nchw[(static_cast<std::size_t>(ch) * h + y) * w + x];
            }
        }
    }
    return nhwc;
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
    const std::string which{(argc > 3) ? argv[3] : "cpu"};
    const std::string dumps{root + "/" + variant};

    if (which != "cpu" && which != "gpu")
    {
        std::printf("usage: test_reference [reference_root] [variant] [cpu|gpu]\n");
        return 2;
    }
#ifndef LITHIUM_CUDA
    if (which == "gpu")
    {
        std::puts("built without LITHIUM_CUDA: the gpu backend is not in this binary");
        return 2;
    }
#endif

    // Declared before the network so it is destroyed after it: storage calls
    // do_deallocate through this pointer on the way out.
#ifdef LITHIUM_CUDA
    cuda_resource managed{};
#endif
    std::pmr::memory_resource* resource{std::pmr::get_default_resource()};
#ifdef LITHIUM_CUDA
    if (which == "gpu")
    {
        resource = &managed;
    }
#endif

    const auto cfg = parse_cfg(root + "/yolov3-tiny-letterbox.cfg");
    if (!cfg)
    {
        std::printf("parse_cfg failed: %s/yolov3-tiny-letterbox.cfg\n", root.c_str());
        return 2;
    }

    auto parsed = parse_weights(root + "/yolov3-tiny.weights", *cfg, resource);
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

    const auto loaded = load(dumps + "/input_c3_h416_w416.bin");
    const auto expected = static_cast<std::size_t>(network->net.channels)
                        * network->net.height * network->net.width;
    if (loaded.size() != expected)
    {
        std::printf("input: wanted %zu floats, %s/input_c3_h416_w416.bin has %zu\n",
                    expected, dumps.c_str(), loaded.size());
        return 2;
    }

    // The input has to live on the same resource as the layer outputs, or a real
    // kernel would be reading host memory for the first convolution.
    const auto input_nhwc = to_nhwc(loaded, network->net.channels,
                                    network->net.height, network->net.width);
    std::pmr::vector<act_t> pixels(input_nhwc.size(), resource);
    std::transform(input_nhwc.begin(), input_nhwc.end(), pixels.begin(), to_act);

    Tensor input{pixels.data(), network->net.channels,
                 network->net.height, network->net.width};

    CPUBackend cpu;
#ifdef LITHIUM_CUDA
    GPUBackend gpu;
#endif
    Backend* backend{&cpu};
#ifdef LITHIUM_CUDA
    if (which == "gpu")
    {
        backend = &gpu;
    }
#endif
    std::printf("%s   [%s backend]\n", dumps.c_str(), which.c_str());

    int status{0};
    for (const bool fuse : {false, true})
    {
        forward(*network, *backend, input, fuse);
        backend->sync();

        std::printf("\n%s\n", fuse ? "fused" : "unfused");
        std::puts("layer  type          shape       max|diff|      scale   diff/scale");

        int failures{0};
        int missing{0};
        int skipped{0};
        for (std::size_t i{0}; i < network->outputs.size(); ++i)
        {
            const Tensor& out = network->outputs[i];
            if (fuse && fuses_maxpool(*network, i))
            {
                ++skipped;
                std::printf("%4zu  %-9s %4dx%3dx%-4d  fused into the next layer\n",
                            i, type_name(network->layers[i].spec.type), out.c, out.h, out.w);
                continue;
            }

            char name_buf[512];
            std::snprintf(name_buf, sizeof(name_buf), "%s/layer_%02zu_c%d_h%d_w%d.bin",
                          dumps.c_str(), i, out.c, out.h, out.w);
            const std::string name{name_buf};

            if (!std::filesystem::exists(name))
            {
                ++missing;
                std::printf("%4zu  %-9s %4dx%3dx%-4d  no reference dump for this shape\n",
                            i, type_name(network->layers[i].spec.type), out.c, out.h, out.w);
                continue;
            }

            const auto raw = load(name);
            if (raw.size() != out.count())
            {
                ++failures;
                std::printf("%4zu  %-9s  reference has %zu floats, layer has %zu\n",
                            i, type_name(network->layers[i].spec.type), raw.size(), out.count());
                continue;
            }
            const auto want = to_nhwc(raw, out.c, out.h, out.w);

            float worst{0.0f};
            float scale{0.0f};
            for (std::size_t k{0}; k < want.size(); ++k)
            {
                worst = std::fmax(worst, std::fabs(want[k] - to_float(out.data[k])));
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

        const int checked{static_cast<int>(network->outputs.size()) - skipped};
        std::printf("\n%s: %d of %d layers match darknet", fuse ? "fused" : "unfused",
                    checked - failures - missing, checked);
        if (skipped != 0)
        {
            std::printf(", %d fused away", skipped);
        }
        if (missing != 0)
        {
            std::printf(", %d had no reference dump", missing);
        }
        std::printf("\n");
        if (failures != 0 || missing != 0)
        {
            status = 1;
        }
    }
    return status;
}
