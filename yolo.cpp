#include "yolo.hpp"

#include <algorithm>
#include <cmath>

static float sigmoid(float x)
{
    return 1.0f / (std::exp(-x) + 1.0f);
} 

static void apply_sigmoid(float* arr, std::size_t start, int len)
{
    for (int i{0}; i < len; ++i)
    {
        arr[start + i] = sigmoid(arr[start + i]); 
    }

}

static lithium::Decoded create(
    const lithium::Tensor& in, int base, int j,
    int classes, std::pair<float, float> anchor,
    int net_w, int net_h, float threshold
)
{
    int hw{in.h * in.w};
    int row{j / in.w}, col{j % in.w};

    lithium::Decoded ret{};
    ret.x = (static_cast<float>(col) + in.data[base + j]) / static_cast<float>(in.w);
    ret.y = (static_cast<float>(row) + in.data[base + j + hw]) / static_cast<float>(in.h);
    ret.w = std::exp(in.data[base + j + 2 * hw]) * anchor.first / static_cast<float>(net_w);
    ret.h = std::exp(in.data[base + j + 3 * hw]) * anchor.second / static_cast<float>(net_h);
    ret.objectness = in.data[base + j + 4 * hw];

    ret.probs.reserve(static_cast<std::size_t>(classes));

    for (int i{0}; i < classes; ++i)
    {
        float prob{in.data[base + j + (5 + i) * hw] * ret.objectness};
        ret.probs.push_back((prob > threshold) ? prob : 0.0f);
    }

    return ret;
    
} 

namespace lithium
{
    void yolo(const NetworkLayer& layer, const Tensor& in, Tensor& out)
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
        return;
    }

    std::vector<Decoded> decode(const Network& network, float threshold)
    {
        std::vector<Decoded> vec{};

        for (std::size_t i{0}; i < network.layers.size(); ++i)
        {
            const LayerSpec& spec{network.layers[i].spec};
            if (spec.type != LayerSpec::LayerType::Yolo)
            {
                continue;
            }

            const Tensor& in{network.outputs[i]};
            int hw{in.h * in.w};
            int entries{5 + spec.yolo_classes};

            for (std::size_t n{0}; n < spec.yolo_mask.size(); ++n)
            {
                auto anchor{spec.yolo_anchors[static_cast<std::size_t>(spec.yolo_mask[n])]};
                int base{static_cast<int>(n) * entries * hw};

                for (int j{0}; j < hw; ++j)
                {
                    if (in.data[base + j + 4 * hw] < threshold)
                    {
                        continue;
                    }
                    vec.push_back(create(in, base, j, spec.yolo_classes, anchor,
                        network.net.width, network.net.height, threshold));
                }
            }
        }
        return vec;
    }
}
