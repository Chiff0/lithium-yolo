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
static float overlap(float x1, float w1, float x2, float w2)
{
    float left  = std::max(x1 - w1/2,  x2 - w2/2);
    float right = std::min(x1 + w1/2,  x2 + w2/2);
    return right - left;
}


static float IOU(const lithium::Decoded& a, const lithium::Decoded& b)
{
    float ox{overlap(a.x, a.w, b.x, b.w)}, oy{overlap(a.y, a.h, b.y, b.h)};
    if (ox <= 0.0f || oy <= 0.0f)
    {
        return 0.0f;
    }

    float intersection{ox * oy};
    float u{a.h * a.w + b.h * b.w - intersection};
    return intersection / u;
}


namespace lithium
{
    void nms(std::vector<Decoded>& predictions, float iou_threshold)
    {
        if (predictions.empty())
        {
            return;
        }

        for (std::size_t i{0}; i < predictions[0].probs.size(); ++i)
        {
            std::sort(predictions.begin(), predictions.end(),
                [i](const Decoded& a, const Decoded& b)
                {
                    return a.probs[i] > b.probs[i];
                });

            for (std::size_t j{0}; j < predictions.size(); ++j)
            {
                if (predictions[j].probs[i] == 0.0f)
                {
                    continue;
                }

                for (std::size_t k{j + 1}; k < predictions.size(); ++k)
                {
                    if (predictions[k].probs[i] == 0.0f)
                    {
                        continue;
                    }

                    if (IOU(predictions[j], predictions[k]) > iou_threshold)
                    {
                        predictions[k].probs[i] = 0.0f;
                    }
                }
            }
        }
    }

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
