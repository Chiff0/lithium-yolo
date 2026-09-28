#pragma once

#include "network.hpp"

#include <vector>




namespace lithium
{
    struct Decoded
    {
        float x{};
        float y{};
        float w{};
        float h{};
        float objectness{};
        std::vector<float> probs{};
    };

    void yolo(const NetworkLayer& layer, const Tensor& in, Tensor& out);
    std::vector<Decoded> decode(const Network& network, float threshold = 0.25f);
    void nms(std::vector<Decoded>& predictions, float iou_threshold = 0.45f);

}
