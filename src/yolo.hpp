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

    std::vector<Decoded> decode(const Network& network, float threshold = 0.25f);
    void nms(std::vector<Decoded>& predictions, float iou_threshold = 0.45f);
    void revert_sizes(std::vector<Decoded>& predictions, int image_w, int image_h,
        int net_w = 416, int net_h = 416);

}
