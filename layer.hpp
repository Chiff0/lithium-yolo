#pragma once


#include "tensor.hpp"
#include <utility>
#include <vector>


namespace lithium 
{
    enum class Activation
    {
        Leaky,
        Linear, 
    };


    struct LayerSpec
    {
        enum class LayerType
        {
            Conv,
            Maxpool,
            Route, 
            Upsample,
            Yolo,
        };

        
        LayerType type{LayerType::Conv};
        int filters{16}, 
            size{3},
            pad{1},
            stride{1};
        bool batch_norm{};
        Activation activation{Activation::Leaky};
        std::vector<int> route_layers{};
        int route_groups{1},
            route_group_id{}; //both used for yolov4
        std::vector<int> yolo_mask{};
        std::vector<std::pair<float, float>> yolo_anchors{};
        int yolo_classes{80};
        Tensor out{};
    };

    struct NetConfig
    {
        int letter_box{};
        int batch{};
        int subdivisions{};
        int width{};
        int height{};
        int channels{};
    };
}