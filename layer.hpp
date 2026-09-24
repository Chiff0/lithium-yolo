#pragma once


#include "tensor.hpp"
#include <memory_resource>
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

    struct LayerWeights
    {
        using allocator_type = std::pmr::polymorphic_allocator<>;

        explicit LayerWeights(allocator_type alloc = {})
            : biases(alloc)
            , scales(alloc)
            , rolling_mean(alloc)
            , rolling_variance(alloc)
            , weights(alloc)
        {
        }

        LayerWeights(const LayerWeights& other, allocator_type alloc = {})
            : biases(other.biases, alloc)
            , scales(other.scales, alloc)
            , rolling_mean(other.rolling_mean, alloc)
            , rolling_variance(other.rolling_variance, alloc)
            , weights(other.weights, alloc)
        {
        }

        LayerWeights(LayerWeights&&) = default;
        LayerWeights(LayerWeights&& other, allocator_type alloc)
            : biases(std::move(other.biases), alloc)
            , scales(std::move(other.scales), alloc)
            , rolling_mean(std::move(other.rolling_mean), alloc)
            , rolling_variance(std::move(other.rolling_variance), alloc)
            , weights(std::move(other.weights), alloc)
        {
        }

        LayerWeights& operator=(const LayerWeights&) = default;
        LayerWeights& operator=(LayerWeights&&) = default;
        ~LayerWeights() = default;

        std::pmr::memory_resource* resource() const { return biases.get_allocator().resource(); }

        std::pmr::vector<float> biases;            // filters
        std::pmr::vector<float> scales;            // filters
        std::pmr::vector<float> rolling_mean;      // filters
        std::pmr::vector<float> rolling_variance;  // filters
        std::pmr::vector<float> weights;           // filters * in_channels * size * size
    };

    struct NetworkLayer
    {
        LayerSpec spec{};
        LayerWeights weights{};
        Tensor out{};  
    };
}