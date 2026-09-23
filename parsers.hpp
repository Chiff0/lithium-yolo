#pragma once

#include "layer.hpp"
#include <iostream>
#include <expected>
#include <cstdint>
#include <string_view>

namespace lithium
{
    enum class parse_error
    {
        invalid_input,
        file_input_error,
    };
    struct ParsedCfg
    {
        NetConfig net;
        std::vector<LayerSpec> layers;
    };

    // one layer's slice of the .weights blob; only convolutional layers carry any,
    // and scales/rolling_* are filled in only when the layer is batch normalised
    struct LayerWeights
    {
        std::vector<float> biases;            // filters
        std::vector<float> scales;            // filters
        std::vector<float> rolling_mean;      // filters
        std::vector<float> rolling_variance;  // filters
        std::vector<float> weights;           // filters * in_channels * size * size
    };

    struct ParsedWeights
    {
        int major{}, minor{}, revision{};
        std::uint64_t seen{};
        std::vector<LayerWeights> layers;  // parallel to ParsedCfg::layers
    };
    
    std::expected<ParsedCfg, parse_error> parse_cfg(std::string_view path);  
    std::expected<ParsedWeights, parse_error> parse_weights(std::string_view path, const ParsedCfg& cfg);
}
