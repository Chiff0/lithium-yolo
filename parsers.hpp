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


    struct ParsedWeights
    {
        int major{}, minor{}, revision{};
        std::uint64_t seen{};


        std::pmr::vector<LayerWeights> layers;

        explicit ParsedWeights(std::pmr::memory_resource* resource) : layers(resource) {}
    };

    std::expected<ParsedCfg, parse_error> parse_cfg(std::string_view path);

    std::expected<ParsedWeights, parse_error> parse_weights(
        std::string_view path,
        const ParsedCfg& cfg,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource());
}
