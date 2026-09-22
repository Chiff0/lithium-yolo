#include "layer.hpp"
#include <iostream>
#include <expected>

namespace lithium
{
    enum class parse_error
    {
        invalid_input,
        overflow
    };
    struct ParsedCfg
    {
        NetConfig net;
        std::vector<LayerSpec> layers;
    };
    
    std::expected<ParsedCfg, parse_error> parse_cfg(std::string_view path);  
    std::expected<std::vector<float>, parse_error> parse_weights(std::string_view path);
}
