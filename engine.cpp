#include "network.cpp"

namespace lithium
{
    std::vector<Decoded> run_inference(int argc, char** argv)
    {
        auto parsed_cfg{parse_cfg(static_cast<std::string_view>(argv[1]))};
        auto parsed_weights{parse_weights(static_cast<std::string_view>(argv[2]), parsed_cfg)};
    }

}