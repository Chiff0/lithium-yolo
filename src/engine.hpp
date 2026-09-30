#pragma once

#include "yolo.hpp"

#include <expected>
#include <vector>


namespace lithium
{
    enum class error
    {
        cfg_error,
        weights_error,
        network_error,
        preprocessing_error,
    };

    std::expected<std::vector<Decoded>, error> run_inference(int argc, char** argv);
}
