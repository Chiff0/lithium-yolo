#pragma once

#include "tensor.hpp"

#include <expected>
#include <string_view>


namespace lithium
{
    enum class preprocess_error
    {
        file_input_error,
        invalid_output,
    };

    struct Loaded
    {
        int image_w{};
        int image_h{};
    };

    std::expected<Loaded, preprocess_error> preprocess(std::string_view path, Tensor& out);
}
