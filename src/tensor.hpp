#pragma once
#include <cstddef>

namespace lithium
{
    struct Tensor
    {
        float* data{nullptr};
        int c{}, h{}, w{};

        std::size_t index(int channel, int x, int y) const
        {
            return (static_cast<std::size_t>(y) * w + x) * c + channel;
        }
        std::size_t count() const
        {
            return static_cast<std::size_t>(h) * w * c;
        }
    };
}

