#pragma once
#include <cstddef>

#ifdef LITHIUM_HALF
#include <cuda_fp16.h>
#endif

namespace lithium
{
#ifdef LITHIUM_HALF
    using act_t = __half;
    inline float to_float(act_t v) { return __half2float(v); }
    inline act_t to_act(float v) { return __float2half(v); }
#else
    using act_t = float;
    inline float to_float(act_t v) { return v; }
    inline act_t to_act(float v) { return v; }
#endif

    struct Tensor
    {
        act_t* data{nullptr};
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
