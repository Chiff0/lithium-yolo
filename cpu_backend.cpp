#include "backend.hpp"


static std::vector<float> im2col(const lithium::Tensor& input, const lithium::Tensor& kernel)
{
    std::vector<float> arr{};
    int nx{input.w - kernel.w + 1}, ny{input.h - kernel.h + 1};
    arr.resize(static_cast<std::size_t>(kernel.h * kernel.w) * input.c * nx * ny);

    std::size_t counter{0};  
    for (int j{0}; j < ny; ++j)
    {
        for (int k{0}; k < nx; ++k)
        {
            for (int i{0}; i < input.c; ++i)
            {
                for (int pos{0}; pos < kernel.h * kernel.w; ++pos)
                {
                    //73% sure this works lol
                    int x{pos % kernel.w + k}, y{pos / kernel.w + j};
                    arr[counter++] = input.data[input.index(i, x, y)];
                }
            }
        }
    }
    return arr;
}

namespace lithium 
{
    struct Backend
    {
        void conv(const NetworkLayer&, const Tensor& in, Tensor& out)
        {
            // where tf do I get the weights from lol
            // next up gemm
        }
    };
}

