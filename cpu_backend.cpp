#include "backend.hpp"
#include <cmath>



static std::vector<float> im2col(
                                const lithium::Tensor& input, 
                                int pad, int stride, int w, int h
                                )
{
    std::vector<float> arr{};
    int nx{(input.w + 2 * pad - w) / stride + 1},
        ny{(input.h + 2 * pad - h) / stride + 1};
    arr.resize(static_cast<std::size_t>(h * w) * input.c * nx * ny);

    std::size_t counter{0};
    for (int j{0}; j < ny; ++j)
    {
        for (int k{0}; k < nx; ++k)
        {
            for (int i{0}; i < input.c; ++i)
            {
                for (int pos{0}; pos < h * w; ++pos)
                {
                    int x{pos % w + k * stride - pad},
                        y{pos / w + j * stride - pad};

                    arr[counter++] = (x < 0 || y < 0 || x >= input.w || y >= input.h)
                        ? 0.0f
                        : input.data[input.index(i, x, y)];
                }
            }
        }
    }
    return arr;
}

static void gemm(
                const float* in, float* out, const float* weights, 
                int filters, int patches, int patch_len
                )
{
    for (int f{0}; f < filters; ++f)
    {
        for (int p{0}; p < patches; ++p)
        {
            float sum{0.0f};
            for (int i{0}; i < patch_len; ++i)
            {
                sum += weights[f * patch_len + i] * in[p * patch_len + i];
            }
            out[f * patches + p] = sum;
        }
    }
    return;
}

static void batch_normalize(float* in, 
                            const lithium::LayerWeights& weights,
                            int len, int c // has to be spec.filters, not number of inpuit channels
                           )
{
    std::size_t size_per_channel{static_cast<std::size_t>(len / c)};
    for (auto i{0}; i < c; ++i)
    {
        auto dev = static_cast<float>(std::sqrt(static_cast<double>(weights.rolling_variance[i] + 1e-5)));
        auto offset{size_per_channel * i};
        auto scale{weights.scales[i]}, bias{weights.biases[i]}, mean{weights.rolling_mean[i]};
        for(auto j{offset}; j < offset + size_per_channel; ++j)
        {
            // I do all 3 there, why not 
            in[j] = (in[j] - mean) / dev; //normalize
            in[j] *= scale; // scale
            in[j] += bias; // add bias

        }
    }
}

namespace lithium 
{
    struct CPUBackend : Backend
    {
        void conv(const NetworkLayer&, const Tensor& in, Tensor& out) override
        {
            

        }
    };
}

