#include "backend.hpp"
#include <cfloat>
#include <cmath>



static std::vector<float> im2col(
                                const lithium::Tensor& input, const lithium::Tensor& out, 
                                int pad, int stride, int h, int w
                                )
{
    std::vector<float> arr{};
    int nx{out.w},
        ny{out.h};
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
    const float* in, lithium::Tensor& out, const float* weights, 
    int patch_len
)
{
    const int patches{out.h * out.w};
    for (int f{0}; f < out.c; ++f)
    {
        for (int p{0}; p < patches; ++p)
        {
            float sum{0.0f};
            for (int i{0}; i < patch_len; ++i)
            {
                sum += weights[f * patch_len + i] * in[p * patch_len + i];
            }
            out.data[f * patches + p] = sum;
        }
    }
    return;
}

static void batch_normalize(lithium::Tensor& out, 
                            const lithium::LayerWeights& weights
)
{
    std::size_t size_per_channel{static_cast<std::size_t>(out.h) * out.w};
    for (auto i{0}; i < out.c; ++i)
    {
        auto dev = static_cast<float>(std::sqrt(static_cast<double>(weights.rolling_variance[i] + 1e-5)));
        auto offset{size_per_channel * i};
        auto scale{weights.scales[i]}, bias{weights.biases[i]}, mean{weights.rolling_mean[i]};
        for(auto j{offset}; j < offset + size_per_channel; ++j)
        {
            // I do all 3 there, why not 
            out.data[j] = (out.data[j] - mean) / dev; //normalize
            out.data[j] *= scale; // scale
            out.data[j] += bias; // add bias
            
        }
    }
}




static void add_bias(lithium::Tensor& out, 
                    const lithium::LayerWeights& weights
)
{
    std::size_t size_per_channel{static_cast<std::size_t>(out.h) * out.w};
    for (auto i{0}; i < out.c; ++i)
    {
        auto offset{size_per_channel * i};
        auto bias{weights.biases[i]};
        for (auto j{offset}; j < offset + size_per_channel; ++j)
        {
            out.data[j] += bias;
        }
    }
}




static float leaky(float x)
{
    return (x > 0) ? x : x * 0.1f; 
}

[[maybe_unused]] static float linear(float x)
{
    return x;
}

static void apply_activation(lithium::Tensor& out, lithium::Activation activation)
{
    switch(activation)
    {
        case lithium::Activation::Leaky:
        for (std::size_t i{0}; i < out.count(); ++i)
        {
            out.data[i] = leaky(out.data[i]);
        }
        break;
        case lithium::Activation::Linear:
        break;
    }
}

static void maxpool(
                   const lithium::Tensor& input, lithium::Tensor& out, 
                   int pad, int stride, int h, int w
                   )
{
    int nx{out.w},
        ny{out.h};

    std::size_t counter{0};
    for (int i{0}; i < out.c; ++i)
    {
        for (int j{0}; j < ny; ++j)
        {
            for (int k{0}; k < nx; ++k)
            {
                float hi{-FLT_MAX};
                for (int pos{0}; pos < h * w; ++pos)
                {
                    int x{pos % w + k * stride - pad / 2},
                        y{pos / w + j * stride - pad / 2};

                    float curr{(x < 0 || y < 0 || x >= input.w || y >= input.h)
                        ? -FLT_MAX
                        : input.data[input.index(i, x, y)]};
                    if (curr > hi)
                        hi = curr;
                }
                out.data[counter++] = hi;
            }
        }
    }
}

static void upsample(
                    const lithium::Tensor& in, lithium::Tensor& out,
                    int stride
                    )
{
    for (int i{0}; i < out.c; ++i)
    {
        for (int j{0}; j < out.h; ++j)
        {
            for (int k{0}; k < out.w; ++k)
            {
                out.data[out.index(i, k, j)] = in.data[in.index(i, k / stride, j / stride)];
            }
        }
    }

    return;
}

static void concat(const std::vector<lithium::Tensor>& ins, lithium::Tensor& out)
{
    int offset{0};
    for (auto& tensor : ins)
    {
        int len{static_cast<int>(tensor.count())};
        std::copy(tensor.data, tensor.data + len, out.data + offset);
        offset += len;
    }

    return;
}


namespace lithium 
{
    struct CPUBackend : Backend
    {
        void conv(const NetworkLayer& layer, const Tensor& in, Tensor& out) override
        {
            auto cols{im2col(
                in, out, 
                layer.spec.pad, layer.spec.stride, 
                layer.spec.size, layer.spec.size
            )};
            gemm(cols.data(), out, layer.weights.weights.data(), 
                in.c * layer.spec.size * layer.spec.size);
            if (layer.spec.batch_norm)
            {
                batch_normalize(out, layer.weights);
            }
            else
            {
                add_bias(out, layer.weights);
            }
            apply_activation(out, layer.spec.activation);
        }

        void maxpool(const NetworkLayer& layer, const Tensor& in, Tensor& out) override
        {
            ::maxpool(in, out, layer.spec.pad, layer.spec.stride, 
                layer.spec.size, layer.spec.size);
        }
        void upsample(const Tensor& in, Tensor& out, int stride) override
        {
            ::upsample(in, out, stride);
        }
        void concat(const std::vector<Tensor>& ins, Tensor& out) override 
        {
            ::concat(ins, out);
        }
        void download(const Tensor& device, float* host) override
        {
            std::copy(device.data, device.data + device.count(), host);
        }


    };
}

