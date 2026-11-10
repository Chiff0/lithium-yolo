#include "network.hpp"

#include <cmath>


#ifdef LITHIUM_CUTLASS
static constexpr int input_channel_multiple{8};
#else
static constexpr int input_channel_multiple{1};
#endif

static std::size_t aligned(std::size_t count)
{
    return (count + 63) / 64 * 64;
}

static std::pair<int, int> conv_dim(int h, int w, int pad, int size, int stride)
{
    int new_h = (h + 2 * pad - size) / stride + 1;
    int new_w = (w + 2 * pad - size) / stride + 1;
    return {new_h, new_w};
}

static std::pair<int, int> maxpool_dim(int h, int w, int pad, int size, int stride)
{
    int new_h = (h + pad - size) / stride + 1;
    int new_w = (w + pad - size) / stride + 1;
    return {new_h, new_w};
}

static std::pair<int, int> route_dim(const std::vector<lithium::Tensor>& outputs,
                                     const std::vector<int>& sources)
{
    return {outputs[sources[0]].h, outputs[sources[0]].w};
}

static int route_channels(const std::vector<lithium::Tensor>& outputs,
                          const std::vector<int>& sources, int groups)
{
    int sum{0};
    for (int source : sources)
    {
        sum += outputs[source].c;
    }
    return sum / groups;
}

static std::pair<int, int> upsample_dim(int h, int w, int stride)
{
    return {h * stride, w * stride};
}




namespace lithium
{
    std::expected<Network, network_error> build_network(const ParsedCfg& cfg, ParsedWeights&& weights)
    {
        if (cfg.layers.size() != weights.layers.size())
        {
            return std::unexpected(network_error::size_not_equal);
        }

        Network network{{}, weights.layers.get_allocator().resource(), {}, {}, std::pmr::vector<act_t>{weights.layers.get_allocator().resource()}};
        network.net = cfg.net; 
        network.image_channels = cfg.net.channels;
        network.net.channels = (cfg.net.channels + input_channel_multiple - 1)
                             / input_channel_multiple * input_channel_multiple;
        network.layers.reserve(cfg.layers.size());
        network.outputs.reserve(cfg.layers.size());
        int prev_h{cfg.net.height}, prev_w{cfg.net.width}, prev_c{network.net.channels};
        int sum{0};
        for (std::size_t i{0}; i < cfg.layers.size(); ++i)
        {
            const auto& layer{cfg.layers[i]};
            network.layers.push_back({layer, std::move(weights.layers[i])});
            
            Tensor tensor{};

            switch (layer.type)
            {
                case LayerSpec::LayerType::Conv:
                {
                    auto pair{conv_dim(prev_h, prev_w, layer.pad, layer.size, layer.stride)};
                    tensor.h = pair.first;
                    tensor.w = pair.second;
                    tensor.c = layer.filters;
                    break;
                }
                case LayerSpec::LayerType::Maxpool:
                {
                    auto pair{maxpool_dim(prev_h, prev_w, layer.pad, layer.size, layer.stride)};
                    tensor.h = pair.first;
                    tensor.w = pair.second;
                    tensor.c = prev_c;
                    break;
                }
                case LayerSpec::LayerType::Upsample:
                {
                    auto pair{upsample_dim(prev_h, prev_w, layer.stride)};
                    tensor.h = pair.first;
                    tensor.w = pair.second;
                    tensor.c = prev_c;
                    break;
                }
                case LayerSpec::LayerType::Route:
                {
                    auto pair{route_dim(network.outputs, layer.route_layers)};
                    tensor.h = pair.first;
                    tensor.w = pair.second;
                    tensor.c = route_channels(network.outputs, layer.route_layers,
                                              layer.route_groups);
                    break;
                }
                case LayerSpec::LayerType::Yolo:
                {
                    tensor.h = prev_h;
                    tensor.w = prev_w;
                    tensor.c = prev_c;
                    break;
                }
            }


            sum += static_cast<int>(aligned(tensor.count()));
            network.outputs.push_back(tensor);
            prev_h = tensor.h;
            prev_w = tensor.w;
            prev_c = tensor.c;

        }
        network.storage.resize(sum);
        std::size_t offset{0};

        for (Tensor& tensor : network.outputs)
        {
            tensor.data = network.storage.data() + offset;
            offset += aligned(tensor.count());
        }

        if (offset != network.storage.size())
        {
            return std::unexpected(network_error::buffer_overlap);
        }

        for (auto& layer : network.layers)
        {
            if (!layer.spec.batch_norm)
                continue;

            std::size_t block_size{static_cast<std::size_t>(layer.weights.weights.size() / layer.spec.filters)};

            for (std::size_t i{}; i < static_cast<std::size_t>(layer.spec.filters); ++i)
            {
                float mul = layer.weights.scales[i] / std::sqrt(layer.weights.rolling_variance[i] + 1e-5f);
                for (std::size_t j{}; j < block_size; ++j)
                {
                    std::size_t idx{block_size * i + j};
                    layer.weights.weights[idx] *= mul;
                }
                layer.weights.biases[i] -= layer.weights.rolling_mean[i] * mul;
            }
            
            layer.spec.batch_norm = false;
        }

        for (auto& layer : network.layers)
        {
            if (layer.spec.type != LayerSpec::LayerType::Conv || layer.weights.weights.empty())
                continue;

            const std::size_t filters{static_cast<std::size_t>(layer.spec.filters)};
            const std::size_t window{static_cast<std::size_t>(layer.spec.size) * layer.spec.size};
            const std::size_t channels{layer.weights.weights.size() / (filters * window)};
            const std::vector<float> source(layer.weights.weights.begin(), layer.weights.weights.end());

            for (std::size_t f{}; f < filters; ++f)
            {
                for (std::size_t c{}; c < channels; ++c)
                {
                    for (std::size_t pos{}; pos < window; ++pos)
                    {
                        layer.weights.weights[(f * window + pos) * channels + c] =
                            source[(f * channels + c) * window + pos];
                    }
                }
            }
        }

        if (network.net.channels != network.image_channels && !network.layers.empty()
            && network.layers.front().spec.type == LayerSpec::LayerType::Conv)
        {
            auto& weights{network.layers.front().weights.weights};
            const std::size_t from{static_cast<std::size_t>(network.image_channels)};
            const std::size_t to{static_cast<std::size_t>(network.net.channels)};
            const std::size_t groups{weights.size() / from};

            std::pmr::vector<float> padded(groups * to, 0.0f, weights.get_allocator());
            for (std::size_t g{}; g < groups; ++g)
            {
                for (std::size_t c{}; c < from; ++c)
                {
                    padded[g * to + c] = weights[g * from + c];
                }
            }
            weights = std::move(padded);
        }

        return network;
    }

    bool fuses_maxpool(const Network& network, std::size_t layer)
    {
        if (layer + 1 >= network.layers.size()
            || network.layers[layer].spec.type != LayerSpec::LayerType::Conv
            || network.layers[layer + 1].spec.type != LayerSpec::LayerType::Maxpool)
        {
            return false;
        }

        for (const auto& other : network.layers)
        {
            for (int source : other.spec.route_layers)
            {
                if (static_cast<std::size_t>(source) == layer)
                {
                    return false;
                }
            }
        }
        return true;
    }

    void forward(Network& network, Backend& backend, const Tensor& input, bool fuse)
    {
        for (std::size_t i{0}; i < network.layers.size(); ++i)
        {
            const auto& layer{network.layers[i]};
            const Tensor& src{(i == 0) ? input : network.outputs[i - 1]};

            switch(layer.spec.type)
            {
                case LayerSpec::LayerType::Conv:
                {
                    if (fuse && fuses_maxpool(network, i))
                    {
                        backend.conv_maxpool(layer, network.layers[i + 1], src,
                                             network.outputs[i], network.outputs[i + 1]);
                        ++i;
                        break;
                    }
                    backend.conv(layer, src, network.outputs[i]);
                    break;
                }
                case LayerSpec::LayerType::Maxpool:
                {
                    backend.maxpool(layer, src, network.outputs[i]);
                    break;
                }
                case LayerSpec::LayerType::Upsample:
                {
                    backend.upsample(src, network.outputs[i], layer.spec.stride);
                    break;
                }
                case LayerSpec::LayerType::Route:
                {
                    std::vector<Tensor> tensors{};
                    for (std::size_t source : layer.spec.route_layers)
                    {
                        tensors.push_back(network.outputs[source]);
                    }
                    backend.concat(tensors, network.outputs[i]);
                    break;
                }
                case LayerSpec::LayerType::Yolo:
                {
                    backend.yolo(layer, src, network.outputs[i]);
                    break;
                }
            }
        }
    }





}