#include "network.hpp"


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

        Network network{{}, weights.layers.get_allocator().resource(), {}, {}, std::pmr::vector<float>{weights.layers.get_allocator().resource()}};
        network.net = cfg.net; 
        network.layers.reserve(cfg.layers.size());
        network.outputs.reserve(cfg.layers.size());
        int prev_h{cfg.net.height}, prev_w{cfg.net.width}, prev_c{cfg.net.channels};
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


            sum += tensor.count();
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
            offset += tensor.count();
        }

        if (offset != network.storage.size())
        {
            return std::unexpected(network_error::buffer_overlap);
        }


        return network;
    }

    void forward(Network& network, Backend& backend, const Tensor& input)
    {
        for (std::size_t i{0}; i < network.layers.size(); ++i)
        {
            const auto& layer{network.layers[i]};
            const Tensor& src{(i == 0) ? input : network.outputs[i - 1]};

            switch(layer.spec.type)
            {
                case LayerSpec::LayerType::Conv:
                {
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
                    yolo(layer, src, network.outputs[i]);
                    break;
                }
            }
        }
    }





}