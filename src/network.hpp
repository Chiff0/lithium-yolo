#pragma once

#include "backend.hpp"
#include "parsers.hpp"
#include "layer.hpp"
#include <expected>
#include <memory_resource>




namespace lithium 
{
    enum class network_error
    {
        size_not_equal,
        buffer_overlap,
    };

    struct Storage
    {
        std::pmr::memory_resource* resource{nullptr};
        act_t* data{nullptr};
        std::size_t count{0};

        Storage() = default;
        Storage(std::pmr::memory_resource* resource, std::size_t count);
        Storage(Storage&& other) noexcept;
        Storage& operator=(Storage&& other) noexcept;
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
        ~Storage();
    };

    struct Network
    {
        NetConfig net{};
        std::pmr::memory_resource* resource{};   
        std::vector<NetworkLayer> layers{};      
        std::vector<Tensor> outputs{};
        Storage storage{};
        int image_channels{};
    
        Tensor output(std::size_t layer) const;
    };
    
    std::expected<Network, network_error> build_network(const ParsedCfg&, ParsedWeights&&,
                                                        std::pmr::memory_resource* activations = nullptr);
    bool fuses_maxpool(const Network&, std::size_t layer);
    void forward(Network&, Backend&, const Tensor& input, bool fuse = true);

}

