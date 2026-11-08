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

    struct Network
    {
        NetConfig net{};
        std::pmr::memory_resource* resource{};   
        std::vector<NetworkLayer> layers{};      
        std::vector<Tensor> outputs{};
        std::pmr::vector<act_t> storage{};       
    
        Tensor output(std::size_t layer) const;
    };
    
    std::expected<Network, network_error> build_network(const ParsedCfg&, ParsedWeights&&);
    bool fuses_maxpool(const Network&, std::size_t layer);
    void forward(Network&, Backend&, const Tensor& input, bool fuse = true);

}

