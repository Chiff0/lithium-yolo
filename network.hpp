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
    };

    struct Network
    {
        NetConfig net{};
        std::pmr::memory_resource* resource{};   
        std::vector<NetworkLayer> layers{};      
        std::vector<Tensor> outputs{};
        std::pmr::vector<float> storage{};       
    
        Tensor output(std::size_t layer) const;
    };
    
    std::expected<Network, network_error> build_network(const ParsedCfg&, ParsedWeights&&);
    void forward(Network&, Backend&, const Tensor& input);

}

