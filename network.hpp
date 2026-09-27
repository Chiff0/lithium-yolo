#include "backend.hpp"
#include "parsers.hpp"
#include "layer.hpp"
#include <expected>
#include <memory_resource>




namespace lithium 
{
    struct Network
    {
        NetConfig net{};
        std::pmr::memory_resource* resource{};   
        std::vector<NetworkLayer> layers{};      
        std::vector<Tensor> activations{};
        std::pmr::vector<float> storage{};       
    
        Tensor output(std::size_t layer) const;
    };
    
    std::expected<Network, parse_error> build_network(const ParsedCfg&, ParsedWeights&&);
    void forward(Network&, Backend&, const Tensor& input);

}

