#include "network.hpp"
#include "preprocess.hpp"
#include "yolo.hpp"
#include "cpu_backend.cpp"

#include <iostream>

namespace lithium
{
    enum class error
    {
        cfg_error,
        weights_error,
        network_error,
        preprocessing_error,
    };

    std::expected<std::vector<Decoded>, error> run_inference(int argc, char** argv)
    {
        
        auto parsed_cfg{parse_cfg(static_cast<std::string_view>(argv[1]))};
        if (!parsed_cfg)
        {
            std::cout<< "Error while parsing CFG";
            return std::unexpected(error::cfg_error);
        }
        auto parsed_weights{parse_weights(static_cast<std::string_view>(argv[2]), *parsed_cfg)};
        if (!parsed_weights)
        {
            std::cout<< "Error while parsing .weights file";
            return std::unexpected(error::weights_error);
        }
        auto network{build_network(*parsed_cfg, std::move(*parsed_weights))};
        if (!network)
        {
            std::cout<< "Error while creating network";
            return std::unexpected(error::network_error);
        }

        Tensor input{};
        auto net_c{(*parsed_cfg).net};
        input.h = net_c.height; 
        input.w = net_c.width;
        input.c = net_c.channels; 
        std::pmr::vector<float> pixels(3 * input.h * input.w, network -> resource);
        input.data = pixels.data();


        auto loaded{preprocess(static_cast<std::string_view>(argv[3]), input)};
        if (!loaded)
        {
            std::cout<< "Error while preprocessing";
            return std::unexpected(error::preprocessing_error);
        }

        CPUBackend backend{};
        forward(*network, backend, input);


        auto predictions{decode(*network)};
        nms(predictions);
        revert_sizes(predictions, (*loaded).image_w, (*loaded).image_h, net_c.width, net_c.height);
        

        return predictions;


    }

}