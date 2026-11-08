#pragma once

#include "backend.hpp"

namespace lithium
{
    struct CPUBackend : Backend
    {
        void conv(const NetworkLayer& layer, const Tensor& in, Tensor& out) override;
        void maxpool(const NetworkLayer& layer, const Tensor& in, Tensor& out) override;
        void conv_maxpool(const NetworkLayer& conv, const NetworkLayer& pool,
                          const Tensor& in, Tensor& conv_out, Tensor& pool_out) override;
        void upsample(const Tensor& in, Tensor& out, int stride) override;
        void concat(const std::vector<Tensor>& ins, Tensor& out) override;
        void yolo(const NetworkLayer& layer, const Tensor& in, Tensor& out) override;
        void download(const Tensor& device, float* host) override;
        void sync() override;
    };
}
