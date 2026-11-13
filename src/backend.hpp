#pragma once

#include "tensor.hpp"
#include "layer.hpp"

#include <functional>

namespace lithium
{
    struct PassKey
    {
        const void* network{nullptr};
        const void* input{nullptr};
        bool fuse{};
        bool operator==(const PassKey&) const = default;
    };

    struct Backend {
        virtual ~Backend() = default;
        virtual void conv(const NetworkLayer&, const Tensor& in, Tensor& out) = 0;
        virtual void maxpool(const NetworkLayer&, const Tensor& in, Tensor& out) = 0;
        virtual void conv_maxpool(const NetworkLayer& conv, const NetworkLayer& pool,
                                  const Tensor& in, Tensor& conv_out, Tensor& pool_out) = 0;
        virtual void upsample(const Tensor& in, Tensor& out, int stride) = 0;
        virtual void concat(const std::vector<Tensor>& ins, Tensor& out) = 0;
        virtual void yolo(const NetworkLayer&, const Tensor& in, Tensor& out) = 0;
        virtual void download(const Tensor&, float* host) = 0;
        virtual void sync() = 0;
        virtual void run(const PassKey&, const std::function<void()>& pass) { pass(); }
        virtual Tensor stage(const Tensor& input) { return input; }
    };
}