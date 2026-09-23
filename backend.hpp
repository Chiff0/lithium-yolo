#include "tensor.hpp"
#include "layer.hpp"

namespace lithium
{
    struct Backend {
        virtual ~Backend() = default;
        virtual void conv(const LayerSpec&, const Tensor& in, Tensor& out) = 0;
        virtual void maxpool(const LayerSpec&, const Tensor& in, Tensor& out) = 0;
        virtual void upsample(const Tensor& in, Tensor& out, int stride) = 0;
        virtual void concat(const std::vector<Tensor>& ins, Tensor& out) = 0;
        virtual void download(const Tensor&, float* host) = 0;
    };
}