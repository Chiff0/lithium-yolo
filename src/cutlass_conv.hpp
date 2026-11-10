#pragma once

#include "layer.hpp"
#include "tensor.hpp"

namespace lithium
{
    struct CutlassConvs;

    CutlassConvs* cutlass_create();
    void cutlass_destroy(CutlassConvs* convs);
    bool cutlass_conv(CutlassConvs* convs, const NetworkLayer& layer, const Tensor& in, Tensor& out,
                      const void* weights);
}
