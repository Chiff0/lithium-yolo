#pragma once

#include "backend.hpp"

#include <cstddef>

struct cublasContext;

namespace lithium
{
    struct GPUBackend : Backend
    {
        void conv(const NetworkLayer& layer, const Tensor& in, Tensor& out) override;
        void maxpool(const NetworkLayer& layer, const Tensor& in, Tensor& out) override;
        void upsample(const Tensor& in, Tensor& out, int stride) override;
        void concat(const std::vector<Tensor>& ins, Tensor& out) override;
        void yolo(const NetworkLayer& layer, const Tensor& in, Tensor& out) override;
        void download(const Tensor& device, float* host) override;
        void sync() override;

        ~GPUBackend() override;

    private:
        float* workspace{nullptr};
        cublasContext* handle{nullptr};
        std::size_t workspace_floats{0};
        void reserve_workspace(std::size_t floats);
        void check_handle();
    };
}