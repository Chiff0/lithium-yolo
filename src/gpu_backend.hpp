#pragma once

#include "backend.hpp"

#include <cstddef>
#include <vector>

struct cublasContext;
struct CUstream_st;
struct CUgraphExec_st;

namespace lithium
{
    struct CutlassConvs;
}

namespace lithium
{
    struct GPUBackend : Backend
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
        void run(const PassKey& key, const std::function<void()>& pass) override;
        Tensor stage(const Tensor& input) override;

        GPUBackend();
        ~GPUBackend() override;

    private:

        struct DeviceCopy
        {
            const float* key{nullptr};
            void*        data{nullptr};
            std::size_t  count{0};
        };
        std::vector<DeviceCopy> device_copies{};
        const void* device_copy(const float* key, std::size_t count, bool half);
        const float* bias_for(const NetworkLayer& layer);

        void* input_device{nullptr};
        std::size_t input_bytes{0};
        const void* input_from{nullptr};
        bool input_on_device{false};

        void* workspace{nullptr}; //cast in the actual backend
        cublasContext* handle{nullptr};
        CutlassConvs* cutlass{nullptr};
        std::size_t workspace_floats{0};
        void reserve_workspace(std::size_t floats);
        void check_handle();
        void convolve(const NetworkLayer& layer, const Tensor& in, Tensor& out);

        const void* weights_for(const NetworkLayer& layer);

        CUstream_st* stream{nullptr};

        struct Graph
        {
            PassKey key{};
            int passes{0};
            CUgraphExec_st* exec{nullptr};
            bool failed{false};
        };
        std::vector<Graph> graphs{};

        const float* layer_0_uploaded{nullptr};
        void upload_layer_0(const NetworkLayer& layer);
    };
}