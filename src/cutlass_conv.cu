#include "cutlass_conv.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
using Layout = cutlass::layout::TensorNHWC;

namespace lithium
{
    struct CutlassConvs
    {
        struct Plan
        {
            const void* key{nullptr};
            bool usable{false};
            ImplicitGemm op{};
            void* workspace{nullptr};
            const void* in{nullptr};
            const void* out{nullptr};
        };
        std::vector<std::unique_ptr<Plan>> plans{};
    };

    static ImplicitGemm::Arguments arguments(const NetworkLayer& layer, const Tensor& in, Tensor& out,
                                             const void* weights)
    {
        const int k{layer.spec.size};
        const int pad{layer.spec.pad};
        const int stride{layer.spec.stride};

        cutlass::conv::Conv2dProblemSize problem(
            {1, in.h, in.w, in.c},
            {out.c, k, k, in.c},
            {pad, pad, pad, pad},
            {stride, stride},
            {1, 1},
            {1, out.h, out.w, out.c},
            cutlass::conv::Mode::kCrossCorrelation,
            1);

        auto* x{reinterpret_cast<cutlass::half_t*>(in.data)};
        auto* w{reinterpret_cast<cutlass::half_t*>(const_cast<void*>(weights))};
        auto* y{reinterpret_cast<cutlass::half_t*>(out.data)};

        return {
            problem,
            {x, Layout::packed({1, in.h, in.w, in.c})},
            {w, Layout::packed({out.c, k, k, in.c})},
            {y, Layout::packed({1, out.h, out.w, out.c})},
            {y, Layout::packed({1, out.h, out.w, out.c})},
            {1.0f, 0.0f}};
    }

    CutlassConvs* cutlass_create()
    {
        return new CutlassConvs{};
    }

    void cutlass_destroy(CutlassConvs* convs)
    {
        if (convs == nullptr)
        {
            return;
        }
        for (const auto& plan : convs->plans)
        {
            if (plan->workspace != nullptr)
            {
                cudaFree(plan->workspace);
            }
        }
        delete convs;
    }

    bool cutlass_conv(CutlassConvs* convs, const NetworkLayer& layer, const Tensor& in, Tensor& out,
                      const void* weights)
    {
        CutlassConvs::Plan* plan{nullptr};
        for (const auto& candidate : convs->plans)
        {
            if (candidate->key == weights)
            {
                plan = candidate.get();
                break;
            }
        }

        const ImplicitGemm::Arguments args{arguments(layer, in, out, weights)};

        if (plan == nullptr)
        {
            convs->plans.push_back(std::make_unique<CutlassConvs::Plan>());
            plan = convs->plans.back().get();
            plan->key = weights;
            plan->usable = ImplicitGemm::can_implement(args) == cutlass::Status::kSuccess;

            if (plan->usable)
            {
                const std::size_t bytes{plan->op.get_workspace_size(args)};
                if (bytes > 0 && cudaMalloc(&plan->workspace, bytes) != cudaSuccess)
                {
                    std::fprintf(stderr, "cutlass workspace alloc (%zu bytes) failed\n", bytes);
                    std::abort();
                }
                const cutlass::Status status{plan->op.initialize(args, plan->workspace)};
                if (status != cutlass::Status::kSuccess)
                {
                    std::fprintf(stderr, "cutlass initialize: %s\n", cutlassGetStatusString(status));
                    std::abort();
                }
                plan->in = in.data;
                plan->out = out.data;
            }

            std::fprintf(stderr, "conv %4dx%3dx%3d -> %4d k%d: %s\n", in.c, in.h, in.w, out.c,
                         layer.spec.size, plan->usable ? "cutlass" : "im2col + cublas");
        }

        if (!plan->usable)
        {
            return false;
        }

        if (plan->in != in.data || plan->out != out.data)
        {
            const cutlass::Status status{plan->op.update(args, plan->workspace)};
            if (status != cutlass::Status::kSuccess)
            {
                std::fprintf(stderr, "cutlass update: %s\n", cutlassGetStatusString(status));
                std::abort();
            }
            plan->in = in.data;
            plan->out = out.data;
        }

        const cutlass::Status status{plan->op()};
        if (status != cutlass::Status::kSuccess)
        {
            std::fprintf(stderr, "cutlass launch: %s\n", cutlassGetStatusString(status));
            std::abort();
        }
        return true;
    }
}
