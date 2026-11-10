#include "cutlass_conv.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using Layout = cutlass::layout::TensorNHWC;

template <int TbM, int TbN, int TbK, int WM, int WN, int WK, int Stages>
using ConvFor = cutlass::conv::device::ImplicitGemmConvolution<
    typename cutlass::conv::kernel::DefaultConv2dFprop<
        cutlass::half_t,
        Layout,
        cutlass::half_t,
        Layout,
        cutlass::half_t,
        Layout,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TbM, TbN, TbK>,
        cutlass::gemm::GemmShape<WM, WN, WK>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized
    >::Kernel>;

namespace lithium
{
    namespace
    {
        template <class Conv>
        typename Conv::Arguments arguments(const NetworkLayer& layer, const Tensor& in, Tensor& out,
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

        void check(cutlass::Status status, const char* what)
        {
            if (status != cutlass::Status::kSuccess)
            {
                std::fprintf(stderr, "cutlass %s: %s\n", what, cutlassGetStatusString(status));
                std::abort();
            }
        }

        struct Runner
        {
            virtual ~Runner() = default;
            virtual void update(const NetworkLayer& layer, const Tensor& in, Tensor& out,
                                const void* weights) = 0;
            virtual void run() = 0;
            const char* name{nullptr};
        };

        template <class Conv>
        struct RunnerFor final : Runner
        {
            Conv op{};
            void* workspace{nullptr};

            ~RunnerFor() override
            {
                if (workspace != nullptr)
                {
                    cudaFree(workspace);
                }
            }

            void update(const NetworkLayer& layer, const Tensor& in, Tensor& out,
                        const void* weights) override
            {
                check(op.update(arguments<Conv>(layer, in, out, weights), workspace), "update");
            }

            void run() override
            {
                check(op(), "launch");
            }
        };

        template <class Conv>
        std::unique_ptr<Runner> make(const char* name, const NetworkLayer& layer, const Tensor& in,
                                     Tensor& out, const void* weights)
        {
            const typename Conv::Arguments args{arguments<Conv>(layer, in, out, weights)};
            if (Conv::can_implement(args) != cutlass::Status::kSuccess)
            {
                return nullptr;
            }

            auto runner{std::make_unique<RunnerFor<Conv>>()};
            runner->name = name;
            const std::size_t bytes{runner->op.get_workspace_size(args)};
            if (bytes > 0 && cudaMalloc(&runner->workspace, bytes) != cudaSuccess)
            {
                std::fprintf(stderr, "cutlass workspace alloc (%zu bytes) failed\n", bytes);
                std::abort();
            }
            if (runner->op.initialize(args, runner->workspace) != cutlass::Status::kSuccess)
            {
                return nullptr;
            }
            return runner;
        }

        using Factory = std::unique_ptr<Runner> (*)(const char*, const NetworkLayer&, const Tensor&,
                                                   Tensor&, const void*);

        struct Candidate
        {
            const char* name;
            Factory make;
        };

        const Candidate candidates[]{
            {"128x128x32/5", &make<ConvFor<128, 128, 32, 64, 64, 32, 5>>},
            {"256x64x32/4",  &make<ConvFor<256, 64, 32, 64, 64, 32, 4>>},
            {"128x64x32/6",  &make<ConvFor<128, 64, 32, 64, 32, 32, 6>>},
            {"64x128x32/6",  &make<ConvFor<64, 128, 32, 32, 64, 32, 6>>},
            {"64x64x32/10",  &make<ConvFor<64, 64, 32, 32, 32, 32, 10>>},
            {"128x32x32/4",  &make<ConvFor<128, 32, 32, 64, 32, 32, 4>>},
            {"256x32x32/4",  &make<ConvFor<256, 32, 32, 64, 32, 32, 4>>},
        };

        float time_ms(Runner& runner)
        {
            for (int i{0}; i < 3; ++i)
            {
                runner.run();
            }
            cudaEvent_t start{}, stop{};
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            constexpr int reps{10};
            cudaEventRecord(start);
            for (int i{0}; i < reps; ++i)
            {
                runner.run();
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms{};
            cudaEventElapsedTime(&ms, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return ms / reps;
        }
    }

    struct CutlassConvs
    {
        struct Plan
        {
            const void* key{nullptr};
            std::unique_ptr<Runner> runner{};
            const void* in{nullptr};
            const void* out{nullptr};
        };
        std::vector<std::unique_ptr<Plan>> plans{};
    };

    CutlassConvs* cutlass_create()
    {
        return new CutlassConvs{};
    }

    void cutlass_destroy(CutlassConvs* convs)
    {
        delete convs;
    }

    bool cutlass_conv(CutlassConvs* convs, const NetworkLayer& layer, const Tensor& in, Tensor& out,
                      const void* weights)
    {
        CutlassConvs::Plan* plan{nullptr};
        for (const auto& existing : convs->plans)
        {
            if (existing->key == weights)
            {
                plan = existing.get();
                break;
            }
        }

        if (plan == nullptr)
        {
            convs->plans.push_back(std::make_unique<CutlassConvs::Plan>());
            plan = convs->plans.back().get();
            plan->key = weights;

            std::vector<std::unique_ptr<Runner>> runners{};
            for (const Candidate& candidate : candidates)
            {
                std::unique_ptr<Runner> runner{candidate.make(candidate.name, layer, in, out, weights)};
                if (runner != nullptr)
                {
                    runners.push_back(std::move(runner));
                }
            }

            std::vector<float> times(runners.size(), 0.0f);
            for (int round{0}; round < 2; ++round)
            {
                for (std::size_t i{0}; i < runners.size(); ++i)
                {
                    const float ms{time_ms(*runners[i])};
                    times[i] = (round == 0) ? ms : std::min(times[i], ms);
                }
            }

            float best_ms{0.0f};
            for (std::size_t i{0}; i < runners.size(); ++i)
            {
                if (plan->runner == nullptr || times[i] < best_ms)
                {
                    best_ms = times[i];
                    plan->runner = std::move(runners[i]);
                }
            }
            plan->in = in.data;
            plan->out = out.data;

            if (plan->runner != nullptr)
            {
                std::fprintf(stderr, "conv %4dx%3dx%3d -> %4d k%d: cutlass %s (%.3f ms)\n", in.c, in.h,
                             in.w, out.c, layer.spec.size, plan->runner->name, best_ms);
            }
            else
            {
                std::fprintf(stderr, "conv %4dx%3dx%3d -> %4d k%d: im2col + cublas\n", in.c, in.h, in.w,
                             out.c, layer.spec.size);
            }
        }

        if (plan->runner == nullptr)
        {
            return false;
        }

        if (plan->in != in.data || plan->out != out.data)
        {
            plan->runner->update(layer, in, out, weights);
            plan->in = in.data;
            plan->out = out.data;
        }

        plan->runner->run();
        return true;
    }
}
