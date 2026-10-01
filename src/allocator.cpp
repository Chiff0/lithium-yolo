#include "allocator.hpp"
#include <cuda_runtime.h>
#include <new>
#include <cstddef>

namespace lithium
{
    void* cuda_resource::do_allocate(std::size_t bytes, std::size_t)
    {
        void* p{};
        if (cudaMallocManaged(&p, bytes) != cudaSuccess)
            throw std::bad_alloc{};
        return p;
    }

    void cuda_resource::do_deallocate(void* p, std::size_t, std::size_t)
    {
        cudaFree(p);
    }

    bool cuda_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept
    {
        return this == &other;
    }

}