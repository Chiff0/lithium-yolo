#pragma once

#include <cstddef>
#include <memory_resource>

namespace lithium
{
    class cuda_resource : public std::pmr::memory_resource
    {
        void* do_allocate(std::size_t bytes, std::size_t alignment) override;
        void do_deallocate(void* p, std::size_t, std::size_t) override;
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;
    };
}
