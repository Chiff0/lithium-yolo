cmake_minimum_required(VERSION 3.22)

project(lithium LANGUAGES CXX)

option(LITHIUM_CUDA       "Build the CUDA backend"                              OFF)
option(LITHIUM_NATIVE     "Tune for the building machine (-march=native)"       OFF)
option(LITHIUM_ASAN       "Address + UB sanitizers, for the test build"         OFF)


option(LITHIUM_FAST_MATH  "Relax IEEE float semantics (-ffast-math)"            OFF)


option(LITHIUM_TF32       "Allow TF32 tensor-core math for FP32 GEMMs"          OFF)


option(LITHIUM_FP16       "Store the GEMM's inputs as FP16 (FP32 accumulate)"  OFF)


if(LITHIUM_FP16 AND LITHIUM_TF32)
    message(FATAL_ERROR
        "LITHIUM_FP16 and LITHIUM_TF32 are mutually exclusive. TF32 is a math "
        "mode for FP32 GEMM inputs; with LITHIUM_FP16 the inputs are __half, "
        "so the TF32 setting has no effect other than to make the startup "
        "banner lie about which path is running. Pick one.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# nvcc 12.x accepts no higher than C++20, so the .cu files build at 20 while the
# host sources stay at 23. This has to be set before add_library(), because a
# target captures the standard properties at creation; setting it later leaves
# CMake asking nvcc for "CUDA23" and the generate step fails. Harmless when
# LITHIUM_CUDA is off, since no CUDA source is in the build.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(lithium_core STATIC
    src/parse_cfg.cpp
    src/parse_weights.cpp
    src/network.cpp
    src/yolo.cpp
    src/letterbox.cpp
    src/preprocess.cpp
    src/cpu_backend.cpp
    src/engine.cpp
)

target_include_directories(lithium_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_compile_options(lithium_core PRIVATE -Wall -Wextra)

if(LITHIUM_NATIVE)
    target_compile_options(lithium_core PUBLIC -march=native)
endif()

if(LITHIUM_FAST_MATH)
    target_compile_options(lithium_core PUBLIC -ffast-math)
endif()

if(LITHIUM_ASAN)
    target_compile_options(lithium_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(lithium_core PUBLIC -fsanitize=address,undefined)
endif()

if(LITHIUM_CUDA)

    find_package(CUDAToolkit REQUIRED)

    target_sources(lithium_core PRIVATE src/allocator.cpp)
    if(LITHIUM_TF32)
        target_compile_definitions(lithium_core PUBLIC LITHIUM_TF32)
    endif()
    if(LITHIUM_FP16)
        target_compile_definitions(lithium_core PUBLIC LITHIUM_FP16)
    endif()
    target_compile_definitions(lithium_core PUBLIC LITHIUM_CUDA)
    target_link_libraries(lithium_core PUBLIC CUDA::cudart CUDA::cublas)


    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_backend.cu)
        # nvcc has its own host compiler, which CMAKE_CXX_COMPILER does not set.
        # Left alone it picks the system g++ (11 on JetPack 6), whose libstdc++
        # lacks the C++20 polymorphic_allocator<> that layer.hpp uses. Point it
        # at the same compiler the C++ sources use.
        if(NOT CMAKE_CUDA_HOST_COMPILER)
            set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        endif()

        enable_language(CUDA)
        target_sources(lithium_core PRIVATE src/gpu_backend.cu)

        # 87 is Orin. Drop it to let CMake probe the building machine instead.
        set_target_properties(lithium_core PROPERTIES CUDA_ARCHITECTURES 87)
    else()
        message(STATUS "LITHIUM_CUDA: src/gpu_backend.cu not present yet "
                       "- building the CUDA allocator only")
    endif()
endif()

add_executable(lithium app/main.cpp app/draw.cpp)
target_link_libraries(lithium PRIVATE lithium_core)

add_executable(test_reference test/test_reference.cpp)
target_link_libraries(test_reference PRIVATE lithium_core)

add_executable(bench test/bench.cpp)
target_link_libraries(bench PRIVATE lithium_core)

enable_testing()
add_test(NAME reference COMMAND test_reference ${CMAKE_CURRENT_SOURCE_DIR}/reference)
