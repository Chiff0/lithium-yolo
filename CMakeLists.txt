cmake_minimum_required(VERSION 3.22)

project(lithium LANGUAGES CXX)

# The CUDA backend is off by default so the project configures and builds on a
# machine with no CUDA toolchain. Turn it on for the Jetson.
option(LITHIUM_CUDA       "Build the CUDA backend"                              OFF)
option(LITHIUM_NATIVE     "Tune for the building machine (-march=native)"       OFF)
option(LITHIUM_ASAN       "Address + UB sanitizers, for the test build"         OFF)

# -ffast-math is what currently buys the ~5x on the CPU backend, but it also
# assumes no NaN/Inf and flushes denormals, so it is a numerics decision and not
# just an optimisation. Off by default: it has to be asked for explicitly, and
# any measurement taken with it on has to say so.
option(LITHIUM_FAST_MATH  "Relax IEEE float semantics (-ffast-math)"            OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # cuda_backend.cu does not exist yet; this is the wiring it will need.
    target_sources(lithium_core PRIVATE src/cuda_backend.cu)
    target_compile_definitions(lithium_core PUBLIC LITHIUM_CUDA)
    target_link_libraries(lithium_core PUBLIC CUDA::cudart CUDA::cublas)

    # 87 is Orin. Leave it unset to let CMake probe the building machine instead.
    set_target_properties(lithium_core PROPERTIES CUDA_ARCHITECTURES 87)
endif()

add_executable(lithium app/main.cpp app/draw.cpp)
target_link_libraries(lithium PRIVATE lithium_core)

add_executable(test_reference test/test_reference.cpp)
target_link_libraries(test_reference PRIVATE lithium_core)

add_executable(bench test/bench.cpp)
target_link_libraries(bench PRIVATE lithium_core)

enable_testing()
add_test(NAME reference COMMAND test_reference ${CMAKE_CURRENT_SOURCE_DIR}/reference)
