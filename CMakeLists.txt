cmake_minimum_required(VERSION 3.16)

project(blockdynamicnetworks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------
# Python
# --------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# --------------------------------------------------
# pybind11
# --------------------------------------------------

execute_process(
    COMMAND Python3_EXECUTABLE -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 CONFIG REQUIRED)

# --------------------------------------------------
# Core C++ library
# --------------------------------------------------

add_library(bdn_core
    bdn_core/graph.cpp
    bdn_core/partition.cpp
    bdn_core/block_stats.cpp
    bdn_core/sbm.cpp
    bdn_core/partition_prior.cpp
    bdn_core/posterior.cpp
    bdn_core/gibbs_sampler.cpp
    bdn_core/split_merge_sampler.cpp
    bdn_core/mcmc_diagnostics.cpp
    bdn_core/hybrid_sampler.cpp
)

target_include_directories(bdn_core
    PUBLIC include
)

# --------------------------------------------------
# C++ test executable
# --------------------------------------------------

add_executable(bdn_test
    bdn_test/test.cpp
)

target_link_libraries(bdn_test
    PRIVATE bdn_core
)

# --------------------------------------------------
# Python module
# --------------------------------------------------

pybind11_add_module(blockdynamicnetworks
    python/bindings.cpp
)

target_link_libraries(blockdynamicnetworks
    PRIVATE bdn_core
)

# --------------------------------------------------
# Installation
# --------------------------------------------------


install(TARGETS blockdynamicnetworks
    LIBRARY DESTINATION blockdynamicnetworks
    RUNTIME DESTINATION blockdynamicnetworks
)