cmake_minimum_required(VERSION 3.16)

project(blockdynamicnetworks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bdn_test
    test.cpp
    src/graph.cpp
    src/partition.cpp
    src/block_stats.cpp
    src/sbm.cpp
    src/partition_prior.cpp
    src/posterior.cpp
    src/gibbs_sampler.cpp
)

target_include_directories(bdn_test PRIVATE include)