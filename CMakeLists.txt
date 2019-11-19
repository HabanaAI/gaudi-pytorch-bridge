cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(habana_pytorch_plugin VERSION 0.5.0 LANGUAGES CXX)

find_package(Torch REQUIRED)

add_subdirectory(habana_device)
add_subdirectory(habana_kernels)
# add_subdirectory(optimization_passes)
