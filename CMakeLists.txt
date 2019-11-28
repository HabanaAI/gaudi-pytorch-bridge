cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(habana_pytorch_plugin VERSION 0.5.0 LANGUAGES CXX)

find_package(Torch REQUIRED)

add_library(Synapse SHARED IMPORTED)
set_target_properties(Synapse PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libSynapse.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HABANA_NPU_STACK_PATH}" )

add_subdirectory(habana_device)
add_subdirectory(habana_kernels)
# add_subdirectory(optimization_passes)
