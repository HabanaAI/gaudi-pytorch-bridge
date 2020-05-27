###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
if(POLICY CMP0076)
  cmake_policy(SET CMP0076 OLD)
endif()
project(habana_pytorch_plugin VERSION 0.8.0 LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(SANITIZER "build sanitizer configuration" OFF)
if(SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fsanitize-address-use-after-scope -O0")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
endif(SANITIZER)
find_program(CCACHE ccache)
if (CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE}")
endif()
find_program(CLANG_TIDY clang-tidy)
if (CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY}")
else ()
  message(AUTHOR_WARNING "clang-tidy was not detected. Please install it and rerun CMake or set CLANG_TIDY to a valid path")
endif ()
find_package(Torch REQUIRED)
add_subdirectory(third_party/pybind11)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(AddNPUStackDependencies)
# workaround for cmake3.5. Starting from 3.13 sources are optional and we can add them in add_subdirectories
add_library(habana_pytorch_plugin SHARED
  habana_device/HPUAllocator.cpp
  habana_helpers/tensor_utils.cpp
  habana_kernels/basic_kernels.cpp
  habana_kernels/binary_kernels.cpp
  habana_kernels/conv_kernels.cpp
  habana_kernels/conv_pool_utils.cpp
  habana_kernels/embedding_kernels.cpp
  habana_kernels/fill_kernels.cpp
  habana_kernels/habana_operator.cpp
  habana_kernels/index_kernels.cpp
  habana_kernels/kernel_utils.cpp
  habana_kernels/linear_kernels.cpp
  habana_kernels/loss_kernels.cpp
  habana_kernels/norm_kernels.cpp
  habana_kernels/pool_kernels.cpp
  habana_kernels/random_gen_kernels.cpp
  habana_kernels/reduction_kernels.cpp
  habana_kernels/scalar.cpp
  habana_kernels/simple_generic_kernel.cpp
  habana_kernels/softmax_kernels.cpp
  habana_kernels/tensor_factories.cpp
  habana_kernels/threshold_kernels.cpp
  habana_kernels/topk_kernels.cpp
  habana_kernels/unary_kernels.cpp
  habana_kernels/create_habana_operator.cpp
)
target_compile_features(habana_pytorch_plugin PRIVATE cxx_std_14)
target_compile_definitions(habana_pytorch_plugin PRIVATE "GENERIC_HELPERS" MAX_DEVICES_PER_BOX=8)
#TODO: remove -Wno-error=deprecated-declarations after removing functions marked [[deprecated]]
target_compile_options(habana_pytorch_plugin PRIVATE -Wall -Wextra -Werror -Wno-error=deprecated-declarations)
# Link against LibTorch and synapse
target_link_libraries(habana_pytorch_plugin PRIVATE "${TORCH_LIBRARIES}" synapse_helpers::synapse_helpers_abi11)
target_include_directories(habana_pytorch_plugin PRIVATE ${CMAKE_SOURCE_DIR})

set(SOURCES habana_bridge/register/register.cpp
                habana_bridge/passes/habana_fuser.cpp
		habana_bridge/kernel/hpu_habana_launch_op_pt.cpp)
pybind11_add_module(hb_torch SHARED ${SOURCES})
target_compile_definitions(hb_torch PRIVATE "GENERIC_HELPERS" MAX_DEVICES_PER_BOX=8)
target_link_libraries(hb_torch PRIVATE "${TORCH_LIBRARIES}" habana_pytorch_plugin synapse_helpers::synapse_helpers_abi11)
target_include_directories(hb_torch PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(hb_torch PRIVATE cxx_std_14)

# Create symlink to latest build from tests dir
add_custom_command(TARGET habana_pytorch_plugin POST_BUILD COMMAND
  $(CMAKE_COMMAND) -E create_symlink $<TARGET_FILE:habana_pytorch_plugin>
  $(CMAKE_SOURCE_DIR)/tests/$<TARGET_FILE_NAME:habana_pytorch_plugin>
)
add_custom_command(TARGET hb_torch POST_BUILD COMMAND
  $(CMAKE_COMMAND) -E create_symlink $<TARGET_FILE:hb_torch>
  $(CMAKE_SOURCE_DIR)/tests/$<TARGET_FILE_NAME:hb_torch>
)
