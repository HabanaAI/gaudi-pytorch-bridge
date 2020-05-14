###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

if(POLICY CMP0076)
  cmake_policy(SET CMP0076 OLD)
endif()

project(habana_pytorch_distributed_plugin VERSION 0.8.0 LANGUAGES CXX)

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

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)
include(AddNPUStackDependencies)

# workaround for cmake3.5. Starting from 3.13 sources are optional and we can add them in add_subdirectories
add_library(habana_pytorch_distributed_plugin SHARED
  hcl_communicator.cpp
)

target_compile_features(habana_pytorch_distributed_plugin PRIVATE cxx_std_14)
target_compile_definitions(habana_pytorch_distributed_plugin PRIVATE "GENERIC_HELPERS" MAX_DEVICES_PER_BOX=8)
#TODO: remove -Wno-error=deprecated-declarations after removing functions marked [[deprecated]]
target_compile_options(habana_pytorch_distributed_plugin PRIVATE -Wall -Wextra -Werror -Wno-error=deprecated-declarations)

# Link against LibTorch and synapse
target_link_libraries(habana_pytorch_distributed_plugin PRIVATE "${TORCH_LIBRARIES}" synapse_helpers::synapse_helpers_abi11)
target_include_directories(habana_pytorch_distributed_plugin PRIVATE ${CMAKE_SOURCE_DIR})

