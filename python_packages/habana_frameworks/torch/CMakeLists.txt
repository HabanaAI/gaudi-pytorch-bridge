# ##############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited. This file contains Habana Labs, Ltd. proprietary and
# confidential information and is subject to the confidentiality and license
# agreements under which it was provided.
#
# ##############################################################################
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(TARGET_NAME "bindings")
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")
add_library(${TARGET_NAME} INTERFACE)
target_include_directories(${TARGET_NAME}
    INTERFACE "${CMAKE_SOURCE_DIR}" $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(${TARGET_NAME} INTERFACE "${PYTHON_INCLUDE_DIR}")
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_compile_options(${TARGET_NAME} INTERFACE -fopenmp -fpermissive)
target_link_libraries(${TARGET_NAME} INTERFACE npu::fmt npu::hllogger npu::hcl npu::specs_external pytorch_helpers)
target_compile_definitions(${TARGET_NAME} INTERFACE MAX_DEVICES_PER_BOX=8)

add_subdirectory(activity_profiler/csrc)
add_subdirectory(core)
add_subdirectory(distributed/hccl)
add_subdirectory(hpex/csrc)
add_subdirectory(hpu/csrc)
add_subdirectory(utils/debug/csrc)
add_subdirectory(utils/experimental/csrc)
add_subdirectory(utils/event_dispatcher/csrc)
add_subdirectory(utils/profiler/csrc)

if ("${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}" VERSION_GREATER_EQUAL "2.0")
    add_subdirectory(dynamo/compile_backend)
endif()
