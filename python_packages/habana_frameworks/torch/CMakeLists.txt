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

set(CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

add_habana_library(bindings INTERFACE)
target_include_directories(
  bindings INTERFACE "${CMAKE_SOURCE_DIR}" $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>
                     "${CMAKE_BINARY_DIR}")

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

include(pt_ver)
set_up_pt_ver_mechanism()

target_compile_options(bindings INTERFACE -fopenmp -fpermissive)
target_link_libraries(
  bindings
  INTERFACE npu::fmt
            npu::hllogger
            npu::hcl
            npu::specs_external
            npu::specs_embedded
            pytorch_helpers
            Python::Python)

add_subdirectory(internal/bridge_config)
add_subdirectory(activity_profiler/csrc)
add_subdirectory(core)
add_subdirectory(distributed/hccl)
add_subdirectory(dynamo/compile_backend)
add_subdirectory(hpex/csrc)
add_subdirectory(hpu/csrc)
add_subdirectory(utils/debug/csrc)
add_subdirectory(utils/experimental/csrc)
add_subdirectory(utils/event_dispatcher/csrc)
add_subdirectory(utils/visualization)
add_subdirectory(utils/profiler/csrc)
add_subdirectory(media_pyt_bridge/csrc)
