###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

set(CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

add_habana_library(bindings INTERFACE)
target_include_directories(
  bindings INTERFACE "${CMAKE_SOURCE_DIR}" $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>
                     "${CMAKE_BINARY_DIR}")

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

if(UPSTREAM_COMPILE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  set(PYBIND_INSTALL_DIRNAME "lib/upstream_pybind")
else()
  set(PYBIND_INSTALL_DIRNAME "lib/fork_pybind")
endif()

include(pt_ver)
set_up_pt_ver_mechanism()

target_compile_options(bindings INTERFACE -fopenmp -fpermissive)
target_link_libraries(
  bindings
  INTERFACE npu::fmt
            npu::hllogger
            npu::hcl
            npu::specs_external
            $<TARGET_NAME_IF_EXISTS:npu::specs_embedded>
            pytorch_helpers
            Python::Python)

add_subdirectory(internal/bridge_config)
add_subdirectory(activity_profiler/csrc)
add_subdirectory(core)
add_subdirectory(distributed/hccl)
add_subdirectory(dynamo/compile_backend)
add_subdirectory(dynamo/debug_utils/visualization)
add_subdirectory(hpex/csrc)
add_subdirectory(hpu/csrc)
add_subdirectory(low_overhead_profiler/csrc)
add_subdirectory(utils/debug/csrc)
add_subdirectory(utils/experimental/csrc)
add_subdirectory(utils/event_dispatcher/csrc)
add_subdirectory(utils/profiler/csrc)
add_subdirectory(media_pyt_bridge/csrc)
