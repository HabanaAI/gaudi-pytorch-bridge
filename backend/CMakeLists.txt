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

add_subdirectory(synapse_helpers)

add_habana_library(habana_pytorch_backend SHARED $<TARGET_OBJECTS:pytorch_synapse_helpers>
  jit_graph_cache.cpp
  jitgraph_utils.cpp
  create_pt_tensor.cpp
  backend_meta.cpp
  habana_operator.cpp
  scalar_cache.cpp
  random.cpp
  kernel/constant_information.cpp
  kernel/control_edges_processing.cpp
  kernel/ds_graph_recompile.cpp
  kernel/hpu_habana_launch_op_pt.cpp
  kernel/hpu_habana_compile_op_pt.cpp
  kernel/hpu_habana_execute_op_pt.cpp
  kernel/hpu_habana_launch_op_pt_lowering.cpp
  kernel/hpu_habana_launch_op_pt_sif.cpp
  kernel/hpu_habana_launch_op_pt_sif_utils.cpp
  kernel/hpu_habana_cache.cpp
  kernel/hpu_recipe_cache.cpp
  kernel/hpu_shape_inference.cpp
  kernel/refinement_engine.cpp
  passes/hpu_habana_persistence_marker_pass.cpp
  passes/replace_inplace_ops_ds.cpp
  habana_device/HPUAllocator.cpp
  habana_device/HPUEvent.cpp
  habana_device/HPUDevice.cpp
  habana_device/HPUGraph.cpp
  habana_device/HPUHooksInterface.cpp
  habana_device/HPUStream.cpp
  habana_device/PinnedMemoryAllocator.cpp
  habana_device/hpu_cached_devices.cpp
  helpers/collective_kernel_info.cpp
  helpers/dynamic_bucket_info.cpp
  helpers/dynamic_bucket_info_utils.cpp
  helpers/dynamic_shape_info.cpp
  helpers/symbolic_expression.cpp
  helpers/dynamic_graph_utils.cpp
  helpers/dynamic_shape_infer.cpp
  helpers/runtime_config.cpp
  helpers/compilation_statistics.cpp
  helpers/collective_utils.cpp
  helpers/lowering_util.cpp
  helpers/tensor_shape.cpp
  helpers/tensor_info.cpp
  helpers/tensor_utils.cpp
  helpers/cast_sequence.cpp
  helpers/create_tensor.cpp
  helpers/event_dispatcher.cpp
  profiling/json_activity_profiler.cpp
  profiling/json_file_parser.cpp
  profiling/profiling.cpp
  profiling/trace_sources/synapse_logger_source.cpp
  profiling/trace_sources/synapse_profiler_source.cpp
  profiling/trace_sources/bridge_logs_source.cpp
  profiling/trace_sources/trace_parser.cpp
  profiling/trace_sources/memory_source.cpp
)

add_custom_command(TARGET habana_pytorch_backend POST_BUILD
  COMMENT "Creating ${TARGET_NAME} symbolic link in $ENV{BUILD_ROOT_LATEST}"
  COMMAND "${CMAKE_COMMAND}" -E create_symlink
  "$<TARGET_FILE:habana_pytorch_backend>" "$ENV{BUILD_ROOT_LATEST}/$<TARGET_FILE_NAME:habana_pytorch_backend>")

target_include_directories(habana_pytorch_backend PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/pytorch_helpers)

target_link_libraries(habana_pytorch_backend
  PUBLIC
  habana_pytorch_backend_ops
  pytorch_synapse_helpers
  c10
  PRIVATE
  Python::Python
  habana_serialization
  torch
  nlohmann_json::nlohmann_json
  absl::strings
  absl::str_format
  ${PYTORCH_SYNAPSE_UTILS_LIB}

  # tensor_comparator_core
  -lstdc++fs
  npu::hllogger
)

target_include_directories(habana_pytorch_backend PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(
  habana_pytorch_backend SYSTEM
  PUBLIC $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:hcl,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_definitions(habana_pytorch_backend PUBLIC GENERIC_HELPERS)
target_compile_features(habana_pytorch_backend PUBLIC cxx_std_14) # still required as this is transitive
set_target_properties(habana_pytorch_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

if(MANYLINUX)
  target_link_libraries(habana_pytorch_backend PUBLIC pytorch_synapse_shim)
else()
  target_link_libraries(habana_pytorch_backend PUBLIC Synapse)
endif()

separate_debug_symbols(habana_pytorch_backend)
install(TARGETS habana_pytorch_backend LIBRARY DESTINATION habana_torch_plugin/lib)
