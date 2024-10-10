###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
  kernel/hpu_shape_inference.cpp
  kernel/refinement_engine.cpp
  passes/remove_inplace_ops.cpp
  passes/hpu_habana_persistence_marker_pass.cpp
  passes/replace_inplace_ops_ds.cpp
  habana_device/HPUAllocator.cpp
  habana_device/HPUEvent.cpp
  habana_device/HPUDevice.cpp
  habana_device/HPUGraph.cpp
  habana_device/HPUStream.cpp
  habana_device/PinnedMemoryAllocator.cpp
  habana_device/hpu_cached_devices.cpp
  helpers/collective_kernel_info.cpp
  helpers/dynamic_bucket_info.cpp
  helpers/dynamic_bucket_info_utils.cpp
  helpers/dynamic_shape_info.cpp
  helpers/symbolic_expression.cpp
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
  profiling/kineto_activity_profiler.cpp
  profiling/profiling.cpp
  profiling/trace_sources/synapse_logger_source.cpp
  profiling/trace_sources/synapse_profiler_source.cpp
  profiling/trace_sources/bridge_logs_source.cpp
  profiling/trace_sources/trace_parser.cpp
  profiling/trace_sources/memory_source.cpp
  program/components/cache.cpp
  program/components/executor.cpp
  program/components/program.cpp
  program/components/splitter.cpp
  program/components/strategy.cpp
  program/create_executor.cpp
  program/scheduling_strategies/naive_rpo_scheduling_strategy.cpp
  program/splitting_strategies/eager_splitting_strategy.cpp
  program/splitting_strategies/lvaspill_splitting_strategy.cpp
  program/utils/jit_bfstopsort.cpp
  program/utils/jit_fixcolors.cpp
  program/utils/jit_fixcycles.cpp
  program/utils/jit_liveness_analysis.cpp
  program/utils/union_find.cpp
)

add_custom_command(TARGET habana_pytorch_backend POST_BUILD
  COMMENT "Creating ${TARGET_NAME} symbolic link in $ENV{BUILD_ROOT_LATEST}"
  COMMAND "${CMAKE_COMMAND}" -E create_symlink
  "$<TARGET_FILE:habana_pytorch_backend>" "$ENV{BUILD_ROOT_LATEST}/$<TARGET_FILE_NAME:habana_pytorch_backend>")

target_include_directories(habana_pytorch_backend PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/pytorch_helpers)

if(DEFINED ENV{KINETO_ROOT})
  target_include_directories(habana_pytorch_backend PRIVATE $ENV{KINETO_ROOT}/libkineto/include)
else()
  message(FATAL_ERROR "KINETO_ROOT is not defined, please update automation repository")
endif()

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
  $<TARGET_PROPERTY:tpc_kernels,INTERFACE_INCLUDE_DIRECTORIES>
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
