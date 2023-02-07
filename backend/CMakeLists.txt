###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

add_library(habana_pytorch_backend SHARED $<TARGET_OBJECTS:pytorch_synapse_helpers>
  jit_graph_cache.cpp
  jitgraph_utils.cpp
  habana_operator.cpp
  kernel/ds_graph_recompile.cpp
  kernel/hpu_habana_launch_op_pt.cpp
  kernel/hpu_habana_launch_op_pt_lowering.cpp
  kernel/hpu_habana_launch_op_pt_sif.cpp
  kernel/hpu_habana_cache.cpp
  kernel/hpu_process_control_edges.cpp
  kernel/hpu_shape_inference.cpp
  kernel/refinement_engine.cpp
  passes/habana_fuser.cpp
  passes/remove_inplace_ops.cpp
  passes/hpu_habana_persistence_marker_pass.cpp
  helpers/collective_kernel_info.cpp
  helpers/dynamic_bucket_info.cpp
  helpers/dynamic_bucket_info_utils.cpp
  helpers/compilation_statistics.cpp
  helpers/tensor_shape.cpp
  helpers/tensor_info.cpp
  helpers/create_tensor.cpp
  helpers/event_dispatcher.cpp
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
  PRIVATE
    habana_serialization
    torch
    nlohmann_json::nlohmann_json
    absl::strings
    absl::str_format
    ${PYTORCH_SYNAPSE_UTILS_LIB}
    # tensor_comparator_core
    -lstdc++fs
    npu::spdlog
)

target_include_directories(habana_pytorch_backend PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(
  habana_pytorch_backend SYSTEM
  PUBLIC $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>
         $<TARGET_PROPERTY:tpc_kernels,INTERFACE_INCLUDE_DIRECTORIES>
         $<TARGET_PROPERTY:hcl,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_definitions(habana_pytorch_backend PRIVATE MAX_DEVICES_PER_BOX=8)
target_compile_definitions(habana_pytorch_backend PUBLIC GENERIC_HELPERS)
target_compile_features(habana_pytorch_backend PUBLIC cxx_std_14) # still required as this is transitive
set_target_properties(habana_pytorch_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

if(MANYLINUX)
target_link_libraries(habana_pytorch_backend PUBLIC pytorch_synapse_shim)
else()
target_link_libraries(habana_pytorch_backend PUBLIC Synapse)
endif()
separate_debug_symbols(habana_pytorch_backend)
