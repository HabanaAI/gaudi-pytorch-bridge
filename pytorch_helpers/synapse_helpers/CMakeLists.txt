###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################
cmake_minimum_required(VERSION 3.10)

add_library(pytorch_synapse_helpers OBJECT
  device.cpp
  env_flags.cpp
  event.cpp
  event_handle_cache.cpp
  device_context.cpp
  device_memory.cpp
  devmem_logger.cpp
  graph.cpp
  habana_tensor.cpp
  hccl_communicator.cpp
  host_memory.cpp
  logger_stub.cpp
  mem_handle.cpp
  memory_defragmentation.cpp
  memory_mapping.cpp
  pool_allocator/utils.cpp
  pool_allocator/PoolAllocator.cpp
  pool_allocator/CoalescedPoolAllocator.cpp
  pool_allocator/CoalescedStringentPoolAllocator.cpp
  RealTimeMemoryLogger.cpp
  recipe.cpp
  recipe_handle_cache.cpp
  status_conversion.cpp
  stream_event_manager.cpp
  session.cpp
  stream.cpp
  tensor_builder_base.cpp)

target_include_directories(pytorch_synapse_helpers
   PUBLIC "${CMAKE_SOURCE_DIR}" $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(pytorch_synapse_helpers
   PUBLIC "${CMAKE_SOURCE_DIR}" $<TARGET_PROPERTY:tpc_kernels,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(pytorch_synapse_helpers
   PUBLIC "${CMAKE_SOURCE_DIR}" $<TARGET_PROPERTY:hcl,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(pytorch_synapse_helpers PUBLIC "$ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers")
target_include_directories(pytorch_synapse_helpers PUBLIC "$ENV{SPDLOG_ROOT}")
target_include_directories(pytorch_synapse_helpers PRIVATE ${PYTHON_INCLUDE_DIR})
target_compile_options(pytorch_synapse_helpers
  PRIVATE -Wall -Wextra -Werror)

set_target_properties(pytorch_synapse_helpers PROPERTIES OUTPUT_NAME "pytorch_synapse_helpers")

target_link_libraries(pytorch_synapse_helpers PUBLIC
 absl::strings
 absl::optional
 absl::flags
 absl::variant
 absl::hash
 absl::raw_hash_set)
target_compile_features(pytorch_synapse_helpers PUBLIC cxx_std_14) # still required as this is transitive
set_target_properties(pytorch_synapse_helpers PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_definitions(pytorch_synapse_helpers PUBLIC GENERIC_HELPERS)

if(MANYLINUX)
target_link_libraries(pytorch_synapse_helpers PUBLIC pytorch_synapse_shim)
else()
target_link_libraries(pytorch_synapse_helpers PUBLIC Synapse)
endif()
separate_debug_symbols(pytorch_synapse_helpers)
export(TARGETS pytorch_synapse_helpers
  NAMESPACE
  synapse_helpers::
  FILE
  synapse_helpersConfig.cmake)
set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE synapse_helpers)
