###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################
cmake_minimum_required(VERSION 3.10)

foreach (FLAVOR abi11)
  set(TARGET_NAME "pytorch_synapse_helpers_${FLAVOR}")
  add_library(${TARGET_NAME} SHARED
    device.cpp
    env_flags.cpp
    event.cpp
    event_handle_cache.cpp
    device_memory.cpp
    devmem_logger.cpp
    graph.cpp
    habana_tensor.cpp
    hcl_communicator.cpp
    host_memory.cpp
    logger_stub.cpp
    mem_handle.cpp
    memory_mapping.cpp
    pool_allocator/utils.cpp
    pool_allocator/PoolAllocator.cpp
    pool_allocator/CoalescedPoolAllocator.cpp
    pool_allocator/CoalescedStringentPoolAllocator.cpp
    recipe.cpp
    recipe_handle_cache.cpp
    stream_event_manager.cpp
    session.cpp
    stream.cpp
    tensor_builder_base.cpp)

  target_include_directories(${TARGET_NAME}
     PUBLIC "${CMAKE_SOURCE_DIR}" $<TARGET_PROPERTY:tpc_kernels,INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(${TARGET_NAME} PUBLIC "$ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers")
  target_compile_options(${TARGET_NAME}
    PRIVATE -Wall -Wextra -Werror)

  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "pytorch_synapse_helpers")

  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMENT "Creating ${TARGET_NAME} symbolic link in $ENV{BUILD_ROOT_LATEST}"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
    "$<TARGET_FILE:${TARGET_NAME}>" "$ENV{BUILD_ROOT_LATEST}/$<TARGET_FILE_NAME:${TARGET_NAME}>")
endforeach ()

target_link_libraries(pytorch_synapse_helpers_abi11 PUBLIC
 Synapse
 absl::strings
 absl::optional
 absl::flags
 absl::variant
 absl::hash
 absl::raw_hash_set)
set_target_properties(pytorch_synapse_helpers_abi11 PROPERTIES SUFFIX ".so")
target_compile_features(pytorch_synapse_helpers_abi11 PUBLIC cxx_std_14) # still required as this is transitive
set_target_properties(pytorch_synapse_helpers_abi11 PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_definitions(pytorch_synapse_helpers_abi11 PUBLIC GENERIC_HELPERS)
target_link_libraries(pytorch_synapse_helpers_abi11 PUBLIC Synapse)
separate_debug_symbols(pytorch_synapse_helpers_abi11)

export(TARGETS pytorch_synapse_helpers_abi11
  NAMESPACE
  synapse_helpers::
  FILE
  synapse_helpersConfig.cmake)
set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE synapse_helpers)
