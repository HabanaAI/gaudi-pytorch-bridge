################################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
################################################################################

add_library(pytorch_synapse_shim STATIC hccl_impl.cpp syn_training_impl.cpp partial_event_emulation.cpp synapse_api_shim.cpp null_hw_api.cpp)
target_include_directories(pytorch_synapse_shim PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(pytorch_synapse_shim PRIVATE "$ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers")
target_include_directories(pytorch_synapse_shim
     PUBLIC "${CMAKE_SOURCE_DIR}" $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(pytorch_synapse_shim
     PUBLIC "${CMAKE_SOURCE_DIR}" $<TARGET_PROPERTY:tpc_kernels,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(pytorch_synapse_shim
     PUBLIC "${CMAKE_SOURCE_DIR}" $<TARGET_PROPERTY:hcl,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(pytorch_synapse_shim PUBLIC "$ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers")
target_include_directories(pytorch_synapse_shim PUBLIC "$ENV{SPDLOG_ROOT}")

target_compile_options(pytorch_synapse_shim PRIVATE -Wall -Wextra -Werror)
target_link_libraries(pytorch_synapse_shim PUBLIC ${CMAKE_DL_LIBS} Synapse)
target_link_libraries(pytorch_synapse_shim PRIVATE absl::variant absl::raw_hash_set)

export(TARGETS pytorch_synapse_shim
  FILE
  pytorch_synapse_shimConfig.cmake
)
