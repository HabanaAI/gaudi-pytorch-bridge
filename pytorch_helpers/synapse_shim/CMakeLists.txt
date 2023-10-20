###############################################################################
# Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

add_habana_library(
  pytorch_synapse_shim
  STATIC
  hccl_impl.cpp
  syn_training_impl.cpp
  partial_event_emulation.cpp
  synapse_api_shim.cpp
  null_hw_api.cpp)

target_include_directories(pytorch_synapse_shim SYSTEM PUBLIC $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>
                                                              $<TARGET_PROPERTY:hcl,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(pytorch_synapse_shim PUBLIC ${PROJECT_SOURCE_DIR}/pytorch_helpers)

target_link_libraries(
  pytorch_synapse_shim
  PUBLIC ${CMAKE_DL_LIBS} Synapse
  PRIVATE absl::variant absl::raw_hash_set)

export(TARGETS pytorch_synapse_shim FILE pytorch_synapse_shimConfig.cmake)
