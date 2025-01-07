###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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
