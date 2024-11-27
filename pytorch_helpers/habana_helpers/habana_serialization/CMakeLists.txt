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
  habana_serialization
  STATIC
  cache_file_handler.cpp
  deserializers.cpp
  serializers.cpp
  recipe_cache.cpp
  inter_host_cache.cpp
  cache_version.cpp
  base_cache_file_handler.cpp
  recipe_cache_config.cpp
  const_section.cpp)

target_compile_features(habana_serialization PUBLIC cxx_std_17)

target_include_directories(
  habana_serialization
  PRIVATE include/habana_serialization/ ${PROJECT_SOURCE_DIR}/pytorch_helpers
  PRIVATE ${CMAKE_BINARY_DIR}
  PUBLIC include/ ${PROJECT_SOURCE_DIR})
target_include_directories(habana_serialization SYSTEM PUBLIC $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(habana_serialization PUBLIC torch c10 absl::span npu::hllogger npu::fmt)
