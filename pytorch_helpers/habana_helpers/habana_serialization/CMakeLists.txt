# ##############################################################################
# Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ##############################################################################

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
