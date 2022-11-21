###############################################################################
# Copyright (C) 2021-2022 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################
cmake_minimum_required(VERSION 3.10)

find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 17)

add_library(habana_serialization STATIC
    cache_file_handler.cpp
    deserializers.cpp
    serializers.cpp
    recipe_cache.cpp
    inter_host_cache.cpp
    cache_version.cpp
    include/habana_serialization/cache_file_handler.h
    include/habana_serialization/deserializers.h
    include/habana_serialization/serializers.h
    include/habana_serialization/recipe_cache.h
    include/habana_serialization/inter_host_cache.h
    include/habana_serialization/cache_version.h
)

target_include_directories(habana_serialization
    PRIVATE include/habana_serialization/
    PUBLIC  include/
)

target_link_libraries(habana_serialization PUBLIC "${TORCH_LIBRARIES}" pytorch_synapse_helpers_abi11)
