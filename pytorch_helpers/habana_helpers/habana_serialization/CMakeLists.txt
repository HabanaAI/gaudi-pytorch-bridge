###############################################################################
# Copyright (C) 2021-2022 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################

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

target_compile_features(habana_serialization PUBLIC cxx_std_17)

target_include_directories(habana_serialization
    PRIVATE include/habana_serialization/
    PUBLIC  include/
)

target_link_libraries(habana_serialization PUBLIC torch c10 pytorch_synapse_helpers_abi11)
