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

add_habana_library(pytorch_synapse_utils_shim STATIC synapse_utils_shim.cpp)

target_include_directories(pytorch_synapse_utils_shim SYSTEM
                           PUBLIC $<TARGET_PROPERTY:SynapseUtils,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(pytorch_synapse_utils_shim PUBLIC ${CMAKE_DL_LIBS} SynapseUtils pytorch_synapse_shim)

export(TARGETS pytorch_synapse_utils_shim FILE pytorch_synapse_utils_shimConfig.cmake)
