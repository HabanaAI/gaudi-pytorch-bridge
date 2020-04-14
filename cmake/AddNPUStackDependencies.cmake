###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################
cmake_minimum_required(VERSION 3.0)

add_library(Synapse SHARED IMPORTED)
set_target_properties(Synapse PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libSynapse.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{SYNAPSE_ROOT}/include"
  )

add_library(synapse_helpers_abi11 SHARED)
set_target_properties(synapse_helpers_abi11 PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libsynapse_helpers.so.1"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{TF_MODULES_ROOT};$ENV{ABSEIL_CPP_INCLUDE_DIR};$ENV{TPC_KERNELS_ROOT}/include"
  INTERFACE_COMPILE_DEFINITIONS GENERIC_HELPERS
  LINKER_LANGUAGE CXX
  )
add_library(synapse_helpers::synapse_helpers_abi11 ALIAS synapse_helpers_abi11)
target_link_libraries(synapse_helpers_abi11 INTERFACE Synapse)
