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

add_library(tpc_kernels SHARED IMPORTED)
set_target_properties(tpc_kernels PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{TPC_KERNELS_ROOT}/include")
