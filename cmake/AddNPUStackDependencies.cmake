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
list(APPEND LIB_INCLUDE_DIRS "$ENV{SYNAPSE_ROOT}/include")
list(APPEND LIB_INCLUDE_DIRS "$ENV{HCL_ROOT}/include")
set_target_properties(Synapse PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libSynapse.so"
  INTERFACE_INCLUDE_DIRECTORIES "${LIB_INCLUDE_DIRS}"
  )

add_library(tpc_kernels SHARED IMPORTED)
set_target_properties(tpc_kernels PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "$ENV{SPECS_EXT_ROOT}")

add_library(hcl SHARED IMPORTED)
set_target_properties(hcl PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HCL_ROOT}/include"
  )