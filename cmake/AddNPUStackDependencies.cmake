###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

add_library(spdlog INTERFACE IMPORTED)
set_target_properties(spdlog PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{SPDLOG_ROOT}")
add_library(npu::spdlog ALIAS spdlog)

add_library(nlohmann_json INTERFACE IMPORTED)
set_target_properties(nlohmann_json PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                               "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/json/single_include")
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

if(MANYLINUX)
  add_library(Synapse INTERFACE IMPORTED)
  add_library(SynapseUtils INTERFACE IMPORTED)
else()
  add_library(Synapse SHARED IMPORTED)
  set_target_properties(Synapse PROPERTIES IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libSynapse.so")
  add_library(SynapseUtils SHARED IMPORTED)
  set_target_properties(SynapseUtils PROPERTIES IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libsynapse_utils.so")
endif()

add_library(hcl INTERFACE IMPORTED)
set_target_properties(hcl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{HCL_ROOT}/include")
add_library(npu::hcl ALIAS hcl)

set_target_properties(Synapse PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{SYNAPSE_ROOT}/include")
target_link_libraries(Synapse INTERFACE npu::hcl)
add_library(npu::Synapse ALIAS Synapse)

add_library(specs_external INTERFACE IMPORTED)
set_target_properties(specs_external PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{SPECS_EXT_ROOT}")
add_library(npu::specs_external ALIAS specs_external)

add_library(tpc_kernels INTERFACE IMPORTED)
set_target_properties(tpc_kernels PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                             "$ENV{TPC_KERNELS_ROOT}/shared_layer/include")
target_link_libraries(tpc_kernels INTERFACE npu::specs_external)
add_library(npu::tpc_kernels ALIAS tpc_kernels)

set_target_properties(SynapseUtils PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{SYNAPSE_UTILS_ROOT}/include")
target_link_libraries(SynapseUtils INTERFACE npu::tpc_kernels npu::Synapse)
add_library(npu::SynapseUtils ALIAS SynapseUtils)
