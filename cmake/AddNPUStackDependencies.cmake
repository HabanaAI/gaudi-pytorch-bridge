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

# define fmt target, so kineto will not compile it from sources. Otherwise we have conflicts
add_library(fmt INTERFACE IMPORTED)
set_target_properties(fmt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{THIRD_PARTIES_ROOT}/fmt-9.1.0/include/")
target_compile_definitions(fmt INTERFACE FMT_HEADER_ONLY)
add_library(npu::fmt ALIAS fmt)

add_library(hllogger SHARED IMPORTED)
set_target_properties(hllogger PROPERTIES IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libhl_logger.so")
set_target_properties(hllogger PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{HL_LOGGER_INCLUDE_DIRS}")
add_library(npu::hllogger ALIAS hllogger)

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

add_library(specs_embedded INTERFACE IMPORTED)
set_target_properties(specs_embedded PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{SPECS_EMBEDDED_ROOT}")
add_library(npu::specs_embedded ALIAS specs_embedded)

add_library(tpc_kernels INTERFACE IMPORTED)
set_target_properties(tpc_kernels PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                             "$ENV{TPC_KERNELS_ROOT}/shared_layer/include")
target_link_libraries(tpc_kernels INTERFACE npu::specs_external npu::specs_embedded)
add_library(npu::tpc_kernels ALIAS tpc_kernels)

set_target_properties(SynapseUtils PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{SYNAPSE_UTILS_ROOT}/include")
target_link_libraries(SynapseUtils INTERFACE npu::tpc_kernels npu::Synapse)
add_library(npu::SynapseUtils ALIAS SynapseUtils)

add_library(Media INTERFACE IMPORTED)
list(APPEND MEDIA_INCLUDE_DIRS "$ENV{MEDIA_ROOT}/include")
set_target_properties(Media PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${MEDIA_INCLUDE_DIRS}")
