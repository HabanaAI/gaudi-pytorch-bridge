###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

include(FetchContent)

FetchContent_Declare(
  exprtk
  GIT_REPOSITORY https://github.com/ArashPartow/exprtk.git
  GIT_TAG 0.0.3-cmake
  GIT_SHALLOW TRUE
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  xxhash
  GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
  GIT_TAG v0.8.3
  GIT_SHALLOW TRUE
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
  GIT_SHALLOW TRUE
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 9.1.0
  GIT_SHALLOW TRUE
  SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/fmt-9.1.0 SYSTEM EXCLUDE_FROM_ALL)
set(FMT_INSTALL ON)

FetchContent_Declare(
  magic_enum
  GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
  GIT_TAG v0.9.7
  GIT_SHALLOW TRUE
  SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/magic_enum-0.9.7 SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  devscripts
  URL https://snapshot.debian.org/archive/debian/20250412T205410Z/pool/main/d/devscripts/devscripts_2.25.9.tar.xz
      SYSTEM EXCLUDE_FROM_ALL)

add_executable(hardening-check IMPORTED)

FetchContent_MakeAvailable(devscripts exprtk xxhash fmt nlohmann_json magic_enum)

set_target_properties(hardening-check PROPERTIES IMPORTED_LOCATION ${devscripts_SOURCE_DIR}/scripts/hardening-check.pl)

add_library(hllogger SHARED IMPORTED)
set_target_properties(hllogger PROPERTIES IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libhl_logger.so")
target_link_libraries(hllogger INTERFACE magic_enum::magic_enum fmt::fmt-header-only)
target_include_directories(hllogger INTERFACE "$ENV{SWTOOLS_SDK_ROOT}/hl_logger/include" "${FETCHCONTENT_BASE_DIR}")
add_library(npu::hllogger ALIAS hllogger)

add_library(Synapse INTERFACE IMPORTED)
add_library(SynapseUtils INTERFACE IMPORTED)

add_library(hcl INTERFACE IMPORTED)
set_target_properties(hcl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{HCL_INCLUDE_DIR}")
add_library(npu::hcl ALIAS hcl)

set_target_properties(Synapse PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{SYNAPSE_INCLUDE_DIR}")
target_link_libraries(Synapse INTERFACE npu::hcl)
add_library(npu::Synapse ALIAS Synapse)

add_library(specs_external INTERFACE IMPORTED)
set_target_properties(specs_external PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{SPECS_EXT_ROOT}")
add_library(npu::specs_external ALIAS specs_external)

if(EXISTS "$ENV{SPECS_EMBEDDED_ROOT}/hlml_shm.h")
  add_library(specs_embedded INTERFACE IMPORTED)
  set_target_properties(specs_embedded PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{SPECS_EMBEDDED_ROOT}")
  target_compile_definitions(specs_embedded INTERFACE PT_HLML_ENABLED)
  add_library(npu::specs_embedded ALIAS specs_embedded)
else()
  if(NOT DEFINED ENV{SPECS_EMBEDDED_ROOT})
    message(STATUS "Environment variable SPECS_EMBEDDED_ROOT is not defined.")
  endif()
  message(STATUS "Embedded specs repo not found. Will build without HLML support")
endif()

set_target_properties(SynapseUtils PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$ENV{SYNAPSE_UTILS_INCLUDE_DIR}")
target_link_libraries(SynapseUtils INTERFACE npu::Synapse npu::specs_external)
add_library(npu::SynapseUtils ALIAS SynapseUtils)

add_library(Media INTERFACE IMPORTED)
list(APPEND MEDIA_INCLUDE_DIRS "$ENV{MEDIA_ROOT}/include")
set_target_properties(Media PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${MEDIA_INCLUDE_DIRS}")

include(FetchContent)
FetchContent_Declare(
  exprtk
  GIT_REPOSITORY https://github.com/ArashPartow/exprtk.git
  GIT_TAG 0.0.3
  SOURCE_SUBDIR "exprtk")

FetchContent_MakeAvailable(exprtk)
include_directories(BEFORE SYSTEM "${FETCHCONTENT_BASE_DIR}/exprtk-src")
