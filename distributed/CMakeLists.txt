###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

if(POLICY CMP0076)
  cmake_policy(SET CMP0076 OLD)
endif()

project(habana_pytorch_distributed_plugin VERSION 0.8.0 LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SANITIZER "build sanitizer configuration" OFF)
if(SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fsanitize-address-use-after-scope -O0")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
endif(SANITIZER)

find_program(CCACHE ccache)
if (CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE}")
endif()

set(DISABLE_CLANG_TIDY "OFF")
if (NOT DEFINED CLANG_TIDY OR CLANG_TIDY)
  find_program(CLANG_TIDY clang-tidy-10)
else (${CLANG_TIDY} STREQUAL "")
  set(DISABLE_CLANG_TIDY "ON")
endif ()
if (CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY}")
elseif (${DISABLE_CLANG_TIDY} STREQUAL "ON")
  message(AUTHOR_WARNING "clang-tidy-10 was disabled")
else ()
  message(FATAL_ERROR "clang-tidy-10 was not detected. Please install it and rerun CMake or set CLANG_TIDY to a valid path")
endif ()

# workaround for cmake3.5. Starting from 3.13 sources are optional and we can add them in add_subdirectories
add_library(habana_pytorch_distributed_plugin SHARED
  hcl_communicator.cpp
)

add_library(Synapse SHARED IMPORTED)
set_target_properties(Synapse PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libSynapse.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{SYNAPSE_ROOT}/include"
  )

add_library(tpc_kernels SHARED IMPORTED)
set_target_properties(tpc_kernels PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "$ENV{SPECS_EXT_ROOT}")

target_include_directories(habana_pytorch_distributed_plugin PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{SYNAPSE_ROOT}/include"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{ABSEIL_CPP_INCLUDE_DIR}"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers"
  )

set_target_properties(habana_pytorch_distributed_plugin PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libpytorch_synapse_helpers.so"
  )

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set( BUILD_DIST "true" )
add_subdirectory(../pytorch_helpers [${CMAKE_BINARY_DIR}])

target_compile_features(habana_pytorch_distributed_plugin PRIVATE cxx_std_14)
target_compile_definitions(habana_pytorch_distributed_plugin PRIVATE "GENERIC_HELPERS" MAX_DEVICES_PER_BOX=8)
#TODO: remove -Wno-error=deprecated-declarations after removing functions marked [[deprecated]]
target_compile_options(habana_pytorch_distributed_plugin PRIVATE -Wall -Wextra -Werror -Wno-error=deprecated-declarations)
target_link_libraries(habana_pytorch_distributed_plugin PRIVATE "${TORCH_LIBRARIES}" pytorch_synapse_helpers_abi11)
