###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
################################################################################
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

include(ExternalProject)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

file(GLOB_RECURSE PT_INTEGRATION_TEST_SOURCES *.cpp)
#list(FILTER PT_INTEGRATION_TEST_SOURCES EXCLUDE REGEX
#   ".*testfile.cpp" example to exclude file from compilation
#)

add_executable(test_pt_integration ${PT_INTEGRATION_TEST_SOURCES})

target_include_directories(
  test_pt_integration
  SYSTEM PUBLIC
  "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/abseil-cpp"
  "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/googletest_1_10/googletest/include"
  "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/json/single_include"
  "$ENV{PYTORCH_MODULES_ROOT_PATH}"
  "$ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers"
  "$ENV{SYNAPSE_ROOT}/include"
)
#add_compile_definitions(MAX_DEVICES_PER_BOX=8)
target_compile_definitions(test_pt_integration PRIVATE "TESTS" MAX_DEVICES_PER_BOX=8)

#find_library(PTPY_LIB "libtorch_python.so"
#  HINTS "${TORCH_DIR}/lib/")
#message("Pytorch Python Path: ${PTPY_LIB}")

file(GLOB PT_INTEGRATION_TEST_JSON_FILES *.json)

execute_process(COMMAND cp ${PT_INTEGRATION_TEST_JSON_FILES} ${CMAKE_BINARY_DIR}/.)
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
target_link_libraries(test_pt_integration PRIVATE
  "${TORCH_LIBRARIES}"
  habana_pytorch_plugin
  habana_serialization
  pytorch_synapse_helpers_abi11
  gtest
  gtest_main
  util
  pthread
  stdc++
  stdc++fs
  dl)

add_custom_command(TARGET test_pt_integration POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:test_pt_integration>
  ${CMAKE_SOURCE_DIR}/tests/$<TARGET_FILE_NAME:test_pt_integration>
)
