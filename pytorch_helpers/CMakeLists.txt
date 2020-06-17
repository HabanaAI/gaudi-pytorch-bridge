###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if (BUILD_DIST)
 message ("Building only pytorch_helpers for the usage in distributed")
 set(CMAKE_POSITION_INDEPENDENT_CODE ON)

 list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
 find_package(PythonLibs REQUIRED)
 find_program(SWIG swig)

 include(SeparateDebugSymbols)
 set(CMAKE_CXX_STANDARD_REQUIRED ON)

 add_subdirectory(cmake/absl_abi98)
 add_subdirectory(cmake/absl_abi11)
 set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
 set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
 set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


 add_subdirectory(synapse_helpers)

 add_subdirectory(synapse_logger)
else ()
 list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
 include(SeparateDebugSymbols)

 add_subdirectory(cmake/absl_abi98)
 add_subdirectory(cmake/absl_abi11)
 add_subdirectory(synapse_helpers)

 find_package(PythonLibs REQUIRED)
 find_package(NumPy)
 find_program(SWIG swig)
 add_subdirectory(synapse_logger)
endif ()
