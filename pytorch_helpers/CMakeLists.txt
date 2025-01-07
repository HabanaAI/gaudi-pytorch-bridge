###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

add_habana_library(pytorch_helpers INTERFACE)
target_include_directories(pytorch_helpers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pytorch_helpers INTERFACE nlohmann_json absl::variant)

if (BUILD_DIST)
 message ("Building only pytorch_helpers for the usage in distributed")
 set(CMAKE_POSITION_INDEPENDENT_CODE ON)

 list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
 find_program(SWIG swig)
 add_subdirectory(../third_party/pybind11 [${CMAKE_BINARY_DIR}])

 include(SeparateDebugSymbols)
 set(CMAKE_CXX_STANDARD_REQUIRED ON)
 set(CMAKE_CXX_STANDARD 17)

 set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
 set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
 set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

 if(MANYLINUX)
 add_subdirectory(synapse_shim)
 endif()
 add_subdirectory(habana_helpers)
 add_subdirectory(synapse_logger)
else ()
 list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
 include(SeparateDebugSymbols)

 add_subdirectory(cmake/absl)
 if(MANYLINUX)
 add_subdirectory(synapse_shim)
 endif()
 add_subdirectory(habana_helpers)

 find_program(SWIG swig)
 add_subdirectory(synapse_logger)
endif ()

add_subdirectory(synapse_utils_shim)

add_subdirectory(dataloader/habana_dataloader)
