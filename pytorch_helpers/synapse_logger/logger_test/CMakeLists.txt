###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

cmake_minimum_required(VERSION 3.5.1)

# prefer clang as it has significant speed advantage to the
# point of rendering g++ useless for larger test.cxx
set(CMAKE_CXX_COMPILER_NAMES clang++ c++ cxx)
set(CMAKE_C_COMPILER_NAMES clang) # gcc icc cc)

project(Test LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

option(SANITIZER "build sanitizer configuration" OFF)
option(GTEST "build gtest executable" OFF)
message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}, SANITIZER = ${SANITIZER}")


# disregard default cmake compilation flags in favor of setup
# that optimizes compilation time and debugability
set(CMAKE_CXX_FLAGS "-fno-var-tracking -O0 -g -fno-omit-frame-pointer")

if (SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fsanitize-address-use-after-scope")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
endif (SANITIZER)

set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS})

list(APPEND CMAKE_MODULE_PATH $ENV{PYTORCH_MODULES_ROOT_PATH}/cmake)

include(ExternalProject)
include(AddNPUStackDependencies)

add_executable(Test test.cxx ../compare.cxx)
target_link_libraries(Test PRIVATE Synapse)
set_target_properties(Test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)


if (EXISTS "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/googletest_1_13" AND GTEST)
  externalproject_add(
    googletest
    URL "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/googletest_1_13/"
    TIMEOUT 10
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -Dgtest_force_shared_crt=ON
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)
  externalproject_get_property(googletest source_dir binary_dir)
  target_include_directories(Test PRIVATE ${source_dir}/googletest/include)
  target_link_libraries(Test PRIVATE ${binary_dir}/googlemock/gtest/libgtest.a)
  target_link_libraries(Test PRIVATE ${binary_dir}/googlemock/gtest/libgtest_main.a)
  add_dependencies(Test googletest)
else ()
  target_include_directories(Test PRIVATE gtest_stub)
endif ()

target_link_libraries(Test PRIVATE pthread)
