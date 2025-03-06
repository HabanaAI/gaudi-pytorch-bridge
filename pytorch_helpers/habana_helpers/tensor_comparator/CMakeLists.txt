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

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(tensor_comparator)

set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0 -fPIC" CACHE STRING "debug flags" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fPIC" CACHE STRING "release flags" FORCE)
set(CMAKE_CXX_CLANG_TIDY "")

if(NOT DEFINED ENV{THIRD_PARTIES_ROOT})
    message(FATAL_ERROR "THIRD_PARTIES_ROOT env var doesn't exist")
endif()

add_subdirectory(src)
