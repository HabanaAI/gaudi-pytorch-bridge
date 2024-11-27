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

set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "")

set(CMAKE_CXX_FLAGS -w)
set(CMAKE_CXX_STANDARD 17)

set(BUILD_TESTING OFF)

set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory($ENV{THIRD_PARTIES_ROOT}/abseil-cpp absl SYSTEM)

include(ExportAbseil)
