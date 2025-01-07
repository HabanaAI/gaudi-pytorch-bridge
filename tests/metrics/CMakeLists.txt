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

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
project(metrics_tests LANGUAGES CXX)
add_habana_library(metrics_tests SHARED metrics_test_op.cpp metrics_test_kernel.cpp)
target_include_directories(metrics_tests PRIVATE $ENV{PYTORCH_MODULES_ROOT_PATH} $ENV{SWTOOLS_SDK_ROOT}/hl_logger/include/ $ENV{THIRD_PARTIES_ROOT}/magic_enum-0.8.1/include $ENV{THIRD_PARTIES_ROOT} $ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers $ENV{PYTORCH_MODULES_ROOT_PATH}/backend/helpers $ENV{THIRD_PARTIES_ROOT}/abseil-cpp $ENV{THIRD_PARTIES_ROOT}/fmt-9.1.0/include $ENV{SYNAPSE_ROOT}/include)
target_link_libraries(metrics_tests "${TORCH_LIBRARIES}")

