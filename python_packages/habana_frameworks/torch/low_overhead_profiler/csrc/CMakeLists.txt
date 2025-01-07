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

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(TARGET_NAME "_lop_profiler_C")
add_library(${TARGET_NAME} SHARED
    bindings.cpp)
target_compile_definitions(${TARGET_NAME} PRIVATE TORCH_EXTENSION_NAME=${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
target_link_libraries(${TARGET_NAME} PRIVATE bindings torch ${TORCH_PYTHON_LIBRARY})
install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION habana_torch_plugin/${PYBIND_INSTALL_DIRNAME})
