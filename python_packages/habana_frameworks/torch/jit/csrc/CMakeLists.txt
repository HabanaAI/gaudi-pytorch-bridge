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

set(TARGET_NAME "_torch_jit_C")
set(TORCH_JIT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_JIT_FORK_ROOT "${TORCH_JIT_ROOT}/jit_fork")

set(JIT_FORK_SRCS)
add_subdirectory(jit_fork)

add_habana_library(${TARGET_NAME} SHARED
    init_python_jit_bindings.cpp
    _C.cpp
    ${JIT_FORK_SRCS}
    )

target_compile_definitions(${TARGET_NAME} PRIVATE TORCH_EXTENSION_NAME=${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
target_link_libraries(${TARGET_NAME} PUBLIC bindings torch ${TORCH_PYTHON_LIBRARY})
target_include_directories(${TARGET_NAME} PRIVATE "${TORCH_JIT_ROOT}")


install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION habana_torch_plugin/${PYBIND_INSTALL_DIRNAME})
