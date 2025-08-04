###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH_FULL
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH_SHORT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND git diff HEAD --quiet
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  RESULT_VARIABLE GIT_TREE_IS_DIRTY)

add_habana_library(_git_info_C SHARED bindings.cpp)
target_compile_definitions(
  _git_info_C PRIVATE TORCH_EXTENSION_NAME=_git_info_C GIT_COMMIT_HASH_SHORT="${GIT_COMMIT_HASH_SHORT}"
                      GIT_COMMIT_HASH_FULL="${GIT_COMMIT_HASH_FULL}" GIT_TREE_IS_DIRTY="${GIT_TREE_IS_DIRTY}")
set_target_properties(_git_info_C PROPERTIES PREFIX "")
target_link_libraries(_git_info_C PUBLIC bindings torch absl::strings ${TORCH_PYTHON_LIBRARY})

install(TARGETS _git_info_C LIBRARY DESTINATION habana_torch_plugin/${PYBIND_INSTALL_DIRNAME})
