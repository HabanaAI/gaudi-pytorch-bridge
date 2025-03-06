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

add_habana_library(graph_writer SHARED)
set_target_properties(
  graph_writer
  PROPERTIES PREFIX ""
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             CXX_INCLUDE_WHAT_YOU_USE "") # TODO: make IWYU work with this target

separate_debug_symbols(graph_writer)
add_custom_command(
  TARGET graph_writer
  POST_BUILD
  COMMENT "Creating graph_writer symbolic link in $ENV{BUILD_ROOT_LATEST}"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:graph_writer>"
          "$ENV{BUILD_ROOT_LATEST}/$<TARGET_FILE_NAME:graph_writer>")
