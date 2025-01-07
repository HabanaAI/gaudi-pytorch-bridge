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

# Hacky workaround because we are stuck with an ancient CMake version.
# TODO: Once we have CMake >= 3.11, revert the whole commit that introduced the hack and use FetchContent instead.
function(set_includes_as_system)
  if (${ARGC} EQUAL 0)
    message(FATAL_ERROR "set_includes_as_system called with no arguments")
  endif ()

  set(empty "")
  foreach (target ${ARGV})
    get_target_property(include_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(${target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${empty}")
    target_include_directories(${target} SYSTEM INTERFACE ${include_dirs})
  endforeach ()
endfunction()
