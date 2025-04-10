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

cmake_minimum_required(VERSION 3.26)

function(separate_debug_symbols target)

  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(target_name $<TARGET_FILE:${target}>)
    if(DEFINED $ENV{target_name})
      add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND strip ${target_name} --only-keep-debug -o ${target_name}.debug
        COMMAND strip ${target_name} --strip-unneeded
        COMMAND objcopy --add-gnu-debuglink=${target_name}.debug ${target_name}
        COMMAND ${CMAKE_COMMAND} -E create_symlink "${target_name}.debug" "$ENV{BUILD_ROOT_LATEST}/${target_name}.debug"
        COMMENT "Separating debug symbols of ${target}")
    endif()
  endif()

endfunction()
