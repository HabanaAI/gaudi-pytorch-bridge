###############################################################################
# Copyright (c) 2025 Intel Corporation
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

include(FetchContent)

function(Offline_provide_dependency method dependency_name)
  set(dependency_path "${absolute_dependencies_path}/${dependency_name}")
  if(IS_DIRECTORY "${dependency_path}")
    message(STATUS "Pre-download dependency found: ${dependency_path}")

    string(TOUPPER ${dependency_name} upper_dependency_name)
    set(FETCHCONTENT_SOURCE_DIR_${upper_dependency_name} ${dependency_path})
    FetchContent_Declare(${dependency_name} ${ARGN})
    FetchContent_MakeAvailable(${dependency_name})
  else()
    message(STATUS "Pre-download dependency not found: ${dependency_path}")
  endif()
endfunction()

cmake_language(SET_DEPENDENCY_PROVIDER Offline_provide_dependency SUPPORTED_METHODS FETCHCONTENT_MAKEAVAILABLE_SERIAL)
