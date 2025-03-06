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

function(find_absl_targets DIRECTORY)
  get_property(
    ABSL_TARGETS_IN_DIRECTORY
    DIRECTORY "${DIRECTORY}"
    PROPERTY BUILDSYSTEM_TARGETS)
  list(APPEND ABSL_TARGETS ${ABSL_TARGETS_IN_DIRECTORY})

  get_property(
    SUBDIRECTORIES
    DIRECTORY "${DIRECTORY}"
    PROPERTY SUBDIRECTORIES)
  foreach(SUBDIRECTORY IN LISTS SUBDIRECTORIES)
    find_absl_targets("${SUBDIRECTORY}")
  endforeach()

  return(PROPAGATE ABSL_TARGETS)
endfunction()

find_absl_targets($ENV{THIRD_PARTIES_ROOT}/abseil-cpp)

export(
  TARGETS ${ABSL_TARGETS}
  NAMESPACE absl::
  FILE abseilConfig.cmake)

export(PACKAGE abseil)
