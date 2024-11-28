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

function(find_keyword KEYWORD RESULT_VAR)
  set(${RESULT_VAR}
      FALSE
      PARENT_SCOPE)

  foreach(arg IN LISTS ARGN)
    if(arg STREQUAL ${KEYWORD})
      set(${RESULT_VAR}
          TRUE
          PARENT_SCOPE)
      break()
    endif()
  endforeach()
endfunction()

function(set_up_warnings TARGET_NAME)
  # TODO: Add -Wconversion
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wno-error=deprecated-declarations)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0.0")
    # According to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80635 GCC older than 11 may trigger
    # bugous maybe-uninitialized warning for std::optional destructor. And this was observed on d10
    # build with gcc8.3.0.
    # As a W/A don't emit error in this case.
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-error=maybe-uninitialized)
  endif()

  if(PROJECT_IS_TOP_LEVEL)
    target_compile_options(${TARGET_NAME} PRIVATE -Werror)
  endif()
endfunction()

function(attach_sanitizers_if_requested TARGET_NAME)
  if(SANITIZER)
    target_compile_options(${TARGET_NAME} PRIVATE -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr
                                                 -fsanitize-address-use-after-scope -Og)
    target_link_options(${TARGET_NAME} PRIVATE -fsanitize=address -fsanitize=leak -fsanitize=undefined)
  endif()

  if(THREAD_SANITIZER)
    target_compile_options(${TARGET_NAME} PRIVATE -O0 -g3 -fsanitize=thread)
  endif()
endfunction()

function(allow_code_coverage_if_requested TARGET_NAME)
  if(CODE_COVERAGE)
    target_compile_options(${TARGET_NAME} PRIVATE --coverage -O0)
    target_link_libraries(${TARGET_NAME} PRIVATE --coverage)
  endif()
endfunction()

function(add_habana_library TARGET_NAME)
  add_library(${TARGET_NAME} ${ARGN})
  add_library(npu::${TARGET_NAME} ALIAS ${TARGET_NAME})

  find_keyword(INTERFACE IS_INTERFACE ${ARGN})

  if(NOT IS_INTERFACE)
    set_up_warnings(${TARGET_NAME})
    attach_sanitizers_if_requested(${TARGET_NAME})
    allow_code_coverage_if_requested(${TARGET_NAME})
  endif()
endfunction()

function(add_habana_executable TARGET_NAME)
  add_executable(${TARGET_NAME} ${ARGN})
  add_executable(npu::${TARGET_NAME} ALIAS ${TARGET_NAME})

  find_keyword(INTERFACE IS_INTERFACE ${ARGN})

  if(NOT IS_INTERFACE)
    set_up_warnings(${TARGET_NAME})
    attach_sanitizers_if_requested(${TARGET_NAME})
    allow_code_coverage_if_requested(${TARGET_NAME})
  endif()
endfunction()


if(SANITIZER)
  message("Building sanitizers configuration")
endif()

if(THREAD_SANITIZER)
  message("Building thread sanitizer configuration")
endif()
