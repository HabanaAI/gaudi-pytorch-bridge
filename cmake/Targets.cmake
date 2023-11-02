# ##############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ##############################################################################

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
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wno-error=deprecated-declarations)

  # TODO: Reenable disabled warnings
  # TODO: Add -Wconversion
  target_compile_options(${TARGET_NAME} PRIVATE
    -Wno-unused-parameter)

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

function(add_habana_library TARGET_NAME)
  add_library(${TARGET_NAME} ${ARGN})
  add_library(npu::${TARGET_NAME} ALIAS ${TARGET_NAME})

  find_keyword(INTERFACE IS_INTERFACE ${ARGN})

  if(NOT IS_INTERFACE)
    set_up_warnings(${TARGET_NAME})
  endif()
endfunction()

function(add_habana_executable TARGET_NAME)
  add_executable(${TARGET_NAME} ${ARGN})
  add_executable(npu::${TARGET_NAME} ALIAS ${TARGET_NAME})

  find_keyword(INTERFACE IS_INTERFACE ${ARGN})

  if(NOT IS_INTERFACE)
    set_up_warnings(${TARGET_NAME})
  endif()
endfunction()
