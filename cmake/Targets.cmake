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
  target_compile_options(${TARGET_NAME} PRIVATE
    -Wno-unused-parameter -Wno-unused-variable -Wno-strict-aliasing -Wno-array-bounds
    -Wno-sign-compare)

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
