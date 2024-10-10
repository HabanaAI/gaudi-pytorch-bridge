###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################
cmake_minimum_required(VERSION 3.10)

function (separate_debug_symbols target)

  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(TARGET_NAME $<TARGET_FILE:${target}>)
    if (DEFINED $ENV{TARGET_NAME})
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND strip ${TARGET_NAME} --only-keep-debug -o ${TARGET_NAME}.debug
          COMMAND strip ${TARGET_NAME} --strip-unneeded
          COMMAND objcopy --add-gnu-debuglink=${TARGET_NAME}.debug ${TARGET_NAME}
          COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${TARGET_NAME}.debug" "$ENV{BUILD_ROOT_LATEST}/${TARGET_NAME}.debug"
          COMMENT "Separating debug symbols of ${target}")
    endif()
  endif()

endfunction()
