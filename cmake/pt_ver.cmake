###############################################################################
# Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

macro(detect_pt_version)
  message(VERBOSE "Detecting PT version...")
  list(
    APPEND
    PT_VER_PRINTER_LINES
    "#include <torch/version.h>"
    "#include <cstdio>"
    ""
    "#define str(a) str_internal(a)"
    "#define str_internal(a) #a"
    ""
    "int main() {"
    "  puts(str(TORCH_VERSION_MAJOR) \"\;\""
    "    str(TORCH_VERSION_MINOR) \"\;\""
    "    str(TORCH_VERSION_PATCH) \"\;\""
    "    str(TORCH_VERSION) \"\;\""
    "    str(PYTORCH_FORK_MAJOR) \"\;\""
    "    str(PYTORCH_FORK_MINOR))\;"
    "}")
  list(JOIN PT_VER_PRINTER_LINES "\n" CMAKE_CONFIGURABLE_FILE_CONTENT)
  unset(PT_VER_PRINTER_LINES)

  configure_file("${CMAKE_ROOT}/Modules/CMakeConfigurableFile.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/pt_version_printer.cpp" @ONLY)
  unset(CMAKE_CONFIGURABLE_FILE_CONTENT)

  try_run(
    PT_VER_RUN_RESULT PT_VER_COMPILE_RESULT "${PROJECT_BINARY_DIR}" SOURCES
    "${CMAKE_CURRENT_BINARY_DIR}/pt_version_printer.cpp"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${TORCH_INCLUDE_DIRS}"
    RUN_OUTPUT_STDOUT_VARIABLE PT_VERSIONS
    RUN_OUTPUT_STDERR_VARIABLE PT_VERSIONS_STDERR
    COMPILE_OUTPUT_VARIABLE PT_VER_COMPILE_OUTPUT)

  if(NOT ${PT_VER_COMPILE_RESULT})
    message(FATAL_ERROR "Could not compile exec for PyTorch version detection. Output: \n${PT_VER_COMPILE_OUTPUT}")
  endif()

  if(NOT PT_VERSIONS_STDERR STREQUAL "")
    message(FATAL_ERROR "Errors while running PyTorch version detection tool: \n${PT_VERSIONS_STDERR}")
  endif()

  list(GET PT_VERSIONS 0 TORCH_VERSION_MAJOR)
  list(GET PT_VERSIONS 1 TORCH_VERSION_MINOR)
  list(GET PT_VERSIONS 2 TORCH_VERSION_PATCH)
  list(GET PT_VERSIONS 3 TORCH_VERSION)
  string(REPLACE "\"" "" TORCH_VERSION "${TORCH_VERSION}")
  list(GET PT_VERSIONS 4 PYTORCH_FORK_MAJOR)
  list(GET PT_VERSIONS 5 PYTORCH_FORK_MINOR)

  message(STATUS "PyTorch version detected: ${TORCH_VERSION}")
endmacro(detect_pt_version)

macro(find_most_recent_pt_ver)
  execute_process(
    COMMAND python3 most_recent_pt_ver.py
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/scripts"
    OUTPUT_VARIABLE PT_VER_NEWEST
    RESULT_VARIABLE PT_VER_NEWEST_FAILED)
  if(PT_VER_NEWEST_FAILED)
    message(
      FATAL_ERROR
        "Failed to pick newest PT version include directory from <root>/pt_ver/*"
    )
  endif()
  message(
    WARNING
      "Version specific include dir for ${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR} is not present. Trying ${PT_VER_NEWEST} instead."
  )
  set(PT_VER_DIR "${PROJECT_SOURCE_DIR}/pt_ver/${PT_VER_NEWEST}")
endmacro()

macro(set_up_pt_ver_mechanism)
  detect_pt_version()
  set(PT_VER_DIR
      "${PROJECT_SOURCE_DIR}/pt_ver/${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}"
  )
  if(NOT EXISTS "${PT_VER_DIR}")
    find_most_recent_pt_ver()
  endif()
  message(VERBOSE "PT_VER includes will be taken from ${PT_VER_DIR}")
  target_include_directories(torch BEFORE INTERFACE "${PT_VER_DIR}")
endmacro()
