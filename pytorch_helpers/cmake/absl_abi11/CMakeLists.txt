cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "")

set(CMAKE_CXX_FLAGS -w)
set(CMAKE_CXX_STANDARD 17)

set(BUILD_TESTING OFF)

add_subdirectory($ENV{THIRD_PARTIES_ROOT}/abseil-cpp absl)

include(SetIncludesAsSystem)

if ("${ABSL_LIB_LIST}" STREQUAL "")
  message(FATAL_ERROR "Abseil target list is empty. Please fill it by first including absl_abi98")
endif ()
set_includes_as_system(${ABSL_LIB_LIST})

include(ExportAbseil)
