cmake_minimum_required(VERSION 3.5)

set(GTEST_DIR "${CMAKE_BINARY_DIR}/gtest")
set(HABANA_BUILD_ROOT "$ENV{BUILD_ROOT_LATEST}")
message("Root Path: $ENV{BUILD_ROOT_LATEST}")
set(PT_MODULES_ROOT "$ENV{PYTORCH_MODULES_ROOT_PATH}")
set(INTERFACE_INCLUDE_DIRECTORIES  "$ENV{SYNAPSE_ROOT}/include")
set(ABSEIL_ROOT "$ENV{THIRD_PARTIES_ROOT}/abseil-cpp")

project(pt_integration_test)

execute_process(COMMAND
  python -c "import os, torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE TORCH_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Torch REQUIRED)

include(ExternalProject)
set_directory_properties(PROPERTIES EP_PREFIX "${GTEST_DIR}")

ExternalProject_Add(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 6f5fd0d7199b9a19faa
  SOURCE_DIR "${GTEST_DIR}/src/googletest-src"
  BINARY_DIR "${GTEST_DIR}/src/googletest-build"
  # Disable install step
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
)

ExternalProject_Get_Property(googletest SOURCE_DIR)

set(PT_INTEGRATION_TEST_SOURCES
  debug_utils_tests.cpp
  eager_kernel_tests.cpp
  post_order_tests.cpp
  lazy_jit_graph_tests.cpp
  lazy_kernel_tests.cpp
  main.cpp
  post_order_tests.cpp
)

add_executable(test_pt_integration ${PT_INTEGRATION_TEST_SOURCES})

target_include_directories(
  test_pt_integration
  SYSTEM PUBLIC
  "${SOURCE_DIR}/googletest/include"
  "${PT_MODULES_ROOT}"
  "${INTERFACE_INCLUDE_DIRECTORIES}"
  "${PT_MODULES_ROOT}/pytorch_helpers"
  "${ABSEIL_ROOT}"
)

add_dependencies(test_pt_integration googletest)

ExternalProject_Get_Property(googletest BINARY_DIR)

find_library(PLUGIN_LIB "libhabana_pytorch_plugin.so"
  HINTS "${HABANA_BUILD_ROOT}")
message("Plugin Path: ${PLUGIN_LIB}")
find_library(PTPY_LIB "libtorch_python.so"
  HINTS "${TORCH_DIR}/lib/")
message("Pytorch Python Path: ${PTPY_LIB}")

SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
target_link_libraries(
  test_pt_integration
  "${TORCH_LIBRARIES}"
  "${BINARY_DIR}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a"
  "${PLUGIN_LIB}"
  -lutil
  -pthread
  -lstdc++
  -ldl)
