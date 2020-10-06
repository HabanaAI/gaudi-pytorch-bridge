include(ExternalProject)

file(GLOB_RECURSE PT_INTEGRATION_TEST_SOURCES *.cpp)

add_executable(test_pt_integration ${PT_INTEGRATION_TEST_SOURCES})

target_include_directories(
  test_pt_integration
  SYSTEM PUBLIC
  "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/abseil-cpp"
  "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/googletest/googletest/include"
  "$ENV{PYTORCH_MODULES_ROOT_PATH}"
  "$ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers"
  "$ENV{SYNAPSE_ROOT}/include"
)

#find_library(PTPY_LIB "libtorch_python.so"
#  HINTS "${TORCH_DIR}/lib/")
#message("Pytorch Python Path: ${PTPY_LIB}")

SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
target_link_libraries(
  test_pt_integration
  "${TORCH_LIBRARIES}"
# "${PTPY_LIB}"
  habana_pytorch_plugin
  gtest
  gtest_main
  util
  pthread
  stdc++
  dl)

add_custom_command(TARGET test_pt_integration POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:test_pt_integration>
  ${CMAKE_SOURCE_DIR}/tests/$<TARGET_FILE_NAME:test_pt_integration>
)
