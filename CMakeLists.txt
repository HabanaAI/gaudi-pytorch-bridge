cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0076 OLD)

project(habana_pytorch_plugin VERSION 0.5.0 LANGUAGES CXX)
find_package(Torch REQUIRED)

# workaround for cmake3.5. Starting from 3.13 sources are optional and we can add them in add_subdirectories
add_library(habana_pytorch_plugin SHARED
  habana_device/HPUAllocator.cpp
  habana_device/HPUContext.cpp
  habana_device/HPUGuardImpl.cpp
  habana_kernels/basic_kernels.cpp
)

# Enable C++11
target_compile_features(habana_pytorch_plugin PRIVATE cxx_std_11)

# Link against LibTorch and synapse
target_link_libraries(habana_pytorch_plugin PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(habana_pytorch_plugin PRIVATE Synapse)

add_library(Synapse SHARED IMPORTED)
set_target_properties(Synapse PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libSynapse.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HABANA_NPU_STACK_PATH}" )

target_include_directories(habana_pytorch_plugin PRIVATE ${CMAKE_SOURCE_DIR})
# target_include_directories(habana_pytorch_plugin PRIVATE DirectoryA)

# add_subdirectory(habana_device) # we've already added all source files
# add_subdirectory(habana_kernels)

# Create symlink to latest build from tests dir
add_custom_command(TARGET habana_pytorch_plugin POST_BUILD COMMAND
  $(CMAKE_COMMAND) -E create_symlink $<TARGET_FILE:habana_pytorch_plugin>
  $(CMAKE_SOURCE_DIR)/tests/$<TARGET_FILE_NAME:habana_pytorch_plugin>)