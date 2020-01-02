cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0076 OLD)

project(habana_pytorch_plugin VERSION 0.5.0 LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory("$ENV{ABSEIL_CPP_INCLUDE_DIR}" absl)
find_package(Torch REQUIRED)
find_package(synapse_helpers REQUIRED)

# workaround for cmake3.5. Starting from 3.13 sources are optional and we can add them in add_subdirectories
add_library(habana_pytorch_plugin SHARED
  habana_device/HPUAllocator.cpp
  habana_device/HPUContext.cpp
  habana_device/HPUGuardImpl.cpp
  habana_helpers/tensor_utils.cpp
  habana_kernels/basic_kernels.cpp
  habana_kernels/conv_kernels.cpp
  habana_kernels/conv_pool_utils.cpp
  habana_kernels/kernel_utils.cpp
  habana_kernels/linear_kernels.cpp
  habana_kernels/pool_kernels.cpp
  habana_kernels/softmax_kernels.cpp
  habana_kernels/unary_kernels.cpp
)

# Enable C++11
target_compile_features(habana_pytorch_plugin PRIVATE cxx_std_11)
target_compile_definitions(habana_pytorch_plugin PRIVATE "GENERIC_HELPERS" MAX_DEVICES_PER_BOX=8)

# Link against LibTorch and synapse
target_link_libraries(habana_pytorch_plugin PRIVATE "${TORCH_LIBRARIES}"
  absl::flags
  absl::variant
  synapse_helpers::synapse_helpers
)

add_library(Synapse SHARED IMPORTED)
set_target_properties(Synapse PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libSynapse.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HABANA_NPU_STACK_PATH}"
)
target_include_directories(habana_pytorch_plugin PRIVATE ${CMAKE_SOURCE_DIR})

# Create symlink to latest build from tests dir
add_custom_command(TARGET habana_pytorch_plugin POST_BUILD COMMAND
  $(CMAKE_COMMAND) -E create_symlink $<TARGET_FILE:habana_pytorch_plugin>
  $(CMAKE_SOURCE_DIR)/tests/$<TARGET_FILE_NAME:habana_pytorch_plugin>
)