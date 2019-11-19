# Define our library target
add_library(habana_kernels SHARED basic_kernels.cpp)
# Enable C++11
target_compile_features(habana_kernels PRIVATE cxx_std_11)
# Link against LibTorch
target_link_libraries(habana_kernels PRIVATE "${TORCH_LIBRARIES}")
# Create symlink to latest build from tests dir
add_custom_command(TARGET habana_kernels POST_BUILD COMMAND
  $(CMAKE_COMMAND) -E create_symlink $<TARGET_FILE:habana_kernels>
  $(CMAKE_SOURCE_DIR)/tests/$<TARGET_FILE_NAME:habana_kernels>)
