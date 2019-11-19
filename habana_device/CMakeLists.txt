# Define our library target
add_library(habana_device SHARED HPUGuardImpl.cpp)
# Enable C++11
target_compile_features(habana_device PRIVATE cxx_std_11)
# Link against LibTorch
target_link_libraries(habana_device PRIVATE "${TORCH_LIBRARIES}")
# Create symlink to latest build from tests dir
add_custom_command(TARGET habana_device POST_BUILD COMMAND
  $(CMAKE_COMMAND) -E create_symlink $<TARGET_FILE:habana_device>
  $(CMAKE_SOURCE_DIR)/tests/$<TARGET_FILE_NAME:habana_device>)
