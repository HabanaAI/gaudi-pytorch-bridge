add_habana_library(graph_writer SHARED)
set_target_properties(
  graph_writer
  PROPERTIES PREFIX ""
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             CXX_INCLUDE_WHAT_YOU_USE "") # TODO: make IWYU work with this target

separate_debug_symbols(graph_writer)
add_custom_command(
  TARGET graph_writer
  POST_BUILD
  COMMENT "Creating graph_writer symbolic link in $ENV{BUILD_ROOT_LATEST}"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:graph_writer>"
          "$ENV{BUILD_ROOT_LATEST}/$<TARGET_FILE_NAME:graph_writer>")
