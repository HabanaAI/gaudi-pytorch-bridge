cmake_minimum_required(VERSION 3.10)

set(GRAPH_WRITER_SRCS
  )
add_library(graph_writer SHARED ${GRAPH_WRITER_SRCS})
set_target_properties(graph_writer PROPERTIES
  PREFIX ""
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  CXX_INCLUDE_WHAT_YOU_USE "")  # TODO: make IWYU work with this target
target_compile_options(graph_writer
  PRIVATE -Wall -Wextra -Werror)
separate_debug_symbols(graph_writer)
add_custom_command(TARGET graph_writer POST_BUILD
  COMMENT "Creating graph_writer symbolic link in $ENV{BUILD_ROOT_LATEST}"
  COMMAND ${CMAKE_COMMAND} -E create_symlink
  "$<TARGET_FILE:graph_writer>" "$ENV{BUILD_ROOT_LATEST}/$<TARGET_FILE_NAME:graph_writer>"
  )
