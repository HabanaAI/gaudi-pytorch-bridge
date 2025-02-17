###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

set(SYNAPSE_LOGGER_SRCS
  ./synapse_logger.cpp
  ./syn_training_impl.cpp
  )

add_habana_library(pytorch_synapse_logger SHARED ${SYNAPSE_LOGGER_SRCS})
set_target_properties(pytorch_synapse_logger PROPERTIES PREFIX "")
target_include_directories(pytorch_synapse_logger PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(pytorch_synapse_logger SYSTEM
                           PRIVATE $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(pytorch_synapse_logger PRIVATE ${CMAKE_DL_LIBS})
add_custom_command(TARGET pytorch_synapse_logger POST_BUILD
  COMMENT "Creating pytorch_synapse_logger symbolic link in $ENV{BUILD_ROOT_LATEST}"
  COMMAND ${CMAKE_COMMAND} -E create_symlink
  "$<TARGET_FILE:pytorch_synapse_logger>" "$ENV{BUILD_ROOT_LATEST}/$<TARGET_FILE_NAME:pytorch_synapse_logger>")
separate_debug_symbols(pytorch_synapse_logger)
target_link_libraries(pytorch_synapse_logger PRIVATE ${CMAKE_DL_LIBS} absl::strings absl::str_format absl::variant)
install(TARGETS pytorch_synapse_logger LIBRARY DESTINATION habana_torch_plugin/lib)

set(PY_SYNAPSE_LOGGER_I ${CMAKE_CURRENT_SOURCE_DIR}/py_synapse_logger.i)
#set(SWIG "SWIG-NOTFOUND")
if (${SWIG} STREQUAL "SWIG-NOTFOUND")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/py_synapse_logger_wrap.cpp ${CMAKE_BINARY_DIR}/py_synapse_logger.py
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/py_synapse_logger_wrap.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/py_synapse_logger_wrap.cpp
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/py_synapse_logger.py
    ${CMAKE_BINARY_DIR}/py_synapse_logger.py
    COMMENT "no swig! will copy stashed version of the interface"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/py_synapse_logger_wrap.cpp ${CMAKE_CURRENT_SOURCE_DIR}/py_synapse_logger.py
  )
else ()
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/py_synapse_logger_wrap.cpp ${CMAKE_BINARY_DIR}/py_synapse_logger.py
    COMMAND ${SWIG} -c++ -o ${CMAKE_CURRENT_BINARY_DIR}/py_synapse_logger_wrap.cpp -python ${PY_SYNAPSE_LOGGER_I}
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/py_synapse_logger.py
    ${CMAKE_BINARY_DIR}/py_synapse_logger.py
    DEPENDS ${PY_SYNAPSE_LOGGER_I} ${CMAKE_CURRENT_SOURCE_DIR}/py_synapse_logger.h
  )
endif()
# don't run linters nor warning checks on autogenerated files
add_library(py_pytorch_synapse_logger_wrap OBJECT ${CMAKE_CURRENT_BINARY_DIR}/py_synapse_logger_wrap.cpp)
set_target_properties(py_pytorch_synapse_logger_wrap PROPERTIES CXX_CLANG_TIDY "" CXX_INCLUDE_WHAT_YOU_USE "")
target_compile_features(py_pytorch_synapse_logger_wrap PRIVATE cxx_std_11)
target_compile_options(py_pytorch_synapse_logger_wrap PRIVATE -fno-exceptions -w)
target_include_directories(py_pytorch_synapse_logger_wrap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(py_pytorch_synapse_logger_wrap PRIVATE Python::Python Python::NumPy)

add_habana_library(_py_pytorch_synapse_logger SHARED ${SYNAPSE_LOGGER_SRCS})
target_compile_features(_py_pytorch_synapse_logger PRIVATE cxx_std_11)
set_target_properties(_py_pytorch_synapse_logger PROPERTIES PREFIX "")
target_compile_options(_py_pytorch_synapse_logger PRIVATE -fno-exceptions)
target_compile_definitions(_py_pytorch_synapse_logger PUBLIC -DBINARY_NAME="_py_pytorch_synapse_logger.so")
target_include_directories(_py_pytorch_synapse_logger PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(_py_pytorch_synapse_logger SYSTEM
                           PRIVATE $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>)

separate_debug_symbols(_py_pytorch_synapse_logger)
target_link_libraries(
  _py_pytorch_synapse_logger
  PRIVATE Python::Python
          Python::NumPy
          $<TARGET_OBJECTS:py_pytorch_synapse_logger_wrap>
          ${CMAKE_DL_LIBS}
          absl::strings
          absl::str_format
          absl::variant)

set_target_properties(pytorch_synapse_logger _py_pytorch_synapse_logger py_pytorch_synapse_logger_wrap PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  # TODO: Add CXX_EXTENSIONS OFF
  )

add_dependencies(_py_pytorch_synapse_logger py_pytorch_synapse_logger_wrap)
add_custom_command(TARGET _py_pytorch_synapse_logger POST_BUILD
  COMMENT "Creating _py_pytorch_synapse_logger symbolic links in $ENV{BUILD_ROOT_LATEST}"
  COMMAND ${CMAKE_COMMAND} -E create_symlink
  "$<TARGET_FILE:_py_pytorch_synapse_logger>" "$ENV{BUILD_ROOT_LATEST}/$<TARGET_FILE_NAME:_py_pytorch_synapse_logger>"
  COMMAND ${CMAKE_COMMAND} -E create_symlink
  "${CMAKE_BINARY_DIR}/py_synapse_logger.py" "$ENV{BUILD_ROOT_LATEST}/py_pytorch_synapse_logger.py")

install(TARGETS _py_pytorch_synapse_logger LIBRARY DESTINATION habana_torch_plugin/lib)
install(FILES $ENV{BUILD_ROOT_LATEST}/py_pytorch_synapse_logger.py DESTINATION habana_torch_plugin/lib)
