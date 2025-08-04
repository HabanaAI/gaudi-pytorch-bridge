###############################################################################
#
#  Copyright (c) 2024-2025 Intel Corporation
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

set(SLRG_COMMON_SRC slrg_functions.cpp shared_layer_report_generator.cpp stack_generator.cpp
                    utils/shared_structures.cpp ${CMAKE_BINARY_DIR}/generated/slrg/registry.cpp)
set(SLRG_PY_SRC ${SLRG_COMMON_SRC} slrg_bindings.cpp)

set(SLRG_LINKS Python::Module torch habana_pytorch_backend habana_pytorch_plugin)

set_source_files_properties(${CMAKE_BINARY_DIR}/generated/slrg/registry.cpp PROPERTIES COMPILE_FLAGS -w)

add_habana_library(slrg_py SHARED ${SLRG_PY_SRC})

target_compile_definitions(slrg_py PRIVATE TORCH_EXTENSION_NAME=slrg_py)
set_target_properties(slrg_py PROPERTIES PREFIX "")
target_link_libraries(slrg_py PUBLIC ${SLRG_LINKS})
target_include_directories(slrg_py PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS slrg_py LIBRARY DESTINATION habana_torch_plugin)
add_dependencies(slrg_py slrg_gen)
