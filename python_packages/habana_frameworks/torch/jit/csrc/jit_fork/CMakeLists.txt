###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

add_habana_library(
  jit_fork
  OBJECT
  frontend/error_report.cpp
  frontend/function_schema_parser.cpp
  frontend/lexer.cpp
  frontend/schema_matching.cpp
  frontend/schema_type_parser.cpp
  frontend/source_range.cpp
  ir/alias_analysis.cpp
  ir/attributes.cpp
  ir/constants.cpp
  ir/graph_utils.cpp
  ir/ir.cpp
  ir/irparser.cpp
  ir/node_hashing.cpp
  ir/scope.cpp
  ir/subgraph_matcher.cpp
  ir/type_hashing.cpp
  ir/type_wrapper.cpp
  operator_upgraders/utils.cpp
  operator_upgraders/version_map.cpp
  passes/canonicalize.cpp
  passes/dead_code_elimination.cpp
  passes/getitem_folding_pass.cpp
  passes/rem_dup_const_pass.cpp
  passes/remove_mutation.cpp
  passes/restore_mutation.cpp
  passes/utils/memory_dag.cpp
  passes/utils/subgraph_utils.cpp
  python/forked_pybind_utils.cpp)

# SynapseAI 1.24.1's installed hl_logger headers define FMT_HEADER_ONLY even
# when fmt::fmt-header-only has already supplied the same command-line define.
# GCC does not attach a suppressible warning category to this diagnostic. The
# JIT fork reaches fmt before and through hl_logger. Normalize the transitive
# value to the empty definition used by that public header so fmt remains
# header-only from the first include and the identical redefinition is quiet.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(jit_fork PRIVATE -UFMT_HEADER_ONLY -DFMT_HEADER_ONLY)
endif()

target_link_libraries(jit_fork PUBLIC bindings torch ${TORCH_PYTHON_LIBRARY})

target_include_directories(jit_fork PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

export(
  TARGETS jit_fork bindings pytorch_helpers nlohmann_json
  NAMESPACE jit_fork::
  FILE jit_forkConfig.cmake)
set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE jit_fork)
