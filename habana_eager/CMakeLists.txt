###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

add_habana_library(habana_pytorch2_frontend OBJECT
          ${CMAKE_SOURCE_DIR}/hpu_ops/cpu_fallback.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/op_validator.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/add_composite_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/arange_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/bernoulli_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/binary_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/index_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/convolution_backward_overrideable_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/convolution_overrideable_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/fill_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/random_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/topk_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/eager/optimizer_lamb_gen.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/op_logger.cpp
          ${CMAKE_SOURCE_DIR}/hpu_ops/supported_dtypes.cpp
          ops/as_strided.cpp
          ops/bincount.cpp
          ops/empty.cpp
          ops/copy_from.cpp
          ops/complex.cpp
          ops/index_put.cpp
          ops/instance_norm.cpp
          ops/local_scalar_dense.cpp
          ops/matmul.cpp
          ops/masked_select.cpp
          ops/nonzero.cpp
          ops/unique.cpp
          ops/unique2.cpp
          ops/set.cpp
          ops/view.cpp
          ops/pin_memory.cpp
          ops/eager_op.cpp
          ops/repeat.cpp
          ops/linear.cpp
          passes/add_attribute_alpha.cpp
          passes/detect_weights_tensors.cpp
          passes/get_outputs_order_in_graph.cpp
          passes/handle_input_views.cpp
          passes/handle_tupple_on_output.cpp
          passes/remove_detach_op.cpp
          passes/remove_dummy_output.cpp
          passes/replace_getitem_with_list_unpack.cpp
          passes/sanitize_graph_input.cpp
          passes/replace_dynamic_ops.cpp
          passes/mark_params_as_const.cpp
          eager_custom_op_register.cpp
          eager_tensor.cpp
          eager_exec.cpp
          eager_view.cpp
          eager_context.cpp
          graph_execs_group.cpp
          graph_exec.cpp
          graph_storage.cpp
          graph_dynamic.cpp
          graph_dynamic_ops.cpp
          graph_weight_permute.cpp
          kernels_impl.cpp
          strided_dynamic_ops.cpp
          unused_link_symbols.cpp
          user_custom_op_frontend.cpp
          ../hpu_ops/op_validator.cpp
          # autogenerated files:
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op0.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op1.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op2.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op3.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op4.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op5.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op6.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op7.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op8.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op9.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op_custom.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op_quant.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/hpu_op_torchvision.cpp
          ${CMAKE_BINARY_DIR}/generated/eager/wrap_kernels_registrations.cpp)

target_include_directories(habana_pytorch2_frontend
  PUBLIC
  ${PROJECT_SOURCE_DIR})
target_include_directories(habana_pytorch2_frontend SYSTEM
  PUBLIC
  $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:SynapseUtils,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:hcl,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_definitions(habana_pytorch2_frontend PUBLIC GENERIC_HELPERS)
set_target_properties(habana_pytorch2_frontend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(habana_pytorch2_frontend PRIVATE ${PROJECT_SOURCE_DIR}/pytorch_helpers)
target_include_directories(habana_pytorch2_frontend PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(habana_pytorch2_frontend
  PRIVATE
  Python::Python
  habana_serialization
  pytorch_synapse_utils_shim
  torch
  nlohmann_json::nlohmann_json
  absl::strings
  absl::str_format
  npu::hllogger
)

# this is to make sure that gen_op.py is invoked before habana_pytorch2_frontend
# Custom command is specified in CMake there, so we cannot otherwise ensure proper order
add_dependencies(habana_pytorch2_frontend habana_pytorch_frontend_ops)
