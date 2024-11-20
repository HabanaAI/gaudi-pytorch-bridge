/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "habana_lazy/lazy_arg_spec.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/passes/pass_utils.h"
namespace habana_lazy {

LazyArgumentSpec::LazyArgumentSpec(
    bool with_grad,
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    size_t post_order_nodes_hash,
    const ir::ValueList& inputs,
    const ir::ValueNodeListMap& value_input_nodes_map,
    const ir::ValueList& outputs,
    const std::vector<size_t>& parent_vec,
    const std::vector<bool>& node_bcast_map) {
  PT_LAZY_TRACE;
  // Create the ArgumentSpec from nodes and inputs
  // ArgumentSpec hash is created based on the inputs
  GetArgSpecKey(with_grad, input_refs, inputs, value_input_nodes_map, outputs);

  m_post_order_nodes_hash = post_order_nodes_hash;
  HABANA_ASSERT(m_post_order_nodes_hash > 0);

  // Create final hash_code by combining the ArgumentSpec
  // and post order nodes hash
  m_hash_code = at::hash_combine(m_hash_code, m_post_order_nodes_hash);

  // Include num_outputs also part of the hash code
  auto num_outputs = outputs.size();
  m_hash_code = at::hash_combine(m_hash_code, num_outputs);

  // Include the initial duplicate information within the hash code
  for (const auto& a : parent_vec) {
    m_hash_code = at::hash_combine(m_hash_code, a);
  }
  if (!node_bcast_map.empty()) {
    std::hash<std::vector<bool>> hash_bcast;
    m_hash_code = at::hash_combine(m_hash_code, hash_bcast(node_bcast_map));
  }
}

torch::jit::Stack LazyArgumentSpec::CreateStack(
    const at::ArrayRef<torch::jit::IValue>& list) {
  // Create a torch::jit::Stack from the IValues
  return torch::jit::Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}

size_t LazyArgumentSpec::GetInputHash(
    const ir::ValueList& inputs,
    const ir::ValueNodeListMap& value_input_nodes_map) {
  PT_LAZY_TRACE;
  size_t hash_val = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    size_t input_connection_hash = i;
    HABANA_ASSERT(value_input_nodes_map.count(inputs[i]) > 0);
    auto nodes = value_input_nodes_map.at(inputs[i]);
    for (auto& node : nodes) {
      HABANA_ASSERT(node);
      input_connection_hash =
          at::hash_combine(input_connection_hash, node->get_hash());
    }
    hash_val = at::hash_combine(hash_val, input_connection_hash);
  }
  return hash_val;
}

size_t LazyArgumentSpec::GetOutputHash(const ir::ValueList& outputs) {
  PT_LAZY_TRACE;
  size_t hash_val = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    size_t output_connection_hash = i;
    if (outputs[i]) {
      auto node = outputs[i].mp_node;
      HABANA_ASSERT(node);
      output_connection_hash =
          at::hash_combine(output_connection_hash, node->get_post_order_pos());
      output_connection_hash =
          at::hash_combine(output_connection_hash, node->get_hash());
      hash_val = at::hash_combine(hash_val, output_connection_hash);
    }
  }
  return hash_val;
}

void LazyArgumentSpec::GetArgSpecKey(
    bool with_grad,
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    const ir::ValueList& inputs,
    const ir::ValueNodeListMap& value_input_nodes_map,
    const ir::ValueList& outputs) {
  // ArgumentSpecCreator requires a JIT graph to be
  // passed, where the JIT graph inputs are the only
  // content used.
  PT_LAZY_TRACE;
  m_hash_code = at::hash_combine(
      m_hash_code, GetInputHash(inputs, value_input_nodes_map));
  m_hash_code = at::hash_combine(m_hash_code, GetOutputHash(outputs));

  auto num_inputs = input_refs.size();

  uint64_t input_hash{};

  torch::jit::ArgumentSpec as(num_inputs, 0);
  for (auto& input : input_refs) {
    as.addTensor(input, with_grad);
  }
  input_hash = as.hashCode();
  m_hash_code = at::hash_combine(m_hash_code, input_hash);

  // Incorporate the memory format of the inputs within hash
  int64_t mf_hash_code{};
  for (auto const& input_ival : input_refs) {
    if (input_ival.isTensor()) {
      auto in_tensor = input_ival.toTensor();
      auto m = in_tensor.suggest_memory_format();
      int64_t m_int =
          static_cast<std::underlying_type<c10::MemoryFormat>::type>(m);
      mf_hash_code =
          at::hash_combine(mf_hash_code, at::get_hash(habana::mod_exp(m_int)));
      if (habana::is_tensor_const_with_valid_const_id(in_tensor)) {
        auto const_id = habana::get_tensor_const_id(in_tensor);
        mf_hash_code = at::hash_combine(mf_hash_code, const_id);
      }
      if (in_tensor.has_storage()) {
        auto hb_tensor = GetHbInternalTensorImpl(in_tensor);
        if (hb_tensor) {
          auto m_lazy = hb_tensor->GetTensorLayout();
          int64_t m_lazy_int =
              static_cast<std::underlying_type<habana::LayoutFormat>::type>(
                  m_lazy);
          mf_hash_code = at::hash_combine(
              mf_hash_code, at::get_hash(habana::mod_exp(m_lazy_int)));
        }
      }
    }
  }
  m_hash_code = at::hash_combine(m_hash_code, mf_hash_code);
}
} // namespace habana_lazy
