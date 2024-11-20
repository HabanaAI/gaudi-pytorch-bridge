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
#pragma once

#include "backend/habana_operator.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ir_utils.h"
namespace habana_lazy {
/**
 * LazyArgumentSpec
 *
 * This spec creates a hash_code for a given post_order graph
 * and input IValues.
 */
class LazyArgumentSpec {
 public:
  LazyArgumentSpec(
      bool with_grad,
      const at::ArrayRef<torch::jit::IValue>& input_refs,
      size_t post_order_nodes_hash,
      const ir::ValueList& inputs,
      const ir::ValueNodeListMap& value_input_nodes_map,
      const ir::ValueList& outputs,
      const std::vector<size_t>& parent_vec,
      const std::vector<bool>& node_bcast_map = {});

  bool operator==(const LazyArgumentSpec& rv) const {
    return m_hash_code == rv.m_hash_code &&
        m_post_order_nodes_hash == rv.m_post_order_nodes_hash;
  }

  bool operator!=(const LazyArgumentSpec& rv) const {
    return !(*this == rv);
  }

  size_t hashCode() const {
    return m_hash_code;
  }

 private:
  torch::jit::Stack CreateStack(const at::ArrayRef<torch::jit::IValue>& list);

  void GetArgSpecKey(
      bool with_grad,
      const at::ArrayRef<torch::jit::IValue>& input_refs,
      const ir::ValueList& inputs,
      const ir::ValueNodeListMap& value_input_nodes_map,
      const ir::ValueList& outputs);

  size_t GetInputHash(
      const ir::ValueList& inputs,
      const ir::ValueNodeListMap& value_input_nodes_map);

  size_t GetOutputHash(const ir::ValueList& outputs);

  size_t m_post_order_nodes_hash;
  size_t m_hash_code = 0;
};
} // namespace habana_lazy
