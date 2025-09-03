/**
 * Copyright (c) 2025 Intel Corporation
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

#include <c10/util/ArrayRef.h>
#include "backend/kernel/hpu_habana_launch_op_pt.h"

namespace habana::graph::pass {

using H2dScalesIndicesNames =
    std::vector<std::pair<std::vector<size_t>, std::string>>;

namespace {
std::vector<size_t> get_scales_indices(std::string_view node_name) {
  static const std::unordered_map<std::string_view, std::vector<size_t>>
      scales_indices_map = {
          {"hpu::cast_to_fp8_v2", {1}},
          {"hpu::cast_from_fp8", {1}},
          {"hpu::fp8_gemm_v2", {6, 7}},
          {"hpu::fp8_sdpa_fwd_dropout_seed", {9, 10, 11, 12, 13, 14}},
          {"hpu::fp8_sdpa_fwd_non_dropout", {8, 9, 10, 11, 12, 13}},
          {"hpu::fp8_sdpa_recomp_fwd_dropout_seed", {10, 11, 12, 13, 14, 15}},
          {"hpu::fp8_sdpa_recomp_fwd_non_dropout", {9, 10, 11, 12, 13, 14}},
          {"hpu::mixture_of_experts.fp8", {6, 7, 8, 9, 10}},
          {"hpu::mixture_of_experts.fp8_fused_weights", {5, 6, 7, 8}},
          {"hpu::mixture_of_experts.fp8_dynamic", {6, 7, 8, 9}},
          {"hpu::mixture_of_experts.fp8_fused_weights_dynamic", {5, 6, 7}}};

  if (const auto it = scales_indices_map.find(node_name);
      it != scales_indices_map.end()) {
    return it->second;
  }
  return {};
}
} // namespace

/**
 * HandleH2dScalesPass
 *
 * This pass replaces CPU scale tensors in some fp8 ops with H2D tensors.
 * Values from CPU tensors are patched into host pointers.
 *
 * Despite using H2D infrastructure, this is not related to dynamic shapes.
 * The goal is to reduce compilation time of fp8 models by patching scale values
 * in runtime, yet maintaining GC hw-scaling optimizations.
 */
struct HandleH2dScalesPass {
  explicit HandleH2dScalesPass(
      std::shared_ptr<habana_torch::jit::Graph> graph,
      H2dScalesIndicesNames& h2d_scales_idx_names,
      AdjacentCastFp8Indices& adjacent_cast_fp8_indices)
      : m_graph(std::move(graph)),
        m_h2d_scales_idx_names(h2d_scales_idx_names),
        m_adjacent_cast_fp8_indices(adjacent_cast_fp8_indices) {}

  void run(torch::jit::Stack& stack) {
    PT_EAGER_TRACE;
    processBlocks(m_graph->block(), stack);
  }

 private:
  void collectScaleIndices(
      const habana_torch::jit::Value* input,
      const torch::jit::Stack& org_stack,
      const GraphInputIndexMap& org_stack_index_map,
      const std::string& node_name,
      std::vector<size_t>& op_scale_indices) {
    // Many fp8 ops allow for None scale, so it needs to be handled here.
    if (not input->type()->cast<torch::jit::TensorType>()) {
      return;
    }
    const auto scale_name = input->debugName();
    const auto scale_idx = org_stack_index_map.at(scale_name);
    const auto scale_ivalue = org_stack[scale_idx];
    const auto scale_tensor = scale_ivalue.toTensor();

    if (scale_tensor.is_cpu()) {
      // Store CPU scales indices for later patching.
      op_scale_indices.emplace_back(scale_idx);
    } else {
      PT_BRIDGE_WARN(
          "H2D scales flow is enabled, but op ",
          node_name,
          " received non cpu scale.");
    }
  }

  void collectIndicesOfAdjacentScales(
      const habana_torch::jit::Node* node,
      const GraphInputIndexMap& org_stack_index_map) {
    static const std::unordered_set<c10::Symbol> m_logical_ops{
        c10::Symbol::fromQualString("aten::reshape"),
        c10::Symbol::fromQualString("aten::view"),
        c10::Symbol::fromQualString("aten::t"),
        c10::Symbol::fromQualString("aten::transpose"),
        c10::Symbol::fromQualString("aten::squeeze"),
        c10::Symbol::fromQualString("aten::unsqueeze"),
        c10::Symbol::fromQualString("aten::permute"),
        c10::Symbol::fromQualString("aten::expand"),
        c10::Symbol::fromQualString("aten::slice"),
        c10::Symbol::fromQualString("aten::clone")};
    static const c10::Symbol m_cast_to_fp8_symbol =
        c10::Symbol::fromQualString("hpu::cast_to_fp8_v2");
    static const c10::Symbol m_cast_from_fp8_symbol =
        c10::Symbol::fromQualString("hpu::cast_from_fp8");

    const auto node_symbol = node->kind();
    if (m_cast_to_fp8_symbol != node_symbol and
        m_cast_from_fp8_symbol != node_symbol) {
      return;
    }
    const auto& inputs = node->inputs();
    const auto input = inputs[0];
    // Logical ops are allowed to be placed between cast_to_fp8 and
    // cast_from_fp8 nodes.
    auto parent_node = input->node();
    while (m_logical_ops.count(parent_node->kind()) == 1) {
      parent_node = parent_node->inputs()[0]->node();
    }
    const auto parent_symbol = parent_node->kind();
    const bool node_is_cast_to = m_cast_to_fp8_symbol == node_symbol;
    if (not((node_is_cast_to and m_cast_from_fp8_symbol == parent_symbol) or
            (not node_is_cast_to and m_cast_to_fp8_symbol == parent_symbol))) {
      return;
    }
    const auto node_scale = inputs[1];
    if (node_scale->type()->cast<at::TensorType>() == nullptr) {
      return;
    }

    const auto parent_scale = parent_node->inputs()[1];
    if (parent_scale->type()->cast<at::TensorType>() == nullptr) {
      return;
    }

    m_adjacent_cast_fp8_indices.emplace_back(
        org_stack_index_map.at(parent_scale->debugName()),
        org_stack_index_map.at(node_scale->debugName()));
  }

  void processBlock(
      const habana_torch::jit::Block* block,
      const torch::jit::Stack& org_stack) {
    PT_EAGER_TRACE;
    HABANA_ASSERT(m_graph->inputs().size() == org_stack.size());

    GraphInputIndexMap org_stack_index_map;
    habana_helpers::createGraphInputStackIndexMap(m_graph, org_stack_index_map);

    for (const auto node : block->nodes()) {
      const auto maybe_schema = node->maybeSchema();
      if (maybe_schema == nullptr) {
        continue;
      }

      const auto& operator_name = maybe_schema->operator_name();
      std::string node_name = operator_name.name;
      const std::string node_overload = operator_name.overload_name;
      if (not node_overload.empty()) {
        node_name += "." + node_overload;
      }

      const auto scale_indices = get_scales_indices(node_name);

      if (scale_indices.empty()) {
        continue;
      }

      std::vector<size_t> op_scale_indices{};
      op_scale_indices.reserve(scale_indices.size());

      for (const size_t idx : scale_indices) {
        const auto scale = node->inputs().at(idx);
        if (scale->node()->kind() == habana_torch::jit::prim::ListConstruct) {
          for (const auto& input : scale->node()->inputs()) {
            collectScaleIndices(
                input,
                org_stack,
                org_stack_index_map,
                node_name,
                op_scale_indices);
          }
        } else {
          collectScaleIndices(
              scale,
              org_stack,
              org_stack_index_map,
              node_name,
              op_scale_indices);
        }
      }
      if (not op_scale_indices.empty()) {
        m_h2d_scales_idx_names.emplace_back(
            std::move(op_scale_indices), std::move(node_name));
      }
      if (GET_ENV_FLAG_NEW(PT_HPU_MARK_NON_RECIPROCAL_CASTS)) {
        collectIndicesOfAdjacentScales(node, org_stack_index_map);
      }
    }
    PT_BRIDGE_DEBUG(
        "Found ",
        m_adjacent_cast_fp8_indices.size(),
        " pairs of adjacent cast_to/from_fp8 nodes in the graph");
  }

  void processBlocks(
      const at::ArrayRef<habana_torch::jit::Block*> blocks,
      const torch::jit::Stack& org_stack) {
    PT_EAGER_TRACE;
    for (auto block : blocks) {
      processBlock(block, org_stack);
    }
  }

  std::shared_ptr<habana_torch::jit::Graph> m_graph;
  H2dScalesIndicesNames& m_h2d_scales_idx_names;
  AdjacentCastFp8Indices& m_adjacent_cast_fp8_indices;
};

void HandleH2dScales(
    std::shared_ptr<habana_torch::jit::Graph> graph,
    torch::jit::Stack& stack,
    H2dScalesIndicesNames& h2d_scales_idx_names,
    AdjacentCastFp8Indices& adjacent_cast_fp8_indices) {
  PT_EAGER_TRACE;
  HandleH2dScalesPass pass{
      graph, h2d_scales_idx_names, adjacent_cast_fp8_indices};
  pass.run(stack);
}

} // namespace habana::graph::pass
