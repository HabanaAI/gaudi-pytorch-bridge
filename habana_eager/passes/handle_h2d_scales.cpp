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

using H2dScalesIndicesNames = std::vector<std::pair<size_t, std::string>>;

namespace {
std::vector<size_t> get_scales_indices(std::string_view node_name) {
  if (node_name == "hpu::cast_to_fp8_v2"sv) {
    return {1};
  } else if (node_name == "hpu::fp8_gemm_v2"sv) {
    return {6, 7};
  } else if (node_name == "hpu::fp8_sdpa_fwd_dropout_seed"sv) {
    return {9, 10, 11, 12, 13, 14};
  } else if (node_name == "hpu::fp8_sdpa_fwd_non_dropout"sv) {
    return {8, 9, 10, 11, 12, 13};
  } else if (node_name == "hpu::fp8_sdpa_recomp_fwd_dropout_seed"sv) {
    return {10, 11, 12, 13, 14, 15};
  } else if (node_name == "hpu::fp8_sdpa_recomp_fwd_non_dropout"sv) {
    return {9, 10, 11, 12, 13, 14};
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
      std::shared_ptr<torch::jit::Graph> graph,
      H2dScalesIndicesNames& h2d_scales_idx_names)
      : m_graph(std::move(graph)),
        m_h2d_scales_idx_names(h2d_scales_idx_names) {}

  bool run(torch::jit::Stack& stack) {
    PT_EAGER_TRACE;
    return processBlocks(m_graph->block(), stack);
  }

 private:
  bool processBlock(torch::jit::Block* block, torch::jit::Stack& org_stack) {
    PT_EAGER_TRACE;
    HABANA_ASSERT(m_graph->inputs().size() == org_stack.size());

    GraphInputIndexMap org_stack_index_map;
    habana_helpers::createGraphInputStackIndexMap(m_graph, org_stack_index_map);
    bool changed{false};

    for (const auto node : block->nodes()) {
      const auto node_kind = node->kind().toQualString();
      const auto scale_indices = get_scales_indices(node_kind);

      if (scale_indices.empty()) {
        continue;
      }

      for (const size_t idx : scale_indices) {
        const auto scale = node->inputs().at(idx);
        const auto scale_name = scale->debugName();

        const auto scale_idx = org_stack_index_map[scale_name];
        const auto scale_ivalue = org_stack[scale_idx];

        if (not scale_ivalue.isTensor()) {
          continue;
        }

        const auto scale_tensor = scale_ivalue.toTensor();

        if (scale_tensor.device().type() != c10::DeviceType::CPU) {
          PT_BRIDGE_WARN(
              "H2D scales flow is enabled, but op ",
              node_kind,
              " received non cpu scale.");
          continue;
        }

        const auto dtype = scale_tensor.scalar_type();
        HABANA_ASSERT(
            dtype == at::ScalarType::Float or dtype == at::ScalarType::BFloat16,
            "CPU scale should be Float or BFloat16, got ",
            dtype);

        // Create H2D tensor and set it as a scale input.
        at::Tensor h2d_tensor =
            createDynamicTensor({1}, HOST_TO_DEVICE_TENSOR, dtype);
        const auto is_float = dtype == at::ScalarType::Float;
        const auto el_size = is_float ? sizeof(float_t) : sizeof(at::BFloat16);
        const auto dt_type = is_float ? habana::HostDataType::FLOAT_T
                                      : habana::HostDataType::BFLOAT16_T;

        auto tmeta{get_tensor_extra_meta(h2d_tensor)};
        tmeta->set_host_size(1);
        tmeta->set_host_el_size(el_size);
        tmeta->set_host_dt_type(dt_type);
        tmeta->set_host_total_elem(2 * el_size);

        org_stack[scale_idx] = torch::jit::IValue(h2d_tensor);

        // Store CPU scales indices for later patching.
        m_h2d_scales_idx_names.emplace_back(scale_idx, node_kind);
        changed = true;

        PT_BRIDGE_DEBUG(
            "Scale CPUTensor ",
            scale_name,
            " of node ",
            node_kind,
            " was converted to H2D tensor.");
      }
    }
    return changed;
  }

  bool processBlocks(
      at::ArrayRef<torch::jit::Block*> blocks,
      torch::jit::Stack& org_stack) {
    PT_EAGER_TRACE;
    bool changed{true};
    for (auto block : blocks)
      changed &= processBlock(block, org_stack);
    return changed;
  }

  std::shared_ptr<torch::jit::Graph> m_graph;
  H2dScalesIndicesNames& m_h2d_scales_idx_names;
};

void HandleH2dScales(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& stack,
    H2dScalesIndicesNames& h2d_scales_idx_names) {
  PT_EAGER_TRACE;
  HandleH2dScalesPass pass{graph, h2d_scales_idx_names};
  bool changed{pass.run(stack)};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
}

} // namespace habana::graph::pass
