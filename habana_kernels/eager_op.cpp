/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "habana_kernels/eager_op.h"
#include "pytorch_helpers/habana_device/hpu_cached_devices.h"

#include <torch/csrc/jit/ir/ir.h>

namespace habana {
namespace eager {

using JitValue = torch::jit::Value;

std::shared_ptr<torch::jit::Graph> create_simple_JIT(
    const at::Symbol& symbol,
    const std::vector<at::Tensor>& inputs,
    const std::vector<OutputSpec>& outputs,
    const habana_lazy::ir::MetaData& metadata) {
  auto graph = std::make_shared<torch::jit::Graph>();

  std::vector<JitValue*> args_vector;

  for (const auto& inp : inputs) {
    auto t = graph->addInput(inp.toString());
    t->setType(c10::TensorType::createContiguous(
        inp.scalar_type(), inp.device(), inp.sizes()));
    // TODO do we need debug names?
    // t->setDebugName(inp.toString());
    args_vector.push_back(t);
  }

  // Total inputs to a node is size of meta data + size of inputs
  // Allocate vector with nulllptr with inputs_size
  std::vector<JitValue*> node_inputs(
      args_vector.size() + metadata.size(), nullptr);

  // Iterate thru each of the metadata and create constant node and
  // assign this to correct index in the input array
  std::for_each(metadata.cbegin(), metadata.cend(), [&](const auto& meta_data) {
    node_inputs[meta_data.first] = graph->insertConstant(meta_data.second);
  });

  // Now we will fill the inputs in the array whereever its null
  size_t j = 0;
  std::for_each(node_inputs.begin(), node_inputs.end(), [&](auto& node) {
    if (nullptr == node) {
      node = args_vector[j++];
    }
  });
  HABANA_ASSERT(j == args_vector.size()); // make sure all the inputs were used

  // TODO Do we need scopes for single-node JIT graphs?
  //   std::shared_ptr<torch::jit::WithCurrentScope> scope_context;
  //       auto scope_name = node->GetModuleName().empty()
  //           ? (node->GetScope() ? *node->GetScope() : "")
  //           : node->GetModuleName();
  //       if (AccThread::IsAccThreadEnabled() ?
  //       !node->GetModuleName().empty()
  //                                           : node->GetScope() != NULL) {
  //         scope_context = std::make_shared<torch::jit::WithCurrentScope>(
  //             *mp_g_,
  //             c10::make_intrusive<torch::jit::Scope>(
  //                 torch::jit::ScopePtr(),
  //                 c10::Symbol::fromQualString("debug::" + scope_name)));
  //       }

  at::ArrayRef<JitValue*> args(node_inputs);
  auto jit_node = graph->create(symbol, args, outputs.size());
  // TODO scope
  //   if (AccThread::IsAccThreadEnabled()) {
  //     jit_node->setScope(c10::make_intrusive<torch::jit::Scope>(
  //         torch::jit::ScopePtr(),
  //         c10::Symbol::fromQualString("debug::" + scope_name)));
  //   }
  if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
    auto one = torch::jit::attr::alpha;
    /*Need to set this node if the deterministic mode is ON*/
    auto& device = synapse_helpers::HPURegistrar::get_device();
    jit_node->i_(one, device.getDeterministic());
    PT_BRIDGE_DEBUG(
        "Deterministic val during Jit Node creation: ", jit_node->i(one));
  }

  graph->insertNode(jit_node);

  // TODO Do we need special handling for prim::ListConstruct?
  //   if (c10::Symbol::fromQualString("prim::ListConstruct") == node->op() ||
  //       node->is_output_tensor_list()) {
  //     auto* list_node = dynamic_cast<ir::ListConstruct*>(node.get());
  //     if (list_node && list_node->isOptional()) {
  //       jit_node->output()->setType(
  //           torch::jit::ListType::create(torch::jit::OptionalType::ofTensor()));
  //     } else {
  //       jit_node->output()->setType(torch::jit::ListType::ofTensors());
  //     }
  //   } else {
  for (size_t idx = 0; idx < jit_node->outputs().size(); idx++) {
    auto jit_value_out = jit_node->output(idx);
    if (jit_node->output(idx)->type()->kind() == c10::TypeKind::TensorType) {
      const auto& out_val = outputs.at(idx);

      jit_value_out->setType(c10::TensorType::createContiguous(
          out_val.scalar_type, out_val.device, out_val.sizes));
      // TODO do we need debug names?
      // jit_value_out->setDebugName(irout_val.ToString());
    }
    graph->registerOutput(jit_value_out);
  }

  return graph;
  // TODO This is part of Create/ConstructJITGraph in HLExec. Do we need it?
  // Optimize(stack);
  // PruneDuplicateGraphInputs(parent_vec, is_duplicate_vec);
}

} // namespace eager
} // namespace habana
