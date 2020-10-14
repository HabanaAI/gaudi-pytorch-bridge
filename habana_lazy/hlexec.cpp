/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "hlexec.h"
#include "ops/constant.h"
#include "ops/convolution.h"

namespace habana_lazy {
namespace exec {
HlExec::HlExec() {
  mp_g_ = std::make_shared<Graph>();
}

HlExec::HlExec(ScopePtr scope) {
  mp_g_ = std::make_shared<Graph>(scope);
}

/*
 * Establish mapping between JIT IR value pointer and
 * habana lazy tensor
 */
void HlExec::Bind(const HabanaLazyTensorPtrList& inputs) {
  //
  // Iterate thru each inputs and create binding for the same
#if 0
  for (const auto& i : inputs) {
    PyValuePtr v = std::make_shared<PyValue>(mp_g_->addInput(i->atTensor().name()));

    if (tensorbind_.count(i) == false) {
      tensorbind_.insert({i, v});
    } else {
      assert(0);
    }
  }
#endif
}

/*
 * Creates the Graph
 */
std::tuple<LazyValueToJitValueMap, LazyValueToJitValueMap> HlExec::Create(
    const ir::NodePtrList nodes,
    const ir::ValueList inputs,
    const ir::ValueList outputs) {
  LazyValueToJitValueMap value_map, input_map, output_map;

  for (auto inp : inputs) {
    auto t = mp_g_->addInput(inp.ToString());
    value_map[inp] = input_map[inp] = t;
  }

  for (auto node : nodes) {
    // Is it a scalar node?
    if (c10::Symbol::fromQualString("prim::constant") == node->op()) {
      // add constant
      auto scalar_node = dynamic_cast<ir::ScalarConstant*>(node.get());
      auto scalar_const = scalar_node->getIValue();
      // TBD: Should we create a constant node, or should it be
      // a 1-element tensor as input?
      // Keeping it as a constant node
      // allows for optimized graph (no DMA required, some optimizations
      // like avoiding multiply with 1 can be removed).
      // Keeping it as a variable (1-elem input) allows to be able to
      // reuse the same graph when the scalar values change.
      auto c = mp_g_->insertConstant(scalar_const);
      value_map[node->GetOutput(0)] = c;
    } else if (node->ToString().find("hpu::input") != std::string::npos) {
      // Its a tensor, should already be there in the value maps
      HABANA_ASSERT(value_map.find(node->GetOutput(0)) != value_map.end());
    } else {
      std::vector<JitValue*> args_vector;
      auto node_input_vals = node->GetInputs();
      std::transform(
          node_input_vals.begin(),
          node_input_vals.end(),
          std::back_inserter(args_vector),
          [&](HabanaLazyValue inp) -> JitValue* {
            auto it = value_map.find(inp);
            HABANA_ASSERT(it != value_map.end());
            return it->second;
          });

      // Total inputs to a node is size of meta data + size of inputs
      // Allocate vector with nulllptr with inputs_size
      std::vector<JitValue*> node_inputs(
          args_vector.size() + node->GetMetaData().size(), nullptr);

      // Iterate thru each of the metadata and create constant node and
      // assign this to correct index in the input array
      std::for_each(
          node->GetMetaData().cbegin(),
          node->GetMetaData().cend(),
          [&](const auto& meta_data) {
            HABANA_ASSERT(node_inputs[meta_data.first] == nullptr);
            node_inputs[meta_data.first] =
                mp_g_->insertConstant(meta_data.second);
          });

      // Now we will fill the inputs in the array whereever its null
      size_t j = 0;
      std::for_each(node_inputs.begin(), node_inputs.end(), [&](auto& node) {
        if (nullptr == node) {
          node = args_vector[j++];
        }
      });
      HABANA_ASSERT(j == args_vector.size());

      at::ArrayRef<JitValue*> args(node_inputs);
      auto jit_node = mp_g_->create(node->op(), args, node->GetNumOutputs());
      mp_g_->insertNode(jit_node);
      auto jit_outputs = jit_node->outputs();
      int i = 0;
      for (const auto jit_output : jit_outputs) {
        value_map[node->GetOutput(i++)] = jit_output;
      }
    }
  }

  for (auto output : outputs) {
    output_map[output] = value_map[output];
    mp_g_->registerOutput(value_map[output]);
  }

  return std::make_tuple(input_map, output_map);
}

} // namespace exec
} // namespace habana_lazy
