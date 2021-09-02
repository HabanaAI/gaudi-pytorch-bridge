/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {

template <>
LazyRandom<at::Tensor&>::LazyRandom(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, {}, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  int64_t seed = get_seed_hpu(inputs.at(1).toOptional<at::Generator>());
  set_inputs({stack_tensor(inputs, 0), seed});
}

template <>
at::Tensor& LazyRandom<at::Tensor&>::get_result_overrideable() {
  return stack_tensor(get_inputs(), 0);
}

std::shared_ptr<void> HabanaOperatorHelper::FillRandomParams(
    const at::Stack& stack,
    size_t& size) {
  static_cast<void>(stack);
  PARAMS_STUB(ns_RandomUniform::Params);
  params->low = 0;
  params->high = std::numeric_limits<float>::max();
  params->seed = stack.at(1).toInt();

  return params;
}

void RandomOp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  kernel_meta_data_.tpc_input_order = {habana::NO_INPUTS};
  HabanaOperatorHelper::AddNode(graph, stack, is_output_persistent_list);
}
} // namespace habana
