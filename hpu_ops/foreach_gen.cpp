/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"

namespace habana {
template <>
ForeachFE<::std::vector<at::Tensor>>::ForeachFE(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<::std::vector<at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
::std::vector<at::Tensor> ForeachFE<
    ::std::vector<at::Tensor>>::get_result_overrideable() {
  ::std::vector<at::Tensor> tensors;
  for (const auto& tensor : get_inputs()[0].toTensorList()) {
    tensors.emplace_back(at::empty_like(tensor));
  }
  return tensors;
}

void Foreach::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out = BuildOp(
        graph, guid_, {syn_in(i)}, {{tensor.sizes(), tensor.scalar_type(), i}});
    syn_out(i) = std::move(out[0]);
  }
}
} // namespace habana
