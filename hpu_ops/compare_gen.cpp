/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/eq.h"
#include "generated/ge.h"
#include "generated/gt.h"
#include "generated/le.h"
#include "generated/logical_and.h"
#include "generated/logical_or.h"
#include "generated/logical_xor.h"
#include "generated/lt.h"
#include "generated/ne.h"

namespace habana {
template <>
LazyCmp<at::Tensor>::LazyCmp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    bool,
    bool,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  auto x = get_inputs();
  // convert scalar input to tensor to avoid cache misses in cases where scalar
  // value changes across iterations
  if (x[1].isScalar()) {
    auto self = x[0].toTensor();
    auto other = x[1].toScalar();
    auto dtype = at::result_type(self, other);
    auto other_tensor = habana_lazy::get_tensor_for_scalar(
        other.toDouble(), self.options().dtype(dtype));
    x[1] = c10::IValue(other_tensor);
    set_inputs(x);
  }
}

template <>
at::Tensor LazyCmp<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  auto shape = BinaryOutputShape(inputs, false)[0];
  return habana_lazy::empty_hpu_lazy(
      shape, t.options().dtype(at::kBool), t.suggest_memory_format(), false);
}

void CompareOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = BinaryOutputShape(stack, true)[0];
  auto result =
      BuildOp(graph, guid_, {syn_in(0), syn_in(1)}, {{outshape, at::kBool, 0}});

  syn_out(0) = std::move(result[0]);
}
} // namespace habana
