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
#include "hpu_op_helper.h"

namespace habana {
template <>
LazyFrexp<std::tuple<at::Tensor, at::Tensor>>::LazyFrexp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
std::tuple<at::Tensor, at::Tensor> LazyFrexp<
    std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  auto t = get_inputs().at(0).toTensor();
  c10::IntArrayRef out_shape = t.sizes();
  at::Tensor mantissa = habana_lazy::empty_hpu_lazy(
      out_shape, t.options(), t.suggest_memory_format(), false);
  at::Tensor exponent = habana_lazy::empty_hpu_lazy(
      out_shape,
      t.options().dtype(c10::ScalarType::Int),
      t.suggest_memory_format(),
      false);
  return {mantissa, exponent};
}

sizes_vec FrexpOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> shape = self.sizes().vec();
  return {{shape, shape}};
}

void Frexp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = FrexpOutputShape(stack)[0];
  auto frexp = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{outshape, c10::ScalarType::Int, 1}, {outshape, ScalarType(), 0}});

  syn_out(0) = std::move(frexp[1]);
  syn_out(1) = std::move(frexp[0]);
}

} // namespace habana
