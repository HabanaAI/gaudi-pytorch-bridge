/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/argmax.h"
#include "generated/argmin.h"
#include "reduction_template.h"

namespace habana {

sizes_vec ArgMinMaxOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);

  auto dim = stack.at(1);
  auto is_dim_none = dim.isNone();
  auto dim_vec =
      is_dim_none ? std::vector<int64_t>{} : std::vector<int64_t>{dim.toInt()};

  const bool keepdim = stack.at(2).toBool();
  auto shape = ReductionOutputShape(self, dim_vec, keepdim);

  return {shape};
}

template <>
ArgminmaxOutputType<at::Tensor>::ArgminmaxOutputType(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  set_scalar_type(c10::ScalarType::Long);
}

template <>
at::Tensor ArgminmaxOutputType<at::Tensor>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return {};
}

void ArgMinMax::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  const bool keepdim = stack.at(2).toBool();

  auto shape = ArgMinMaxOutputShape(stack)[0];
  auto dtype = torch::kInt;
  auto dim = stack.at(1);
  auto is_dim_none = dim.isNone();

  auto dim_vec =
      is_dim_none ? std::vector<int64_t>{} : std::vector<int64_t>{dim.toInt()};

  auto op = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {syn_in(0)},
      dim_vec,
      keepdim,
      guid_,
      {{shape, dtype, 0}});

  syn_out(0) = std::move(op[0]);
}
} // namespace habana
