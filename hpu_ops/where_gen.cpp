/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/where.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {
FALLBACK_CHECK(
    WhereFallbackCheck,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  if (condition.scalar_type() != torch::kBool) {
    return false;
  }

  // After type promotion, it should pick one of these guids
  //  where_fwd_i8
  //  where_fwd_i32
  //  where_fwd_bf16
  //  where_fwd_f32
  auto result_type = at::result_type(self, other);
  switch (result_type) {
    case torch::kBool:
    case torch::kInt32:
    case torch::kBFloat16:
    case torch::kFloat32:
      return true;
    default:
      return false;
  }
}

sizes_vec WhereOutputShape(const at::Stack& stack, bool) {
  at::IntArrayRef cond = stack_tensor(stack, 0).sizes();
  at::IntArrayRef self = stack_tensor(stack, 1).sizes();
  at::IntArrayRef other = stack_tensor(stack, 2).sizes();
  return {at::infer_size(at::infer_size(cond, self), other)};
}

template <>
WhereFrontend<at::Tensor>::WhereFrontend(
    const std::string& qualstring,
    const std::vector<at::IValue>& stack,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, stack, out_shapes_fn) {
  const auto& self = stack_tensor(stack, 1);
  const auto& other = stack_tensor(stack, 2);
  set_scalar_type(at::result_type(self, other));
}

template <>
at::Tensor WhereFrontend<at::Tensor>::get_result_overrideable() {
  return {};
}

template <>
WhereFrontend<at::Tensor&>::WhereFrontend(
    const std::string& qualstring,
    const std::vector<at::IValue>& stack,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, stack, out_shapes_fn) {}

template <>
at::Tensor& WhereFrontend<at::Tensor&>::get_result_overrideable() {
  return get_inputs().back().toTensor();
}

void WhereBackend::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto shape = ComputeOutputShapes(stack, true)[0];
  const auto& self = stack_tensor(stack, 1);
  const auto& other = stack_tensor(stack, 2);

  c10::optional<const at::IValue*> output = IsOutputAvailable()
      ? c10::make_optional<const at::IValue*>(&stack.back())
      : c10::nullopt;

  auto dtype_helper =
      habana_helpers::DTypeHelper::binary_op_with_type_promotion(
          {stack.at(1), stack.at(2)}, output, false);

  c10::ScalarType result_type = dtype_helper.get_result_dtype();

  std::vector<synapse_helpers::tensor> cast;
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1), syn_in(2)};

  if (self.scalar_type() != result_type) {
    cast.emplace_back(CastHelper(
        graph, syn_in(1), self.sizes(), self.scalar_type(), result_type));
    inputs[1] = cast[0].get();
  } else if (other.scalar_type() != result_type) {
    cast.emplace_back(CastHelper(
        graph, syn_in(2), other.sizes(), other.scalar_type(), result_type));
    inputs[2] = cast[0].get();
  }

  update_guid_dtype(guid_, result_type);

  auto result = BuildOp(graph, guid_, inputs, {{shape, result_type, 0}});
  syn_out(0) = std::move(result[0]);
}
} // namespace habana
