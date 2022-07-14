/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/multinomial.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/reduction_kernels.h"

namespace habana {

template <>
LazyRandomMulti<at::Tensor>::LazyRandomMulti(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  // Seed is always at the end for all variants
  get_inputs().back() = static_cast<int64_t>(
      get_seed_hpu(inputs.back().toOptional<at::Generator>()));
}

template <>
at::Tensor LazyRandomMulti<at::Tensor>::get_result_overrideable() {
  auto t = get_inputs().at(0).toTensor();
  auto num_samples = get_inputs().at(1).toInt();
  c10::IntArrayRef out_shape;
  auto dim = t.dim() == 1 ? 1 : t.sizes()[0];
  if (t.dim() == 1) {
    int64_t data[1];
    data[0] = num_samples;
    c10::IntArrayRef shape(data, 1);
    out_shape = shape;
    PT_KERNEL_DEBUG(__func__, " Output dims:: ", out_shape);
  } else {
    int64_t data[] = {dim, num_samples};
    c10::IntArrayRef shape(data, 2);
    out_shape = shape;
    PT_KERNEL_DEBUG(__func__, " Output dims:: ", out_shape);
  }

  at::Tensor result = habana_lazy::empty_hpu_lazy(
      out_shape,
      t.options().dtype(c10::ScalarType::Long),
      t.suggest_memory_format(),
      false);

  return result;
}

template <>
LazyRandomMultiOut<at::Tensor&>::LazyRandomMultiOut(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn, -1) {
  // Seed is  at the 3rd position
  get_inputs().at(3) = static_cast<int64_t>(
      get_seed_hpu(inputs.at(3).toOptional<at::Generator>()));
}

template <>
at::Tensor& LazyRandomMultiOut<at::Tensor&>::get_result_overrideable() {
  return stack_tensor(get_inputs(), 0);
}

sizes_vec MultinomialOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& t = stack_tensor(stack, 0);
  int64_t num_samples = stack.at(1).toInt();
  auto dim = t.sizes()[0];
  if (t.dim() == 1) {
    return {{num_samples}};
  }
  return {{dim, num_samples}};
}

std::shared_ptr<void> FillMultinomialParams(
    const at::Stack& stack,
    size_t& size) {
  at::ScalarType type = stack_tensor(stack, 0).scalar_type();
  float num_samples = stack.at(1).toInt();
  bool replacement = stack.at(2).toBool();
  int seed = stack.at(3).toInt();
  PARAMS_STUB(ns_RandomMultinomial::ParamsV2);
  params->seed = seed;

  switch (type) {
    case at::ScalarType::Float:
    case at::ScalarType::BFloat16:
      params->num_samples = num_samples;
      params->replacement = replacement;
      break;
    default:
      TORCH_CHECK(false, "Unsupported type for random multinomial: ", type);
      break;
  }

  PT_KERNEL_DEBUG(
      __func__,
      " num_samples: ",
      params->num_samples,
      " replacement: ",
      params->replacement,
      " seed: ",
      params->seed);

  return params;
}

void MultinomialIntSeedInput::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  auto outshape = MultinomialOutputShape(stack, true)[0];
  auto params = FillMultinomialParams(stack, size);
  auto multinomial = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{outshape, torch::kInt, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(multinomial[0]);
}
} // namespace habana
