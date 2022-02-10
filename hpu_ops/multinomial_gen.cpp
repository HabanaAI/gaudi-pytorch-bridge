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

static std::shared_ptr<void> FillExponentParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomExponential::Params);
  float lambd = 1;
  uint32_t seed = stack.at(3).to<uint32_t>();
  params->beta = 1.0 / lambd;
  params->seed = seed;
  return params;
}

static std::shared_ptr<void> FillArgMaxParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_Reduction::Params);
  auto r_dim = -1;
  auto ndim = stack.at(0).toTensor().dim();
  r_dim = c10::maybe_wrap_dim(r_dim, ndim, /*keepdim=*/true);
  auto dim = ndim - 1 - r_dim;
  params->reductionDimension = dim;
  return params;
}

static std::shared_ptr<void> FillTopkParams(
    const at::Stack& stack,
    size_t& size) {
  auto ndim = stack.at(0).toTensor().dim();
  long int num_samples = stack.at(1).toInt();
  int64_t dim = at::maybe_wrap_dim(/*dim=*/-1, ndim, /*wrap_scalar=*/true);
  HPU_PARAMS_STUB(synBeamParams);
  params->bsw = num_samples;
  params->axis = ndim - dim - 1;
  params->bottomK = false; // bottomK = !largest
  return params;
}

static std::shared_ptr<void> FillReduceSumParams(size_t& size) {
  PARAMS_STUB(ns_Reduction::Params);
  params->reductionDimension = 0;
  return params;
}

void MultinomialIntSeedInput::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = ComputeOutputShapes(stack, true)[0];
  auto self = stack.at(0).toTensor();
  size_t size = 0;
  auto params_redsum = FillReduceSumParams(size);
  auto params = FillParams(stack, size);
  bool with_replacement = stack.at(2).toBool();
  auto shape = stack_tensor(stack, 0).sizes();
  if (with_replacement) {
    // random_multinomial kernel requires normalized input
    auto reduce_sum_shape = ReduceOperator::compute_output_shape(self, 1, true);
    auto reduce_sum = BuildOp(
        graph,
        "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{reduce_sum_shape, ScalarType()}},
        params_redsum.get(),
        size);
    auto norm_inp = BuildOp(
        graph,
        "div_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), reduce_sum[0].get()},
        {{shape, ScalarType()}});
    auto multinomial = BuildOp(
        graph,
        "random_multinomial_fwd_" +
            habana_helpers::name_suffix_from_type(ScalarType()),
        {norm_inp[0].get()},
        {{outshape, ScalarType(), 0}},
        params.get(),
        size);
    syn_out(0) = std::move(multinomial[0]);
  } else {
    long int num_samples = stack.at(1).toInt();
    auto params = FillExponentParams(stack, size);
    // exponential distribution
    auto exponential = BuildOp(
        graph,
        "random_exponential_fwd_" +
            habana_helpers::name_suffix_from_type(ScalarType()),
        {},
        {{shape, ScalarType()}},
        params.get(),
        size);
    // exponential / input tensor
    auto div_out = BuildOp(
        graph,
        "div_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {exponential[0].get(), syn_in(0)},
        {{shape, ScalarType()}});
    if (num_samples == 1) {
      const auto& params = FillArgMaxParams(stack, size);
      auto argshape = ReduceOperator::compute_output_shape(self, {-1}, true);
      // argmax of div_out
      auto result = BuildOp(
          graph,
          "argmax_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
          {div_out[0].get()},
          {{argshape, ScalarType(), 0}},
          params.get(),
          size);
      syn_out(0) = std::move(result[0]);
    } else {
      const auto& params = FillTopkParams(stack, size);
      std::vector<int64_t> topkshape = {shape[0], num_samples};
      auto dtype = c10::ScalarType::Int;
      // top k sample of argmax
      auto Topk = BuildOp(
          graph,
          "topk",
          {div_out[0].get()},
          {{topkshape, dtype}, {topkshape, dtype, 0}},
          params.get(),
          size);
      syn_out(0) = std::move(Topk[1]);
    }
  }
}
} // namespace habana
