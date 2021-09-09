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
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  // Seed is always at the end for all variants
  get_inputs().back() = static_cast<int64_t>(
      get_seed_hpu(inputs.back().toOptional<at::Generator>()));
}

template <>
at::Tensor& LazyRandom<at::Tensor&>::get_result_overrideable() {
  return stack_tensor(get_inputs(), 0);
}

static std::shared_ptr<void> RandomUniformParams(
    at::ScalarType type,
    at::optional<float> from,
    at::optional<float> to,
    int seed,
    size_t& size) {
  PARAMS_STUB(ns_RandomUniform::Params);
  params->seed = seed;

  switch (type) {
    case at::ScalarType::Float:
    case at::ScalarType::BFloat16:
      params->low = from.has_value() ? *from : 0;
      params->high = to.has_value() ? *to : std::numeric_limits<float>::max();
      break;
    case at::ScalarType::Int:
      params->low = from.has_value() ? *from : 0;
      params->high = to.has_value()
          ? *to
          : static_cast<float>(std::numeric_limits<int>::max());
      break;
    default:
      TORCH_CHECK(false, "Got unsupported type for random uniform: ", type);
      break;
  }

  PT_KERNEL_DEBUG(
      __func__,
      " low: ",
      params->low,
      " high: ",
      params->high,
      " seed: ",
      params->seed);

  return params;
}

std::shared_ptr<void> HabanaOperatorHelper::FillRandomParams(
    const at::Stack& stack,
    size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(),
      c10::nullopt,
      c10::nullopt,
      stack.at(1).toInt(),
      size);
}

std::shared_ptr<void> HabanaOperatorHelper::FillRandomFromParams(
    const at::Stack& stack,
    size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(),
      stack.at(1).isNone() ? c10::nullopt
                           : c10::make_optional<float>(stack.at(1).toInt()),
      c10::make_optional<float>(stack.at(2).toInt()),
      stack.at(3).toInt(),
      size);
}

std::shared_ptr<void> HabanaOperatorHelper::FillRandomToParams(
    const at::Stack& stack,
    size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(),
      c10::nullopt,
      c10::make_optional<float>(stack.at(1).toInt()),
      stack.at(2).toInt(),
      size);
}

void RandomOp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  if (ScalarType() == c10::ScalarType::Int) {
    size_t size = 0;
    auto rand_params = FillParams(stack, size);
    auto rand = BuildOp(
        graph,
        "random_uniform_fwd_f32",
        {},
        {{stack_tensor(stack, 0).sizes()}},
        rand_params.get(),
        size);

    PARAMS_STUB(ns_CastKernel::IntParams);
    params->cast_mode = _CastIntMode_t::TRUNCATE;
    auto cast = BuildOp(
        graph,
        "cast_f32_to_i32",
        {rand[0].get()},
        {{stack_tensor(stack, 0).sizes(),
          ScalarType(),
          is_output_persistent_list[0],
          true}},
        params.get(),
        size);
    syn_out(0) = std::move(cast[0]);
    return;
  }

  kernel_meta_data_.tpc_input_order = {habana::NO_INPUTS};
  HabanaOperatorHelper::AddNode(graph, stack, is_output_persistent_list);
}
} // namespace habana
