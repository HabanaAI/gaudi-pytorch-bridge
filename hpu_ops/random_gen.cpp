/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/bernoulli.h"
#include "generated/poisson.h"
#include "generated/random.h"
#include "generated/uniform.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {

template <typename T>
LazyTensorSeed<T>::LazyTensorSeed(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  LazyTensorSeed<T>::get_inputs().back() =
      get_seed_tensor_hpu(inputs.back().toOptional<at::Generator>());
}

template <typename T>
T LazyTensorSeed<T>::get_result_overrideable() {
  return stack_tensor(LazyTensorSeed<T>::get_inputs(), 0);
}

template struct LazyTensorSeed<at::Tensor&>;
template struct LazyTensorSeed<at::Tensor>;

template <typename T>
LazyTensorOutSeed<T>::LazyTensorOutSeed(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  LazyTensorOutSeed<T>::get_inputs().at(1) =
      get_seed_tensor_hpu(inputs.at(1).toOptional<at::Generator>());
}

template <typename T>
T LazyTensorOutSeed<T>::get_result_overrideable() {
  return stack_tensor(LazyTensorOutSeed<T>::get_inputs(), 0);
}

template struct LazyTensorOutSeed<at::Tensor&>;
template struct LazyTensorOutSeed<at::Tensor>;

static std::shared_ptr<void> RandomUniformParams(
    at::ScalarType type,
    at::optional<float> from,
    at::optional<float> to,
    size_t& size) {
  PARAMS_STUB(ns_RandomUniform::Params);
  /*
  NOTE: As per PyTorch specification, for floating point types, if unspecified,
  range will be [0, 2^mantissa] to ensure that every value is representable. For
  example, torch.tensor(1, dtype=torch.double).random_() will be uniform in [0,
  2^53].
  */
  switch (type) {
    case at::ScalarType::Float:
      params->low = from.has_value() ? *from : 0;
      params->high = to.has_value() ? *to : 1 << 24;
      break;
    case at::ScalarType::BFloat16:
      params->low = from.has_value() ? *from : 0;
      params->high = to.has_value() ? *to : 1 << 8;
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

  PT_KERNEL_DEBUG(__func__, " low: ", params->low, " high: ", params->high);

  return params;
}

std::shared_ptr<void> FillRandomParams(const at::Stack& stack, size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(), c10::nullopt, c10::nullopt, size);
}

std::shared_ptr<void> FillRandomFromParams(
    const at::Stack& stack,
    size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(),
      stack.at(1).isNone() ? c10::nullopt
                           : c10::make_optional<float>(stack.at(1).toInt()),
      c10::make_optional<float>(stack.at(2).toInt()),
      size);
}

std::shared_ptr<void> FillRandomToParams(const at::Stack& stack, size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(),
      c10::nullopt,
      c10::make_optional<float>(stack.at(1).toInt()),
      size);
}

std::shared_ptr<void> FillUniformParams(const at::Stack& stack, size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(),
      c10::make_optional<float>(stack.at(1).toDouble()),
      c10::make_optional<float>(stack.at(2).toDouble()),
      size);
}

void RandomSeedTensorInput::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // Discard self tensor, input is seed tensor only
  p_context_->syn_inputs_.pop_front();
  HABANA_ASSERT(p_context_->syn_inputs_.size() == 1);

  if (ScalarType() == c10::ScalarType::Int) {
    auto outshape = stack_tensor(stack, 0).sizes();
    size_t size = 0;
    auto rand_params = FillParams(stack, size);
    auto rand = BuildOp(
        graph,
        update_guid_dtype(guid_, "f32"),
        {syn_in(0)},
        {{outshape}},
        rand_params.get(),
        size);

    PARAMS_STUB(ns_CastKernel::Params);
    auto cast = BuildOp(
        graph,
        "cast_f32_to_i32",
        {rand[0].get()},
        {{outshape, ScalarType(), 0}},
        params.get(),
        size);
    syn_out(0) = std::move(cast[0]);
    return;
  }

  OpBackend::AddNode(graph, stack);
}
} // namespace habana
