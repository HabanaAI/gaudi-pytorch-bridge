/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/bernoulli.h"
#include "generated/backend/poisson.h"
#include "generated/backend/random.h"
#include "generated/backend/uniform.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {

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
  auto outshape = stack_tensor(stack, 0).sizes();
  // The following kernels have an optional tesor input before seed tensor
  // (also optional) input. Eg stddev tensor. If we are not passing this tensor
  // to TPC we should set it as null. This is because TPC requires that all
  // leading unused optional tensors are passed as null if any valid tensor
  // (eg. seed in this case) follows them.
  static std::vector<std::string> guids_seed_tensor_pos_check = {
      "random_normal", // TPC spec tensor list : {stddev(opt), seed(opt)}
      "log_normal"}; // TPC spec tensor list : {stddev(opt), seed(opt)}

  std::vector<synTensor> inputs;

  for (size_t i = 0; i < guids_seed_tensor_pos_check.size(); i++) {
    if (guid_.find(guids_seed_tensor_pos_check[i]) != std::string::npos) {
      inputs.push_back(nullptr);
      break;
    }
  }

  inputs.push_back(syn_in(1)); // insert seed tensor
  CreateShapeTensorInput(
      graph,
      ScalarType() == c10::ScalarType::Int ? at::kFloat : ScalarType(),
      outshape,
      inputs);
  size_t size = 0;
  auto rand_params = FillParams(stack, size);

  if (ScalarType() == c10::ScalarType::Int) {
    auto rand = BuildOp(
        graph,
        update_guid_dtype(guid_, "f32"),
        inputs,
        {{outshape}},
        rand_params.get(),
        size);

    PARAMS_STUB(ns_CastKernel::Params);
    // Round down so that the upper limit is not included in the generated seq.
    // The assumption is that the float vaues dont include the upper limit.
    params->round_mode = CAST_ROUND_DOWN;
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

  auto rand = BuildOp(
      graph,
      guid_,
      inputs,
      {{outshape, ScalarType(), 0}},
      rand_params.get(),
      size);
  syn_out(0) = std::move(rand[0]);
}
} // namespace habana
