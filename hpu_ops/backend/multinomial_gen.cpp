/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "generated/backend/multinomial.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_ops/habana_random_ops.h"

namespace habana {

std::vector<int64_t> MultinomialOutputShape(const at::Stack& stack) {
  const torch::Tensor& t = stack_tensor(stack, 0);
  int64_t num_samples = stack.at(1).toInt();
  auto dim = t.sizes()[0];
  if (t.dim() == 1) {
    return {num_samples};
  }
  return {dim, num_samples};
}

static std::shared_ptr<void> MultinomialParams(
    const at::Stack& stack,
    size_t& size,
    unsigned idx_shift = 0) {
  at::ScalarType type = stack_tensor(stack, 0 + idx_shift).scalar_type();
  float num_samples = stack.at(1 + idx_shift).toInt();
  bool replacement = stack.at(2 + idx_shift).toBool();
  const torch::Tensor& t = stack_tensor(stack, 0 + idx_shift);

  PARAMS_STUB(ns_RandomMultinomial::ParamsV2);

  switch (type) {
    case at::ScalarType::Float:
    case at::ScalarType::BFloat16:
    case at::ScalarType::Half:
      params->num_samples = num_samples;
      params->replacement = replacement;
      params->outcomes = t.sizes()[0];
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
      params->replacement);

  return params;
}

OutputMetaDataVector MultinomialMeta(const at::Stack& stack) {
  return {OutputMetaData(at::ScalarType::Long, MultinomialOutputShape(stack))};
}

std::shared_ptr<void> FillMultinomialParams(
    const at::Stack& stack,
    size_t& size) {
  return MultinomialParams(stack, size);
}

std::shared_ptr<void> FillHabanaMultinomialParams(
    const at::Stack& stack,
    size_t& size) {
  return MultinomialParams(stack, size, 1);
}

OutputMetaDataVector HabanaMultinomialMeta(const at::Stack& stack) {
  const auto& t = stack_tensor(stack, 1);
  const int64_t num_samples = stack.at(2).toInt();

  OutputMetaData meta;
  meta.shape = t.dim() == 1 ? std::vector<int64_t>{num_samples}
                            : std::vector<int64_t>{t.sizes()[0], num_samples};
  meta.dtype = at::ScalarType::Long;
  return {meta};
}

HabanaMultinomial::HabanaMultinomial(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "random_multinomial_pt_fwd",
          scalar_type,
          {1},
          {},
          {},
          false) {
  SetOutputMetaFn(HabanaMultinomialMeta);
  SetFillParams(FillHabanaMultinomialParams);
  kernel_meta_data_.tpc_input_order = {1, 0};
}

void HabanaMultinomial::CustomHandler(
    synapse_helpers::graph&,
    at::Stack& stack) {
  SetGuid(get_guid_with_precision(
      "random_multinomial_pt_fwd", stack_tensor(stack, 1).scalar_type()));
}
} // namespace habana

static const auto& HabanaMultinomialKernelRegistry =
    habana::KernelRegistry().add(
        "hpu::habana_multinomial",
        KERNEL_FN_GLOBAL(habana::HabanaMultinomial));
