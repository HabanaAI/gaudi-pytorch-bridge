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

OutputMetaData HabanaMultinomialMetaCommon(const at::Stack& stack) {
  const auto& t = stack_tensor(stack, 1);
  const int64_t num_samples = stack.at(2).toInt();

  OutputMetaData meta;
  meta.shape = t.dim() == 1 ? std::vector<int64_t>{num_samples}
                            : std::vector<int64_t>{t.sizes()[0], num_samples};
  meta.dtype = at::ScalarType::Long;
  return meta;
}

OutputMetaDataVector HabanaMultinomialMeta(const at::Stack& stack) {
  return {HabanaMultinomialMetaCommon(stack)};
}

OutputMetaDataVector HabanaMultinomialCheckpointMeta(const at::Stack& stack) {
  return {SeedOutputMeta(), HabanaMultinomialMetaCommon(stack)};
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

HabanaMultinomialCheckpoint::HabanaMultinomialCheckpoint(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "random_multinomial",
          scalar_type,
          {0, 1},
          {},
          {},
          false) {
  SetOutputMetaFn(HabanaMultinomialCheckpointMeta);
  SetFillParams(FillHabanaMultinomialParams);
}

void HabanaMultinomialCheckpoint::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = GetOutputMetaData();
  const auto& seed_meta = meta[0];
  const auto& multinomial_meta = meta[1];
  auto seed = BuildOp(
      graph, "identity", {syn_in(0)}, {{seed_meta.shape, seed_meta.dtype, 0}});
  syn_out(0) = std::move(seed[0]);

  size_t size = 0;
  auto params = FillParams(stack, size);

  auto output = BuildOp(
      graph,
      get_guid_with_precision(
          "random_multinomial", stack_tensor(stack, 1).scalar_type()),
      {syn_in(1), syn_in(0)},
      {{multinomial_meta.shape, multinomial_meta.dtype, 1}},
      params.get(),
      size);
  syn_out(1) = std::move(output[0]);
}
} // namespace habana

static const auto& HabanaMultinomialKernelRegistry =
    habana::KernelRegistry().REGISTER_RANDOM_OP(multinomial, Multinomial);
