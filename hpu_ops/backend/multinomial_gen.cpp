/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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

SharedMetaDataVector MultinomialSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto self = stack_tensor(stack, 0);
  const auto selfDtype = self.scalar_type();
  const auto rank = self.dim();
  const auto& seed = stack.at(3);
  SharedMetaTensor seedSharedTensor = {1, c10::ScalarType::Int};
  if (seed.isTensor()) {
    const auto seedTensor = seed.toTensor();
    seedSharedTensor = {seedTensor.dim(), seedTensor.scalar_type()};
  }

  const auto outputDtype = selfDtype == c10::ScalarType::Float
      ? c10::ScalarType::Int
      : c10::ScalarType::Short;

  SharedMetaData multinomialSharedMeta{"random_multinomial_pt_fwd"};
  multinomialSharedMeta.inputs_data.emplace_back(rank, selfDtype);
  multinomialSharedMeta.inputs_data.push_back(seedSharedTensor);
  multinomialSharedMeta.outputs_data.emplace_back(rank, outputDtype);

  return {multinomialSharedMeta};
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
          "random_multinomial_pt_fwd",
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
          "random_multinomial_pt_fwd", stack_tensor(stack, 1).scalar_type()),
      {syn_in(1), syn_in(0)},
      {{multinomial_meta.shape, multinomial_meta.dtype, 1}},
      params.get(),
      size);
  syn_out(1) = std::move(output[0]);
}
} // namespace habana

static const auto& HabanaMultinomialKernelRegistry =
    habana::KernelRegistry().REGISTER_RANDOM_OP(multinomial, Multinomial);
