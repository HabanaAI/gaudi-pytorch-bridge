/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "hpu_ops/habana_random_ops.h"
#include "pytorch_helpers/habana_helpers/conversion.h"
#include "pytorch_helpers/habana_helpers/logging.h"

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

static FillParamsT MultinomialParams(
    const at::Stack& stack,
    unsigned idx_shift = 0) {
  at::ScalarType type = stack_tensor(stack, 0 + idx_shift).scalar_type();
  auto num_samples =
      safe_convert<int>(stack.at(1 + idx_shift).toInt(), "num_samples"sv);
  bool replacement = stack.at(2 + idx_shift).toBool();
  const torch::Tensor& t = stack_tensor(stack, 0 + idx_shift);

  PARAMS_STUB(ns_RandomMultinomial::ParamsV2);

  switch (type) {
    case at::ScalarType::Float:
    case at::ScalarType::BFloat16:
    case at::ScalarType::Half:
      params->num_samples = num_samples;
      params->replacement = replacement;
      params->outcomes = safe_convert<int>(t.sizes()[0], "outcomes"sv);
      break;
    default:
      HABANA_ASSERT(false, "Unsupported type for random multinomial: ", type);
      break;
  }

  PT_KERNEL_DEBUG(
      __func__,
      " num_samples: ",
      params->num_samples,
      " replacement: ",
      params->replacement);

  return paramsT;
}

OutputMetaDataVector MultinomialMeta(const at::Stack& stack) {
  return {OutputMetaData(at::ScalarType::Long, MultinomialOutputShape(stack))};
}

SharedMetaDataVector MultinomialSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
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

FillParamsT FillMultinomialParams(const at::Stack& stack) {
  return MultinomialParams(stack);
}

FillParamsT FillHabanaMultinomialParams(const at::Stack& stack) {
  return MultinomialParams(stack, 1);
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
    : HabanaRandomBase(
          device_id,
          "random_multinomial_pt_fwd",
          scalar_type,
          {1}) {
  SetOutputMetaFn(HabanaMultinomialMeta);
  SetFillParams(FillHabanaMultinomialParams);
  kernel_meta_data_.tpc_input_order = {1, 0};
}

using namespace std::literals;

void HabanaMultinomial::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  SetGuid(get_guid_with_precision(
      "random_multinomial_pt_fwd"sv, stack_tensor(stack, 1).scalar_type()));
  OpBackend::AddNode(graph, stack);
}
} // namespace habana

static const auto& HabanaMultinomialKernelRegistry =
    habana::KernelRegistry().REGISTER_HABANA_RANDOM_OP(
        multinomial,
        Multinomial);
