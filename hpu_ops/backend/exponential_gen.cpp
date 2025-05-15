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

#include "generated/backend/exponential.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/habana_random_ops.h"

namespace habana {

namespace {

OutputMetaDataVector ExponentialMetaCommon(
    const at::Stack& stack,
    size_t self_idx) {
  const auto& self = stack.at(self_idx).toTensor();
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

FillParamsT FillExponentialParamsCommon(
    const at::Stack& stack,
    size_t lambd_idx) {
  PARAMS_STUB(ns_RandomExponential::Params);
  float lambd = stack.at(lambd_idx).toScalar().toFloat();
  HABANA_ASSERT(
      lambd > 0.0,
      "exponential_ expects lambda > 0.0, but found lambda=",
      lambd);
  params->beta = 1.0f / lambd;
  return paramsT;
}

} // namespace

OutputMetaDataVector ExponentialMeta(const at::Stack& stack) {
  return ExponentialMetaCommon(stack, 0);
}

SharedMetaDataVector ExponentialSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto input = stack_tensor(stack, 0);
  auto dtype = input.scalar_type();
  auto rank = input.dim();
  auto seed = stack.at(2);

  SharedMetaData randomSharedMeta("random_exponential_fwd");
  randomSharedMeta.inputs_data.emplace_back(
      1, seed.isTensor() ? seed.toTensor().scalar_type() : at::ScalarType::Int);
  randomSharedMeta.outputs_data.emplace_back(rank, dtype);
  return {randomSharedMeta};
}

FillParamsT FillExponentialParams(const at::Stack& stack) {
  return FillExponentialParamsCommon(stack, 1);
}

void ExponentialSeedTensorInput::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = ExponentialMeta(stack)[0];
  auto params = FillExponentialParams(stack);
  std::vector<synTensor> inputs;

  if (stack.at(2).isTensor())
    inputs.push_back(syn_in(1));
  else
    inputs.push_back(syn_seed());

  CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  using namespace std::literals;
  auto exponential = BuildOp(
      graph,
      get_guid_with_precision("random_exponential_fwd"sv, meta.dtype),
      std::move(inputs),
      {{meta.shape, meta.dtype, 0}},
      params.ptr(),
      params.size());
  syn_out(0) = std::move(exponential[0]);
}

//===----------------------------------------------------------------------===//
// This is the implementation of custom exponential op in `torch.compile`
//===----------------------------------------------------------------------===//

OutputMetaDataVector HabanaExponentialMeta(const at::Stack& stack) {
  return ExponentialMetaCommon(stack, 1);
}

FillParamsT FillHabanaExponentialParams(const at::Stack& stack) {
  return FillExponentialParamsCommon(stack, 2);
}

HabanaExponential::HabanaExponential(int device_id, c10::ScalarType scalar_type)
    : HabanaRandomBase(device_id, "random_exponential_fwd", scalar_type, {1}) {
  SetOutputMetaFn(HabanaExponentialMeta);
  SetFillParams(FillHabanaExponentialParams);
}
} // namespace habana

static const auto& HabanaExponentialKernelRegistry =
    habana::KernelRegistry().REGISTER_HABANA_RANDOM_OP(
        exponential,
        Exponential);
