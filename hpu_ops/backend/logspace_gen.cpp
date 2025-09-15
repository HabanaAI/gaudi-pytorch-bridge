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

#include "generated/backend/logspace.h"
#include "habana_helpers/conversion.h"

namespace habana {

OutputMetaDataVector LogspaceMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = {stack.at(2).toInt()};

  meta.dtype = stack.at(4).toOptional<at::ScalarType>().value_or(
      torch::get_default_dtype_as_scalartype());
  meta.layout =
      stack.at(5).toOptional<at::Layout>().value_or(at::Layout::Strided);

  const auto device = stack.at(6).toOptional<at::Device>().value_or(at::kHPU);
  TORCH_INTERNAL_ASSERT(device.is_hpu());

  const bool pin_memory = stack.at(7).toOptional<bool>().value_or(false);
  HABANA_ASSERT(!pin_memory, "Only dense CPU tensors can be pinned");

  return {meta};
}

OutputMetaDataVector LogspaceOutMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = {stack.at(2).toInt()};
  meta.dtype = stack.at(4).toTensor().scalar_type();

  return {meta};
}

// AddNode function is needed as the bridge cannot infer correctly the precision
// type in a compile mode.
void LogSpace::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  auto params = FillParams(stack);

  syn_out(0) = std::move(BuildOp(
      graph,
      get_guid_with_precision("logspace_fwd"sv, meta.dtype),
      {},
      {{meta.shape, meta.dtype, 0}},
      params.ptr(),
      params.size())[0]);
}

FillParamsT FillLogspaceFwdParams(const at::Stack& stack) {
  PARAMS_STUB(ns_Logspace::Params);
  params->start = stack[0].toScalar().to<float>();
  params->end = stack[1].toScalar().to<float>();
  using namespace std::literals;
  params->steps = safe_convert<int>(stack[2].toScalar().to<int32_t>());
  params->base = stack[3].toScalar().to<float>();

  return paramsT;
}

SharedMetaDataVector LogspaceSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  c10::ScalarType dtype;
  if (stack.at(4).isTensor())
    dtype = stack.at(4).toTensor().scalar_type();
  else
    dtype = stack.at(4).toOptional<at::ScalarType>().value_or(
        torch::get_default_dtype_as_scalartype());

  SharedMetaData powSharedMeta{"logspace_fwd"};
  powSharedMeta.outputs_data.emplace_back(1, dtype);
  return {powSharedMeta};
}
} // namespace habana
