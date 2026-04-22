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
#include "generated/backend/isfinite.h"
#include "generated/backend/isinf.h"
#include "generated/backend/isnan.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

SharedMetaDataVector IsFiniteSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return IsFiniteInfNanSharedMeta(stack, "isfinite_fwd");
}

SharedMetaDataVector IsInfSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return IsFiniteInfNanSharedMeta(stack, "isinf_fwd");
}

SharedMetaDataVector IsNanSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  auto shared_meta = IsFiniteInfNanSharedMeta(stack, "isnan_fwd");

  const auto& input = stack_tensor(stack, 0);
  auto dtype = input.scalar_type();

  if (dtype == at::ScalarType::Float8_e5m2 or
      dtype == at::ScalarType::Float8_e4m3fn) {
    shared_meta[0].inputs_data[0].second = at::ScalarType::BFloat16;
  }

  return shared_meta;
}

OutputMetaDataVector IsFiniteInfNanMeta(const at::Stack& stack) {
  const at::Tensor& self = stack_tensor(stack, 0);
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = self.sizes().vec();
  meta.dtype = at::kBool;
  return metaVec;
}

void _IsFiniteInfNan::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto params = FillParams(stack);
  auto meta = IsFiniteInfNanMeta(stack)[0];
  auto dtype = stack_tensor(stack, 0).scalar_type();
  // use cguid autocast
  if (c10::isIntegralType(dtype, true)) {
    update_guid_dtype(guid_, c10::ScalarType::Int);
  }

  if (guid_ == "isnan_fwd_hf8" or guid_ == "isnan_fwd_f8") {
    update_guid_dtype(guid_, c10::ScalarType::BFloat16);
  }

  auto result = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{meta.shape, meta.dtype, 0}},
      params.ptr(),
      params.size());
  syn_out(0) = std::move(result[0]);
}

} // namespace habana
