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
#include "generated/backend/isfinite.h"
#include "generated/backend/isinf.h"
#include "generated/backend/isnan.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

SharedMetaDataVector IsFiniteSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return IsFiniteInfNanSharedMeta(stack, "isfinite_fwd");
}

SharedMetaDataVector IsInfSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return IsFiniteInfNanSharedMeta(stack, "isinf_fwd");
}

SharedMetaDataVector IsNanSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return IsFiniteInfNanSharedMeta(stack, "isnan_fwd");
}

OutputMetaDataVector IsFiniteInfNanMeta(const at::Stack& stack) {
  const at::Tensor& self = stack_tensor(stack, 0);
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = at::kBool;
  return {meta};
}

void _IsFiniteInfNan::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto meta = IsFiniteInfNanMeta(stack)[0];
  auto dtype = stack_tensor(stack, 0).scalar_type();
  // use cguid autocast
  if (c10::isIntegralType(dtype, true)) {
    update_guid_dtype(guid_, c10::ScalarType::Int);
  }

  auto result = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(result[0]);
}

} // namespace habana
