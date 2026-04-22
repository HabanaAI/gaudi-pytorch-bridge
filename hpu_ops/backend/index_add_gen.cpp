/**
 * Copyright (c) 2025 Intel Corporation
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
#include "generated/backend/index_add.h"

using namespace std::literals;

namespace habana {

FillParamsT FillIndexAddParams(const at::Stack& stack) {
  PARAMS_STUB(ns_IndexAdd::Params);

  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toScalar().to<int>();

  params->axis = get_dim_in_tpc_order(dim, self.dim());
  params->alpha = stack.at(4).toScalar().to<double>();
  return paramsT;
}

OutputMetaDataVector IndexAddMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype = input.scalar_type();
  meta.shape = input.sizes().vec();
  return metaVec;
}

void IndexAdd::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto dtype = ScalarType();
  if (c10::isIntegralType(dtype, true)) {
    update_guid_dtype(guid_, c10::ScalarType::Int);
  }

  const auto params = FillIndexAddParams(stack);
  const auto meta = IndexAddMeta(stack)[0];

  auto indexAddResult = BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1), syn_in(2)},
      {{meta.shape, meta.dtype, 0}},
      params.ptr(),
      params.size());

  syn_out(0) = std::move(indexAddResult[0]);
}

} // namespace habana
