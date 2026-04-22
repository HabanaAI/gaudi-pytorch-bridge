/**
 * Copyright (c) 2021-2026 Intel Corporation
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

#include "generated/backend/scatter_reduce.h"
#include "habana_helpers/conversion.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {

const unsigned SELF_INDEX = 0;
const unsigned DIM_INDEX = 1;
const unsigned REDUCE_INDEX = 4;
const unsigned INCLUDE_SELF_INDEX = 5;

FillParamsT ScatterReduceParams(const at::Stack& stack) {
  PARAMS_STUB(ns_ScatterReduceKernel::Params);
  const auto dim = stack.at(DIM_INDEX).toInt();
  auto reduce = stack.at(REDUCE_INDEX).to<std::string_view>();
  auto baseScatterOp = (reduce == "add" || reduce == "multiply");
  auto includeSelf =
      baseScatterOp ? true : stack.at(INCLUDE_SELF_INDEX).toBool();

  ScatterReduceMode_t mode;

  static const std::unordered_map<std::string_view, ScatterReduceMode_t>
      reduceModes = {
          {"sum", ScatterReduceMode_t::SCATTER_REDUCE_SUM},
          {"add", ScatterReduceMode_t::SCATTER_REDUCE_SUM},
          {"prod", ScatterReduceMode_t::SCATTER_REDUCE_PROD},
          {"multiply", ScatterReduceMode_t::SCATTER_REDUCE_PROD},
          {"mean", ScatterReduceMode_t::SCATTER_REDUCE_MEAN},
          {"amax", ScatterReduceMode_t::SCATTER_REDUCE_AMAX},
          {"amin", ScatterReduceMode_t::SCATTER_REDUCE_AMIN},
      };

  auto it = reduceModes.find(reduce);
  if (it != reduceModes.end()) {
    mode = it->second;
  } else {
    HABANA_ASSERT(false, "Unsupported reduce: ", reduce);
  }
  params->dim = safe_convert<int>(dim);
  params->include_self = includeSelf;
  params->mode = mode;

  return paramsT;
}

OutputMetaDataVector ScatterReduceMeta(const at::Stack& stack) {
  const auto& self = stack.at(SELF_INDEX);
  std::vector<int64_t> outputShape;
  at::ScalarType dtype;

  if (self.isTensor()) {
    const auto& selfTensor = self.toTensor();
    outputShape = selfTensor.sizes().vec();
    dtype = selfTensor.scalar_type();
  } else {
    dtype = self.toScalar().type();
  }

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype = dtype;
  meta.shape = outputShape;

  return metaVec;
}

} // namespace habana
