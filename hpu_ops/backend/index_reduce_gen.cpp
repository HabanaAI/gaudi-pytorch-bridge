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

#include "generated/backend/index_reduce.h"

namespace habana {

FillParamsT IndexReduceFillParams(const at::Stack& stack) {
  PARAMS_STUB(ns_IndexReduce::Params);
  params->axis = stack.at(1).toScalar().toInt();
  if (params->axis < 0) {
    params->axis += static_cast<int>(stack.at(0).toTensor().dim());
  }

  const auto mode = stack.at(4).toStringView();
  static const std::unordered_map<std::string_view, IndexReduceMode_t>
      reduceModes = {
          {"amax", IndexReduceMode_t::INDEX_REDUCE_AMAX},
          {"amin", IndexReduceMode_t::INDEX_REDUCE_AMIN},
          {"mean", IndexReduceMode_t::INDEX_REDUCE_MEAN},
          {"prod", IndexReduceMode_t::INDEX_REDUCE_PROD},
      };

  auto it = reduceModes.find(mode);
  HABANA_ASSERT(it != reduceModes.end(), "Unsupported reduce: ", mode)

  params->mode = it->second;
  params->include_self = stack.at(5).toBool();
  return paramsT;
}

} // namespace habana
