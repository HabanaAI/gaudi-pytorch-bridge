
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
#include <perf_lib_layer_params.h>
#include "generated/backend/put.h"

namespace habana {

std::shared_ptr<void> FillPutParams(const at::Stack& stack, size_t& size) {
  const bool accumulate = stack.at(3).toBool();
  PARAMS_STUB(ns_PutKernel::Params);
  params->accumulate = accumulate;
  return params;
}
} // namespace habana
