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
#include "generated/backend/_log_softmax_backward_data.h"

namespace habana {
std::shared_ptr<void> FillLogSoftmaxParams(
    const at::Stack& stack,
    size_t& size) {
  bool half_to_float = stack.at(2).toBool();
  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");
  auto self = stack.at(0).toTensor();
  PARAMS_STUB(ns_Softmax::Params);
  params->dim = get_dim_in_tpc_order(
      /*dim*/ stack.at(1).toInt(),
      /*max dims*/ self.dim());
  return params;
}

std::shared_ptr<void> FillLogSoftmaxBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  auto self = stack.at(0).toTensor();
  PARAMS_STUB(ns_Softmax::Params);
  params->dim = get_dim_in_tpc_order(stack.at(2).toInt(), self.dim());
  return params;
}

} // namespace habana
