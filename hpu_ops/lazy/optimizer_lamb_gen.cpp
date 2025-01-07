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

#include "hpu_ops/optimizer_lamb_gen.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR(
    habana_lazy::LazyOp,
    LazyOptimizerLambNorm,
    -1,
    at::Tensor) {}

HPU_OP_FRONTEND_CREATE_RESULT_ONLY(
    habana_lazy::LazyOp,
    LazyOptimizerLambNorm,
    at::Tensor) {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensorList().get(0);
  return habana_lazy::empty_hpu_lazy(
      {1}, t.options(), t.suggest_memory_format(), false);
}

} // namespace habana