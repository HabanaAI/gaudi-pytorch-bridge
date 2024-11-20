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
#include "generated/eager/wrap_kernels_declarations.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR(
    eager::EagerOp,
    EagerOptimizerLambNorm,
    -1,
    at::Tensor) {}

HPU_OP_FRONTEND_CREATE_RESULT_ONLY(
    eager::EagerOp,
    EagerOptimizerLambNorm,
    at::Tensor) {
  const auto& inputs = get_inputs();
  const auto& t = inputs.at(0).toTensorList().get(0);

  return hpu_wrap::empty(
      {1},
      t.scalar_type(),
      t.options().layout_opt(),
      t.options().device_opt(),
      t.options().pinned_memory_opt(),
      t.suggest_memory_format());
}

} // namespace habana
