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

#include "generated/lazy/div.h"
#include "hpu_ops/common/scalar_dtype_range.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    BinaryScalarFE,
    at::Tensor&) {
  update_other_scalar_if_out_of_scalar_type_range(inputs, get_inputs());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    BinaryScalarFE,
    at::Tensor) {
  update_other_scalar_if_out_of_scalar_type_range(inputs, get_inputs());
}

} // namespace habana
