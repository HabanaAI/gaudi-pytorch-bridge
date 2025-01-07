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

#include "generated/lazy/eq.h"
#include "generated/lazy/ge.h"
#include "generated/lazy/gt.h"
#include "generated/lazy/le.h"
#include "generated/lazy/lt.h"
#include "generated/lazy/ne.h"

namespace habana {
HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    CompareScalarToTensor,
    at::Tensor) {
  // convert scalar input to tensor to avoid cache misses in cases where scalar
  // value changes across iterations
  auto& x = get_inputs();
  auto self = x[0].toTensor();
  auto other = x[1].toScalar();
  auto dtype = at::result_type(self, other);
  x[1] = habana_lazy::get_tensor_for_scalar(
      other.toDouble(), self.options().dtype(dtype));
}
} // namespace habana
