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
#include "habana_kernels/repeat.h"
#include "habana_eager/ops/eager_op.h"
namespace habana {
namespace eager {
at::Tensor repeat_hpu(const at::Tensor& self, at::SymIntArrayRef _repeats) {
  PT_EAGER_TRACE;
  auto repeats = C10_AS_INTARRAYREF_SLOW(_repeats);
  EagerOp<at::Tensor> k{
      "aten::repeat",
      {self, repeats},
      {RepeatOperator::compute_output_shape(self, repeats)},
      0};
  return k.call();
}
} // namespace eager
} // namespace habana
