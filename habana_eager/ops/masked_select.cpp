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
#include "habana_eager/ops/masked_select.h"
#include <c10/core/SymIntArrayRef.h>
#include "habana_kernels/resize.h"
#include "hpu_ops/op_logger.h"
namespace habana {
namespace eager {

at::Tensor masked_select_eager(const at::Tensor& self, const at::Tensor& mask) {
  TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool,
              "masked_select: expected BoolTensor for mask");
  auto new_size = at::infer_size(self.sizes(), mask.sizes());
  auto new_mask = at::broadcast_to(mask, new_size);
  auto new_self = at::broadcast_to(self, new_size);

  return at::index(new_self, {new_mask});
}

at::Tensor& masked_select_out_eager(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& out) {
  auto output = masked_select_eager(self, mask);
  std::vector<int64_t> out_shape{output.sizes().vec()[0]};
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = out.unsafeGetTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(
        c10::IntArrayRef(out_shape));
  }
  out.copy_(output);
  return out;
}

} // namespace eager
} // namespace habana