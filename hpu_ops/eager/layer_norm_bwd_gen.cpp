/**
 * Copyright (c) 2025 Intel Corporation
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

#include "backend/habana_device/HPUEvent.h"
#include "generated/eager/native_layer_norm_backward.h"
#include "generated/eager/wrap_kernels_declarations.h"
#include "habana_eager/ops/eager_op.h"

namespace habana {
std::tuple<at::Tensor, at::Tensor, at::Tensor> LayerNormBwdOutputMaskFn(
    std::tuple<at::Tensor, at::Tensor, at::Tensor> res,
    std::array<bool, 3> output_mask) {
  at::Tensor in_grad;
  at::Tensor weight_grad;
  at::Tensor bias_grad;
  if (output_mask[0]) {
    in_grad = std::get<0>(res);
  }
  if (output_mask[1]) {
    weight_grad = std::get<1>(res);
  }
  if (output_mask[2]) {
    bias_grad = std::get<2>(res);
  }
  std::tuple<at::Tensor, at::Tensor, at::Tensor> final_res = std::make_tuple(
      std::move(in_grad), std::move(weight_grad), std::move(bias_grad));
  return final_res;
}
} // namespace habana
