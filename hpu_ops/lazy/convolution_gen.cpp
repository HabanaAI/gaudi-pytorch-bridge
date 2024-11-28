/**
 * Copyright (c) 2024 Intel Corporation
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
#include "hpu_ops/common/convolution_gen.h"
#include "generated/lazy/convolution.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/view_utils.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    ConvolutionFE,
    at::Tensor) {
  FRONTEND_CONVOLUTION_COMMON(0)

  const auto hl_t = habana_lazy::GetHbLazyTensor(weight);
  const bool is_view_tensor = hl_t.getDataPtr()->stride_params.has_value();

  if (!is_view_tensor and weight.dim() != 3) {
    at::Tensor weight_hpu =
        habana_lazy::HbLazyTensorViews::HandleViewsD2H(weight);

    habana_lazy::PermuteTensors::permuteWeight(weight_hpu);
  }
}

} // namespace habana
