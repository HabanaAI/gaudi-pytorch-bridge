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
#include "generated/eager/convolution_backward.h"
#include "hpu_ops/common/convolution_gen.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    eager::EagerOp,
    ConvolutionBackwardFE,
    ::std::tuple<at::Tensor, at::Tensor, at::Tensor>) {
  FRONTEND_CONVOLUTION_COMMON(1)
}

} // namespace habana
