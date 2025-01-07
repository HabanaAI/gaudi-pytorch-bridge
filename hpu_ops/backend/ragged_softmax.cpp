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

#include "generated/backend/_softmax.h"

namespace habana {

struct RaggedSoftmax : OpBackend {
  RaggedSoftmax(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            "ragged_softmax_fwd",
            scalar_type,
            {0},
            {},
            {},
            false) {
    SetFillParams(FillSoftmaxForwardParams);
  }
};

} // namespace habana

static auto& KernelRegistry = habana::KernelRegistry().add(
    "hpu::ragged_softmax",
    KERNEL_FN(RaggedSoftmax));
