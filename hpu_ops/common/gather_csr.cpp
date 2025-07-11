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

#include "hpu_ops/common/gather_csr.h"
#include <torch/torch.h>

namespace habana {

at::Tensor gather_csr_common(
    at::Tensor src,
    at::Tensor indptr,
    std::optional<at::Tensor> optional_out) {
  const int64_t output_size = optional_out.has_value()
      ? optional_out.value().numel()
      : indptr[indptr.sizes()[0] - 1].item<int64_t>();

  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("hpu::gather_csr", "")
          .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t)>();

  at::Tensor output;
  if (optional_out.has_value()) {
    output = optional_out.value(); // Now both point to the same storage
  } else {
    output = at::empty({output_size}, src.options());
  }

  output.fill_(op.call(src, indptr, output_size));

  return output;
}

} // namespace habana
