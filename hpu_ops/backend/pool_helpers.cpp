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

#include <ATen/native/Pool.h>

#include "backend/synapse_helpers/layout_utils.h"
#include "hpu_ops/backend/pool_helpers.h"

namespace habana {
static std::vector<int> get_params_vector(
    const at::IntArrayRef params,
    bool is_3d) {
  std::vector<int> result;
  int first = at::native::safe_downcast<int, int64_t>(params[0]);
  int dims = is_3d ? 3 : 2;
  for (int i = 0; i < dims; i++) {
    result.push_back(
        params.size() == 1
            ? first
            : at::native::safe_downcast<int, int64_t>(params[i]));
  }
  return result;
}
std::vector<int64_t> compute_pool_kernel_output_shape(
    const at::Tensor& input,
    const at::IntArrayRef kernel_size,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    bool ceil_mode,
    bool is_3d) {
  // depending on 2d or 3d variant of op, dimensions are NCHW or NCDHW
  std::vector<int> filters = get_params_vector(kernel_size, is_3d);
  std::vector<int> strides =
      stride.empty() ? filters : get_params_vector(stride, is_3d);
  std::vector<int> pads = get_params_vector(padding, is_3d);
  std::vector<int> dilations = get_params_vector(dilation, is_3d);

  // Op accepts tensor without batches (3d in 2D op variant and 4d in 3D op
  // variant). In this case, the batch size should be assumed as 1
  bool noBatchesVariant =
      ((is_3d && input.dim() == 4) || (!is_3d && input.dim() == 3));
  int offset = noBatchesVariant ? 1 : 0;
  const int64_t N = noBatchesVariant ? 1 : input.size(0);
  const int64_t C = input.size(1 - offset);
  std::vector<int64_t> outshape{N, C};

  const size_t dims = is_3d ? 3 : 2;
  for (size_t i = 0; i < dims; i++) {
    outshape.push_back(at::native::pooling_output_shape<int64_t>(
        input.size(i + 2 - offset),
        filters[i],
        pads[i],
        strides[i],
        dilations[i],
        ceil_mode));
  }

  return outshape;
}
} // namespace habana