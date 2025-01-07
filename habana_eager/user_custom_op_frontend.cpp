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
#include "habana_eager/ops/eager_op.h"
#include "include/habanalabs/hpu_custom_op_pt2.h"

namespace habana {
namespace custom_op {

std::vector<at::Tensor> UserCustomOpDescriptor::execute(
    const std::vector<c10::IValue>& inputs) {
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<at::ScalarType> output_dtypes;
  for (const auto& meta : output_meta_fn_(inputs)) {
    output_shapes.push_back(meta.shape);
    output_dtypes.push_back(meta.dtype);
  }
  habana::eager::EagerOp<std::vector<at::Tensor>> hpu_op{
      schema_, inputs, std::move(output_shapes)};
  hpu_op.set_scalar_types(std::move(output_dtypes));
  return hpu_op.call();
}

} // namespace custom_op
} // namespace habana
