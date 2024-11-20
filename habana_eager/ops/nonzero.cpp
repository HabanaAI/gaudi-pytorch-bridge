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
#include <ATen/core/DimVector.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/SymIntArrayRef.h>

#include "backend/backend_meta.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/nonzero.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/nonzero.h"
#include "hpu_ops/op_logger.h"

namespace habana {
namespace eager {

at::Tensor nonzero_eager(const at::Tensor& self) {
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  int elements = self.numel();
  at::TensorOptions hb_options = self.options();
  hb_options = hb_options.dtype(c10::ScalarType::Long);
  // Handle case for empty tensor where we return empty tensor with size
  if (elements == 0) {
    auto shape = c10::DimVector({0, dimensions});
    auto output = at::empty(shape, hb_options, c10::nullopt);
    return output;
  }

  std::vector<int64_t> shape_tensor_shape{5};

  auto NonzeroMeta = [](const at::Stack& stack) {
    const auto& self = stack_tensor(stack, 0);
    OutputMetaDataVector meta(2);
    NonZeroParams_t self_params;
    self_params.dtype = self.scalar_type();
    self_params.sizes = self.sizes().vec();
    self_params.numel = self.numel();
    meta.at(0).shape = compute_nonzero_output_shape(self_params);
    meta.at(0).dtype = c10::ScalarType::Long;
    meta.at(1).shape = {5};
    meta.at(1).dtype = at::ScalarType::Int;
    return meta;
  };
  NonZeroParams_t self_params;
  self_params.dtype = self.scalar_type();
  self_params.sizes = self.sizes().vec();
  self_params.numel = self.numel();
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::nonzero_eager",
      {self},
      {compute_nonzero_output_shape(self_params), shape_tensor_shape},
      0};
  hpu_op.SetOutputMetaFn(NonzeroMeta);
  auto result_nonzero = hpu_op.call();
  auto where_tensor = std::get<0>(result_nonzero);
  auto shape_tensor = std::get<1>(result_nonzero);
  // Select second element from shape tensor
  // auto end_tensor = slice_shape_tensor(shape_tensor);
  auto end_tensor = at::select(shape_tensor, 0, 1);
  // .item() internally triggers a mark_step
  auto end = end_tensor.item<int64_t>();
  // Handle case for all False where we return empty tensor with size
  if (end == 0) {
    auto shape = c10::DimVector({0, dimensions});
    auto output = at::empty(shape, hb_options, c10::nullopt);
    return output;
  }
  // Handle case for nonzero scalar input
  if (elements == 1 && dimensions == 0 && end != 0) {
    auto shape = c10::DimVector({end, 0});
    auto output = at::empty(shape, hb_options, c10::nullopt);
    return output;
  }
  // Add a slice node to capture relevent elements from nonzero node
  // in case we have relevant elements
  auto result = at::slice(where_tensor, 0, 0, end, 1);
  return result;
}

at::Tensor& nonzero_out_eager(
    [[maybe_unused]] const at::Tensor& self,
    at::Tensor& out) {
  // TBD
  return out;
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def("nonzero_eager(Tensor self) -> (Tensor, Tensor)");
}
} // namespace eager
} // namespace habana
