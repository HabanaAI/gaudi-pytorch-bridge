/*******************************************************************************
 * Copyright 2024 Intel Corporation.
 *
 * This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 * This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in
 * the License.
 *******************************************************************************
 */

#include "habana_eager/ops/batch_as_strided.h"
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include <vector>
#include "habana_eager/ops/as_strided.h"
#include "habana_helpers/logging.h"

namespace habana {
namespace eager {
std::vector<at::Tensor> batch_as_strided(
    at::TensorList inputs,
    c10::ArrayRef<std::vector<int64_t>> sizes,
    c10::ArrayRef<std::vector<int64_t>> strides,
    at::OptionalIntArrayRef storage_offsets) {
  auto inputs_count = inputs.size();
  HABANA_ASSERT(
      sizes.size() == inputs_count,
      "Length of sizes array doesn't match the number of provided input tensors");
  HABANA_ASSERT(
      strides.size() == inputs_count,
      "Length of strides array doesn't match the number of provided input tensors");
  if (storage_offsets.has_value())
    HABANA_ASSERT(
        storage_offsets.value().size() == inputs_count,
        "Length of storage_offsets array doesn't match the number of provided input tensors");

  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs_count);
  for (size_t i = 0; i < inputs_count; i++)
    outputs.push_back(at::as_strided(
        inputs[i],
        sizes[i],
        strides[i],
        storage_offsets.has_value() ? storage_offsets.value()[i] : 0));

  return outputs;
}

TORCH_LIBRARY_IMPL(hpu, AutogradHPU, m) {
  m.impl("hpu::batch_as_strided", batch_as_strided);
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::batch_as_strided", batch_as_strided);
}

} // namespace eager
} // namespace habana
