/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "hpu_ops/bincount.h"
#include "habana_eager/ops/bincount.h"
#include "habana_eager/ops/eager_op.h"

namespace habana {
namespace eager {

at::Tensor cast_to_32(const at::Tensor& self) {
  if (self.scalar_type() == c10::ScalarType::Long ||
      self.scalar_type() == c10::ScalarType::Byte ||
      self.scalar_type() == c10::ScalarType::Char ||
      self.scalar_type() == c10::ScalarType::Short) {
    return self.to(at::ScalarType::Int);
  } else if (
      self.scalar_type() == c10::ScalarType::Double ||
      self.scalar_type() == c10::ScalarType::Half ||
      self.scalar_type() == c10::ScalarType::BFloat16) {
    return self.to(at::ScalarType::Float);
  }
  return self;
}

c10::optional<at::Tensor> cast_weights(
    const c10::optional<at::Tensor>& weights) {
  if (weights.has_value() &&
      (weights.value().dtype() != c10::ScalarType::Int &&
       weights.value().dtype() != c10::ScalarType::Float)) {
    return c10::make_optional<at::Tensor>(cast_to_32(weights.value()));
  }
  return weights;
}

c10::ScalarType bincount_output_dtype(
    const c10::optional<at::Tensor>& weights) {
  if (!weights.has_value()) {
    return c10::ScalarType::Long;
  }
  return (weights.value().scalar_type() == c10::ScalarType::Float)
      ? c10::ScalarType::Float
      : c10::ScalarType::Double;
}

at::Tensor bincount_eager(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weights,
    int64_t minlength) {
  PT_EAGER_TRACE;
  TORCH_CHECK(
      minlength >= 0 && minlength <= std::numeric_limits<int32_t>::max(),
      "Invalid length. Possible over or underflow.");
  // Handle case for empty tensor where we return empty tensor with size
  if (self.numel() == 0) {
    auto output =
        at::zeros({minlength}, self.options().dtype(c10::ScalarType::Long));
    return output;
  }

  // .item() internally triggers a mark_step
  auto max_in_input = static_cast<int64_t>(at::max(self).item<int64_t>());
  auto length = std::max(max_in_input + 1, minlength);
  std::vector<int64_t> shape{length};
  auto out_dtype = bincount_output_dtype(weights);

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::bincount_backend",
      {cast_to_32(self), length, cast_weights(weights)},
      {shape},
      0};
  hpu_op.SetOutputMetaFn(BinCountMeta);
  auto result_bincount = hpu_op.call();
  if (result_bincount.scalar_type() != out_dtype)
    return result_bincount.to(out_dtype);
  return result_bincount;
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def(
      "hpu::bincount_backend(Tensor self, int length, Tensor? weights) -> (Tensor)");
}
} // namespace eager
} // namespace habana
