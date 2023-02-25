/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/library.h>

namespace habana {
namespace eager {
std::tuple<at::Tensor&, at::Tensor&> cast_to_fp8(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "hpu::cast_to_fp8 is not available in Eager mode.");
}

at::Tensor cast_from_fp8(const at::Tensor&, const at::Tensor&, at::ScalarType) {
  TORCH_CHECK(false, "hpu::cast_from_fp8 is not available in Eager mode.");
}

TORCH_LIBRARY(hpu, m) {
  m.def(
      "hpu::cast_to_fp8(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax) -> (Tensor(a!), Tensor(b!))");
  m.def(
      "hpu::cast_from_fp8(Tensor input, Tensor scale, ScalarType out_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::cast_to_fp8", cast_to_fp8);
  m.impl("hpu::cast_from_fp8", cast_from_fp8);
}

} // namespace eager
} // namespace habana