/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels_ver/lazy_kernels_declarations.h"

using namespace habana;
using namespace at;

namespace habana_lazy {

std::tuple<Tensor, Tensor, Tensor> native_group_norm_hpu_lazy(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t num_groups,
    double eps) {
  return native_group_norm_hpu_lazy(
      input,
      weight_opt,
      bias_opt,
      c10::SymInt{N},
      c10::SymInt{C},
      c10::SymInt{HxW},
      num_groups,
      eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_group_norm_backward_hpu_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> output_mask) {
  return native_group_norm_backward_hpu_lazy(
      grad_out,
      input,
      mean,
      rstd,
      weight,
      c10::SymInt{N},
      c10::SymInt{C},
      c10::SymInt{HxW},
      group,
      output_mask);
}

} // namespace habana_lazy
