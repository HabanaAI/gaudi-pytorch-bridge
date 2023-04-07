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
#include "hpu_ops/optimizer_lamb_gen.h"

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

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> fp8_cast_transpose(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_cast_transpose is not available in Eager mode.");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
fp8_cast_transpose_bgrad(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(
      false, "hpu::fp8_cast_transpose_bgrad is not available in Eager mode.");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
fp8_cast_transpose_bgrad_dgelu(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    bool,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(
      false,
      "hpu::fp8_cast_transpose_bgrad_dgelu is not available in Eager mode.");
}

at::Tensor cast_from_fp8(const at::Tensor&, const at::Tensor&, at::ScalarType) {
  TORCH_CHECK(false, "hpu::cast_from_fp8 is not available in Eager mode.");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> fp8_gelu(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_gelu is not available in Eager mode.");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> fp8_layernorm(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double,
    const at::Tensor&,
    bool,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_layernorm is not available in Eager mode.");
}

at::Tensor& fp8_gemm(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    const at::Tensor&,
    const at::Tensor&,
    bool,
    const at::Tensor&,
    at::ScalarType,
    const c10::optional<at::Tensor>&,
    bool,
    at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_gemm is not available in Eager mode.");
}

at::Tensor& fp8_transpose(const at::Tensor&, at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_transpose is not available in Eager mode.");
}

at::Tensor optimizer_lamb_fused_norm(
    const std::vector<at::Tensor>& grad,
    double max_grad_norm) {
  PT_OP_TRACE;
  PT_EAGER_TRACE;

  EagerOptimizerLambFusedNorm<at::Tensor> hpu_op{
      "hpu::optimizer_lamb_fused_norm", {grad, max_grad_norm}};
  return hpu_op.call();
}

TORCH_LIBRARY(hpu, m) {
  m.def(
      "hpu::cast_to_fp8(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax) -> (Tensor(a!), Tensor(b!))");
  m.def(
      "hpu::fp8_cast_transpose(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax, Tensor(c!) transposed) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::fp8_cast_transpose_bgrad(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax, Tensor(c!) transposed, Tensor(d!) bgrad) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::fp8_cast_transpose_bgrad_dgelu(Tensor grad, Tensor input, Tensor scale, Tensor? retain, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax, Tensor(c!) transposed, Tensor(d!) bgrad) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::cast_from_fp8(Tensor input, Tensor scale, ScalarType out_dtype) -> Tensor");
  m.def(
      "hpu::fp8_gelu(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax, Tensor(c!) retain) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::fp8_layernorm(Tensor input, Tensor weight, Tensor bias, float eps, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax, Tensor(c!) mean, Tensor(d!) istd) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::fp8_gemm(Tensor A, Tensor A_scale_inv, bool trans_A, Tensor B, Tensor B_scale_inv, bool trans_B, Tensor D, ScalarType out_dtype, Tensor? bias, bool accumulate, Tensor(a!) out) -> Tensor(a!)");
  m.def("hpu::fp8_transpose(Tensor input, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "hpu::optimizer_lamb_fused_norm(Tensor[] grad, float max_norm) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::cast_to_fp8", cast_to_fp8);
  m.impl("hpu::fp8_cast_transpose", fp8_cast_transpose);
  m.impl("hpu::fp8_cast_transpose_bgrad", fp8_cast_transpose_bgrad);
  m.impl("hpu::fp8_cast_transpose_bgrad_dgelu", fp8_cast_transpose_bgrad_dgelu);
  m.impl("hpu::cast_from_fp8", cast_from_fp8);
  m.impl("hpu::fp8_gelu", fp8_gelu);
  m.impl("hpu::fp8_layernorm", fp8_layernorm);
  m.impl("hpu::fp8_gemm", fp8_gemm);
  m.impl("hpu::fp8_transpose", fp8_transpose);
  m.impl("hpu::optimizer_lamb_fused_norm", optimizer_lamb_fused_norm);
}

} // namespace eager
} // namespace habana