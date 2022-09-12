/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#include "pytorch_helpers/habana_helpers/pt_version_check.h"

#if IS_PYTORCH_FORK_AT_LEAST(1, 0)

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include "autocast_helpers.h"

namespace at {
namespace autocast {
namespace {

TORCH_LIBRARY_IMPL(_, AutocastHPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastHPU, m) {
  // lower_precision_fp
  KERNEL(
      ADD_NS(conv1d),
      "conv1d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL(
      ADD_NS(conv2d),
      "conv2d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL(
      ADD_NS(conv3d),
      "conv3d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL(
      ADD_NS(conv_transpose1d),
      "conv_transpose1d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      lower_precision_fp)
  KERNEL(
      ADD_NS(conv_transpose2d),
      "conv_transpose2d.input",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      lower_precision_fp)
  KERNEL(
      ADD_NS(conv_transpose3d),
      "conv_transpose3d.input",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      lower_precision_fp)
  KERNEL(
      ADD_NS(addmm),
      "addmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision_fp)
  KERNEL(
      ADD_NS(mean),
      "mean",
      Tensor(const at::Tensor&, c10::optional<at::ScalarType>),
      lower_precision_fp)
  KERNEL(
      ADD_NS(matmul),
      "matmul",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL(
      ADD_NS(mm),
      "mm",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL(
      ADD_NS(mul),
      "mul.Tensor",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL(
      ADD_NS(mv),
      "mv",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL(
      ADD_NS(linear),
      "linear",
      Tensor(const Tensor&, const Tensor&, const c10::optional<Tensor>&),
      lower_precision_fp)
  KERNEL(
      ADD_NS(bmm),
      "bmm",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL(
      ADD_NS(leaky_relu),
      "leaky_relu",
      Tensor(const at::Tensor&, const at::Scalar&),
      lower_precision_fp)
  KERNEL(ADD_NS(relu), "relu", Tensor(const at::Tensor&), lower_precision_fp)
  KERNEL(ADD_NS(t), "t", Tensor(const at::Tensor&), lower_precision_fp)
  KERNEL(
      ADD_NS(dot),
      "dot",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL(
      ADD_NS(dropout),
      "dropout",
      Tensor(const at::Tensor&, double, bool),
      lower_precision_fp)

  // fp32
  KERNEL(ADD_NS(log), "log", Tensor(const Tensor&), fp32)
  KERNEL(ADD_NS(log2), "log2", Tensor(const Tensor&), fp32)
  KERNEL(
      ADD_NS(nll_loss),
      "nll_loss",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t,
          int64_t),
      fp32)
  KERNEL(
      ADD_NS(smooth_l1_loss),
      "smooth_l1_loss",
      Tensor(const Tensor&, const Tensor&, int64_t, double),
      fp32)
  KERNEL(
      ADD_NS(binary_cross_entropy_with_logits),
      "binary_cross_entropy_with_logits",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          int64_t),
      fp32)
  KERNEL(
      ADD_NS(binary_cross_entropy),
      "binary_cross_entropy",
      Tensor(
          const Tensor&, const Tensor&, const c10::optional<Tensor>&, int64_t),
      fp32)
  KERNEL(ADD_NS(div), "div.Tensor", Tensor(const Tensor&, const Tensor&), fp32)
  KERNEL(
      ADD_NS(div),
      "div.Tensor_mode",
      Tensor(const Tensor&, const Tensor&, c10::optional<c10::string_view>),
      fp32)
  KERNEL(ADD_NS(div), "div.Scalar", Tensor(const Tensor&, const Scalar&), fp32)
  KERNEL(
      ADD_NS(divide),
      "divide.Tensor",
      Tensor(const Tensor&, const Tensor&),
      fp32)
  KERNEL(
      ADD_NS(divide),
      "divide.Tensor_mode",
      Tensor(const Tensor&, const Tensor&, c10::optional<c10::string_view>),
      fp32)
  KERNEL(
      ADD_NS(divide),
      "divide.Scalar",
      Tensor(const Tensor&, const Scalar&),
      fp32)
  KERNEL(
      ADD_NS(divide),
      "divide.Scalar_mode",
      Tensor(const Tensor&, const Scalar&, c10::optional<c10::string_view>),
      fp32)
  KERNEL(
      ADD_NS(true_divide),
      "true_divide.Tensor",
      Tensor(const Tensor&, const Tensor&),
      fp32)
  KERNEL(
      ADD_NS(true_divide),
      "true_divide.Scalar",
      Tensor(const Tensor&, const Scalar&),
      fp32)
  KERNEL(
      ADD_NS(softmax),
      "softmax.int",
      Tensor(const Tensor&, int64_t, c10::optional<ScalarType>),
      fp32)
  KERNEL(
      ADD_NS(softmax),
      "softmax.Dimname",
      Tensor(const Tensor&, Dimname, c10::optional<ScalarType>),
      fp32)
  KERNEL(
      ADD_NS(log_softmax),
      "log_softmax.int",
      Tensor(const Tensor&, int64_t, c10::optional<ScalarType>),
      fp32)
  KERNEL(
      ADD_NS(log_softmax),
      "log_softmax.Dimname",
      Tensor(const Tensor&, Dimname, c10::optional<ScalarType>),
      fp32)
  // The macro doesn't like this one (I think it chokes on commas inside <>) so
  // write it manually
  m.impl(
      TORCH_SELECTIVE_NAME("aten::embedding_bag"),
      TORCH_FN((&WrapFunction<
                CastPolicy::fp32,
                std::tuple<Tensor, Tensor, Tensor, Tensor>(
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    bool,
                    int64_t,
                    bool,
                    const c10::optional<at::Tensor>&,
                    bool),
                &ADD_NS(embedding_bag)>::type::call)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::embedding_bag.padding_idx"),
      TORCH_FN((&WrapFunction<
                CastPolicy::fp32,
                std::tuple<Tensor, Tensor, Tensor, Tensor>(
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    bool,
                    int64_t,
                    bool,
                    const c10::optional<at::Tensor>&,
                    bool,
                    c10::optional<int64_t>),
                &ADD_NS(embedding_bag)>::type::call)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::topk"),
      TORCH_FN((&WrapFunction<
                CastPolicy::fp32,
                std::tuple<Tensor, Tensor>(
                    const at::Tensor&, int64_t, int64_t, bool, bool),
                &ADD_NS(topk)>::type::call)));

  // promote
  KERNEL(ADD_NS(exp), "exp", Tensor(const Tensor&), promote)
  KERNEL(
      ADD_NS(pow),
      "pow.Tensor_Scalar",
      Tensor(const Tensor&, const Scalar&),
      promote)
  KERNEL(
      ADD_NS(pow),
      "pow.Tensor_Tensor",
      Tensor(const Tensor&, const Tensor&),
      promote)
  KERNEL(
      ADD_NS(pow), "pow.Scalar", Tensor(const Scalar&, const Tensor&), promote)
  KERNEL(
      ADD_NS(sub),
      "sub.Tensor",
      Tensor(const Tensor&, const Tensor&, const at::Scalar&),
      promote)
  KERNEL(
      ADD_NS(sub),
      "sub.Scalar",
      Tensor(const Tensor&, const Scalar&, const at::Scalar&),
      promote)
  KERNEL(ADD_NS(stack), "stack", Tensor(at::TensorList, int64_t), promote)
  KERNEL(
      ADD_NS(addcdiv),
      "addcdiv",
      Tensor(const Tensor&, const Tensor&, const Tensor&, const Scalar&),
      promote)
  KERNEL(
      ADD_NS(addcmul),
      "addcmul",
      Tensor(const Tensor&, const Tensor&, const Tensor&, const Scalar&),
      promote)
  KERNEL(ADD_NS(cat), "cat", Tensor(at::TensorList, int64_t), promote)
  KERNEL(ADD_NS(cat), "cat.names", Tensor(at::TensorList, at::Dimname), promote)
  KERNEL(
      ADD_NS(add),
      "add.Tensor",
      Tensor(const at::Tensor&, const at::Tensor&, const at::Scalar&),
      promote)
  KERNEL(
      ADD_NS(add),
      "add.Scalar",
      Tensor(const at::Tensor&, const at::Scalar&, const at::Scalar&),
      promote)

  // lower_first_arg
  KERNEL(
      ADD_NS(layer_norm),
      "layer_norm",
      Tensor(
          const Tensor&,
          IntArrayRef,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          double,
          bool),
      lower_first_arg)
  KERNEL(
      ADD_NS(group_norm),
      "group_norm",
      Tensor(
          const Tensor&,
          int64_t,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          double,
          bool),
      lower_first_arg)
  KERNEL(
      ADD_NS(instance_norm),
      "instance_norm",
      Tensor(
          const at::Tensor&,
          const c10::optional<at::Tensor>&,
          const c10::optional<at::Tensor>&,
          const c10::optional<at::Tensor>&,
          const c10::optional<at::Tensor>&,
          bool,
          double,
          double,
          bool),
      lower_first_arg)
  KERNEL(
      ADD_NS(batch_norm),
      "batch_norm",
      Tensor(
          const at::Tensor&,
          const c10::optional<at::Tensor>&,
          const c10::optional<at::Tensor>&,
          const c10::optional<at::Tensor>&,
          const c10::optional<at::Tensor>&,
          bool,
          double,
          double,
          bool),
      lower_first_arg)
}

} // namespace
} // namespace autocast
} // namespace at

#endif
