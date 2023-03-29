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
#include <c10/core/TensorOptions.h>
#include <c10/util/Optional.h>
#include <c10_ver/core/SymIntArrayRef.h>
#include "backend/synapse_helpers/layout_utils.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_logger.h"

#if IS_PYTORCH_AT_LEAST(2, 0)
#define TO_SYMINT_MAYBE(x) c10::fromIntArrayRefSlow(x)
#else
#define TO_SYMINT_MAYBE(x) x
#endif
namespace habana {
namespace eager {

at::Tensor convolution_overrideable(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups) {
  using namespace at;
  using namespace habana;
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "HpuOp convolution_overrideable :",
      " input=",
      to_string(input),
      " weight=",
      to_string(weight),
      " bias=",
      to_string(bias),
      " stride=",
      to_string(stride),
      " padding=",
      to_string(padding),
      " dilation=",
      to_string(dilation),
      " transposed=",
      transposed,
      " output_padding=",
      to_string(output_padding),
      " groups=",
      groups);

  FALLBACK_UNSUPPORTED_OP2(
      convolution,
      PARAMS2(
          input,
          weight,
          bias,
          stride,
          TO_SYMINT_MAYBE(padding),
          dilation,
          transposed,
          TO_SYMINT_MAYBE(output_padding),
          groups));
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor>
convolution_backward_overrideable(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    ::std::array<bool, 3> output_mask) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "HpuOp convolution_backward_overrideable :",
      " grad_output=",
      to_string(grad_output),
      " input=",
      to_string(input),
      " weight=",
      to_string(weight),
      " stride=",
      to_string(stride),
      " padding=",
      to_string(padding),
      " dilation=",
      to_string(dilation),
      " transposed=",
      to_string(transposed),
      " output_padding=",
      to_string(output_padding),
      " groups=",
      to_string(groups),
      " output_mask=",
      to_string(output_mask));

  FALLBACK_UNSUPPORTED_OP2(
      convolution_backward,
      PARAMS2(
          grad_output,
          input,
          weight,
          at::OptionalSymIntArrayRef{},
          stride,
          TO_SYMINT_MAYBE(padding),
          dilation,
          transposed,
          TO_SYMINT_MAYBE(output_padding),
          groups,
          output_mask));
}

} // namespace eager
} // namespace habana

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl(
      "convolution_backward_overrideable",
      static_cast<::std::tuple<at::Tensor, at::Tensor, at::Tensor> (*)(
          const at::Tensor&,
          const at::Tensor&,
          const at::Tensor&,
          at::IntArrayRef,
          at::IntArrayRef,
          at::IntArrayRef,
          bool,
          at::IntArrayRef,
          int64_t,
          ::std::array<bool, 3>)>(
          &habana::eager::convolution_backward_overrideable));
  m.impl(
      "convolution_overrideable",
      static_cast<at::Tensor (*)(
          const at::Tensor&,
          const at::Tensor&,
          const c10::optional<at::Tensor>&,
          at::IntArrayRef,
          at::IntArrayRef,
          at::IntArrayRef,
          bool,
          at::IntArrayRef,
          int64_t)>(&habana::eager::convolution_overrideable));
}
