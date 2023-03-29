
/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "habana_eager/helpers.h"
#include "habana_eager/ops/as_strided.h"
#include "habana_eager/ops/empty.h"
#include "habana_eager/ops/set.h"
#include "habana_eager/ops/view.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_logger.h"

#include "habana_helpers/logging.h"

using namespace at;
using namespace habana;

Tensor hpu_wrap::empty(
    SymIntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  PT_EAGER_TRACE;
  return habana::eager::empty(
      size, dtype, layout, device, pin_memory, optional_memory_format);
}

Tensor hpu_wrap::empty_strided(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  PT_EAGER_TRACE;
  return habana::eager::empty_strided(
      size, stride, dtype, layout, device, pin_memory);
}

Tensor hpu_wrap::_reshape_alias(
    const Tensor& self,
    SymIntArrayRef size,
    SymIntArrayRef stride) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "_reshape_alias :",
      " self=",
      to_string(self),
      " size=",
      to_string(size),
      " stride",
      to_string(stride));
  FALLBACK_IF_UNSUPPORTED_OP(
      _reshape_alias, PARAMS1(self), PARAMS2(self, size, stride))
  return habana::eager::view_hpu(self, size);
}
