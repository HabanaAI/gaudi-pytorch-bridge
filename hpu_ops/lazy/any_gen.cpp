/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/lazy/all.h"
#include "generated/lazy/any.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/lazy/reduction_template.h"

namespace habana {

template <>
AllAnyOutputType<at::Tensor>::AllAnyOutputType(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, 0) {
  set_scalar_types({at::kBool});
}

template <>
at::Tensor AllAnyOutputType<at::Tensor>::get_result_overrideable() {
  return {};
}
} // namespace habana
