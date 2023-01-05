/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/isnan.h"

namespace habana {

template <>
LazyIsNan<at::Tensor>::LazyIsNan(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  set_scalar_type(torch::kBool);
}

template <>
at::Tensor LazyIsNan<at::Tensor>::get_result_overrideable() {
  return {};
}
} // namespace habana
