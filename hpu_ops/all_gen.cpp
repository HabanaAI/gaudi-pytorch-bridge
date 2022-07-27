/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/hpu_op.h"

namespace habana {
sizes_vec AllOutputShape(const at::Stack&, bool lowering) {
  std::vector<int64_t> shape_out{};
  if (lowering) {
    shape_out.push_back(1);
  }
  return {shape_out};
}

template <>
LazyAll<at::Tensor>::LazyAll(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  set_scalar_type(c10::ScalarType::Bool);
}

template <>
at::Tensor LazyAll<at::Tensor>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return {};
}
} // namespace habana
