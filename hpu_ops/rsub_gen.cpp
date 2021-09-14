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

template <>
LazyRsub<at::Tensor>::LazyRsub(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>("aten::sub", inputs, out_shapes_fn) {
  static_cast<void>(qualstring);
  auto& sub_inputs = get_inputs();
  std::swap(sub_inputs.at(0), sub_inputs.at(1));
  const auto& self = sub_inputs.at(0).toTensor();
  const auto& other = sub_inputs.at(1).toTensor();
  const auto& result_type = at::result_type(self, other);
  if (self.scalar_type() != result_type) {
    sub_inputs.at(0) = self.to(result_type);
  }
  if (other.scalar_type() != result_type) {
    sub_inputs.at(1) = other.to(result_type);
  }
}

template <>
at::Tensor LazyRsub<at::Tensor>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return {};
}
} // namespace habana
