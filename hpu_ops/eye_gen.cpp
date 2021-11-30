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
#include "hpu_op_helper.h"

namespace habana {
sizes_vec EyeOutputShape(const at::Stack& stack, bool) {
  const int64_t n = stack.at(0).toInt();
  if (stack.size() == 3) {
    const int64_t m = stack.at(1).toInt();
    return {{n, m}};
  }
  return {{n, n}};
}

void EyeOpOut::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  std::vector<synapse_helpers::tensor> eye_out;
  auto outshape = EyeOutputShape(stack)[0];

  auto constant = ConstantHelper(graph, 1.0f, ScalarType(), outshape);

  eye_out = BuildOp(
      graph,
      "matrix_diagonal_fwd_" +
          habana_helpers::name_suffix_from_type(ScalarType()),
      {constant.get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  syn_out(0) = std::move(eye_out[0]);
}
} // namespace habana
