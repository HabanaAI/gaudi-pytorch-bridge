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

sizes_vec LogspaceOutputShape(const at::Stack& stack, bool) {
  int64_t step = stack.at(2).toInt();
  return {{step}};
}

std::shared_ptr<void> RangeParams(const at::Stack& stack, size_t& size) {
  float first = stack[0].toScalar().to<float>();
  float end = stack[1].toScalar().to<float>();
  int64_t len = stack[2].toScalar().to<int64_t>();
  float step = 0.0;

  TORCH_CHECK(len, "0 steps is not supported as delta cannot be 0.");

  if (len > 1)
    step = (end - first) / (len - 1);
  else
    step = (end - first);

  PARAMS_STUB(ns_RangeKernel::Params);

  get<float>(params->start) = first;
  if (len != 1)
    get<float>(params->limit) = end + step;
  else
    get<float>(params->limit) = end;
  get<float>(params->delta) = step;

  return params;
}

void LogSpace::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = LogspaceOutputShape(stack)[0];
  size_t size = 0;
  auto params = RangeParams(stack, size);

  auto range = BuildOp(
      graph,
      "range_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {},
      {{outshape, ScalarType()}},
      params.get(),
      size);

  auto constant = ConstantHelper(graph, stack[3].toScalar(), ScalarType());

  auto pow = BuildOp(
      graph,
      "pow_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {constant.get(), range[0].get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(pow[0]);
}
} // namespace habana
