/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/linspace.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

sizes_vec LinspaceOutputShape(const at::Stack& stack) {
  int64_t step = stack.at(2).toInt();
  return {{step}};
}

std::shared_ptr<void> LinspaceRangeParams(
    const at::Stack& stack,
    size_t& size) {
  float start = stack[0].toScalar().to<float>();
  float end = stack[1].toScalar().to<float>();
  int64_t step = stack[2].toScalar().to<int64_t>();

  TORCH_CHECK(step, "0 steps is not supported as delta cannot be 0.");
  float endValueModification = 0.000001;
  int64_t arange_step = step;

  float delta = (end - start);
  if (1.0 != arange_step) {
    delta /= (arange_step - 1.0);
  }
  if (arange_step != 1) {
    endValueModification = delta / 2.0;
  }

  end += endValueModification;
  PARAMS_STUB(ns_RangeKernel::Params);

  get<float>(params->start) = start;
  get<float>(params->limit) = end;
  get<float>(params->delta) = delta;

  return params;
}

void LinspaceOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = LinspaceOutputShape(stack)[0];

  float start = stack[0].toScalar().to<float>();
  float end = stack[1].toScalar().to<float>();

  if (start != end) {
    size_t size = 0;
    auto params = LinspaceRangeParams(stack, size);

    auto range = BuildOp(
        graph,
        "range_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {},
        {{outshape, ScalarType(), 0}},
        params.get(),
        size);

    syn_out(0) = std::move(range[0]);
  } else {
    auto result = ConstantHelper(graph, start, ScalarType(), outshape, 0);
    syn_out(0) = std::move(result);
  }
}
} // namespace habana
