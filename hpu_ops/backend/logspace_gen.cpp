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

#include "generated/backend/logspace.h"

namespace habana {

sizes_vec LogspaceOutputShape(const at::Stack& stack) {
  int64_t step = stack.at(2).toInt();
  return {{step}};
}

std::shared_ptr<void> RangeParams(const at::Stack& stack, size_t& size) {
  float start = stack[0].toScalar().to<float>();
  float end = stack[1].toScalar().to<float>();
  int64_t step = stack[2].toScalar().to<int64_t>();

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

void LogSpace::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = LogspaceOutputShape(stack)[0];
  float start = stack[0].toScalar().to<float>();
  float end = stack[1].toScalar().to<float>();
  int64_t len = stack[2].toScalar().to<int64_t>();
  float base = stack[3].toScalar().to<float>();
  if (len == 0) {
    auto result = habana::OpBackend::BuildOp(
        graph, "memset", {}, {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(result[0]);
  } else if (base == 1.f) {
    auto result = ConstantHelper(graph, 1.f, ScalarType(), outshape, 0);
    syn_out(0) = std::move(result);
  } else {
    std::vector<synapse_helpers::tensor> range;
    if (start != end && len != 1) {
      size_t size = 0;
      auto params = RangeParams(stack, size);

      range = BuildOp(
          graph,
          get_guid_with_precision("range", ScalarType()),
          {},
          {{outshape, ScalarType()}},
          params.get(),
          size);
    } else {
      range.push_back(ConstantHelper(graph, start, ScalarType(), outshape));
    }

    auto constant = ConstantHelper(graph, stack[3].toScalar(), ScalarType());

    auto pow = BuildOp(
        graph,
        get_guid_with_precision("pow_fwd", ScalarType()),
        {constant.get(), range[0].get()},
        {{outshape, ScalarType(), 0}});

    syn_out(0) = std::move(pow[0]);
  }
}
} // namespace habana
