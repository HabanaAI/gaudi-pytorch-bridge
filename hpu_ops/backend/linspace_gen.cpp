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

#include "generated/backend/linspace.h"

namespace habana {

OutputMetaDataVector LinspaceMeta(const at::Stack& stack) {
  OutputMetaData meta;
  auto out_tensor = stack.at(3).toTensor();

  meta.dtype = out_tensor.scalar_type();
  meta.shape = {stack.at(2).toInt()};

  return {meta};
}

std::shared_ptr<void> LinspaceRangeParams(
    const at::Stack& stack,
    size_t& size) {
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

void LinspaceOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  OutputMetaData meta = LinspaceMeta(stack)[0];
  auto outshape = meta.shape;

  float start = stack[0].toScalar().to<float>();
  float end = stack[1].toScalar().to<float>();
  int64_t step = stack[2].toScalar().to<int64_t>();

  if (step == 0) {
    // return empty tensor if zero step
    auto result = habana::OpBackend::BuildOp(
        graph, "memset", {}, {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(result[0]);
  } else {
    if (start != end && step != 1) {
      size_t size = 0;
      auto params = LinspaceRangeParams(stack, size);

      auto range = BuildOp(
          graph,
          get_guid_with_precision("range", ScalarType()),
          {},
          {{outshape, ScalarType(), 0}},
          params.get(),
          size);

      syn_out(0) = std::move(range[0]);
    } else {
      // return start when start == end or step == 1
      auto result = ConstantHelper(graph, start, ScalarType(), outshape, 0);
      syn_out(0) = std::move(result);
    }
  }
}
} // namespace habana
