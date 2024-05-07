/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/full.h"
#include "hpu_ops/full_bool.h"

namespace habana {

const unsigned SIZE_INDEX = 0;
const unsigned FILL_VALUE_INDEX = 1;
const unsigned DTYPE_INDEX = 2;

OutputMetaDataVector FullMeta(const at::Stack& stack) {
  auto optionalDtype = stack.at(DTYPE_INDEX).toOptional<at::ScalarType>();
  at::ScalarType dtype;
  if (optionalDtype.has_value()) {
    dtype = optionalDtype.value();
  } else {
    auto fillValue = stack.at(FILL_VALUE_INDEX);
    if (fillValue.isBool())
      dtype = torch::kBool;
    else
      dtype = stack.at(FILL_VALUE_INDEX).isInt() ? torch::kLong : torch::kFloat;
  }

  OutputMetaData meta;
  meta.dtype = dtype;
  meta.shape = stack.at(SIZE_INDEX).toIntVector();

  return {meta};
}

static synapse_helpers::tensor CommonFull(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto fillValue = stack.at(FILL_VALUE_INDEX).toScalar();
  const auto meta = FullMeta(stack)[0];
  return OpBackend::BuildConstant(
      op, graph, fillValue, meta.dtype, meta.shape, 0);
}

void FullBE::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  syn_out(0) = std::move(CommonFull(this, graph, stack));
}

FullBool::FullBool(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "constant", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(FullMeta);
}

void FullBool::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  syn_out(0) = std::move(CommonFull(this, graph, stack));
}

FullBE::FullBE(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "constant", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(FullMeta);
}

} // namespace habana

/* This is a temporary solution that will be removed once "RuntimeError: Tensor
 * Data not present in ExecutedCachedGraph in YOLOv8" is resolved in lazy mode
 */
TORCH_LIBRARY_FRAGMENT(aten, m) {
  m.def(
      "full.bool(int[] size, bool fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor(a)");
}

static const auto& HabanaRandomKernelRegistry = habana::KernelRegistry().add(
    "aten::full.bool",
    KERNEL_FN_GLOBAL(habana::FullBool));
