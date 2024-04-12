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

FullBE::FullBE(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "constant", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(FullMeta);
}

void FullBE::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto fillValue = stack.at(FILL_VALUE_INDEX).toScalar();
  const auto meta = FullMeta(stack)[0];
  auto result = ConstantHelper(graph, fillValue, meta.dtype, meta.shape, 0);
  syn_out(0) = std::move(result);
}

} // namespace habana

static const auto& HabanaFullKernelRegistry = habana::KernelRegistry().add(
    "aten::full",
    KERNEL_FN_GLOBAL(habana::FullBE));
