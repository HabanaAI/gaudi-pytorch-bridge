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

// #include "generated/backend/full.h"
#include "hpu_ops/dynamic_op.h"
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
  // convert tensor to shape vector
  if (stack.at(SIZE_INDEX).isTensor()) {
    meta.shape = stack.at(SIZE_INDEX).toTensor().sizes().vec();
  } else {
    meta.shape = stack.at(SIZE_INDEX).toIntVector();
  }
  return {meta};
}

void FullBE::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto fillValue = stack.at(FILL_VALUE_INDEX).toScalar();
  const auto meta = FullMeta(stack)[0];
  auto result = ConstantHelper(graph, fillValue, meta.dtype, meta.shape, 0);
  syn_out(0) = std::move(result);
}

FullBE::FullBE(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "constant", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(FullMeta);
}

bool FullDSSTMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  PT_BRIDGE_DEBUG("FullDSSTMeta called");
  static_cast<void>(inputs);
  static_cast<void>(outputs);
  if (inputs[0].isTensor()) {
    auto t_size = inputs[0].getTensorShape();
    PT_BRIDGE_DEBUG("FullDSSTMeta constant shape ", t_size);
    habana_helpers::UpdateSTShapeInfo(t_size);
  } else {
    PT_BRIDGE_DEBUG("Full DS meta not supported non tensor input !!!");
    return false;
  }

  return true;
}

void FullOperatorDS::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto fillValue = stack.at(FILL_VALUE_INDEX).toScalar();
  const auto meta = FullMeta(stack)[0];
  auto result = ConstantHelper(graph, fillValue, meta.dtype, meta.shape, 0);
  syn_out(0) = std::move(result);
}

FullOperatorDS::FullOperatorDS(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "full", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(FullMeta);
  SetSTMetaFn(FullDSSTMeta);
}
} // namespace habana

static const auto& HabanaFullKernelRegistry = habana::KernelRegistry().add(
    "aten::full",
    KERNEL_FN_GLOBAL(habana::FullBE));
static const auto& FullOpKernelRegistry = habana::KernelRegistry().add(
    "hpu::full_ds",
    KERNEL_FN_GLOBAL(habana::FullOperatorDS));