/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "backend/create_pt_tensor.h"
#include "hpu_ops/empty.h"

namespace habana {
OutputMetaDataVector EmptyStridedMeta(const at::Stack& stack) {
  OutputMetaData meta;
  const auto size = stack.at(0).toIntList().vec();
  const auto strides = stack.at(1).toIntList().vec();
  const auto dtype =
      stack.at(2).isNone() ? at::kFloat : stack.at(2).toScalarType();
  const auto layout =
      stack.at(3).isNone() ? at::kStrided : stack.at(3).toLayout();
  const auto device = stack.at(4).isNone() ? at::kHPU : stack.at(4).toDevice();
  const auto pin_memory = stack.at(5).isNone() ? false : stack.at(5).toBool();
  TORCH_CHECK(!pin_memory, "Only dense CPU tensors can be pinned");

  TORCH_INTERNAL_ASSERT(device.is_hpu());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout == at::Layout::Strided);

  meta.shape = size;
  meta.dtype = dtype;
  meta.strides = strides;
  meta.mem_format = at::MemoryFormat::Contiguous;

  return {meta};
}

EmptyStrided::EmptyStrided(int device_id, c10::ScalarType scalar_type)
    : Empty(device_id, scalar_type) {
  SetOutputMeta(EmptyStridedMeta);
}

OutputMetaDataVector EmptyMeta(const at::Stack& stack) {
  OutputMetaData meta;
  const auto size = stack.at(0).toIntList().vec();
  const auto dtype =
      stack.at(1).isNone() ? at::kFloat : stack.at(1).toScalarType();
  const auto layout =
      stack.at(2).isNone() ? at::kStrided : stack.at(2).toLayout();
  const auto device = stack.at(3).isNone() ? at::kHPU : stack.at(3).toDevice();
  const auto pin_memory = stack.at(4).isNone() ? false : stack.at(4).toBool();
  const auto memory_format = stack.at(5).isNone()
      ? at::MemoryFormat::Contiguous
      : stack.at(5).toMemoryFormat();
  TORCH_CHECK(!pin_memory, "Only dense CPU tensors can be pinned");

  TORCH_INTERNAL_ASSERT(device.is_hpu());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout == at::Layout::Strided);

  meta.shape = size;
  meta.dtype = dtype;
  meta.mem_format = memory_format;

  return {meta};
}

void Empty::CustomHandler(synapse_helpers::graph& graph, at::Stack& stack) {
  const auto& metadata = GetOutputMetaData(0);
  const auto& dtype = metadata.dtype;
  const auto& mem_format = metadata.mem_format;
  const auto& sizes = metadata.shape;
  auto strides = metadata.strides;
  if (strides.empty()) {
    strides = CalculateStrides(sizes, mem_format);
  }

  at::Tensor output;
  if (metadata.persistent) {
    at::TensorOptions options =
        at::TensorOptions().dtype(dtype).device(at::kHPU);
    output = at::empty_strided(sizes, strides, options);
  } else {
    output = habana::nonPersistentTensor(
        sizes, strides, mem_format, scalarTypeToTypeMeta(dtype));
  }
  AllocateSynapseOutput(graph, output, metadata);
}

void Empty::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& metadata = GetOutputMetaData(0);
  at::Scalar val = 0;
  auto zero_constant = ConstantHelper(graph, val, metadata.dtype);

  auto final_result_index =
      metadata.persistent ? c10::make_optional<int>(0) : c10::nullopt;
  auto broadcast = BroadcastHelper(
      graph,
      zero_constant.get(),
      metadata.shape,
      metadata.dtype,
      final_result_index);

  // output of broadcast is the output of this op
  syn_out(0) = std::move(broadcast);
}

Empty::Empty(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "None_", scalar_type, {}, {}, {}, false) {
  SetOutputMeta(EmptyMeta);
}
} // namespace habana

static const auto& EmptyKernelRegistry =
    habana::KernelRegistry()
        .add("aten::empty", KERNEL_FN_GLOBAL(habana::Empty))
        .add("aten::empty.memory_format", KERNEL_FN_GLOBAL(habana::Empty))
        .add("aten::empty_strided", KERNEL_FN_GLOBAL(habana::EmptyStrided));
