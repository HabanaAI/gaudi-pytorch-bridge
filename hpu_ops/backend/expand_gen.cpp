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
#include <ATen/ExpandUtils.h>
#include "habana_kernels/kernel_utils.h"
#include "hpu_ops/expand.h"

namespace habana {
InferOutputMetaRetType ExpandOp::OutputShapeInf(
    const torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  InferOutputMetaRetType out;
  if (inputs[1].isIntList()) {
    auto size = inputs[1].toIntList();
    std::tie(expandedSizes, expandedStrides) = at::inferExpandGeometry(
        self.sizes(), self.strides(), at::IntArrayRef(size.vec()));

    habana_helpers::recalc_strides(expandedStrides, expandedSizes);
    out.AddShapeTensor(TensorMetaData(
        expandedSizes,
        expandedStrides,
        self.scalar_type(),
        self.suggest_memory_format()));
  } else {
    auto expand_shape = inputs[1].toTensor();
    expandedSizes = expand_shape.sizes().vec();
    expandedStrides = HabanaOperator::CalculateStrides(
        expandedSizes, self.suggest_memory_format());
  }

  out.AddOutputTensor(TensorMetaData(
      expandedSizes,
      expandedStrides,
      self.scalar_type(),
      self.suggest_memory_format()));
  return out;
}

void ExpandOp::AddNode(
    synapse_helpers::graph& graph,
    [[maybe_unused]] const at::Stack& inputs) {
  const auto& metadata = GetOutputMetaData(0);
  auto final_result_index =
      metadata.persistent ? c10::make_optional<int>(0) : c10::nullopt;
  const auto exp_bias =
      habana_helpers::get_tensor_exp_bias(inputs[0].toTensor());
  auto broadcast = BroadcastHelper(
      graph,
      syn_in(0),
      metadata.shape,
      metadata.dtype,
      final_result_index,
      exp_bias);
  syn_out(0) = std::move(broadcast);
}

OutputMetaDataVector ExpandMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);
  auto size = stack.at(1).toIntList();
  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = at::inferExpandGeometry(
      self.sizes(), self.strides(), at::IntArrayRef(size.vec()));

  OutputMetaDataVector meta(1);
  meta.at(0).shape = expandedSizes;
  meta.at(0).dtype = self.scalar_type();
  habana_helpers::set_output_hw_scaling_meta(self, meta.at(0));
  return meta;
}

ExpandOp::ExpandOp(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "broadcast", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(ExpandMeta);
}
} // namespace habana

static const auto& ExpandKernelRegistry = habana::KernelRegistry().add(
    "aten::expand",
    KERNEL_FN_GLOBAL(habana::ExpandOp));
