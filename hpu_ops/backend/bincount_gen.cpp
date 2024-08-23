/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/bincount.h"

namespace habana {

OutputMetaDataVector BinCountMeta(const at::Stack& stack) {
  int64_t length = stack.at(1).toInt();
  auto weights = stack.at(2).toOptional<at::Tensor>();

  OutputMetaData meta;
  meta.shape = {length};
  meta.dtype = weights.has_value() ? weights.value().scalar_type()
                                   : c10::ScalarType::Int;
  return {meta};
}

BinCount::BinCount(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(BinCountMeta);
}

void BinCount::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(
      !graph.is_dynamic_graph(), "Dynamic graph is not supported for bincount");

  StackGetter stackGetter(stack, "Bincount::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto length = getNextInput<int32_t>(stackGetter);
  auto weights = getNextInput<c10::optional<TensorsPair>>(stackGetter);

  ns_BinCountKernel::Params params{
      weights.has_value() ? BinCountMode_t::USE_WEIGHT
                          : BinCountMode_t::NO_WEIGHT};
  auto meta = BinCountMeta(stack).at(0);

  auto constantHolder = ConstantHelper(graph, length, c10::ScalarType::Int);
  std::vector<synTensor> inputs{self.syn_t, constantHolder.get()};
  if (weights.has_value()) {
    inputs.push_back(weights.value().syn_t);
  }

  auto bincount = BuildOp(
      graph,
      get_guid_with_precision("bincount", meta.dtype),
      std::move(inputs),
      {{meta.shape, meta.dtype, 0}},
      (void*)&params,
      sizeof(params));

  syn_out(0) = std::move(bincount.at(0));
}

} // namespace habana

static const auto& BinCountKernelRegistry = habana::KernelRegistry().add(
    "hpu::bincount_backend",
    KERNEL_FN_GLOBAL(habana::BinCount));
