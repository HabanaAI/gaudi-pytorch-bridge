/******************************************************************************
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

#include "generated/backend/_fused_dropout.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
std::shared_ptr<void> FillFusedDropoutParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_DropoutKernel::Params);
  params->ratio = stack.at(1).toScalar().toDouble();
  return params;
}

OutputMetaDataVector FusedDropoutMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto shape = self.sizes().vec();

  OutputMetaDataVector metas(2);
  metas[0].shape = shape;
  metas[0].dtype = self.scalar_type();
  metas[1].shape = shape;
  metas[1].dtype = at::kChar;

  return metas;
}

void FusedDropout::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto seed = stack.at(2);
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto metas = FusedDropoutMeta(stack);

  std::vector<synTensor> inputTensors = {syn_in(0)};
  if (seed.isTensor())
    inputTensors.push_back(syn_in(1));
  else
    inputTensors.push_back(syn_seed());

  auto dropout = BuildOp(
      graph,
      get_guid_with_precision("dropout_fwd", metas[0].dtype),
      inputTensors,
      {NodeAttr::NodeOutputAttr{metas[0].shape, metas[0].dtype, 0},
       NodeAttr::NodeOutputAttr{metas[1].shape, metas[1].dtype, 1}},
      params.get(),
      size);

  syn_out(0) = std::move(dropout[0]);
  syn_out(1) = std::move(dropout[1]);
}
} // namespace habana
