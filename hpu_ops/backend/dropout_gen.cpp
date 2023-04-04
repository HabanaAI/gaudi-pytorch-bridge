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

#include "hpu_ops/backend/dropout.h"

namespace habana {
std::vector<synapse_helpers::tensor> BuildDropout(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const std::vector<OpBackend::TensorsPair>& inTensors,
    const std::vector<NodeAttr::NodeOutputAttr>& outAttr,
    ns_DropoutKernel::Params* params,
    const size_t paramsSize) {
  HABANA_ASSERT(
      1 <= inTensors.size() and inTensors.size() <= 3,
      "Number of input tensors must be in the range [1, 3]");
  HABANA_ASSERT(
      inTensors[0].pt_t.scalar_type() == outAttr[0].dtype,
      "Input/output feature maps tensors must be of the same datatype");
  HABANA_ASSERT(
      inTensors[0].pt_t.sizes() == outAttr[0].sizes,
      "Input/output feature maps tensors must be of same size");
  if (inTensors.size() > 1) {
    HABANA_ASSERT(
        inTensors[1].pt_t.numel() == 1, "Seed tensor must be \"scalar\"");
  }
  if (outAttr.size() > 1) {
    HABANA_ASSERT(
        outAttr[1].dtype == at::kChar, "The mask tensor is of int8 type");
    HABANA_ASSERT(
        inTensors[0].pt_t.sizes() == outAttr[1].sizes,
        "Input/output and mask feature maps tensors must be of same size");
  }

  const std::string guid = "dropout_fwd_" +
      habana_helpers::name_suffix_from_type(inTensors[0].pt_t.scalar_type());

  std::vector<synTensor> inputTensors;
  inputTensors.reserve(3);
  for (const auto& tensor : inTensors)
    inputTensors.push_back(tensor.syn_t);

  return OpBackend::BuildNode(
      op,
      graph,
      {std::move(guid),
       std::move(inputTensors),
       std::move(outAttr),
       params,
       paramsSize});
}
} // namespace habana
