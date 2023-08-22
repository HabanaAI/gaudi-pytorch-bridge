/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/threshold_kernels.h"

using namespace torch;

void habana::ThresholdBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for threshold operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg3 type expected to be scalar");

  auto grad_output = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto threshold = inputs[2].toScalar();

  TORCH_CHECK(
      threshold.to<float>() == 0.0,
      "Threshold values other than 0 are not supported")

  auto grad_input =
      habana::createPTTensor(self, output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, grad_input, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

habana::InferOutputMetaRetType habana::ThresholdBackwardOperator::
    InferOutputMeta(torch::jit::Stack& inputs) {
  auto self = inputs[1].toTensor();
  habana::InferOutputMetaRetType out;
  out.AddOutputTensor(habana::TensorMetaData(
      self.sizes().vec(),
      HabanaOperator::CalculateStrides(
          self.sizes().vec(), self.suggest_memory_format()),
      self.scalar_type(),
      self.suggest_memory_format()));
  return out;
}
