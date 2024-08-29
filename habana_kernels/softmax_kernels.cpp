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
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include <algorithm>
#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/recipe.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/softmax_kernels.h"

using namespace torch;
using namespace habana;

void SoftmaxIntOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Softmax operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be tensor for Softmax operator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input type expected to be int for Softmax operator");
  TORCH_CHECK(
      inputs[2].isScalar() || inputs[2].isInt() || inputs[2].isNone(),
      "Input type expected to be Scalar for SoftmaxInt operator");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  if (!inputs[2].isNone()) {
    if (inputs[2].isScalar()) {
      [[maybe_unused]] auto dtype = inputs[2].toScalar();
    } else if (inputs[2].isInt()) {
      [[maybe_unused]] auto dtype = inputs[2].toInt();
    }
  }
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

  ns_Softmax::Params params{static_cast<int>(self.ndimension() - 1 - dim)};

  if (self.scalar_type() == c10::ScalarType::Int ||
      self.scalar_type() == c10::ScalarType::Bool) {
    // Cast Input tensor to Float tensor
    std::string node_type;
    if (self.scalar_type() == c10::ScalarType::Int) {
      node_type = "cast_i32_to_f32";
    } else if (self.scalar_type() == c10::ScalarType::Bool) {
      node_type = "cast_i8_to_f32";
    }

    // Create the operator
    auto intToFloatOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    intToFloatOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    // Build Params for the graph
    std::vector<c10::IValue> stack{
        IValue(self), IValue(c10::ScalarType::Float)};
    intToFloatOp->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));

    synapse_helpers::tensor& float_syn_tensor =
        intToFloatOp->GetSynOutputs()[0];
    auto output_float = intToFloatOp->GetOutputs()[0];

    auto output =
        habana::createPTTensor(output_float, output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    std::vector<synTensor> syn_in{float_syn_tensor.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    at::ScalarType scalar_type = output_float.scalar_type();
    node_type = get_guid_with_precision("softmax_fwd", scalar_type);
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &params,
        sizeof(params),
        std::move(node_type),
        nullptr,
        nullptr,
        nullptr,
        deterministic,
        getContextHints());
  } else {
    // Softmax Operator
    p_context_->params_.emplace<ns_Softmax::Params>(params);
    p_context_->params_size_ = sizeof(params);

    auto output =
        habana::createPTTensor(self, output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

void SoftmaxIntOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Softmax operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be tensor for Softmax operator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input type expected to be int for Softmax operator");
  TORCH_CHECK(
      inputs[2].isScalar(),
      "Input type expected to be Bool for SoftmaxInt operator");

  auto self = inputs[0].toTensor();
  if (self.dtype() == c10::ScalarType::BFloat16) {
    auto output = at::empty(
        self.sizes(),
        self.options().dtype(c10::ScalarType::BFloat16),
        self.suggest_memory_format());
    HabanaOperator::SetPTOutput(output);
  } else {
    auto output = at::empty(
        self.sizes(),
        self.options().dtype(c10::ScalarType::Float),
        self.suggest_memory_format());
    HabanaOperator::SetPTOutput(output);
  }
}

static auto& SoftmaxKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("aten::softmax", KERNEL_FN(SoftmaxIntOperator))
        .add("aten::softmax.int", KERNEL_FN(SoftmaxIntOperator));
