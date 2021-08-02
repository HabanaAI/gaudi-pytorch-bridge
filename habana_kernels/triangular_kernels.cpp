/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_kernels/triangular_kernels.h"
#include <torch/script.h>
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;
using namespace habana;

void DiagOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input arguments for Diag Out Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg 1 for Diag Out op needs to be tensor type");
  TORCH_CHECK(inputs[1].isInt(), "Input arg 2 for Diag Out op needs to be Int");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg 3 for Diag Out op needs to be tensor type");

  static_cast<void>(is_output_persistent);
  auto self = inputs[0].toTensor();
  auto diagonal = inputs[1].toInt();
  auto output = inputs[2].toTensor();

  TORCH_CHECK(self.dim() <= 2, "Input tensor should have a dimension 1 or 2");

  // node parameters
  ns_MatrixDiag::Params params = {};
  params.kMin = diagonal;
  params.kMax = diagonal;

  // GUID depends on the number of input dimensions
  std::string guid;
  if (self.dim() == 1) {
    guid = "matrix_diagonal_fwd_";
  } else {
    guid = "matrix_diag_part_fwd_";
  }
  guid += habana_helpers::name_suffix_from_type(self.scalar_type());
  SetGuid(guid);

  p_context_->syn_outputs_.emplace_back(std::move(p_context_->syn_inputs_[1]));
  p_context_->pt_outputs_.emplace_back(output);

  std::vector<synTensor> syn_in;
  syn_in.emplace_back(
      static_cast<synapse_helpers::tensor&>(p_context_->syn_inputs_[0]).get());

  std::vector<synTensor> syn_out;
  syn_out.emplace_back(
      static_cast<synapse_helpers::tensor&>(p_context_->syn_outputs_[0]).get());

  graph.add_node(
      std::move(syn_in),
      std::move(syn_out),
      &params,
      sizeof(params),
      std::move(guid));
}

void DiagOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input arguments for Diag Operator");
  TORCH_CHECK(
      inputs[0].isTensor(), "Input arg 1 for Diag op needs to be tensor type");
  TORCH_CHECK(inputs[1].isInt(), "Input arg 2 for Diag op needs to be Int");

  auto self = inputs[0].toTensor();
  auto diagonal = inputs[1].toInt();

  auto out_shape = compute_output_shape(self, diagonal);

  auto output = habana_helpers::createPTTensor(
      self,
      out_shape,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);

  // Addding to stack, pt_inputs to be suitable
  // for out variant's AllocateAbdAddSynapseNoe
  inputs.emplace_back(output);
  AllocateSynapseInput(graph, output, is_output_persistent);

  DiagOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

std::vector<int64_t> DiagOutOperator::compute_output_shape(
    const Tensor& self,
    int64_t& diagonal) {
  auto sizes = self.sizes().vec();
  std::vector<int64_t> output_shape;
  if (self.dim() == 1) {
    output_shape.push_back(self.sizes().vec()[0] + abs(diagonal));
    output_shape.push_back(self.sizes().vec()[0] + abs(diagonal));
  }
  if (self.dim() == 2) {
    int64_t m = self.sizes().vec()[0];
    int64_t n = self.sizes().vec()[1];
    int size;
    if (diagonal > 0) {
      size = n - abs(diagonal);
    } else {
      size = m - abs(diagonal);
    }
    TORCH_CHECK(
        size > 0,
        "Invalid inputs!",
        "Diagonal value",
        diagonal,
        "Input size",
        self.sizes().vec());
    output_shape.push_back(size);
  }
  return output_shape;
}

Tensor DiagOutOperator::AllocateOutputTensor(
    const Tensor& self,
    int64_t& diagonal,
    bool is_output_persistent) {
  auto shape = compute_output_shape(self, diagonal);

  // allocate output tensor
  auto output = habana_helpers::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);

  return output;
}

void DiagOutOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto diagonal = inputs[1].toInt();

  auto output = AllocateOutputTensor(self, diagonal, true);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

at::Tensor diag_hpu(const at::Tensor& self, int64_t diagonal) {
  at::ScalarType scalar_type = self.scalar_type();
  size_t device_id = self.device().index();
  // Create the operator
  DiagOperator Op(device_id, scalar_type);
  std::string node_type =
      "diag_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {IValue(self), IValue(diagonal)};

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

at::Tensor& diag_hpu_out(
    const at::Tensor& self,
    int64_t diagonal,
    at::Tensor& result) {
  at::ScalarType scalar_type = self.scalar_type();
  size_t device_id = self.device().index();
  // Create the operator
  DiagOutOperator Op(device_id, scalar_type);
  std::string node_type =
      "diag_out" + habana_helpers::name_suffix_from_type(scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, result};
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(diagonal), IValue(result)};

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
  PT_KERNEL_END;
  return result;
}

/**
 * @brief Fill matrix_band_part params structure
 */
ns_MatrixBandPartKernel::Params matrix_band_part_params_builder(
    const int64_t diagonal,
    const std::string& callerOp) {
  ns_MatrixBandPartKernel::Params param;
  if (callerOp == "TriuOperator") {
    if (diagonal > 0) {
      param.numLower = -1;
      param.numUpper = diagonal - 1;
    } else {
      param.numLower = abs(diagonal);
      param.numUpper = -1;
    }
  } else {
    if (diagonal >= 0) {
      param.numLower = -1;
      param.numUpper = diagonal;
    } else {
      param.numLower = abs(diagonal) - 1;
      param.numUpper = -1;
    }
  }
  return param;
}

void MatrixBandPartOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  const int64_t correctInputSize = 2;
  TORCH_CHECK(
      inputs.size() == correctInputSize,
      "Incorrect size ",
      inputs.size(),
      " provided as input, while expected size is ",
      correctInputSize,
      " for MatrixBandPartOperator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for MatrixBandPartOperator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input arg2 expected to be Int for MatrixBandPartOperator");

  auto self = inputs[0].toTensor();
  auto diagonal = inputs[1].toInt();

  ns_MatrixBandPartKernel::Params param =
      matrix_band_part_params_builder(diagonal, this->callerOp_);

  p_context_->params_.emplace<ns_MatrixBandPartKernel::Params>(param);
  p_context_->params_size_ = sizeof(param);

  auto out = habana_helpers::createPTTensor(self, is_output_persistent);
  AllocateSynapseOutput(graph, out, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void TriuOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  const int64_t correctInputSize = 2;
  TORCH_CHECK(
      inputs.size() == correctInputSize,
      "Incorrect size ",
      inputs.size(),
      " provided as input, while expected size is ",
      correctInputSize,
      " for TriuOperator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for TriuOperator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input arg2 expected to be Int for TriuOperator");

  auto self = inputs[0].toTensor();
  auto diagonal = inputs[1].toInt();

  if (diagonal > 0) {
    auto matrixBandPartOperatorOp = make_operator<MatrixBandPartOperator>(
        this->p_context_->device_id_, self.scalar_type(), "TriuOperator");
    auto& syn_arg0 = matrixBandPartOperatorOp->SetSynapseInput(
        std::move(p_context_->syn_inputs_[0]));
    matrixBandPartOperatorOp->AllocateAndAddSynapseNode(graph, inputs, false);
    p_context_->syn_inputs_[0] = std::move(syn_arg0);

    auto subOp = make_operator<SubOperator>(
        this->p_context_->device_id_, self.scalar_type());
    auto& syn_arg1 =
        subOp->SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    UNUSED auto& syn_arg2 = subOp->SetSynapseInput(
        std::move(matrixBandPartOperatorOp->GetSynOutputs()[0]));

    Scalar alpha = 1.0;
    torch::jit::Stack subOp_stack{
        IValue(self),
        IValue(matrixBandPartOperatorOp->GetOutputs()[0]),
        IValue(alpha)};
    subOp->AllocateAndAddSynapseNode(graph, subOp_stack, is_output_persistent);

    p_context_->syn_inputs_[0] = std::move(syn_arg1);

    p_context_->pt_outputs_.emplace_back(std::move(subOp->GetOutputs()[0]));
    p_context_->syn_outputs_.emplace_back(std::move(subOp->GetSynOutputs()[0]));
  } else {
    auto triuOp = make_operator<MatrixBandPartOperator>(
        this->p_context_->device_id_, self.scalar_type(), "TriuOperator");
    auto& syn_arg0 = triuOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    triuOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg0);
    p_context_->pt_outputs_.emplace_back(triuOp->GetOutputs()[0]);
    p_context_->syn_outputs_.emplace_back(
        std::move(triuOp->GetSynOutputs()[0]));
  }
}

void TrilOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  const int64_t correctInputSize = 2;
  TORCH_CHECK(
      inputs.size() == correctInputSize,
      "Incorrect size ",
      inputs.size(),
      " provided as input, while expected size is ",
      correctInputSize,
      " for TrilOperator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for TrilOperator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input arg2 expected to be Int for TrilOperator");

  auto self = inputs[0].toTensor();
  auto diagonal = inputs[1].toInt();

  if (diagonal >= 0) {
    auto triuOp = make_operator<MatrixBandPartOperator>(
        this->p_context_->device_id_, self.scalar_type(), "TrilOperator");
    auto& syn_arg0 = triuOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    triuOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg0);
    p_context_->pt_outputs_.emplace_back(triuOp->GetOutputs()[0]);
    p_context_->syn_outputs_.emplace_back(
        std::move(triuOp->GetSynOutputs()[0]));
  } else {
    auto matrixBandPartOperatorOp = make_operator<MatrixBandPartOperator>(
        this->p_context_->device_id_, self.scalar_type(), "TrilOperator");
    auto& syn_arg0 = matrixBandPartOperatorOp->SetSynapseInput(
        std::move(p_context_->syn_inputs_[0]));
    matrixBandPartOperatorOp->AllocateAndAddSynapseNode(graph, inputs, false);
    p_context_->syn_inputs_[0] = std::move(syn_arg0);

    auto subOp = make_operator<SubOperator>(
        this->p_context_->device_id_, self.scalar_type());
    auto& syn_arg1 =
        subOp->SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    UNUSED auto& syn_arg2 = subOp->SetSynapseInput(
        std::move(matrixBandPartOperatorOp->GetSynOutputs()[0]));

    Scalar alpha = 1.0;
    torch::jit::Stack subOp_stack{
        IValue(self),
        IValue(matrixBandPartOperatorOp->GetOutputs()[0]),
        IValue(alpha)};
    subOp->AllocateAndAddSynapseNode(graph, subOp_stack, is_output_persistent);

    p_context_->syn_inputs_[0] = std::move(syn_arg1);

    p_context_->pt_outputs_.emplace_back(std::move(subOp->GetOutputs()[0]));
    p_context_->syn_outputs_.emplace_back(std::move(subOp->GetSynOutputs()[0]));
  }
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::diag",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<DiagOperator>(device_id, node_type);
            })
        .add(
            "hpu::diag_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<DiagOutOperator>(device_id, node_type);
            })
        .add(
            "aten::triu",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<TriuOperator>(device_id, node_type);
            })
        .add("aten::tril", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<TrilOperator>(device_id, node_type);
        });