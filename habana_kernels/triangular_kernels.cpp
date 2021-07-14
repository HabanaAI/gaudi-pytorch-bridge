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
#include "habana_kernels/embedding_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;
using namespace habana;

void MatrixDiagonalOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  auto self = inputs[0].toTensor();

  std::vector<int64_t> shape_out = {
      self.sizes().vec()[0], self.sizes().vec()[0]};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&self});
  auto output = habana_helpers::createPTTensor(
      self, shape_out, self.options(), memory_format, is_output_persistent);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
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

  TORCH_CHECK(self.dim() <= 2, "Input tensor should have a dimension 1 or 2");

  torch::jit::Stack stack;

  if (self.dim() == 1) {
    /*
    example for Matrix Diagonal Operator functinality
    [x, y, z] -----> [[x, 0, 0],
                      [0, y, 0],
                      [0, 0, z]]

    dimension of the resultant matrix from matrix diagonal op is 2

    Pad with diagonal value 1

    pad_values = {diagonal(1), 0, 0, diagonal(1)} - we have two values for each
    of our dimensions( 4 for 2 in our eg.). The first two values corresponds to
    0th dimension in TPC kernel(n - 1 - 0 in the pytorch case)

    first two values of pad_values - diagonal(1), 0
    we need to add diagonal(1) number of 0s infront of each row and
    0 number of 0s at the end of each row.

    which will make
    [[x, 0, 0],      [[0, x, 0, 0],
     [0, y, 0], --->  [0, 0, y, 0],
     [0, 0, z]]       [0, 0, 0, z]]

    last two values of pad_values - 0, diagonal(1)

    we need to add 0 number of 0s infront of each column and
    diagonal(1) number of 0s at the end of each column.

    [[0, x, 0, 0],       [[0, x, 0, 0],
     [0, 0, y, 0],  --->  [0, 0, y, 0],
     [0, 0, 0, z]]        [0, 0, 0, z],
                          [0, 0, 0, 0]]

    which is the desired output!!!

    similarly,
    for pad with diagonal value -1 the pad_values are
    {0, -diagonal, -diagonal, 0}. With the similar explanation
    as above, we will get the matrix as following
    [[0, 0, 0, 0],
     [x, 0, 0, 0],
     [0, y, 0, 0],
     [0, 0, z, 0]]

    */
    auto matrix_diagonal_op = make_operator<MatrixDiagonalOperator>(
        self.device().index(), self.scalar_type());
    matrix_diagonal_op->SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    stack = {c10::IValue(self)};
    if (diagonal != 0) {
      matrix_diagonal_op->AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();
      // Build Params for pad and call pad
      auto pad_op =
          make_operator<PadOperator>(self.device().index(), self.scalar_type());
      pad_op->SetPTInputs({matrix_diagonal_op->GetOutputs()[0]});
      pad_op->SetSynapseInput(
          std::move(matrix_diagonal_op->GetSynOutputs()[0]));
      float value = 0.0f;
      std::vector<int64_t> pad;
      if (diagonal < 0)
        pad = {0, -diagonal, -diagonal, 0};
      else
        pad = {diagonal, 0, 0, diagonal};

      stack = {
          IValue(matrix_diagonal_op->GetOutputs()[0]),
          IValue(pad),
          IValue(value)};
      pad_op->AllocateAndAddSynapseNode(graph, stack, true);
      stack.clear();
      p_context_->syn_outputs_.emplace_back(
          std::move(pad_op->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(pad_op->GetOutputs()[0]);
    } else {
      matrix_diagonal_op->AllocateAndAddSynapseNode(graph, stack, true);
      stack.clear();
      p_context_->syn_outputs_.emplace_back(
          std::move(matrix_diagonal_op->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(matrix_diagonal_op->GetOutputs()[0]);
    }
  }

  if (self.dim() == 2) {
    /*
    Given 3x3 matrix/tensor
    [[x1, y1, z1],
     [x2, y2, z2],
     [x3, y3, z3]]

    reshape the tensor to 1 x 9
    [[x1, y1, z1, x2, y2, z2, x3, y3, z3]]

    if diagonal value is 1, then slice the values from start = diagonal(1)
    to end = 3 * (3 * 2) + 1 = 7 with steps of 4 in the reshaped tensor
    so, the resultant tensor will be [[y1, z2]].

    Now, reshape the tensor to 2, so we wil get [y1, z2],
    which is the desired output.
    Similary we can obtain output from diagonal values less than 0 with
    appropriate slicing.
    */
    TORCH_CHECK(
        diagonal < self.sizes().vec()[0],
        "Diagonal value should be in -self.sizes().vec()[0] < diagonal < self.sizes().vec()[0] for 2D matrix");
    TORCH_CHECK(
        -diagonal < self.sizes().vec()[0],
        "Diagonal value should be in -self.sizes().vec()[0] < diagonal < self.sizes().vec()[0] for 2D matrix");
    // Add Reshape node to graph
    auto reshape_op = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, self.scalar_type());
    reshape_op->SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    std::vector<int64_t> reshape_dim;
    reshape_dim.push_back(self.sizes().vec()[0] * self.sizes().vec()[0]);
    reshape_dim.push_back(1);
    stack = {c10::IValue(self), c10::IValue(reshape_dim)};
    reshape_op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto slice_op =
        make_operator<SliceOperator>(self.device().index(), self.scalar_type());
    slice_op->SetPTInputs({reshape_op->GetOutputs()[0]});
    slice_op->SetSynapseInput(std::move(reshape_op->GetSynOutputs()[0]));
    int dim = 0;
    int start, end, step;
    int n = self.sizes().vec()[0];
    step = n + 1;
    if (diagonal >= 0) {
      start = diagonal;
      end = n * (n - diagonal) + diagonal;
    } else {
      start = -diagonal * n;
      end = n * (n - 1) + n + diagonal;
    }
    stack = {
        IValue(reshape_op->GetOutputs()[0]),
        IValue(dim),
        IValue(start),
        IValue(end),
        IValue(step)};
    slice_op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto reshape_op2 = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, self.scalar_type());
    reshape_op2->SetSynapseInput(std::move(slice_op->GetSynOutputs()[0]));
    reshape_dim.clear();
    auto output_shape = compute_output_shape(self, diagonal);

    stack = {c10::IValue(slice_op->GetOutputs()[0]), c10::IValue(output_shape)};
    reshape_op2->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(
        std::move(reshape_op2->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(reshape_op2->GetOutputs()[0]);
  }
}

std::vector<int64_t> DiagOperator::compute_output_shape(
    const Tensor& self,
    int64_t& diagonal) {
  auto sizes = self.sizes().vec();
  std::vector<int64_t> output_shape;
  if (self.dim() == 1) {
    output_shape.push_back(self.sizes().vec()[0] + abs(diagonal));
    output_shape.push_back(self.sizes().vec()[0] + abs(diagonal));
  }
  if (self.dim() == 2) {
    output_shape.push_back(self.sizes().vec()[0] - abs(diagonal));
  }
  return output_shape;
}

Tensor DiagOperator::AllocateOutputTensor(
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

void DiagOperator::SetPTOutputs(torch::jit::Stack& inputs) {
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
            "aten::triu",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<TriuOperator>(device_id, node_type);
            })
        .add("aten::tril", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<TrilOperator>(device_id, node_type);
        });