/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include <ATen/InferSize.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/habana_device/tensor_builder.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_ops/common/batched_matmul_output_shape.h"

using namespace torch;

static void check_matmul_params(
    const Tensor& mat1,
    const Tensor& mat2,
    bool mat1_transposed,
    bool mat2_transposed,
    c10::optional<const at::Tensor*> bias) {
  TORCH_CHECK(mat1.ndimension() == 2, "matmul_hpu supports only 2d matrices");
  TORCH_CHECK(mat2.ndimension() == 2, "matmul_hpu supports only 2d matrices");

  if (mat1_transposed == false && mat2_transposed == false) {
    TORCH_CHECK(
        mat1.size(1) == mat2.size(0), "matmul inner dimensions doesn't match");
  } else if (mat1_transposed == true && mat2_transposed == false) {
    TORCH_CHECK(
        mat1.size(0) == mat2.size(0), "matmul inner dimensions doesn't match");
  } else if (mat1_transposed == false && mat2_transposed == true) {
    TORCH_CHECK(
        mat1.size(1) == mat2.size(1), "matmul inner dimensions doesn't match");
  } else {
    TORCH_CHECK(false, "matmul_hpu won't support both transposed");
  }

  /*TORCH_CHECK(
      mat1.size(1) == mat2.size(0), "matmul inner dimensions doesn't match"); */
  TORCH_CHECK(
      static_cast<int>(mat1.is_contiguous()) + mat2.is_contiguous() > 0,
      "Only one matrix can me non contiguous.",
      "\nmat1.is_contiguous() returned: ",
      mat1.is_contiguous(),
      "\nmat2.is_contiguous() returned: ",
      mat2.is_contiguous(),
      "\nmat1 sizes: ",
      mat1.sizes(),
      "mat1 strides: ",
      mat1.strides(),
      "\nmat2 sizes: ",
      mat2.sizes(),
      "mat2 strides: ",
      mat2.strides());
  if (bias)
    TORCH_CHECK(
        bias.value()->ndimension() == 1, "matmul_hpu supports only 1d bias");
}

std::vector<int64_t> habana::MMOperator::compute_output_shape(
    at::Tensor self,
    at::Tensor other,
    bool self_transposed,
    bool other_transposed) {
  if (self_transposed == false && other_transposed == false)
    return {self.size(0), other.size(1)};
  else if (self_transposed == true && other_transposed == false)
    return {self.size(1), other.size(1)};
  else if (self_transposed == false && other_transposed == true)
    return {self.size(0), other.size(0)};
  else
    return {self.size(1), other.size(0)};
}

void habana::MMOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  // Bias is the 5th input
  TORCH_CHECK(
      ((inputs.size() == 2) || (inputs.size() == 4) || (inputs.size() == 5)),
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  auto mat1 = inputs[0].toTensor();
  auto mat2 = inputs[1].toTensor();

  bool mat1_transposed = false;
  bool mat2_transposed = false;
  if (inputs.size() > 2) {
    TORCH_CHECK(inputs[2].isBool(), "Input tranpose flag expected to be bool");
    TORCH_CHECK(inputs[3].isBool(), "Input tranpose flag expected to be bool");
    mat1_transposed = inputs[2].toBool();
    mat2_transposed = inputs[3].toBool();
  }

  if (inputs.size() > 4) {
    TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");
  }

  check_matmul_params(
      mat1, mat2, mat1_transposed, mat2_transposed, c10::nullopt);

  auto shape_out = habana::MMOperator::compute_output_shape(
      mat1, mat2, mat1_transposed, mat2_transposed);
  auto output = habana::createPTTensor(
      mat1,
      shape_out,
      mat1.options(),
      mat1.suggest_memory_format(),
      output_metadata.at(0).dtype,
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  synGEMMParams params{mat1_transposed, mat2_transposed};
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void habana::BmmOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      ((inputs.size() >= 3) || (inputs.size() == 5) || (inputs.size() == 6)),
      "Incorrect size of inputs expected for BmmOut operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");

  auto out = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto mat2 = inputs[2].toTensor();
  bool mat1_transposed = false;
  bool mat2_transposed = false;
  if (inputs.size() == 5) {
    mat1_transposed = inputs[3].toBool();
    mat2_transposed = inputs[4].toBool();
  } else if (inputs.size() == 6) {
    mat1_transposed = inputs[4].toBool();
    mat2_transposed = inputs[5].toBool();
  }

  AllocateSynapseOutput(graph, out, output_metadata.at(0));
  synGEMMParams params{mat1_transposed, mat2_transposed};
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

std::vector<int64_t> habana::BmmOperator::compute_output_shape(
    const Tensor& self,
    const Tensor& mat2,
    bool mat1_transposed,
    bool mat2_transposed) {
  auto self_sizes = self.sizes();
  auto mat2_sizes = mat2.sizes();
  auto self_dims = self.dim();
  auto mat2_dims = mat2.dim();
  auto self_end_iter = self_sizes.end();
  auto mat2_end_iter = mat2_sizes.end();

  HABANA_ASSERT(self_dims >= 3 && "BMM Input1 should be at least 3D")
  HABANA_ASSERT(mat2_dims >= 2 && "BMM Input2 should be at least 2D")
  if (!mat1_transposed && !mat2_transposed) {
    HABANA_ASSERT(
        (*(self_end_iter - 1) == *(mat2_end_iter - 2)),
        "BMM inner dimensions doesn't match at (mat1, mat2) indices -1 & -2: ",
        *(self_end_iter - 1),
        ", ",
        *(mat2_end_iter - 2))
  } else if (mat1_transposed && !mat2_transposed) {
    HABANA_ASSERT(
        (*(self_end_iter - 2) == *(mat2_end_iter - 2)),
        "BMM inner dimensions doesn't match at (mat1, mat2) indices  -2 & -2: ",
        *(self_end_iter - 2),
        ", ",
        *(mat2_end_iter - 2))
  } else if (!mat1_transposed && mat2_transposed) {
    HABANA_ASSERT(
        (*(self_end_iter - 1) == *(mat2_end_iter - 1)),
        "BMM inner dimensions doesn't match at (mat1, mat2) indices  -1 & -1: ",
        *(self_end_iter - 1),
        ", ",
        *(mat2_end_iter - 1))
  } else { // if (mat1_transposed && mat2_transposed)
    HABANA_ASSERT(
        (*(self_end_iter - 2) == *(mat2_end_iter - 1)),
        "BMM inner dimensions doesn't match at (mat1, mat2) indices  -2 & -1: ",
        *(self_end_iter - 2),
        ", ",
        *(mat2_end_iter - 2))
  }

  return habana::getBatchMatmulOutShape(
      self_sizes, mat2_sizes, mat1_transposed, mat2_transposed);
}

void habana::BmmOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      ((inputs.size() == 2) || (inputs.size() == 4) || (inputs.size() == 5)),
      "Incorrect size of inputs expected for Bmm operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");

  auto self = inputs[0].toTensor();
  auto mat2 = inputs[1].toTensor();
  // inputs[2] and inputs[3] are optional mat1 and mat2 transpose flags
  bool mat1_transposed = false;
  bool mat2_transposed = false;
  if (inputs.size() == 4) {
    mat1_transposed = inputs[2].toBool();
    mat2_transposed = inputs[3].toBool();
  } else if (inputs.size() == 5) {
    mat1_transposed = inputs[3].toBool();
    mat2_transposed = inputs[4].toBool();
  }

  auto shape_out = habana::BmmOperator::compute_output_shape(
      self, mat2, mat1_transposed, mat2_transposed);

  auto output = habana::createPTTensor(
      self,
      shape_out,
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).dtype,
      output_metadata.at(0).persistent);
  inputs.insert(inputs.begin(), IValue(output));

  habana::BmmOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, output_metadata);
}

/*****************************************************************************************************
*@brief Implements torch.dot(vector,vector)
self - 1D m
other - 1D m
output - 0-D tensor
*****************************************************************************************************/
void habana::DotOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  auto mat1 = inputs[0].toTensor(); // size of mat1 is m
  auto mat2 = inputs[1].toTensor(); // size of mat2 is m

  std::vector<c10::IValue> stack;
  // ReShape Operator to covert 1d tensor to 2d for mat1
  int64_t data_m1[2];
  data_m1[0] = 1;
  data_m1[1] = mat1.numel();
  c10::IntArrayRef shape_m1(data_m1, 2);
  auto ReShapeOp_m1 = make_operator<ReshapeOperator>(
      this->p_context_->device_id_, mat1.scalar_type());
  ReShapeOp_m1->SetSynapseInput(p_context_->syn_inputs_[0]);
  // Build Params for the graph
  stack.emplace_back(IValue(mat1));
  stack.emplace_back(IValue(shape_m1));
  ReShapeOp_m1->AllocateAndAddSynapseNode(
      graph, stack, habana::OutputMetaDataVector(1));
  stack.clear();

  // ReShape Operator to covert 1d tensor to 2d for mat2
  int64_t data_m2[2];
  data_m2[0] = mat2.numel();
  data_m2[1] = 1;
  c10::IntArrayRef shape_m2(data_m2, 2);
  // Create the operator
  auto ReShapeOp_m2 = make_operator<ReshapeOperator>(
      this->p_context_->device_id_, mat2.scalar_type());
  ReShapeOp_m2->SetSynapseInput(p_context_->syn_inputs_[1]);
  // Build Params for the graph
  stack.emplace_back(IValue(mat2));
  stack.emplace_back(IValue(shape_m2));
  ReShapeOp_m2->AllocateAndAddSynapseNode(
      graph, stack, habana::OutputMetaDataVector(1));
  stack.clear();

  // Matmul Operator (1xn) * (nx1) = (1x1)
  auto mmOp = make_operator<MMOperator>(this->p_context_->device_id_);
  mmOp->SetSynapseInput(ReShapeOp_m1->GetSynOutputs()[0]);
  mmOp->SetSynapseInput(ReShapeOp_m2->GetSynOutputs()[0]);
  // Build Params for the graph
  stack.emplace_back(IValue(ReShapeOp_m1->GetOutputs()[0]));
  stack.emplace_back(IValue(ReShapeOp_m2->GetOutputs()[0]));
  OutputMetaData mm_output_metadata{};
  mm_output_metadata.dtype = output_metadata.at(0).dtype;
  mmOp->AllocateAndAddSynapseNode(graph, stack, {mm_output_metadata});
  stack.clear();

  // ReShape Operator to covert 2d tensor to 1d for output
  int64_t data[1];
  data[0] = mmOp->GetOutputs()[0].numel();
  c10::IntArrayRef shape(data, 1);
  auto ReShapeOp_out = make_operator<ReshapeOperator>(
      this->p_context_->device_id_, mmOp->GetOutputs()[0].scalar_type());
  ReShapeOp_out->SetSynapseInput(mmOp->GetSynOutputs()[0]);
  // Build Params for the graph
  stack.emplace_back(IValue(mmOp->GetOutputs()[0]));
  stack.emplace_back(IValue(shape));
  ReShapeOp_out->AllocateAndAddSynapseNode(graph, stack, output_metadata);

  p_context_->syn_outputs_.emplace_back(
      std::move(ReShapeOp_out->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(
      std::move(ReShapeOp_out->GetOutputs()[0]));
}

/*****************************************************************************************************
*@brief Implements torch.mv(tensor,vector)
self - 2D nxm
other - 1D m
output - 1D n
*****************************************************************************************************/
void habana::MvOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  auto mat1 = inputs[0].toTensor(); // mxn
  auto mat2 = inputs[1].toTensor(); // size of mat2 is n

  // ReShape Operator to covert n to nx1 for mat2
  int64_t data[2];
  data[0] = mat2.numel();
  data[1] = 1;
  c10::IntArrayRef shape(data, 2);
  // Create the operator
  auto ReShapeOp = make_operator<ReshapeOperator>(
      this->p_context_->device_id_, mat2.scalar_type());
  ReShapeOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  // Build Params for the graph
  std::vector<c10::IValue> stack{IValue(mat2), IValue(shape)};
  ReShapeOp->AllocateAndAddSynapseNode(
      graph, stack, habana::OutputMetaDataVector(1));
  stack.clear();

  // Matmul Operator (mxn) * (nx1) = (mx1)
  auto mmOp = make_operator<MMOperator>(this->p_context_->device_id_);
  mmOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  mmOp->SetSynapseInput(ReShapeOp->GetSynOutputs()[0]);
  // Build Params for the graph
  stack.emplace_back(IValue(mat1));
  stack.emplace_back(IValue(ReShapeOp->GetOutputs()[0]));
  OutputMetaData mm_output_metadata{};
  mm_output_metadata.dtype = output_metadata.at(0).dtype;
  mmOp->AllocateAndAddSynapseNode(graph, stack, {mm_output_metadata});
  stack.clear();

  // PT expects 1-D
  // ReShape Operator to covert mx1 to 1xm for output of Matmul Operator
  int64_t data2[1];
  data2[0] = mmOp->GetOutputs()[0].numel();
  c10::IntArrayRef shape2(data2, 1);
  auto ReShapeOp_2 = make_operator<ReshapeOperator>(
      this->p_context_->device_id_, mmOp->GetOutputs()[0].scalar_type());
  ReShapeOp_2->SetSynapseInput(mmOp->GetSynOutputs()[0]);
  // Build Params for the graph
  stack.emplace_back(IValue(mmOp->GetOutputs()[0]));
  stack.emplace_back(IValue(shape2));
  ReShapeOp_2->AllocateAndAddSynapseNode(graph, stack, output_metadata);

  p_context_->syn_outputs_.emplace_back(
      std::move(ReShapeOp_2->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(ReShapeOp_2->GetOutputs()[0]));
}

std::vector<int64_t> habana::MatMulOperator::compute_output_shape(
    const Tensor& self,
    const Tensor& other,
    bool other_transposed) {
  /*
  The output shape for depends on the dimensionality of the input Tensors as
  follows:
  - 1D x 1D:
    dot product (scalar) is returned.
    (a) x (a)
    => ()

  - 2D x 2D:
    matrix-matrix product is returned.
    (a, b) x (b, c)
    => (a, c)

  - 1D x 2D:
    1 is prepended to dimension of 1st Tensor, then mm() is performed.
    Then 1 is removed from 1st dimension after multiplication is done.
    (a) x (a, b)
    => (1, a) x (a, b)
    => (1, b)
    => (b)

  - 2D x 1D:
    the matrix-vector product is returned.
    (a, b) x (b)
    => (a)

  - (MD x ND) || (ND x MD) :
    M > 1 and N > 2
    In this case bmm() is performed
    Check habana::BmmOperator::compute_output_shape() for cases like:
    5D x 5D
    4D x 3D
    2D x 4D

  - (1D x ND) || (ND x 1D) : N > 2
    1D x ND
      1 is prepended to dimension of first tensor
      1D x ND => 2D x ND
      Now its a case of above one. bmm() is performed.
      Finally 1 removed.
    ND x 1D
      1 is appended to dimension of 2nd tensor
      bmm() performed
      1 removed

    The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be
  broadcastable). e.g., (j, 1, n, m) x (k, m, p) => (j, k, n, p)
  */
  return habana::getBatchMatmulOutShape(
      self.sizes(), other.sizes(), false, other_transposed);
}

bool habana::MatMulOperator::is_gmemm_with_transpose_possible(
    const Tensor& t1,
    const Tensor& t2) {
  auto dim_tensor1 = t1.dim();
  auto dim_tensor2 = t2.dim();

  if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) &&
      (dim_tensor1 >= 3 || dim_tensor2 >= 3))
    return true;
  else
    return false;
}

void habana::MatMulOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      ((inputs.size() == 2) || (inputs.size() == 3) || (inputs.size() == 4) ||
       (inputs.size() == 5)),
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  bool reshape_3d_2d = habana_helpers::IsMatmul3d2dReshapeEnabled();
  auto tensor1 = inputs[0].toTensor();
  auto tensor2 = inputs[1].toTensor();
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();
  bool bias1d_present_for_bmm =
      (((inputs.size() == 3) || (inputs.size() == 5)) &&
       (inputs[2].toTensor().dim() == 1))
      ? true
      : false;
  bool mat1_transposed = false;
  bool mat2_transposed = false;
  if (inputs.size() == 4) {
    mat1_transposed = inputs[2].toBool();
    mat2_transposed = inputs[3].toBool();
  } else if (inputs.size() == 5) {
    mat1_transposed = inputs[3].toBool();
    mat2_transposed = inputs[4].toBool();
  }

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    auto dot_op = make_operator<habana::DotOperator>(tensor1.device().index());
    dot_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    dot_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack stack = {IValue(tensor1), IValue(tensor2)};
    dot_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_.emplace_back(
        std::move(dot_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(dot_op->GetOutputs()[0]));
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    auto mv_op = make_operator<habana::MvOperator>(tensor1.device().index());
    mv_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    mv_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack stack = {IValue(tensor1), IValue(tensor2)};
    mv_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_.emplace_back(std::move(mv_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(mv_op->GetOutputs()[0]));
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    Tensor t1 = tensor1;
    // tensor1.unsqueeze(-1)
    auto reshape_ten1 = make_operator<ReshapeOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    std::vector<int64_t> shape_in{tensor1.sizes().vec()};
    shape_in.insert(shape_in.cbegin(), 1);
    reshape_ten1->SetSynapseInput(p_context_->syn_inputs_[0]);
    torch::jit::Stack stack = {c10::IValue(tensor1), c10::IValue(shape_in)};
    reshape_ten1->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    t1 = reshape_ten1->GetOutputs()[0];
    stack.clear();

    auto mm_op = make_operator<habana::MMOperator>(tensor1.device().index());
    mm_op->SetSynapseInput(reshape_ten1->GetSynOutputs()[0]);
    mm_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    stack = {IValue(t1), IValue(tensor2)};
    OutputMetaData mm_output_metadata{};
    mm_output_metadata.dtype = output_metadata.at(0).dtype;
    mm_op->AllocateAndAddSynapseNode(graph, stack, {mm_output_metadata});
    stack.clear();

    // reshape the output
    auto output = mm_op->GetOutputs()[0];
    std::vector<int64_t> shape_out{output.sizes().vec()};
    shape_out.erase(shape_out.begin());
    auto reshape_out = make_operator<ReshapeOperator>(
        tensor2.device().index(), tensor2.scalar_type());
    reshape_out->SetSynapseInput(mm_op->GetSynOutputs()[0]);
    stack = {c10::IValue(output), c10::IValue(shape_out)};
    reshape_out->AllocateAndAddSynapseNode(graph, stack, output_metadata);

    p_context_->syn_outputs_.emplace_back(
        std::move(reshape_out->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(reshape_out->GetOutputs()[0]));
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    auto mm_op = make_operator<habana::MMOperator>(tensor1.device().index());
    mm_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    mm_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack stack = {IValue(tensor1), IValue(tensor2)};
    mm_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_.emplace_back(std::move(mm_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(mm_op->GetOutputs()[0]));
  } else if (dim_tensor1 >= 3 && dim_tensor2 == 1) {
    Tensor t2 = tensor2;
    // tensor2.unsqueeze(-1)
    auto reshape_ten2 = make_operator<ReshapeOperator>(
        tensor2.device().index(), tensor2.scalar_type());
    std::vector<int64_t> shape_in{tensor2.sizes().vec()};
    shape_in.push_back(1);
    reshape_ten2->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack stack = {c10::IValue(tensor2), c10::IValue(shape_in)};
    reshape_ten2->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    t2 = reshape_ten2->GetOutputs()[0];
    stack.clear();

    auto bmm_op = make_operator<BmmOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    bmm_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    bmm_op->SetSynapseInput(reshape_ten2->GetSynOutputs()[0]);
    stack = {IValue(tensor1), IValue(t2)};
    OutputMetaData bmm_output_metadata{};
    bmm_output_metadata.dtype = output_metadata.at(0).dtype;
    bmm_op->AllocateAndAddSynapseNode(graph, stack, {bmm_output_metadata});
    stack.clear();

    // reshape the output
    auto output = bmm_op->GetOutputs()[0];
    std::vector<int64_t> shape_out{output.sizes().vec()};
    shape_out.pop_back();
    auto reshape_out = make_operator<ReshapeOperator>(
        tensor2.device().index(), tensor2.scalar_type());
    reshape_out->SetSynapseInput(bmm_op->GetSynOutputs()[0]);
    stack = {c10::IValue(output), c10::IValue(shape_out)};
    reshape_out->AllocateAndAddSynapseNode(graph, stack, output_metadata);

    p_context_->syn_outputs_.emplace_back(
        std::move(reshape_out->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(reshape_out->GetOutputs()[0]));
  } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
    // for 2x3 case: swap inner dimensions of both inputs, call BMM on
    // swapped arguments and then swap inner dimensions of output
    // for 1x3 case: unsqueeze(-1) on t1 and swap inner dimensions of t2,
    // call BMM on swapped arguments and then squeeze(-1) on output
    auto reshape_in = make_operator<ReshapeOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    auto t1_op = make_operator<TransposeOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    if (dim_tensor1 == 1) {
      std::vector<int64_t> shape_in{tensor1.sizes().vec()};
      shape_in.push_back(1);
      reshape_in->SetSynapseInput(p_context_->syn_inputs_[0]);
      torch::jit::Stack stack = {c10::IValue(tensor1), c10::IValue(shape_in)};
      reshape_in->AllocateAndAddSynapseNode(
          graph, stack, habana::OutputMetaDataVector(1));
      stack.clear();
    } else {
      torch::jit::Stack stack = {IValue(tensor1), IValue(-1), IValue(-2)};
      t1_op->SetSynapseInput(p_context_->syn_inputs_[0]);
      t1_op->AllocateAndAddSynapseNode(
          graph, stack, habana::OutputMetaDataVector(1));
      stack.clear();
    }
    auto bmm_op = make_operator<BmmOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    bmm_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    bmm_op->SetSynapseInput(
        (dim_tensor1 == 1) ? reshape_in->GetSynOutputs()[0]
                           : t1_op->GetSynOutputs()[0]);
    torch::jit::Stack stack = {
        IValue(tensor2),
        IValue(
            (dim_tensor1 == 1) ? reshape_in->GetOutputs()[0]
                               : t1_op->GetOutputs()[0]),
        IValue(true) /*mat1_transpose*/,
        IValue(false) /*mat2_transpose*/};
    if (bias1d_present_for_bmm) {
      bmm_op->SetSynapseInput(p_context_->syn_inputs_[2]);
      stack.insert(stack.cbegin() + 2, IValue(inputs[2].toTensor()));
    }
    OutputMetaData bmm_output_metadata{};
    bmm_output_metadata.dtype = output_metadata.at(0).dtype;
    bmm_op->AllocateAndAddSynapseNode(graph, stack, {bmm_output_metadata});
    stack.clear();

    auto reshape_out = make_operator<ReshapeOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    auto tout_op = make_operator<TransposeOperator>(
        tensor2.device().index(), tensor2.scalar_type());
    if (dim_tensor1 == 1) {
      std::vector<int64_t> shape_out{bmm_op->GetOutputs()[0].sizes().vec()};
      shape_out.pop_back();
      reshape_out->SetSynapseInput(bmm_op->GetSynOutputs()[0]);
      torch::jit::Stack stack = {
          c10::IValue(bmm_op->GetOutputs()[0]), c10::IValue(shape_out)};
      reshape_out->AllocateAndAddSynapseNode(graph, stack, output_metadata);
      stack.clear();
    } else {
      torch::jit::Stack stack = {
          IValue(bmm_op->GetOutputs()[0]), IValue(-1), IValue(-2)};
      tout_op->SetSynapseInput(bmm_op->GetSynOutputs()[0]);
      tout_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
      stack.clear();
    }
    p_context_->syn_outputs_.emplace_back(std::move(
        (dim_tensor1 == 1) ? reshape_out->GetSynOutputs()[0]
                           : tout_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(
        (dim_tensor1 == 1) ? reshape_out->GetOutputs()[0]
                           : tout_op->GetOutputs()[0]));
  } else if (
      (dim_tensor1 == 4 && dim_tensor2 == 3) ||
      (dim_tensor1 == 3 && dim_tensor2 == 4)) {
    // Broadcast along batch to make dimensions consistent before calling
    // BMM, since GC supports BMM for 4D tensors only when dimensions are
    // consistent
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    IntArrayRef batch_tensor1(
        tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
    int64_t p = tensor2.size(-1);
    IntArrayRef batch_tensor2(
        tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion =
        at::infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

    auto bcastOpTens1 = make_operator<BroadcastOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    torch::jit::Stack stack = {
        IValue(tensor1), IValue(tensor1_expand_size), IValue(false)};
    bcastOpTens1->SetSynapseInput(p_context_->syn_inputs_[0]);
    bcastOpTens1->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    stack.clear();
    // expand tensor2
    auto bcastOpTens2 = make_operator<BroadcastOperator>(
        tensor2.device().index(), tensor2.scalar_type());
    stack = {IValue(tensor2), IValue(tensor2_expand_size), IValue(false)};
    bcastOpTens2->SetSynapseInput(p_context_->syn_inputs_[1]);
    bcastOpTens2->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    stack.clear();

    auto bmm_op = make_operator<BmmOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    bmm_op->SetSynapseInput(bcastOpTens1->GetSynOutputs()[0]);
    bmm_op->SetSynapseInput(bcastOpTens2->GetSynOutputs()[0]);
    stack = {
        IValue(bcastOpTens1->GetOutputs()[0]),
        IValue(bcastOpTens2->GetOutputs()[0])};
    bmm_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);

    p_context_->syn_outputs_.emplace_back(
        std::move(bmm_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(bmm_op->GetOutputs()[0]));

  } else if (reshape_3d_2d && (dim_tensor1 == 3) && (dim_tensor2 == 2)) {
    auto mat1_sizes = tensor1.sizes().vec();
    auto mat2_sizes = tensor2.sizes().vec();
    std::vector<int64_t> shape_in{mat1_sizes[0] * mat1_sizes[1], mat1_sizes[2]};

    auto reshape_ten1 = make_operator<ReshapeOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    reshape_ten1->SetSynapseInput(p_context_->syn_inputs_[0]);
    torch::jit::Stack stack = {c10::IValue(tensor1), c10::IValue(shape_in)};
    reshape_ten1->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    auto t1 = reshape_ten1->GetOutputs()[0];
    stack.clear();

    auto mm_op = make_operator<habana::MMOperator>(tensor1.device().index());
    mm_op->SetSynapseInput(reshape_ten1->GetSynOutputs()[0]);
    mm_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    /*stack = {
        IValue(t1),
        IValue(tensor2),
        IValue(mat1_transposed),
        IValue(mat2_transposed)};*/
    stack = {IValue(t1), IValue(tensor2), mat1_transposed, mat2_transposed};
    if (bias1d_present_for_bmm) {
      mm_op->SetSynapseInput(p_context_->syn_inputs_[2]);
      stack.emplace_back(IValue(inputs[2].toTensor()));
    }
    OutputMetaData mm_output_metadata{};
    mm_output_metadata.dtype = output_metadata.at(0).dtype;
    mm_op->AllocateAndAddSynapseNode(graph, stack, {mm_output_metadata});
    /*mm_op->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));*/
    stack.clear();

    // reshape the output
    auto output = mm_op->GetOutputs()[0];
    int64_t d2 = mat2_transposed ? mat2_sizes[0] : mat2_sizes[1];
    std::vector<int64_t> shape_out{mat1_sizes[0], mat1_sizes[1], d2};
    auto reshape_out = make_operator<ReshapeOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    reshape_out->SetSynapseInput(mm_op->GetSynOutputs()[0]);
    stack = {c10::IValue(output), c10::IValue(shape_out)};
    reshape_out->AllocateAndAddSynapseNode(graph, stack, output_metadata);

    p_context_->syn_outputs_.emplace_back(
        std::move(reshape_out->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(reshape_out->GetOutputs()[0]));

  } else if (
      (dim_tensor1 >= 1 && dim_tensor2 >= 1) &&
      (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // 4x4; 3x3; 3x2 cases handled here. GC does not support other cases
    // directly, therefore they are handled as special cases above
    auto bmm_op = make_operator<BmmOperator>(
        tensor1.device().index(), tensor1.scalar_type());
    bmm_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    bmm_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack stack = {
        IValue(tensor1),
        IValue(tensor2),
        IValue(mat1_transposed),
        IValue(mat2_transposed)};
    if (bias1d_present_for_bmm) {
      bmm_op->SetSynapseInput(p_context_->syn_inputs_[2]);
      stack.insert(stack.cbegin() + 2, IValue(inputs[2].toTensor()));
    }
    bmm_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);

    p_context_->syn_outputs_.emplace_back(
        std::move(bmm_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(bmm_op->GetOutputs()[0]));
  } else {
    PT_KERNEL_FATAL(
        "matmul with following dimension is not supported ",
        dim_tensor1,
        "D and ",
        dim_tensor2,
        "D");
  }
}

void habana::MatmulBackwardOperator::MatBwTranspose(
    synapse_helpers::graph& graph,
    HabanaOperatorPtr Op,
    Tensor& mat,
    synapse_helpers::tensor& syn_input) {
  auto dim = mat.dim();
  if (dim == 1) {
    // Add a identity node to graph (to get out = mat)
    Op->SetSynapseInput(syn_input);
    torch::jit::Stack stack = {IValue(mat)};
    Op->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
  } else if (dim == 2) {
    // Add a transpose node to graph
    Op->SetSynapseInput(syn_input);
    torch::jit::Stack stack = {IValue(mat), IValue(0), IValue(1)};
    Op->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
  } else {
    // Add a permute node to graph (since permute is only on last
    // 2 dims we use transpose node to simplify code). Keep the
    // commented code below, just in case we need to go back to
    // permute again for some reason.

    // std::vector<int64_t> dims_v(dim);
    // std::iota(std::begin(dims_v), std::end(dims_v), 0);
    // dims_v[dims_v.size() - 1] = dim - 2;
    // dims_v[dims_v.size() - 2] = dim - 1;
    // IntArrayRef dims(dims_v.data(), dims_v.size());
    Op->SetSynapseInput(syn_input);
    torch::jit::Stack stack = {IValue(mat), IValue(dim - 2), IValue(dim - 1)};
    Op->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
  }
}

void habana::MatmulBackwardOperator::MatBwReshape(
    synapse_helpers::graph& graph,
    Tensor& mat,
    std::vector<int64_t> sizes,
    synapse_helpers::tensor& syn_input) {
  auto reshape =
      make_operator<ReshapeOperator>(mat.device().index(), mat.scalar_type());
  reshape->SetSynapseInput(syn_input);
  torch::jit::Stack stack = {IValue(mat), IValue(sizes)};
  reshape->AllocateAndAddSynapseNode(
      graph, stack, habana::OutputMetaDataVector(1));
  ReshapeOpList.push_back(reshape);
}

void habana::MatmulBackwardOperator::MatBwSpecialFold(
    synapse_helpers::graph& graph,
    HabanaOperatorPtr Op,
    Tensor& mat1,
    Tensor& mat2,
    synapse_helpers::tensor& syn_input1,
    synapse_helpers::tensor& syn_input2,
    const OutputMetaData& output_metadata,
    bool avoid_mat1_transpose) {
  /*
    # In matmul backward case of [b, m, n] * [b, n, p] => [m, p],
    # instead of doing [b, m, p] and then reduce to [m, p]
    # whice potentially uses large intermediate of size b*m*p,
    # we do [m, bn] * [bn, p] to avoid having the large
    # intermediate, thus reduces max memory usage.
  */
  auto transpose1 = make_operator<TransposeOperator>(
      mat1.device().index(), mat1.scalar_type());
  Tensor transpose_out;
  if (!avoid_mat1_transpose) {
    MatBwTranspose(graph, transpose1, mat1, syn_input1);
    transpose_out = transpose1->GetOutputs()[0];
  }

  auto ReshapeListSizeIn = ReshapeOpList.size();
  int64_t reshape_size = (!avoid_mat1_transpose)
      ? transpose_out.size(transpose_out.dim() - 1)
      : mat1.size(mat1.dim() - 1);
  std::vector<int64_t> reshape_mat1_sizes{-1, reshape_size};
  if (avoid_mat1_transpose) {
    MatBwReshape(graph, mat1, reshape_mat1_sizes, syn_input1);
  } else {
    synapse_helpers::tensor& tsyn = transpose1->GetSynOutputs()[0];
    MatBwReshape(graph, transpose1->GetOutputs()[0], reshape_mat1_sizes, tsyn);
  }
  std::vector<int64_t> reshape_mat2_sizes{-1, mat2.size(mat2.dim() - 1)};
  MatBwReshape(graph, mat2, reshape_mat2_sizes, syn_input2);

  Op->SetSynapseInput(ReshapeOpList.at(ReshapeListSizeIn)->GetSynOutputs()[0]);
  Op->SetSynapseInput(
      ReshapeOpList.at(ReshapeListSizeIn + 1)->GetSynOutputs()[0]);
  torch::jit::Stack stack = {
      IValue(ReshapeOpList.at(ReshapeListSizeIn)->GetOutputs()[0]),
      IValue(ReshapeOpList.at(ReshapeListSizeIn + 1)->GetOutputs()[0]),
      IValue(true) /*mat1_transpose*/,
      IValue(false) /*mat2_transpose*/};
  OutputMetaData mm_output_metadata{};
  mm_output_metadata.dtype = output_metadata.dtype;
  Op->AllocateAndAddSynapseNode(graph, stack, {mm_output_metadata});
  stack.clear();

  ReshapeOpList.pop_back();
  ReshapeOpList.pop_back();
}

void habana::MatmulBackwardOperator::MatBwSize(
    synapse_helpers::graph& graph,
    HabanaOperatorPtr Op,
    Tensor& mat1,
    Tensor& mat2,
    IntArrayRef sizes,
    synapse_helpers::tensor& syn_input1,
    synapse_helpers::tensor& syn_input2,
    const OutputMetaData& output_metadata,
    bool avoid_mat1_transpose) {
  auto dim_out = sizes.size();
  auto dim1 = mat1.dim();
  auto dim2 = mat2.dim();

  if (is_specialfold_without_reshape_case(dim1, dim2, dim_out)) {
    /* out = AD_matmul_bw_special_fold(mat1, mat2) */
    auto mm = make_operator<habana::MMOperator>(mat1.device().index());
    MatBwSpecialFold(
        graph,
        mm,
        mat1,
        mat2,
        syn_input1,
        syn_input2,
        output_metadata,
        avoid_mat1_transpose);
    Op->SetSynapseInput(mm->GetSynOutputs()[0]);
    torch::jit::Stack stack = {IValue(mm->GetOutputs()[0]), IValue(sizes)};
    Op->AllocateAndAddSynapseNode(graph, stack, {output_metadata});
  } else if ((dim1 + dim2) == static_cast<int64_t>(dim_out)) {
    /*
    if dim2 == 1:
        target_dim2 = 0
    else:
        target_dim2 = -2
    out = torch.matmul(mat1.unsqueeze(dim1), mat2.unsqueeze(target_dim2))
    */
    std::vector<int64_t> reshape1_sizes{mat1.sizes().vec()};
    reshape1_sizes.push_back(1);
    MatBwReshape(graph, mat1, reshape1_sizes, syn_input1);

    std::vector<int64_t> reshape2_sizes{mat2.sizes().vec()};
    reshape2_sizes.insert(reshape2_sizes.cbegin(), 1);
    MatBwReshape(graph, mat2, reshape2_sizes, syn_input2);

    auto matmul = make_operator<habana::MatMulOperator>(mat1.device().index());
    matmul->SetSynapseInput(ReshapeOpList.at(0)->GetSynOutputs()[0]);
    matmul->SetSynapseInput(ReshapeOpList.at(1)->GetSynOutputs()[0]);
    torch::jit::Stack stack = {
        IValue(ReshapeOpList.at(0)->GetOutputs()[0]),
        IValue(ReshapeOpList.at(1)->GetOutputs()[0])};
    OutputMetaData mm_output_metadata{};
    mm_output_metadata.dtype = output_metadata.dtype;
    matmul->AllocateAndAddSynapseNode(graph, stack, {mm_output_metadata});
    stack.clear();

    ReshapeOpList.clear();

    Op->SetSynapseInput(matmul->GetSynOutputs()[0]);
    stack = {IValue(matmul->GetOutputs()[0]), IValue(sizes)};
    Op->AllocateAndAddSynapseNode(graph, stack, {output_metadata});
  } else if ((dim_out == 1) && (dim1 - dim2) == 1 && (dim1 >= 3)) {
    /*
    elif dim_out == 1 and dim1 - dim2 == 1 and dim1 >= 3:
        mat2_unsqueeze = mat2.unsqueeze(-1)
        out = AD_matmul_bw_special_fold(mat1, mat2_unsqueeze)
        out = out.squeeze(-1)
    */
    std::vector<int64_t> reshape2_sizes{mat2.sizes().vec()};
    reshape2_sizes.push_back(1);
    MatBwReshape(graph, mat2, reshape2_sizes, syn_input2);

    // Using MM directly since MatMul is anyway going to call MM for a 2x2 case
    auto matmul = make_operator<habana::MMOperator>(mat1.device().index());
    MatBwSpecialFold(
        graph,
        matmul,
        mat1,
        ReshapeOpList.at(0)->GetOutputs()[0],
        syn_input1,
        ReshapeOpList.at(0)->GetSynOutputs()[0],
        output_metadata);

    std::vector<int64_t> reshape1_sizes{matmul->GetOutputs()[0].sizes().vec()};
    reshape1_sizes.pop_back();
    MatBwReshape(
        graph,
        matmul->GetOutputs()[0],
        reshape1_sizes,
        matmul->GetSynOutputs()[0]);

    Op->SetSynapseInput(ReshapeOpList.at(1)->GetSynOutputs()[0]);
    torch::jit::Stack stack = {
        IValue(ReshapeOpList.at(1)->GetOutputs()[0]), IValue(sizes)};
    Op->AllocateAndAddSynapseNode(graph, stack, {output_metadata});

    ReshapeOpList.clear();
  } else if (static_cast<int64_t>(dim_out) == (dim1 - dim2)) {
    /* out = torch.matmul(mat1, mat2.unsqueeze(dim2)).squeeze(-1) */
    std::vector<int64_t> reshape2_sizes{mat2.sizes().vec()};
    reshape2_sizes.push_back(dim2);
    MatBwReshape(graph, mat2, reshape2_sizes, syn_input2);

    auto matmul = make_operator<habana::MatMulOperator>(mat1.device().index());
    matmul->SetSynapseInput(syn_input1);
    matmul->SetSynapseInput(ReshapeOpList.at(0)->GetSynOutputs()[0]);
    torch::jit::Stack stack = {
        IValue(mat1), IValue(ReshapeOpList.at(0)->GetOutputs()[0])};
    OutputMetaData mm_output_metadata{};
    mm_output_metadata.dtype = output_metadata.dtype;
    matmul->AllocateAndAddSynapseNode(graph, stack, {mm_output_metadata});
    stack.clear();

    std::vector<int64_t> reshape1_sizes{matmul->GetOutputs()[0].sizes().vec()};
    reshape1_sizes.pop_back();
    MatBwReshape(
        graph,
        matmul->GetOutputs()[0],
        reshape1_sizes,
        matmul->GetSynOutputs()[0]);

    Op->SetSynapseInput(ReshapeOpList.at(1)->GetSynOutputs()[0]);
    stack = {IValue(ReshapeOpList.at(1)->GetOutputs()[0]), IValue(sizes)};
    Op->AllocateAndAddSynapseNode(graph, stack, {output_metadata});

    ReshapeOpList.clear();
  } else if (static_cast<int64_t>(dim_out) == (dim2 - dim1)) {
    /* out = torch.matmul(mat1.unsqueeze(-2), mat2).squeeze(-2) */
    std::vector<int64_t> reshape1_sizes{mat1.sizes().vec()};
    reshape1_sizes.insert(reshape1_sizes.cbegin(), 1);
    MatBwReshape(graph, mat1, reshape1_sizes, syn_input1);

    auto matmul = make_operator<habana::MatMulOperator>(mat1.device().index());
    matmul->SetSynapseInput(ReshapeOpList.at(0)->GetSynOutputs()[0]);
    matmul->SetSynapseInput(syn_input2);
    torch::jit::Stack stack = {
        IValue(ReshapeOpList.at(0)->GetOutputs()[0]), IValue(mat2)};
    OutputMetaData mm_output_metadata{};
    mm_output_metadata.dtype = output_metadata.dtype;
    matmul->AllocateAndAddSynapseNode(graph, stack, {mm_output_metadata});
    stack.clear();

    std::vector<int64_t> reshape2_sizes{matmul->GetOutputs()[0].sizes().vec()};
    reshape2_sizes.erase(reshape2_sizes.begin());
    MatBwReshape(
        graph,
        matmul->GetOutputs()[0],
        reshape2_sizes,
        matmul->GetSynOutputs()[0]);

    ReshapeOpList.clear();

    Op->SetSynapseInput(matmul->GetSynOutputs()[0]);
    stack = {IValue(matmul->GetOutputs()[0]), IValue(sizes)};
    Op->AllocateAndAddSynapseNode(graph, stack, {output_metadata});
  } else {
    /* out = torch.matmul(mat1, mat2) */
    auto matmul = make_operator<habana::MatMulOperator>(mat1.device().index());
    matmul->SetSynapseInput(syn_input1);
    matmul->SetSynapseInput(syn_input2);
    torch::jit::Stack stack = {IValue(mat1), IValue(mat2)};
    OutputMetaData mm_output_metadata{};
    mm_output_metadata.dtype = output_metadata.dtype;
    matmul->AllocateAndAddSynapseNode(graph, stack, {mm_output_metadata});
    stack.clear();

    Op->SetSynapseInput(matmul->GetSynOutputs()[0]);
    stack = {IValue(matmul->GetOutputs()[0]), IValue(sizes)};
    Op->AllocateAndAddSynapseNode(graph, stack, {output_metadata});
  }
}

void habana::MatmulBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      ((inputs.size() == 3) || (inputs.size() == 4)),
      "Incorrect size of inputs expected for matmul backward operator");

  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input1 type expected to be tensor for matmul backward operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input2 type expected to be tensor for matmul backward operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input3 type expected to be tensor for matmul backward operator");

  auto grad_out = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto other = inputs[2].toTensor();

  auto matmul_bwd_op = make_operator<habana::MatMulBwdOperator>(
      self.device().index(), self.scalar_type());
  matmul_bwd_op->SetSynapseInput(p_context_->syn_inputs_[0]);
  matmul_bwd_op->SetSynapseInput(p_context_->syn_inputs_[1]);
  matmul_bwd_op->SetSynapseInput(p_context_->syn_inputs_[2]);
  torch::jit::Stack stack = {
      c10::IValue(grad_out), c10::IValue(self), c10::IValue(other)};
  if (inputs.size() == 4) {
    stack.push_back(inputs[3]);
  }
  matmul_bwd_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
  p_context_->syn_outputs_.emplace_back(
      std::move(matmul_bwd_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(
      std::move(matmul_bwd_op->GetOutputs()[0]));
  p_context_->syn_outputs_.emplace_back(
      std::move(matmul_bwd_op->GetSynOutputs()[1]));
  p_context_->pt_outputs_.emplace_back(
      std::move(matmul_bwd_op->GetOutputs()[1]));
}

void habana::MatMulBwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      ((inputs.size() == 3 || inputs.size() == 4)),
      "Incorrect size of inputs expected for linear backward operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input0 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input1 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input2 type expected to be tensor");
  auto grad_out = inputs[0].toTensor();
  auto input_a = inputs[1].toTensor();
  auto input_b = inputs[2].toTensor();
  ns_MatmulBwdKernel::Params params;
  params.skip_other_transpose = inputs.size() == 4 ? inputs[3].toBool() : false;

  std::vector<int64_t> shape_out = input_a.sizes().vec();
  // output1
  auto grad_a = habana::createPTTensor(
      input_a,
      shape_out,
      input_a.options(),
      input_a.suggest_memory_format(),
      output_metadata.at(0).dtype,
      output_metadata.at(0).persistent);
  // output 2
  std::vector<int64_t> shape_out2 = input_b.sizes().vec();
  auto grad_b = habana::createPTTensor(
      input_b,
      shape_out2,
      input_b.options(),
      input_b.suggest_memory_format(),
      output_metadata.at(1).dtype,
      output_metadata.at(1).persistent);

  AllocateSynapseOutput(graph, grad_a, output_metadata.at(0));
  AllocateSynapseOutput(graph, grad_b, output_metadata.at(1));

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

static auto& LinearKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::mm_t", KERNEL_FN_DROP_ARG2(MMOperator))
        .add(
            "aten::matmul_backward",
            KERNEL_FN_DROP_ARG2(MatmulBackwardOperator))
        .add(
            "hpu::matmul_backward",
            KERNEL_FN_DROP_ARG2(MatmulBackwardOperator))
        .add("aten::matmul", KERNEL_FN_DROP_ARG2(MatMulOperator));
