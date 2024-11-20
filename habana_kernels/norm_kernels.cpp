/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <ATen/ExpandUtils.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <memory>
#include <tuple>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/lowering_util.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "generated/backend/ne.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_inplace_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/repeat.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/unary_kernels.h"

using namespace torch;
using namespace habana;

void BatchNormInfOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& in_stack,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(in_stack[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(in_stack[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(in_stack[5].isBool(), "Input type expected to be bool");
  TORCH_CHECK(in_stack[6].isDouble(), "Input type expected to be double");
  TORCH_CHECK(in_stack[7].isDouble(), "Input type expected to be double");

  auto input = in_stack[0].toTensor();
  const auto momentum = in_stack[6].toDouble();
  const auto eps = in_stack[7].toDouble();

  auto output = habana::createPTTensor(input, output_metadata.at(0).persistent);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  ns_BatchNormKernel::Params params{};
  params.threshold.f = 0.0;
  params.momentum = static_cast<float>(momentum);
  params.epsilon = static_cast<float>(eps);
  p_context_->params_.emplace<ns_BatchNormKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

InferOutputMetaRetType BatchNormInfOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto input = inputs[0].toTensor();

  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      input.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input.sizes(), input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));
  return out;
}

std::vector<int64_t> NormOperator::compute_output_shape(
    const Tensor& self,
    at::IntArrayRef dim,
    bool keepdim) {
  if (dim.size() == 0)
    return {};
  auto sizes = self.sizes().vec();
  std::vector<int64_t> wrapped_dims;
  for (unsigned i = 0; i < dim.size(); i++)
    wrapped_dims.emplace_back(at::maybe_wrap_dim(dim[i], self.dim()));
  unsigned removed_count = 0;
  for (unsigned i = 0; i < wrapped_dims.size(); i++) {
    if (keepdim) {
      sizes[wrapped_dims[i]] = 1;
    } else {
      sizes.erase(sizes.cbegin() + wrapped_dims[i] - removed_count);
      removed_count++;
    }
  }
  return sizes;
}

void NormOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() >= 2 && inputs.size() <= 4,
      "Incorrect size of inputs expected for Norm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for Norm Operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be Scalar for Norm Operator");

  auto self = inputs[0].toTensor();
  auto shape = NormOperator::compute_output_shape(self, {}, 0);
  auto output = at::empty(shape, self.options(), c10::nullopt);
  HabanaOperator::SetPTOutput(output);
}

struct NotEqualScalar : NE {
  NotEqualScalar(int device_id, c10::ScalarType scalar_type)
      : NE(device_id, "None_", scalar_type, {0}, {}, {1}, false) {
    EnableTypePromotion();
  }
};

void NormOperator::AddL0NormNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaData& output_metadata) {
  auto self = inputs[0].toTensor();
  std::vector<int64_t> dims;
  bool keepdim;
  if (inputs.size() == 2) { // reduce along all dims
    for (unsigned i = 0; i < inputs[0].toTensor().sizes().size(); i++)
      dims.emplace_back(i);
    keepdim = false;
  } else { // reduce along given dims
    dims = inputs[2].toIntList().vec();
    keepdim = inputs[3].toBool();
  }
  auto device_id = self.device().index();
  auto scalar_type = self.scalar_type();
  torch::jit::Stack stack;
  std::shared_ptr<HabanaOperator> ne_op =
      make_operator<NotEqualScalar>(device_id, scalar_type);
  ne_op->SetSynapseInput(p_context_->syn_inputs_[0]);
  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(0.0));
  ne_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();
  std::string node_type = get_guid_with_precision("cast_i8_to", scalar_type);
  auto cast1 = make_operator<CastOperator>(device_id, node_type);
  cast1->SetSynapseInput(ne_op->GetSynOutputs()[0]);
  stack.emplace_back(IValue(ne_op->GetOutputs()[0]));
  stack.emplace_back(IValue(scalar_type));

  OutputMetaData md;
  md.dtype = scalar_type;
  cast1->AllocateAndAddSynapseNode(graph, stack, {md});
  stack.clear();
  // Reduction operation - Create the operator
  auto sum_dim_op =
      make_operator<SumDimOperator>(this->p_context_->device_id_, scalar_type);
  sum_dim_op->SetSynapseInput(cast1->GetSynOutputs()[0]);
  stack.emplace_back(IValue(cast1->GetOutputs()[0]));
  stack.emplace_back(IValue(dims));
  stack.emplace_back(IValue(keepdim));
  stack.emplace_back(IValue(scalar_type));
  sum_dim_op->AllocateAndAddSynapseNode(graph, stack, {output_metadata});
  stack.clear();
  p_context_->syn_outputs_.emplace_back(
      std::move(sum_dim_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(sum_dim_op->GetOutputs()[0]);
}

void NormOperator::AddLInfNormNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaData& output_metadata) {
  auto self = inputs[0].toTensor();
  auto p = inputs[1].toScalar();
  std::vector<int64_t> dims;
  bool keepdim;
  if (inputs.size() == 2) { // reduce along all dims
    for (unsigned i = 0; i < inputs[0].toTensor().sizes().size(); i++)
      dims.emplace_back(i);
    keepdim = false;
  } else { // reduce along given dims
    dims = inputs[2].toIntList().vec();
    keepdim = inputs[3].toBool();
  }
  auto device_id = self.device().index();
  auto scalar_type = self.scalar_type();
  torch::jit::Stack stack;
  auto abs_op = make_operator<AbsOperator>(device_id, scalar_type);
  abs_op->SetSynapseInput(p_context_->syn_inputs_[0]);
  stack.emplace_back(IValue(self));
  abs_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();
  // Reduction operation - Create the operator
  std::shared_ptr<ReduceOperator> reduce_op;
  if (p.toFloat() == LoweringUtil::FP_INFINITY) {
    reduce_op = make_operator<ReduceMultiOutputOperator>(
        this->p_context_->device_id_, scalar_type, "max");
  } else if (p.toFloat() == LoweringUtil::FP_NEG_INFINITY) {
    reduce_op = make_operator<ReduceMultiOutputOperator>(
        this->p_context_->device_id_, scalar_type, "min");
  } else {
    HABANA_ASSERT(0, "Call to AddLInfNormNode with invalid p value");
  }
  reduce_op->SetSynapseInput(abs_op->GetSynOutputs()[0]);
  stack.emplace_back(IValue(abs_op->GetOutputs()[0]));
  stack.emplace_back(IValue(dims));
  stack.emplace_back(IValue(keepdim));
  stack.emplace_back(IValue(scalar_type));
  reduce_op->AllocateAndAddSynapseNode(graph, stack, {output_metadata});
  stack.clear();
  p_context_->syn_outputs_.emplace_back(
      std::move(reduce_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(reduce_op->GetOutputs()[0]);
}

void NormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() >= 2 && inputs.size() <= 5,
      "Incorrect size of inputs expected for Norm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for Norm Operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be Scalar for Norm Operator");
  auto self = inputs[0].toTensor();
  auto p = inputs[1].toScalar();
  if (p.toFloat() == 0.0) {
    // L0 Norm
    AddL0NormNode(graph, inputs, output_metadata.at(0));
    return;
  } else if (
      p.toFloat() == LoweringUtil::FP_INFINITY ||
      p.toFloat() == LoweringUtil::FP_NEG_INFINITY) {
    // LInf and LNegInf Norms
    AddLInfNormNode(graph, inputs, output_metadata.at(0));
    return;
  }
  if ((p.toFloat() == 2.0) && (inputs.size() < 3)) {
    if (self.dim() <= 1 || self.sizes()[0] == 1) {
      auto device_id = self.device().index();
      auto scalar_type = self.scalar_type();
      // x^2 implemented as x*x. Identity node used to create aliased tensor
      // since GC/TPC does not like giving same tensor as both inputs to a
      // binary op
      auto identityOp = make_operator<IdentityOperator>(
          this->p_context_->device_id_, scalar_type);
      identityOp->SetSynapseInput(p_context_->syn_inputs_[0]);
      torch::jit::Stack stack = {IValue(self)};
      identityOp->AllocateAndAddSynapseNode(
          graph, stack, OutputMetaDataVector(1));
      stack.clear();

      auto mulOp = make_operator<habana::MulOperator>(
          this->p_context_->device_id_, scalar_type);
      mulOp->SetSynapseInput(p_context_->syn_inputs_[0]);
      mulOp->SetSynapseInput(identityOp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(self));
      stack.emplace_back(IValue(identityOp->GetOutputs()[0]));
      mulOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();
      // add node to compute reduce_sum
      auto sum_lp = make_operator<SumOperator>(device_id, scalar_type);
      sum_lp->SetSynapseInput(mulOp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(mulOp->GetOutputs()[0]));
      stack.emplace_back(IValue(scalar_type));
      sum_lp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

      auto sqrt_op = make_operator<SqrtOperator>(device_id, scalar_type);
      sqrt_op->SetSynapseInput(sum_lp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(sum_lp->GetOutputs()[0]));
      sqrt_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
      stack.clear();
      // synapse_helpers::tensor& sum_syn_tensor = sum_lp.GetSynOutputs()[0];
      p_context_->syn_outputs_.emplace_back(
          std::move(sqrt_op->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(sqrt_op->GetOutputs()[0]);
    } else {
      at::ScalarType scalar_type = self.scalar_type();
      std::vector<c10::IValue> stack{};
      // LpNorm Operator
      // Create the operator
      auto LpNormFrobeniusOp = make_operator<LpNormFrobeniusOperator>(
          this->p_context_->device_id_, scalar_type);
      LpNormFrobeniusOp->SetSynapseInput(p_context_->syn_inputs_[0]);

      // Build Params for the graph
      stack.emplace_back(IValue(self));
      LpNormFrobeniusOp->AllocateAndAddSynapseNode(
          graph, stack, output_metadata);

      stack.clear();
      p_context_->syn_outputs_.emplace_back(
          std::move(LpNormFrobeniusOp->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(LpNormFrobeniusOp->GetOutputs()[0]);
    }
  } else if (inputs.size() < 3) { // no dims given - reduce all dims
    // ReShape Operator
    at::ScalarType scalar_type = self.scalar_type();
    auto shape = {self.numel()};
    auto ReShapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, scalar_type);
    ReShapeOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    // Build Params for the graph
    std::vector<c10::IValue> stack{
        IValue(self), IValue(c10::IntArrayRef(shape))};
    ReShapeOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

    auto output_reshape = ReShapeOp->GetOutputs()[0];
    stack.clear();
    // LpNorm Operator
    // Create the operator
    auto LpNormOp = make_operator<LpNormOperator>(
        this->p_context_->device_id_, scalar_type);
    LpNormOp->SetSynapseInput(ReShapeOp->GetSynOutputs()[0]);
    // Build Params for the graph
    stack.emplace_back(IValue(output_reshape));
    stack.emplace_back(IValue(p));
    stack.emplace_back(IValue(0)); // dim
    LpNormOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_.emplace_back(
        std::move(LpNormOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(LpNormOp->GetOutputs()[0]));
  } else { // reduce along specified dim
    at::ScalarType scalar_type = self.scalar_type();
    std::vector<int64_t> dims = inputs[2].toIntList().vec();
    bool keepdim = inputs[3].toBool();
    std::vector<c10::IValue> stack;
    if (dims.size() == 1) {
      auto LpNormOp = make_operator<LpNormOperator>(
          this->p_context_->device_id_, scalar_type);
      LpNormOp->SetSynapseInput(p_context_->syn_inputs_[0]);
      stack = {IValue(self), IValue(p), IValue(dims[0])};
      LpNormOp->AllocateAndAddSynapseNode(
          graph, stack, keepdim ? output_metadata : OutputMetaDataVector(1));

      if (!keepdim) {
        auto new_sizes = LpNormOp->GetOutputs()[0].sizes().vec();
        new_sizes.erase(new_sizes.cbegin() + dims[0]);
        auto reshape_op = make_operator<ReshapeOperator>(
            this->p_context_->device_id_, scalar_type);
        reshape_op->SetSynapseInput(LpNormOp->GetSynOutputs()[0]);
        stack = {IValue(LpNormOp->GetOutputs()[0]), IValue(new_sizes)};
        reshape_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
        p_context_->syn_outputs_.emplace_back(
            std::move(reshape_op->GetSynOutputs()[0]));
        p_context_->pt_outputs_.emplace_back(
            std::move(reshape_op->GetOutputs()[0]));
      } else {
        p_context_->syn_outputs_.emplace_back(
            std::move(LpNormOp->GetSynOutputs()[0]));
        p_context_->pt_outputs_.emplace_back(
            std::move(LpNormOp->GetOutputs()[0]));
      }
    } else {
      HABANA_ASSERT(
          0, "NormOperator doesn't support more than single dim reduction");
    }

    return;
  }
}

void LpNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for LpNorm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for LpNorm Operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be Scalar for LpNorm Operator");
  auto self = inputs[0].toTensor();
  auto p = inputs[1].toScalar();
  auto dim = inputs[2].toInt();

  TORCH_CHECK(p.toFloat() > 0.0, "norm with p > 0.0 is only supported");
  auto lpnorm_output = habana::createPTTensor(self, false);
  auto retain = habana::createPTTensor(self, false);

  ns_LpNormKernel::Params params{};
  params.p = p.to<float>();
  params.dim = self.dim() - dim - 1;
  params.eps = 1e-5; // arbitrarily small value
  std::vector<at::Tensor> outputs{lpnorm_output, retain};
  AllocateSynapseOutputs(graph, outputs, OutputMetaDataVector(2));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
  std::vector<c10::IValue> stack;
  // Reciprocal Operator is used since what we require is reciprocal of retain
  auto reciprocalOp = make_operator<ReciprocalOperator>(
      this->p_context_->device_id_, self.scalar_type());
  reciprocalOp->SetSynapseInput(p_context_->syn_outputs_[1]);
  // Build Params for the graph
  stack.emplace_back(IValue(retain));
  reciprocalOp->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  stack.clear();
  // take just the first element of Reciprocal since all values would be
  // repeated
  auto slice_op = make_operator<SliceOperator>(
      this->p_context_->device_id_, self.scalar_type());
  slice_op->SetSynapseInput(reciprocalOp->GetSynOutputs()[0]);
  stack.emplace_back(IValue(reciprocalOp->GetOutputs()[0]));
  int start = 0;
  int end = 1;
  int step = 1;
  stack.emplace_back(IValue(dim));
  stack.emplace_back(IValue(start));
  stack.emplace_back(IValue(end));
  stack.emplace_back(IValue(step));
  slice_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
  p_context_->syn_outputs_.erase(p_context_->syn_outputs_.begin());
  p_context_->syn_outputs_.insert(
      p_context_->syn_outputs_.begin(),
      std::move(slice_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.erase(p_context_->pt_outputs_.begin());
  p_context_->pt_outputs_.insert(
      p_context_->pt_outputs_.begin(), std::move(slice_op->GetOutputs()[0]));
}

void LpNormFrobeniusOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for LpNorm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for LpNorm Operator");

  auto self = inputs[0].toTensor();
  auto output = habana::createPTTensor(
      self,
      {1},
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      output_metadata.at(0).persistent);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

std::shared_ptr<SliceOperator> FusedNormOperator::compute_clip_coeff(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto gradients = inputs[0].toTensorList();
  auto max_grad_norm = inputs[1].toTensor();
  auto norm_type = inputs[2].toScalar();
  float eps = 1e-6;
  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();
  auto num_params = static_cast<unsigned int>(gradients.size());

  auto slice_op = make_operator<SliceOperator>(device_id, scalar_type);

  if (norm_type.toFloat() == 2.0) {
    torch::jit::Stack stack;
    std::vector<Tensor> cat_input;
    auto cat_grad_norms = make_operator<CatOperator>(device_id, scalar_type);
    std::vector<int64_t> shape{1, 1};
    for (unsigned int i = 0; i < num_params; i++) {
      // Add node to compute norm on each gradient tensor
      auto norm_lp = make_operator<NormOperator>(device_id, scalar_type);
      norm_lp->SetSynapseInput(p_context_->syn_inputs_[i]);
      stack.emplace_back(IValue(gradients.get(i)));
      stack.emplace_back(IValue(2.0));
      norm_lp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();
      // each grad_norm connected to cat node
      cat_input.push_back(norm_lp->GetOutputs()[0]);
      cat_grad_norms->SetSynapseInput(norm_lp->GetSynOutputs()[0]);
      stack.clear();
    }

    // grad_norms are concatened into a single big tensor of shape
    // {num_params,1}
    stack.emplace_back(IValue(cat_input));
    stack.emplace_back(IValue(0));
    cat_grad_norms->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // node to do compute total_norm
    auto norm_final = make_operator<NormOperator>(device_id, scalar_type);
    norm_final->SetSynapseInput(cat_grad_norms->GetSynOutputs()[0]);
    stack.emplace_back(IValue(cat_grad_norms->GetOutputs()[0]));
    stack.emplace_back(IValue(2.0));
    norm_final->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();
    p_context_->syn_outputs_.emplace_back(
        std::move(norm_final->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(norm_final->GetOutputs()[0]));

    /*
    Now use the total_norm calculated to update grads
    max_norm = float(max_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
      for p in parameters:
        p.grad.detach().mul_(clip_coef)
    */

    // total_norm + 1e-6
    auto add_op1 = make_operator<AddOperator>(device_id, scalar_type);
    add_op1->SetSynapseInput(p_context_->syn_outputs_[0]);
    stack.emplace_back(IValue(p_context_->pt_outputs_[0]));
    stack.emplace_back(IValue(Scalar(eps)));
    stack.emplace_back(IValue(1.0));
    add_op1->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // clip_coef = max_norm / (total_norm + 1e-6)
    auto div_final = make_operator<DivOperator>(device_id, scalar_type);
    div_final->SetSynapseInput(p_context_->syn_inputs_[num_params]);
    div_final->SetSynapseInput(add_op1->GetSynOutputs()[0]);
    stack.emplace_back(IValue(max_grad_norm));
    stack.emplace_back(IValue(add_op1->GetOutputs()[0]));
    div_final->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // mask = total_norm > max_grad_norm
    auto gt_op = make_operator<GtOperator>(device_id, scalar_type);
    gt_op->SetSynapseInput(p_context_->syn_outputs_[0]);
    gt_op->SetSynapseInput(p_context_->syn_inputs_[num_params]);
    stack.emplace_back(IValue(p_context_->pt_outputs_[0]));
    stack.emplace_back(IValue(max_grad_norm));
    gt_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    std::string node_type = "cast_i8_to_f32";
    auto cast1 = make_operator<CastOperator>(device_id, node_type);
    cast1->SetSynapseInput(gt_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(gt_op->GetOutputs()[0]));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    OutputMetaData md;
    md.dtype = at::kFloat;
    cast1->AllocateAndAddSynapseNode(graph, stack, {md});
    stack.clear();

    // mul1 = mask * clip_coef
    auto mul1 = make_operator<MulOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(cast1->GetSynOutputs()[0]);
    mul1->SetSynapseInput(div_final->GetSynOutputs()[0]);
    stack.emplace_back(IValue(cast1->GetOutputs()[0]));
    stack.emplace_back(IValue(div_final->GetOutputs()[0]));
    mul1->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
    // imask = (mask == 0)
    auto eq_op = make_operator<EqOperator>(device_id, scalar_type);
    eq_op->SetSynapseInput(cast1->GetSynOutputs()[0]);
    stack.emplace_back(IValue(cast1->GetOutputs()[0]));
    stack.emplace_back(IValue(0.0));
    eq_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    node_type = "cast_i8_to_f32";
    auto cast2 = make_operator<CastOperator>(device_id, node_type);
    cast2->SetSynapseInput(eq_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(eq_op->GetOutputs()[0]));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    cast2->AllocateAndAddSynapseNode(graph, stack, {md});
    stack.clear();

    // mask*clip_coef + imask
    auto add_op2 = make_operator<AddOperator>(device_id, scalar_type);
    add_op2->SetSynapseInput(mul1->GetSynOutputs()[0]);
    add_op2->SetSynapseInput(cast2->GetSynOutputs()[0]);
    stack.emplace_back(IValue(mul1->GetOutputs()[0]));
    stack.emplace_back(IValue(cast2->GetOutputs()[0]));
    stack.emplace_back(IValue(1.0));
    add_op2->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
    // take just the first element of add since all values would be repeated

    slice_op->SetSynapseInput(add_op2->GetSynOutputs()[0]);
    stack.emplace_back(IValue(add_op2->GetOutputs()[0]));
    stack.emplace_back(IValue(0));
    stack.emplace_back(IValue(0));
    stack.emplace_back(IValue(1));
    stack.emplace_back(IValue(1));
    slice_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
  } else {
    // other norm_type not supported for now
    // BERT Hugging-face uses norm_type = 2.0
    // therefore supporting only that for now.
    HABANA_ASSERT(0, "unsupported norm_type for fused norm");
  }

  return slice_op;
}

void FusedNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for FusedNorm Operator");

  auto gradients = inputs[0].toTensorList();
  auto num_params = static_cast<unsigned int>(gradients.size());
  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();

  auto slice_op = compute_clip_coeff(graph, inputs, output_metadata);

  torch::jit::Stack stack;

  // p.grad.detach().mul_(clip_coef)
  for (unsigned int i = 0; i < num_params; i++) {
    auto mul1 = make_operator<MulInplaceOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(p_context_->syn_inputs_[i]);
    mul1->SetSynapseInput(slice_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(slice_op->GetOutputs()[0]));
    auto out_metadata = SelectVectorIndices(output_metadata, {i + 1u});
    mul1->AllocateAndAddSynapseNode(graph, stack, out_metadata);
    stack.clear();
    // Add grads to output lists to satisfy GC (since grad updation is
    // inplace)
    p_context_->syn_outputs_.emplace_back(std::move(mul1->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(mul1->GetOutputs()[0]);
  }
}

void FusedNormLazyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for FusedNorm Operator");

  auto gradients = inputs[0].toTensorList();
  auto num_params = static_cast<unsigned int>(gradients.size());
  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();

  auto slice_op = compute_clip_coeff(graph, inputs, output_metadata);

  torch::jit::Stack stack;

  // clipped_grad = p.grad.detach().mul(clip_coef)
  for (unsigned int i = 0; i < num_params; i++) {
    auto mul1 = make_operator<MulOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(p_context_->syn_inputs_[i]);
    mul1->SetSynapseInput(slice_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(slice_op->GetOutputs()[0]));
    mul1->AllocateAndAddSynapseNode(
        graph, stack, SelectVectorIndices(output_metadata, {i + 1}));
    stack.clear();

    p_context_->syn_outputs_.emplace_back(std::move(mul1->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(mul1->GetOutputs()[0]);
  }
}

////////////////////////////////////////New BN

/*********************************************************************************
 * @brief - This op is used in lazy mode to avoid memcopy nodes for RMV
 *********************************************************************************/
at::Tensor BatchNormForwardOperator::create_or_return_tensor_bn(
    synapse_helpers::graph& graph,
    const Tensor& input,
    uint size,
    Device device,
    int syn_index) {
  Tensor ret_tensor;
  if (!input.defined()) {
    // If optional tensor is undefined, we create one and synapse tensor
    // appended info is patching info passed back to the kernel for this new
    // tensor which lowering kernel is unaware of
    ret_tensor = at::empty({size}, device);
    auto syn_tensor = habana_helpers::create_tensor(
        ret_tensor, graph, true, false, c10::nullopt);
    auto it = p_context_->syn_inputs_.begin() + syn_index;
    p_context_->syn_inputs_.insert(it, std::move(syn_tensor));

    appended_tensor_infos.emplace_back(
        std::make_tuple(syn_tensor.name(), ret_tensor, syn_tensor.id()));
  } else if (input.defined() && input.device() != DeviceType::HPU) {
    ret_tensor = input.to(DeviceType::HPU);
  } else {
    return input;
  }

  return ret_tensor;
}
at::Tensor BatchNormForwardOperator::create_or_return_pt_tensor_bn(
    const at::Tensor& input,
    uint size,
    Device device) {
  Tensor ret_tensor;
  if (!input.defined()) {
    ret_tensor = at::empty({size}, device);
  } else if (input.defined() && input.device() != DeviceType::HPU) {
    ret_tensor = input.to(DeviceType::HPU);
  } else {
    return input;
  }
  return ret_tensor;
}

void BatchNormForwardOperator::preProcessInputs(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect number of inputs against expected count for BatchNormForward operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[5].isBool(), "Input type expected to be bool");
  TORCH_CHECK(inputs[6].isDouble(), "Input type expected to be double");
  TORCH_CHECK(inputs[7].isDouble(), "Input type expected to be double");

  const auto input = inputs[0].toTensor();
  const auto weight = inputs[1].toTensor();
  const auto bias = inputs[2].toTensor();
  const auto running_mean = inputs[3].toTensor();
  const auto running_var = inputs[4].toTensor();
  const auto training = inputs[5].toBool();
  TORCH_CHECK(
      training, "BN Forward training flag should be set to 1 in training mode");

  Tensor wt_hpu, bias_hpu;
  auto device = DeviceType::HPU;
  auto channel_dim = synapse_helpers::layouts::INPUT_C_IDX;

  if (training) {
    wt_hpu = create_or_return_tensor_bn(
        graph, weight, input.sizes()[channel_dim], device, (uint)1);
    bias_hpu = create_or_return_tensor_bn(
        graph, bias, input.sizes()[channel_dim], device, (uint)2);
  } else {
    bias_hpu = create_or_return_tensor_bn(
        graph, bias, input.sizes()[channel_dim], device, (uint)1);
    wt_hpu = create_or_return_tensor_bn(
        graph, weight, input.sizes()[channel_dim], device, (uint)2);
  }

  Tensor running_mean_hpu = create_or_return_tensor_bn(
      graph, running_mean, input.sizes()[channel_dim], device, (uint)3);
  Tensor running_var_hpu = create_or_return_tensor_bn(
      graph, running_var, input.sizes()[channel_dim], device, (uint)4);

  pre_inputs = {
      std::move(input),
      std::move(wt_hpu),
      std::move(bias_hpu),
      std::move(running_mean_hpu),
      std::move(running_var_hpu)};
  pt_inputs = {
      pre_inputs[0],
      pre_inputs[1],
      pre_inputs[2],
      pre_inputs[3],
      pre_inputs[4]};

  p_context_->pt_inputs_.clear();
  SetPTInputs(pt_inputs);
}

void BatchNormForwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& in_stack,
    const OutputMetaDataVector& output_metadata) {
  preProcessInputs(graph, in_stack);
  TORCH_CHECK(
      output_metadata.size() == 5,
      "BatchNormForwardOperator: output_metadata should be 5 in training mode");
  const auto training = in_stack[5].toBool();
  const auto momentum = in_stack[6].toDouble();
  const auto eps = in_stack[7].toDouble();
  auto output =
      habana::createPTTensor(pre_inputs[0], output_metadata.at(0).persistent);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  auto current_mean =
      habana::createPTTensor(pre_inputs[3], output_metadata.at(1).persistent);
  auto current_istd =
      habana::createPTTensor(pre_inputs[4], output_metadata.at(2).persistent);
  OutputMetaDataVector out_12_metadata =
      SelectVectorIndices(output_metadata, {1, 2});
  AllocateSynapseOutputs(graph, {current_mean, current_istd}, out_12_metadata);

  auto running_mean_out =
      habana::createPTTensor(pre_inputs[3], output_metadata.at(3).persistent);
  auto running_var_out =
      habana::createPTTensor(pre_inputs[4], output_metadata.at(4).persistent);
  OutputMetaDataVector out_34_metadata =
      SelectVectorIndices(output_metadata, {3, 4});
  AllocateSynapseOutputs(
      graph, {running_mean_out, running_var_out}, out_34_metadata);

  ns_BatchNormKernel::ParamsV2 params;
  params.momentum = static_cast<float>(momentum),
  params.epsilon = static_cast<float>(eps);
  params.threshold.f = 0.0;
  params.isTraining = training;
  p_context_->params_.emplace<ns_BatchNormKernel::ParamsV2>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

InferOutputMetaRetType BatchNormForwardOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  const auto input = inputs[0].toTensor();
  const auto running_mean = inputs[3].toTensor();
  const auto running_var = inputs[4].toTensor();

  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      input.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input.sizes(), input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));

  auto mean_tensor_data = TensorMetaData(
      running_mean.sizes().vec(),
      HabanaOperator::CalculateStrides(
          running_mean.sizes(), running_mean.suggest_memory_format()),
      running_mean.scalar_type(),
      running_mean.suggest_memory_format());

  auto var_tensor_data = TensorMetaData(
      running_var.sizes().vec(),
      HabanaOperator::CalculateStrides(
          running_var.sizes(), running_var.suggest_memory_format()),
      running_var.scalar_type(),
      running_var.suggest_memory_format());

  // current_mean and current_istd
  out.AddOutputTensor(mean_tensor_data);
  out.AddOutputTensor(var_tensor_data);

  // running_mean and running_var
  out.AddOutputTensor(mean_tensor_data);
  out.AddOutputTensor(var_tensor_data);

  return out;
}

// BN New Backward
void BatchNormBackwardOperator::create_opt_input_tensor_bn_bwd(
    synapse_helpers::graph& graph,
    const Tensor& input,
    uint size,
    Device device,
    int pos) {
  Tensor ret_tensor;
  if (!input.defined()) {
    ret_tensor = at::empty({size}, device);
    auto syn_tensor = habana_helpers::create_tensor(
        ret_tensor, graph, true, false, c10::nullopt);
    appended_tensor_infos.emplace_back(
        std::make_tuple(syn_tensor.name(), ret_tensor, syn_tensor.id()));
    // if input is not defined, we get dummy tensor from wrapper
    // create new one and place it to the original position
    p_context_->syn_inputs_.emplace_back(std::move(syn_tensor));
    if ((uint)pos < p_context_->pt_inputs_.size()) {
      p_context_->pt_inputs_[pos] = ret_tensor;
    } else {
      // this is for bias which is not present in input list
      // pushed to the end
      p_context_->pt_inputs_.push_back(ret_tensor);
    }
  }
}

void BatchNormBackwardOperator::preProcessInputs(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect number of inputs against expected count for BatchNormBackwardOperator preProcessInputs: expected 5 got ",
      inputs.size());
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");

  const auto input = inputs[0].toTensor();
  const auto mean = inputs[2].toTensor();
  const auto invstd = inputs[3].toTensor();
  const auto weight = inputs[4].toTensor();

  auto device = input.device();
  auto channel_dim = synapse_helpers::layouts::INPUT_C_IDX;

  create_opt_input_tensor_bn_bwd(
      graph, mean, input.sizes()[channel_dim], device, 2);
  create_opt_input_tensor_bn_bwd(
      graph, invstd, input.sizes()[channel_dim], device, 3);
  create_opt_input_tensor_bn_bwd(
      graph, weight, input.sizes()[channel_dim], device, 4);

  SetProprocessingDone();
}

void BatchNormBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect number of inputs against expected count for BatchNormBackwardOperator AllocateAndAddSynapseNode");
  TORCH_CHECK(
      inputs[6].isDouble(), "Input type for eps is expected to be double");
  TORCH_CHECK(
      output_metadata.size() == 3,
      "BatchNormBackwardOperator: #output_metadata should be 3");
  if (CheckProprocessingDone() == false) {
    Stack preprocess_in = {};
    for (auto& input : inputs) {
      if (input.isTensor()) {
        preprocess_in.insert(preprocess_in.end(), input);
      }
    }
    preProcessInputs(graph, preprocess_in);
  }
  const auto input = inputs[0].toTensor();
  const auto weight = inputs[4].toTensor();
  const auto training = inputs[5].toBool();
  const auto eps = inputs[6].toDouble();
  const auto momentum = inputs[7].toDouble();
  // Prepare output tensor vector
  auto grad_in_nhwc =
      habana::createPTTensor(input, output_metadata.at(0).persistent);
  auto grad_beta =
      habana::createPTTensor(weight, output_metadata.at(1).persistent);
  auto grad_gamma =
      habana::createPTTensor(weight, output_metadata.at(2).persistent);

  AllocateSynapseOutputs(
      graph, {grad_in_nhwc, grad_beta, grad_gamma}, output_metadata);

  ns_BatchNormKernel::ParamsV2 params;
  params.momentum = static_cast<float>(momentum),
  params.epsilon = static_cast<float>(eps);
  params.threshold.f = 0.0;
  params.isTraining = training;
  p_context_->params_.emplace<ns_BatchNormKernel::ParamsV2>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

InferOutputMetaRetType BatchNormBackwardOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  const auto input = inputs[0].toTensor();
  const auto weight = inputs[4].toTensor();

  InferOutputMetaRetType out;
  // grad_in_nhwc
  out.AddOutputTensor(TensorMetaData(
      input.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input.sizes(), input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));
  // grad_gamma
  out.AddOutputTensor(TensorMetaData(
      weight.sizes().vec(),
      HabanaOperator::CalculateStrides(
          weight.sizes(), weight.suggest_memory_format()),
      weight.scalar_type(),
      weight.suggest_memory_format()));
  // grad_beta
  out.AddOutputTensor(TensorMetaData(
      weight.sizes().vec(),
      HabanaOperator::CalculateStrides(
          weight.sizes(), weight.suggest_memory_format()),
      weight.scalar_type(),
      weight.suggest_memory_format()));
  return out;
}

///////////////////////////////////////
TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def(
      "fused_norm_(Tensor(a!)[] grad, Tensor max_norm, float norm_type) -> Tensor");
  m.def(
      "fused_norm_lazy(Tensor(a!)[] grad, Tensor max_norm, float norm_type) -> Tensor");
}

static auto& NormKernelsKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::native_batch_norm_training",
            KERNEL_FN(BatchNormForwardOperator))
        .add("hpu::native_batch_norm_inf", KERNEL_FN(BatchNormInfOperator))
        .add(
            "hpu::native_batch_norm_backward",
            KERNEL_FN(BatchNormBackwardOperator))
        .add("hpu::fused_norm_", KERNEL_FN(FusedNormOperator))
        .add("hpu::fused_norm_lazy", KERNEL_FN(FusedNormLazyOperator));
