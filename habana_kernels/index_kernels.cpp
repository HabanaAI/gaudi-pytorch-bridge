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
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "backend/backend_meta.h"
#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/dynamic_bucket_info.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "backend/synapse_helpers/tensor_builder_base.h"
#include "habana_helpers/frontend_utils.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/logging_pt.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/nonzero_kernel.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/topk_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/tensor_impl.h"
#include "hpu_ops/common/index.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/hpu_op_helper.h"
#include "pytorch_helpers/habana_helpers/dtype_helpers.h"

using namespace torch;
using namespace habana;
using tensor_name_generator = synapse_helpers::detail::tensor_name_generator;

/*************************************************************************
 * @brief This helper function makes the size of index tensor to be same as
 * value tensor, with broadcast of indices (within index tensor)
 ************************************************************************/
int ArangeOperator::GetOutputSize(Scalar start_, Scalar end_, Scalar step_) {
  auto start = start_.to<double>();
  auto end = end_.to<double>();
  auto step = step_.to<double>();

  TORCH_CHECK(step != 0, "step value can not be 0.");
  TORCH_CHECK(!((start > end) && (step > 0)), "step must be negative.");
  TORCH_CHECK(!((start < end) && (step < 0)), "step must be positive.");

  float max, min, abs_del;
  int depth;
  max = start > end ? start : end;
  min = start > end ? end : start;
  abs_del = std::abs(step);
  depth = std::ceil((max - min) / abs_del);
  return depth;
}

Tensor GatherOperator::AllocateOutput(
    torch::jit::Stack& inputs,
    const OutputMetaData& output_metadata) {
  auto self = inputs[0].toTensor();
  auto dim_ = inputs[1].toInt();
  auto index = inputs[2].toTensor();

  auto shape = ComputeGatherOperatorOutputShape(self, dim_, index);

  auto output = habana::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      output_metadata.persistent);
  return output;
}

void GatherOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  OutputMetaData md;
  md.persistent = true;
  auto output = AllocateOutput(inputs, md);
  HabanaOperator::SetPTOutput(output);
}

void GatherOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of input expected for Gather operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be Tensor for Gather operator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input type expected to be Int for Gather operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input type expected to be Tensor for Gather operator");
  TORCH_CHECK(
      inputs[3].isBool(), "Input type expected to be Bool for Gather operator");

  auto self = inputs[0].toTensor();
  auto dim_ = inputs[1].toInt();
  auto index = inputs[2].toTensor();
  auto sparse_grad = inputs[3].toBool();

  TORCH_CHECK(sparse_grad == false, "spare_grad is not supported")
  if (index.dim() == 0) {
    SET_SIZE_STRIDE_1D(index);
  }

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  auto output = AllocateOutput(inputs, output_metadata.at(0));

  ns_GatherKernel::Params params;
  params.axis = get_dim_in_tpc_order(dim, self.dim());

  p_context_->params_.emplace<ns_GatherKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

InferOutputMetaRetType GatherOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto index = inputs[2].toTensor();
  if (index.dim() == 0) {
    SET_SIZE_STRIDE_1D(index);
  }
  auto output = AllocateOutput(inputs, OutputMetaData());
  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      output.sizes().vec(),
      HabanaOperator::CalculateStrides(
          output.sizes().vec(), output.suggest_memory_format()),
      output.scalar_type(),
      output.suggest_memory_format()));
  return out;
}

std::vector<int64_t> GatherElemOperator::compute_output_shape(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index) {
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto shape = self.sizes().vec();
  if (shape.size()) {
    // for gather op, output size is same as index
    if (self.dim() == index.dim()) {
      shape = index.sizes().vec();
    } else {
      // for index_select and other index ops
      shape.erase(shape.begin() + dim);
      shape.insert(shape.begin() + dim, index.numel());
    }
  }
  return shape;
}

Tensor GatherElemOperator::AllocateOutput(
    torch::jit::Stack& inputs,
    const OutputMetaData& output_metadata) {
  auto self = inputs[0].toTensor();
  auto dim_ = inputs[3].toInt();
  auto index = inputs[1].toTensor();

  auto shape = GatherElemOperator::compute_output_shape(self, dim_, index);

  auto output = habana::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      output_metadata.persistent);
  return output;
}

void GatherElemOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of input expected for Gather operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be Tensor for Gather operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input type expected to be Int for Gather operator");
  TORCH_CHECK(
      inputs[2].isTensor() or inputs[2].isNone(),
      "Input type expected to be Tensor for Gather operator");
  TORCH_CHECK(
      inputs[4].isBool(), "Input type expected to be Bool for Gather operator");

  auto self = inputs[0].toTensor();
  auto dim_ = inputs[3].toInt();

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  auto output = AllocateOutput(inputs, output_metadata.at(0));

  ns_GatherElementsKernel::Params params;
  params.axis = get_dim_in_tpc_order(dim, self.dim());

  p_context_->params_.emplace<ns_GatherElementsKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

std::vector<int64_t> ScatterWrapperOperator::compute_output_shape(
    const Tensor& self) {
  return self.sizes().vec();
}

Tensor ScatterWrapperOperator::AllocateOutput(
    torch::jit::Stack& inputs,
    const OutputMetaData& output_metadata) {
  auto self = inputs[0].toTensor();
  auto output = habana::createPTTensor(self, output_metadata.persistent);
  return output;
}

void ScatterWrapperOperator::SetPTOutput(torch::jit::Stack& inputs) {
  OutputMetaData md;
  md.persistent = true;
  auto output = AllocateOutput(inputs, md);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

void ScatterWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of input expected for Scatter operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be Tensor for Scatter operator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input type expected to be Int for Scatter operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input type expected to be Tensor for Scatter operator");
  TORCH_CHECK(
      inputs[3].isTensor(),
      "nput type expected to be Int for Scatter operator");

  auto self = inputs[0].toTensor();
  auto dim_ = inputs[1].toInt();
  auto index = inputs[2].toTensor();
  // auto src = inputs[3].toTensor();
  if (index.dim() == 0) {
    SET_SIZE_STRIDE_1D(index);
  }

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  if (!inplace) {
    auto output = AllocateOutput(inputs, output_metadata.at(0));
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
  } else {
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
    p_context_->pt_outputs_.emplace_back(self);
  }

  ns_ScatterKernel::Params params;
  params.axis = get_dim_in_tpc_order(dim, self.dim());

  p_context_->params_.emplace<ns_ScatterKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void ScatterAddOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of input expected for Scatter Add operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be Tensor for Scatter Add operator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input type expected to be Int for Scatter Add operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input type expected to be Tensor for Scatter Add operator");
  TORCH_CHECK(
      inputs[3].isTensor(),
      "Input type expected to be Int for Scatter Add operator");

  auto self = inputs[0].toTensor();
  auto dim_ = inputs[1].toInt();
  auto index = inputs[2].toTensor();
  auto src = inputs[3].toTensor();

  if (index.dim() == 0) {
    SET_SIZE_STRIDE_1D(index);
  }

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  if (!inplace) {
    auto output = AllocateOutput(inputs, output_metadata.at(0));
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
  } else {
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
    p_context_->pt_outputs_.emplace_back(self);
  }

  ns_ScatterKernel::Params params;
  params.axis = get_dim_in_tpc_order(dim, self.dim());
  p_context_->params_.emplace<ns_ScatterKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  if (GET_ENV_FLAG_NEW(PT_HPU_USE_UNSORTED_SCATTER_ADD) &&
      HPUGlobalConfig::get().getDeterministic() == false &&
      at::globalContext().deterministicAlgorithms() == false &&
      HPUDeviceContext::get_device().type() != synDeviceType::synDeviceGaudi) {
    if (self.scalar_type() == c10::ScalarType::BFloat16) {
      auto cast_op1 = make_operator<CastOperator>(
          self.device().index(), "cast_bf16_to_f32");
      auto cast_op2 = make_operator<CastOperator>(
          self.device().index(), "cast_bf16_to_f32");
      c10::ScalarType cast_scalar_type = c10::ScalarType::Float;
      auto md = OutputMetaDataVector(1);
      md[0].dtype = cast_scalar_type;

      std::vector<c10::IValue> cast_stack1{
          IValue(self), IValue(cast_scalar_type)};
      cast_op1->SetSynapseInput(p_context_->syn_inputs_[0]);
      cast_op1->AllocateAndAddSynapseNode(graph, cast_stack1, md);
      std::vector<c10::IValue> cast_stack2{
          IValue(src), IValue(cast_scalar_type)};
      cast_op2->SetSynapseInput(p_context_->syn_inputs_[2]);
      cast_op2->AllocateAndAddSynapseNode(graph, cast_stack2, md);
      auto unsorted_scatter_add_op = make_operator<UnsortedScatterAddOperator>(
          self.device().index(), c10::ScalarType::Float);
      std::vector<c10::IValue> stack{
          IValue(cast_op1->GetOutputs()[0]),
          IValue(dim),
          IValue(index),
          IValue(cast_op2->GetOutputs()[0])};
      unsorted_scatter_add_op->SetSynapseInput(cast_op1->GetSynOutputs()[0]);
      unsorted_scatter_add_op->SetSynapseInput(p_context_->syn_inputs_[1]);
      unsorted_scatter_add_op->SetSynapseInput(cast_op2->GetSynOutputs()[0]);
      unsorted_scatter_add_op->AllocateAndAddSynapseNode(graph, stack, md);

      cast_scalar_type = c10::ScalarType::BFloat16;
      // since cast is the last operator in this path, we need to use the
      // incoming output_metadata for the cast_op3->AllocateAndAddSynapseNode.
      // At the same time, cast operator needs the dtpe to be set in output meta
      // data. dtype may not be set in the incoming output_metadata if
      // ScatterAdd operator is used as an intermediate op. Eg. in
      // EmbeddingDenseBackward operator. So, use the incoming output_metadata,
      // but set the dtype correctly.
      OutputMetaData md_updated = output_metadata[0];
      md_updated.dtype = cast_scalar_type;
      auto cast_op3 = make_operator<CastOperator>(
          self.device().index(), "cast_f32_to_bf16");
      std::vector<c10::IValue> cast_stack3{
          IValue(unsorted_scatter_add_op->GetOutputs()[0]),
          IValue(cast_scalar_type)};
      cast_op3->SetSynapseInput(unsorted_scatter_add_op->GetSynOutputs()[0]);
      cast_op3->AllocateAndAddSynapseNode(graph, cast_stack3, {md_updated});
      p_context_->syn_outputs_[0] = std::move(cast_op3->GetSynOutputs()[0]);
      p_context_->pt_outputs_[0] = std::move(cast_op3->GetOutputs()[0]);
      return;
    }
    SetGuid("unsorted_scatter_add_fwd_f32");
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  } else { // On Gaudi1
    // "To sort or not to sort, that is the question..."!!
    // Sorting of index tensor is needed on G1 for functional correctness.
    // But it may have an impact on perf. So enable sorting using env var
    // for now.
    if (GET_ENV_FLAG_NEW(PT_HPU_SORT_INDEX_IN_SCATTER_ADD)) {
      auto sorted_scatter_add_op = make_operator<SortedScatterAddOperator>(
          self.device().index(),
          self.scalar_type()); // TO DO: use common self.device().index(
      std::vector<synapse_helpers::tensor_or_ref> syn_src;
      std::vector<at::Tensor> pt_src;
      // sort only if index.sizes()[dim] > 1
      // TODO: this check can have issues in dynamic shapes scenario.
      // Remove this check once SW-124506 is addressed.
      if (index.sizes()[dim] != 1) {
        auto topkOp =
            make_operator<TopkOperator>(this->p_context_->device_id_, "topk");
        int64_t topk_dim = dim;
        bool largest = true;
        bool sorted = true;
        topkOp->SetSynapseInput(p_context_->syn_inputs_[1]);
        std::vector<c10::IValue> topk_stack{
            IValue(index),
            IValue(index.sizes()[dim]),
            IValue(topk_dim),
            IValue(largest),
            IValue(sorted)};
        topkOp->AllocateAndAddSynapseNode(
            graph, topk_stack, OutputMetaDataVector(2));

        // Node: reordered source
        // If we get to scatter add via the python scatter_add()
        // operator, src and index tensors will have same dimensions.
        // But getting here via some other path like scatter_add in
        // embedding_dense_backward can have src.dim() != index.dim()
        // When src.dim() == index.dim(), GatherElem(gather_elements guid)
        // operator will work. But not when src.dim() !=index.dim().
        // In this case we need to use Gather(gather guid) operator.
        if (src.dim() == index.dim()) {
          auto gatherOp = make_operator<GatherElemOperator>(
              this->p_context_->device_id_, src.scalar_type());
          gatherOp->SetSynapseInput(p_context_->syn_inputs_[2]);
          gatherOp->SetSynapseInput(topkOp->GetSynOutputs()[1]);
          bool sparse_grad = false;
          std::vector<c10::IValue> gather_stack{
              IValue(src),
              IValue(topkOp->GetOutputs()[1]),
              IValue(c10::nullopt),
              IValue(dim),
              IValue(sparse_grad)};
          gatherOp->AllocateAndAddSynapseNode(
              graph, gather_stack, OutputMetaDataVector(1));
          pt_src.push_back(std::move(gatherOp->GetOutputs()[0]));
          syn_src.push_back(std::move(gatherOp->GetSynOutputs()[0]));
        } else {
          auto gatherOp = make_operator<GatherOperator>(
              this->p_context_->device_id_, src.scalar_type());
          gatherOp->SetSynapseInput(p_context_->syn_inputs_[2]);
          gatherOp->SetSynapseInput(topkOp->GetSynOutputs()[1]);
          bool sparse_grad = false;
          std::vector<c10::IValue> gather_stack{
              IValue(src),
              IValue(dim),
              IValue(topkOp->GetOutputs()[1]),
              IValue(sparse_grad)};
          gatherOp->AllocateAndAddSynapseNode(
              graph, gather_stack, OutputMetaDataVector(1));
          pt_src.push_back(std::move(gatherOp->GetOutputs()[0]));
          syn_src.push_back(std::move(gatherOp->GetSynOutputs()[0]));
        }

        std::vector<c10::IValue> stack{
            IValue(self),
            IValue(dim),
            IValue(topkOp->GetOutputs()[0]),
            IValue(pt_src[0])};
        sorted_scatter_add_op->SetSynapseInput(p_context_->syn_inputs_[0]);
        sorted_scatter_add_op->SetSynapseInput(topkOp->GetSynOutputs()[0]);
        sorted_scatter_add_op->SetSynapseInput(syn_src[0]);
        sorted_scatter_add_op->AllocateAndAddSynapseNode(
            graph, stack, output_metadata);
      } else {
        std::vector<c10::IValue> stack{
            IValue(self), IValue(dim), IValue(index), IValue(src)};
        sorted_scatter_add_op->SetSynapseInput(p_context_->syn_inputs_[0]);
        sorted_scatter_add_op->SetSynapseInput(p_context_->syn_inputs_[1]);
        sorted_scatter_add_op->SetSynapseInput(p_context_->syn_inputs_[2]);
        sorted_scatter_add_op->AllocateAndAddSynapseNode(
            graph, stack, output_metadata);
      }
      p_context_->syn_outputs_[0] =
          std::move(sorted_scatter_add_op->GetSynOutputs()[0]);
      p_context_->pt_outputs_[0] =
          std::move(sorted_scatter_add_op->GetOutputs()[0]);
    } else {
      AddNodeToSynapseGraph(graph, &params, sizeof(params));
    }
  }
}

void IndexAddOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 5, "Incorrect size of inputs for index_add operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input 0 type expected to be Tensor for index_add operator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input 1 type expected to be int64_t for index_add operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input 2 type expected to be Tensor for index_add operator");
  TORCH_CHECK(
      inputs[3].isTensor(),
      "Input 3 type expected to be Tensor for index_add operator");
  TORCH_CHECK(
      inputs[4].isScalar(),
      "Input 4 type expected to be Scalar for index_add operator");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto index = inputs[2].toTensor();
  auto value = inputs[3].toTensor();
  auto alpha = inputs[4].toScalar();

  std::vector<synapse_helpers::tensor_or_ref> addSynOutput;
  torch::jit::Stack temp_stack;
  dim = at::maybe_wrap_dim(dim, self.dim());

  ////auto slice = at::index_select(self, 0, indices[0]);
  auto index_selectOp = make_operator<IndexSelectOperator>(
      this->p_context_->device_id_, self.scalar_type());
  temp_stack = {IValue(self), IValue(dim), IValue(index)};
  index_selectOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  index_selectOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  index_selectOp->AllocateAndAddSynapseNode(
      graph, temp_stack, OutputMetaDataVector(1));
  temp_stack.clear();

  ////alpha_value = value * alpha;
  auto mulOp = make_operator<MulOperator>(
      this->p_context_->device_id_, value.scalar_type());
  temp_stack = {IValue(value), IValue(alpha)};
  mulOp->SetSynapseInput(p_context_->syn_inputs_[2]);
  mulOp->AllocateAndAddSynapseNode(graph, temp_stack, OutputMetaDataVector(1));
  temp_stack.clear();

  ////value_acc += slice;
  auto addOp = make_operator<AddOperator>(
      this->p_context_->device_id_, value.scalar_type());
  temp_stack = {
      IValue(mulOp->GetOutputs()[0]),
      IValue(index_selectOp->GetOutputs()[0]),
      IValue(Scalar(1.0))};
  addOp->SetSynapseInput(mulOp->GetSynOutputs()[0]);
  addOp->SetSynapseInput(index_selectOp->GetSynOutputs()[0]);
  addOp->AllocateAndAddSynapseNode(graph, temp_stack, OutputMetaDataVector(1));
  addSynOutput.push_back(std::move(addOp->GetSynOutputs()[0]));
  temp_stack.clear();

  // Expand 1D index tensor to same number of dimensions as value tensor
  auto expanded_sizes = std::vector<int64_t>(value.ndimension(), 1);
  expanded_sizes[dim] = index.sizes()[0];

  ////auto index_expanded = index.view(expanded_sizes)
  auto reshapeOp = make_operator<ReshapeOperator>(
      this->p_context_->device_id_, index.scalar_type());
  temp_stack = {IValue(index), IValue(expanded_sizes)};
  reshapeOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  reshapeOp->AllocateAndAddSynapseNode(
      graph, temp_stack, OutputMetaDataVector(1));
  temp_stack.clear();

  // Broadcast index tensor to same shape as value tensor
  bool implicit =
      false; // The value of implicit is currently ignored in broadcast kernel
  auto bcastOp = make_operator<BroadcastOperator>(
      this->p_context_->device_id_, reshapeOp->GetOutputs()[0].scalar_type());
  temp_stack = {
      IValue(reshapeOp->GetOutputs()[0]),
      IValue(value.sizes()),
      IValue(implicit)};
  bcastOp->SetSynapseInput(reshapeOp->GetSynOutputs()[0]);
  bcastOp->AllocateAndAddSynapseNode(
      graph, temp_stack, OutputMetaDataVector(1));
  temp_stack.clear();

  auto scatterOp = make_operator<ScatterHelperOperator>(
      this->p_context_->device_id_, self.scalar_type());
  temp_stack = {
      IValue(self),
      IValue(dim),
      IValue(bcastOp->GetOutputs()[0]),
      IValue(value)};

  scatterOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  scatterOp->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
  scatterOp->SetSynapseInput(addSynOutput[0]);
  scatterOp->AllocateAndAddSynapseNode(graph, temp_stack, output_metadata);

  p_context_->syn_outputs_.emplace_back(
      std::move(scatterOp->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(scatterOp->GetOutputs()[0]));
}

/*
 Implementation to take care of duplicate entries in index tensor and also
 the case where index tensor size can be greater than the self tensor size at
 the relevant dim.
*/
void IndexAddV2Operator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 5, "Incorrect size of inputs for index_add operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input 0 type expected to be Tensor for index_add operator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input 1 type expected to be int64_t for index_add operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input 2 type expected to be Tensor for index_add operator");
  TORCH_CHECK(
      inputs[3].isTensor(),
      "Input 3 type expected to be Tensor for index_add operator");
  TORCH_CHECK(
      inputs[4].isScalar(),
      "Input 4 type expected to be Scalar for index_add operator");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto index = inputs[2].toTensor();
  auto value = inputs[3].toTensor();
  auto self_is_int32 = self.scalar_type() == c10::ScalarType::Int;
  auto alpha = inputs[4].toScalar();

  std::vector<synapse_helpers::tensor_or_ref> syn_value;
  torch::jit::Stack temp_stack;
  ////alpha_value = value * alpha;
  auto mulOp = make_operator<MulOperator>(
      this->p_context_->device_id_, value.scalar_type());
  temp_stack = {IValue(value), IValue(alpha)};
  mulOp->SetSynapseInput(p_context_->syn_inputs_[2]);
  mulOp->AllocateAndAddSynapseNode(graph, temp_stack, OutputMetaDataVector(1));
  temp_stack.clear();
  auto pt_index_into_reshape = index;
  syn_value.push_back(std::move(mulOp->GetSynOutputs()[0]));
  auto pt_value = mulOp->GetOutputs()[0];

  // TPC mandates indices (and hence the source as well) to be sorted for
  // scatter_add_fwd guid to work accurately. So sort the indices using topk and
  // based on sorted indices re order the source using gather.

  // TODO: if Gaudi2 can support scatter_add without sorting, add a check and
  // avoid sorting on G2

  // Create cast operator for indices->Float = node_type = "cast_i32_to_f32" for
  // topk
  auto castIndicesToFloatOp = make_operator<CastOperator>(
      this->p_context_->device_id_, "cast_i32_to_f32");
  castIndicesToFloatOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  c10::ScalarType cast_scalar_type = c10::ScalarType::Float;
  std::vector<c10::IValue> cast_stack{IValue(index), IValue(cast_scalar_type)};
  castIndicesToFloatOp->AllocateAndAddSynapseNode(
      graph, cast_stack, OutputMetaDataVector(1));
  cast_stack.clear();
  // Node: topk_idx = at::topk(cast_indices_pt_tensor, numel);
  auto topkOp =
      make_operator<TopkOperator>(this->p_context_->device_id_, "topk");
  int64_t topk_dim = 0;
  bool largest = true;
  bool sorted = true;
  topkOp->SetSynapseInput(castIndicesToFloatOp->GetSynOutputs()[0]);
  std::vector<c10::IValue> topk_stack{
      IValue(castIndicesToFloatOp->GetOutputs()[0]),
      IValue(index.numel()),
      IValue(topk_dim),
      IValue(largest),
      IValue(sorted)};
  topkOp->AllocateAndAddSynapseNode(graph, topk_stack, OutputMetaDataVector(2));
  // output[0] -> topk_values
  // output[1] -> topk_indices
  // Create cast operator for topk_values = node_type = "cast_f32_to_i32"
  auto castTopkValsOp = make_operator<CastOperator>(
      this->p_context_->device_id_, "cast_f32_to_i32");
  castTopkValsOp->SetSynapseInput(topkOp->GetSynOutputs()[0]);
  cast_scalar_type = c10::ScalarType::Int;
  std::vector<c10::IValue> cast_stack1{
      IValue(topkOp->GetOutputs()[0]), IValue(cast_scalar_type)};
  castTopkValsOp->AllocateAndAddSynapseNode(
      graph, cast_stack1, OutputMetaDataVector(1));

  pt_index_into_reshape = castTopkValsOp->GetOutputs()[0];
  // Node: reordered source(i.e, alpha_value tensor) =
  // at::gather(alpha_value, 0, topk_indices);
  auto gatherOp = make_operator<GatherOperator>(
      this->p_context_->device_id_, value.scalar_type());
  gatherOp->SetSynapseInput(syn_value[0]);
  gatherOp->SetSynapseInput(topkOp->GetSynOutputs()[1]);
  bool sparse_grad = false;
  std::vector<c10::IValue> gather_stack{
      IValue(mulOp->GetOutputs()[0]),
      IValue(dim),
      IValue(topkOp->GetOutputs()[1]),
      IValue(sparse_grad)};
  gatherOp->AllocateAndAddSynapseNode(
      graph, gather_stack, OutputMetaDataVector(1));
  pt_value = gatherOp->GetOutputs()[0];

  // Expand 1D index tensor to same number of dimensions as value tensor
  auto expanded_sizes = std::vector<int64_t>(value.ndimension(), 1);
  expanded_sizes[dim] = index.sizes()[0];
  ////auto index_expanded = index.view(expanded_sizes)
  auto reshapeOp = make_operator<ReshapeOperator>(
      this->p_context_->device_id_, index.scalar_type());
  temp_stack = {IValue(pt_index_into_reshape), IValue(expanded_sizes)};
  reshapeOp->SetSynapseInput(castTopkValsOp->GetSynOutputs()[0]);
  reshapeOp->AllocateAndAddSynapseNode(
      graph, temp_stack, OutputMetaDataVector(1));
  temp_stack.clear();

  // Broadcast index tensor to same shape as value tensor
  bool implicit =
      false; // The value of implicit is currently ignored in broadcast kernel
  auto bcastOp = make_operator<BroadcastOperator>(
      this->p_context_->device_id_, reshapeOp->GetOutputs()[0].scalar_type());
  temp_stack = {
      IValue(reshapeOp->GetOutputs()[0]),
      IValue(value.sizes()),
      IValue(implicit)};
  bcastOp->SetSynapseInput(reshapeOp->GetSynOutputs()[0]);
  bcastOp->AllocateAndAddSynapseNode(
      graph, temp_stack, OutputMetaDataVector(1));
  temp_stack.clear();
  // TPC does not support int for scatter_add;
  // Cast self and alpha_value to float
  if (self_is_int32) {
    // cast self
    c10::ScalarType cast_scalar_type = c10::ScalarType::Float;
    auto castSelfToFloatOp = make_operator<CastOperator>(
        this->p_context_->device_id_, "cast_i32_to_f32");
    castSelfToFloatOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    std::vector<c10::IValue> cast_stack{IValue(self), IValue(cast_scalar_type)};
    castSelfToFloatOp->AllocateAndAddSynapseNode(
        graph, cast_stack, OutputMetaDataVector(1));
    cast_stack.clear();
    // cast value
    auto castValueToFloatOp = make_operator<CastOperator>(
        this->p_context_->device_id_, "cast_i32_to_f32");
    castValueToFloatOp->SetSynapseInput(gatherOp->GetSynOutputs()[0]);
    cast_stack = {IValue(pt_value), IValue(cast_scalar_type)};
    castValueToFloatOp->AllocateAndAddSynapseNode(
        graph, cast_stack, OutputMetaDataVector(1));
    cast_stack.clear();

    // scatter_add(self, dim, index_broadcast, alpha_value);
    auto scatterAddOp = make_operator<ScatterAddOperator>(
        this->p_context_->device_id_, c10::ScalarType::Float);
    temp_stack = {
        IValue(castSelfToFloatOp->GetOutputs()[0]),
        IValue(dim),
        IValue(bcastOp->GetOutputs()[0]),
        IValue(castValueToFloatOp->GetOutputs()[0])};

    scatterAddOp->SetSynapseInput(castSelfToFloatOp->GetSynOutputs()[0]);
    scatterAddOp->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    scatterAddOp->SetSynapseInput(castValueToFloatOp->GetSynOutputs()[0]);
    scatterAddOp->AllocateAndAddSynapseNode(
        graph, temp_stack, OutputMetaDataVector(1));
    // cast self back to int
    cast_scalar_type = c10::ScalarType::Int;
    auto castSelfToIntOp = make_operator<CastOperator>(
        this->p_context_->device_id_, "cast_f32_to_i32");
    castSelfToIntOp->SetSynapseInput(scatterAddOp->GetSynOutputs()[0]);
    cast_stack = {
        IValue(scatterAddOp->GetOutputs()[0]), IValue(cast_scalar_type)};
    castSelfToIntOp->AllocateAndAddSynapseNode(
        graph, cast_stack, output_metadata);

    p_context_->syn_outputs_.emplace_back(
        std::move(castSelfToIntOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(castSelfToIntOp->GetOutputs()[0]));

  } else {
    // scatter_add(self, dim, index_broadcast, alpha_value);
    auto scatterAddOp = make_operator<ScatterAddOperator>(
        this->p_context_->device_id_, self.scalar_type());
    temp_stack = {
        IValue(self),
        IValue(dim),
        IValue(bcastOp->GetOutputs()[0]),
        IValue(pt_value)};

    scatterAddOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    scatterAddOp->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    scatterAddOp->SetSynapseInput(gatherOp->GetSynOutputs()[0]);
    scatterAddOp->AllocateAndAddSynapseNode(graph, temp_stack, output_metadata);

    p_context_->syn_outputs_.emplace_back(
        std::move(scatterAddOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(scatterAddOp->GetOutputs()[0]));
  }
}

// brodcast index tensor shape and get the correct shape and size
std::vector<int64_t> IndexPutOperator::broadcast_size(at::TensorList indices) {
  auto size = indices[0].sizes().vec();
  for (size_t i = 1; i < indices.size(); i++) {
    size = infer_size(size, indices[i].sizes());
  }
  return size;
}

void IndexPutOperator::AllocateAndAddSynapseNodeBoolIndices(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto self = inputs[0].toTensor();
  auto indices = inputs[1].toTensorList().vec();
  auto values = inputs[2].toTensor();
  auto accumulate = inputs[3].toBool();

  auto max_size = broadcast_size(indices);
  auto device_id = this->p_context_->device_id_;
  Stack stack;

  auto non_zero_op =
      make_operator<NonZeroOperator>(device_id, c10::ScalarType::Bool);
  stack = {IValue(indices[0]), IValue{c10::nullopt}};
  non_zero_op->SetSynapseInput(p_context_->syn_inputs_[1]);

  non_zero_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(2));
  stack.clear();

  // Calculate the dimensionality of updates for broadcasting
  auto rank_inp = self.ndimension();
  auto rank_idx = non_zero_op->GetOutputs()[0].sizes().vec()[1];
  std::vector<int64_t> value_upd_dim;

  if (accumulate ||
      (values.numel() >
       1)) { // if values has more than 1 elem, we have to assume the valid
             // count in indices will match values numel
    if (indices[0].dim() != self.dim() &&
        values.dim() != (1 + (self.dim() - indices[0].dim()))) {
      value_upd_dim.push_back(non_zero_op->GetOutputs()[0].sizes().vec()[0]);
      for (int i = rank_idx; i < rank_inp; i++)
        value_upd_dim.push_back(self.sizes().vec()[i]);
    } else {
      for (int i = 0; i < values.dim(); i++)
        value_upd_dim.push_back(values.sizes().vec()[i]);
    }
  } else { // We are assuming uses passes value shapes correctly for scatter
    value_upd_dim.push_back(non_zero_op->GetOutputs()[0].sizes().vec()[0]);
    for (int i = rank_idx; i < rank_inp; i++)
      value_upd_dim.push_back(self.sizes().vec()[i]);
  }

  auto values_scalar_type = values.scalar_type();
  auto bcastOp =
      make_operator<BroadcastOperator>(device_id, values_scalar_type);
  stack = {IValue(values), IValue(value_upd_dim), IValue(false)};
  bcastOp->SetSynapseInput(p_context_->syn_inputs_[indices.size() + 1]);

  bcastOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

  auto broadcasted_values = bcastOp->GetOutputs()[0];
  stack.clear();

  std::shared_ptr<HabanaOperator> scatter_op;
  auto self_scalar_type = self.scalar_type();
  scatter_op =
      make_operator<ScatterNdONNXOperator>(device_id, self_scalar_type);

  if (!accumulate) {
    stack = {
        IValue(self),
        IValue(non_zero_op->GetOutputs()[0]),
        IValue(bcastOp->GetOutputs()[0]),
        IValue(non_zero_op->GetOutputs()[1])};

    scatter_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    scatter_op->SetSynapseInput(non_zero_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(non_zero_op->GetSynOutputs()[1]);

    scatter_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();
    p_context_->syn_outputs_.emplace_back(
        std::move(scatter_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(scatter_op->GetOutputs()[0]));
  } else if (
      self.scalar_type() == c10::ScalarType::Bool ||
      self.scalar_type() == c10::ScalarType::Char) {
    std::shared_ptr<HabanaOperator> castOp1;
    std::shared_ptr<HabanaOperator> castOp2;
    std::string node1_type = "cast_i8_to_i16";
    self_scalar_type = c10::ScalarType::Short;

    castOp1 =
        make_operator<CastOperator>(this->p_context_->device_id_, node1_type);
    castOp1->SetSynapseInput(p_context_->syn_inputs_[0]);
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    auto md = OutputMetaDataVector(1);
    md[0].dtype = stack[1].toScalarType();
    castOp1->AllocateAndAddSynapseNode(graph, stack, md);
    stack.clear();

    node1_type = "cast_i8_to_i16";
    castOp2 =
        make_operator<CastOperator>(this->p_context_->device_id_, node1_type);
    castOp2->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(broadcasted_values));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    md[0].dtype = stack[1].toScalarType();
    castOp2->AllocateAndAddSynapseNode(graph, stack, md);
    stack.clear();

    auto zero_op = make_operator<ConstantOperator>(device_id, self_scalar_type);
    auto zero_t = habana::createPTTensor(
        self,
        self.sizes().vec(),
        self.options(),
        at::MemoryFormat::Contiguous,
        false);

    stack = {IValue(zero_t), IValue(0)};
    zero_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
    stack = {
        IValue(zero_t),
        IValue(non_zero_op->GetOutputs()[0]),
        IValue(broadcasted_values),
        IValue(non_zero_op->GetOutputs()[1])};

    scatter_op->SetSynapseInput(zero_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(non_zero_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(non_zero_op->GetSynOutputs()[1]);

    scatter_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto add_op = make_operator<AddOperator>(device_id, self_scalar_type);
    stack = {IValue(self), IValue(scatter_op->GetOutputs()[0]), IValue(1.0)};
    add_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    add_op->SetSynapseInput(scatter_op->GetSynOutputs()[0]);

    add_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto out_type = self.scalar_type();
    node1_type = "cast_i16_to_i8";
    std::shared_ptr<HabanaOperator> castOpOut =
        make_operator<CastOperator>(this->p_context_->device_id_, node1_type);
    castOpOut->SetSynapseInput(add_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(add_op->GetOutputs()[0]));
    stack.emplace_back(IValue(out_type));
    castOpOut->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();
    p_context_->syn_outputs_.emplace_back(
        std::move(castOpOut->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(castOpOut->GetOutputs()[0]));
  } else {
    auto zero_op = make_operator<ConstantOperator>(device_id, self_scalar_type);
    auto zero_t = habana::createPTTensor(
        self,
        self.sizes().vec(),
        self.options(),
        at::MemoryFormat::Contiguous,
        false);

    stack = {IValue(zero_t), IValue(0)};
    zero_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
    scatter_op =
        make_operator<ScatterNdONNXOperator>(device_id, self_scalar_type);
    stack = {
        IValue(zero_t),
        IValue(non_zero_op->GetOutputs()[0]),
        IValue(broadcasted_values),
        IValue(non_zero_op->GetOutputs()[1])};

    scatter_op->SetSynapseInput(zero_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(non_zero_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(non_zero_op->GetSynOutputs()[1]);

    scatter_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto add_op = make_operator<AddOperator>(device_id, self_scalar_type);
    stack = {IValue(self), IValue(scatter_op->GetOutputs()[0]), IValue(1)};
    add_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    add_op->SetSynapseInput(scatter_op->GetSynOutputs()[0]);

    add_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();
    p_context_->syn_outputs_.emplace_back(
        std::move(add_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(add_op->GetOutputs()[0]));
  }
  return;
}

void IndexPutOperator::AllocateAndAddSynapseNodeNonBoolIndices(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto self = inputs[0].toTensor();
  auto indices = inputs[1].toTensorList().vec();
  auto values = inputs[2].toTensor();
  auto accumulate = inputs[3].toBool();

  auto max_size = broadcast_size(indices);
  auto device_id = this->p_context_->device_id_;
  auto indices_scalar_type = indices[0].scalar_type();
  std::vector<Tensor> cat_input;
  auto pre_cat_op = make_operator<CatOperator>(device_id, indices_scalar_type);
  Stack stack;
  for (size_t i = 0; i < indices.size(); i++) {
    // broadcast index tensor to largest index tensor size
    auto bcastOp =
        make_operator<BroadcastOperator>(device_id, indices_scalar_type);
    stack = {IValue(indices[i]), IValue(max_size), IValue(false)};
    bcastOp->SetSynapseInput(p_context_->syn_inputs_[i + 1]);
    bcastOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

    stack.clear();

    // auto shape_broadcasted = bcastOp->GetOuptuts()[0].sizes().vec();
    // Reshape broadcasted indices to [N, 1] for concatenation
    auto flattened_size = std::accumulate(
        std::begin(max_size), std::end(max_size), 1, std::multiplies<size_t>());

    std::vector<int64_t> expanded_size = {flattened_size, 1};
    stack.clear();

    auto ReshapeOp =
        make_operator<ReshapeOperator>(device_id, indices_scalar_type);
    stack = {IValue(bcastOp->GetOutputs()[0]), IValue(expanded_size)};
    ReshapeOp->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    ReshapeOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

    cat_input.emplace_back(std::move(ReshapeOp->GetOutputs()[0]));
    pre_cat_op->SetSynapseInput(ReshapeOp->GetSynOutputs()[0]);
  }

  // Create index tensor of shape [num_updates, dimensionality of indices]
  // Handle conversion of negative indices to positive indices since some
  // TPC guids like 'gather' doesn't support negative indexing as of now.
  // positive_index_tensor = (input_index_tensor +
  // numel(self_sizes[indexing_dim])) % numel(self_sizes[indexing_dim])
  std::shared_ptr<HabanaOperator> concatinated_pos_indices_op;
  stack = {IValue(cat_input), IValue(-1)};
  if (accumulate && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_NEGATIVE_INDEXING)) {
    pre_cat_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    auto pre_concatenated_indices = pre_cat_op->GetOutputs()[0];
    auto pre_concatenated_indices_shape =
        pre_concatenated_indices.sizes().vec();
    stack.clear();
    // Compute const factor for each dimension for handling negative indices
    // based on corresponding dim size of self tensor
    auto self_shape = self.sizes().vec();
    std::vector<int> const_factor_v;
    for (size_t i = 0; i < indices.size(); i++)
      const_factor_v.push_back(self_shape[i]);

    std::vector<Tensor> cat_input_neg_ind;
    auto neg_to_pos_const_constructor_op =
        make_operator<CatOperator>(device_id, indices_scalar_type);

    for (size_t i = 0; i < const_factor_v.size(); i++) {
      auto constOp =
          make_operator<ConstantOperator>(device_id, indices_scalar_type);
      auto const_shape_tensor = habana::createPTTensor(
          indices[0],
          {pre_concatenated_indices_shape[0], 1},
          indices[0].options(),
          at::MemoryFormat::Contiguous,
          false);
      Stack stack = {IValue(const_shape_tensor), IValue(const_factor_v[i])};
      constOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();
      cat_input_neg_ind.emplace_back(std::move(constOp->GetOutputs()[0]));
      neg_to_pos_const_constructor_op->SetSynapseInput(
          constOp->GetSynOutputs()[0]);
    }

    stack = {IValue(cat_input_neg_ind), IValue(-1)};
    neg_to_pos_const_constructor_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto add_to_pre_concat_ind_op =
        make_operator<AddOperator>(device_id, indices_scalar_type);
    stack = {
        IValue(pre_concatenated_indices),
        IValue(neg_to_pos_const_constructor_op->GetOutputs()[0]),
        IValue(1)};
    add_to_pre_concat_ind_op->SetSynapseInput(pre_cat_op->GetSynOutputs()[0]);
    add_to_pre_concat_ind_op->SetSynapseInput(
        neg_to_pos_const_constructor_op->GetSynOutputs()[0]);

    add_to_pre_concat_ind_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();
    concatinated_pos_indices_op =
        make_operator<RemainderOperator>(device_id, indices_scalar_type);
    stack = {
        IValue(add_to_pre_concat_ind_op->GetOutputs()[0]),
        IValue(neg_to_pos_const_constructor_op->GetOutputs()[0]),
        IValue(1)};
    concatinated_pos_indices_op->SetSynapseInput(
        add_to_pre_concat_ind_op->GetSynOutputs()[0]);
    concatinated_pos_indices_op->SetSynapseInput(
        neg_to_pos_const_constructor_op->GetSynOutputs()[0]);
  } else {
    concatinated_pos_indices_op = pre_cat_op;
  }
  concatinated_pos_indices_op->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  auto concatenated_indices = concatinated_pos_indices_op->GetOutputs()[0];
  stack.clear();

  // Calculate the dimensionality of updates for broadcasting
  auto rank_inp = self.ndimension();
  auto rank_idx = concatenated_indices.sizes().vec()[1];
  std::vector<int64_t> value_upd_dim{concatenated_indices.sizes().vec()[0]};

  if (((int)indices.size() == self.dim()) && (values.numel() > 1)) {
    value_upd_dim.clear();
    value_upd_dim = values.sizes().vec();
  }

  for (int i = rank_idx; i < rank_inp; i++)
    value_upd_dim.push_back(self.sizes().vec()[i]);
  auto values_scalar_type = values.scalar_type();
  // value_upd_dim is the final shape we want for values tensor to match
  // scatter_nd_onnx requirements. Either broadcast of reshape input values
  // tensor to get that shape.
  std::shared_ptr<HabanaOperator> values_bcast_or_reshape_op;
  if (values.dim() <= static_cast<int>(value_upd_dim.size())) {
    values_bcast_or_reshape_op =
        make_operator<BroadcastOperator>(device_id, values_scalar_type);
    stack = {IValue(values), IValue(value_upd_dim), IValue(false)};
  } else {
    values_bcast_or_reshape_op =
        make_operator<ReshapeOperator>(device_id, values_scalar_type);
    stack = {IValue(values), IValue(value_upd_dim)};
  }

  values_bcast_or_reshape_op->SetSynapseInput(
      p_context_->syn_inputs_[indices.size() + 1]);
  values_bcast_or_reshape_op->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  stack.clear();

  std::shared_ptr<HabanaOperator> reshape_or_identity_op;
  if ((int)indices.size() == self.dim()) {
    reshape_or_identity_op =
        make_operator<ReshapeOperator>(device_id, values_scalar_type);
    std::vector<int64_t> reshape_bcast_size(
        {concatenated_indices.sizes().vec()[0]});
    stack = {
        IValue(values_bcast_or_reshape_op->GetOutputs()[0]),
        IValue(reshape_bcast_size)};

    reshape_or_identity_op->SetSynapseInput(
        values_bcast_or_reshape_op->GetSynOutputs()[0]);
    reshape_or_identity_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();
  } else {
    reshape_or_identity_op = values_bcast_or_reshape_op;
  }
  std::shared_ptr<HabanaOperator> scatter_op;
  auto self_scalar_type = self.scalar_type();

  if (!accumulate) {
    scatter_op =
        make_operator<ScatterNdONNXOperator>(device_id, self_scalar_type);
    stack = {
        IValue(self),
        IValue(concatenated_indices),
        IValue(reshape_or_identity_op->GetOutputs()[0])};
    scatter_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    scatter_op->SetSynapseInput(
        concatinated_pos_indices_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(reshape_or_identity_op->GetSynOutputs()[0]);

    scatter_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();
    p_context_->syn_outputs_.emplace_back(
        std::move(scatter_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(scatter_op->GetOutputs()[0]));
  } else {
    // Convert indices to values (ravelling indices) for sorting
    std::vector<int64_t> indices_shape;
    for (int i = 0; i < concatenated_indices.sizes().vec()[1]; i++)
      indices_shape.push_back(self.sizes().vec()[i]);

    // Compute multiplication factor for each dimension
    std::vector<int> mul_factor_v{1};
    for (size_t i = 0; i < indices_shape.size() - 1; i++)
      mul_factor_v.push_back(mul_factor_v[i] * indices_shape[i]);

    // auto mul_factor = torch::from_blob(
    //     mul_factor_v.data(), {1, int64_t(mul_factor_v.size())}, torch::kInt);
    // auto multiplied_indices = at::mul(concatenated_indices, mul_factor);
    std::vector<Tensor> cat_input2;
    auto cat_op2 = make_operator<CatOperator>(device_id, indices_scalar_type);

    for (size_t i = 0; i < mul_factor_v.size(); i++) {
      auto constOp =
          make_operator<ConstantOperator>(device_id, indices_scalar_type);
      auto const_shape_tensor = habana::createPTTensor(
          indices[0],
          {1},
          indices[0].options(),
          at::MemoryFormat::Contiguous,
          false);
      Stack stack = {IValue(const_shape_tensor), IValue(mul_factor_v[i])};
      constOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();
      cat_input2.emplace_back(std::move(constOp->GetOutputs()[0]));
      cat_op2->SetSynapseInput(constOp->GetSynOutputs()[0]);
    }

    stack = {IValue(cat_input2), IValue(-1)};
    cat_op2->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    std::vector<int64_t> reshape_size({1, (int64_t)mul_factor_v.size()});
    auto ReshapeOp2 =
        make_operator<ReshapeOperator>(device_id, indices_scalar_type);
    stack = {IValue(cat_op2->GetOutputs()[0]), IValue(reshape_size)};
    ReshapeOp2->SetSynapseInput(cat_op2->GetSynOutputs()[0]);
    ReshapeOp2->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();
    auto mul_factor_const_t = ReshapeOp2->GetOutputs()[0];
    auto mul_op = make_operator<MulOperator>(device_id, indices_scalar_type);
    stack = {
        IValue(concatinated_pos_indices_op->GetOutputs()[0]),
        IValue(mul_factor_const_t)};

    mul_op->SetSynapseInput(concatinated_pos_indices_op->GetSynOutputs()[0]);
    mul_op->SetSynapseInput(
        ReshapeOp2->GetSynOutputs()[0]); // const_tensor for mul_factor

    mul_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
    std::vector<int64_t> dim_arr({1});
    auto dtype = mul_op->GetOutputs()[0].scalar_type();
    auto sum_op = make_operator<SumDimOperator>(device_id, dtype);
    stack = {
        IValue(mul_op->GetOutputs()[0]),
        IValue(dim_arr),
        IValue(false),
        IValue(dtype)};
    sum_op->SetSynapseInput(mul_op->GetSynOutputs()[0]);
    sum_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto ravelled_indices = sum_op->GetOutputs()[0];
    auto sort_op = make_operator<TopkOperator>(device_id, "topk");
    sort_op->SetSynapseInput(sum_op->GetSynOutputs()[0]);
    stack = {
        IValue(ravelled_indices),
        IValue(ravelled_indices.sizes()[0]),
        IValue(ravelled_indices.dim() - 1),
        IValue(true),
        IValue(true)};
    sort_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(2));
    stack.clear();

    auto sorted_results = sort_op->GetOutputs()[0];
    auto permutation = sort_op->GetOutputs()[1].to(torch::kInt);
    auto index_select_op =
        make_operator<IndexSelectOperator>(device_id, indices_scalar_type);
    //  auto grouped_indices =
    //     at::index_select(concatenated_indices, 0, permutation);
    stack = {IValue(concatenated_indices), IValue(0), IValue(permutation)};
    index_select_op->SetSynapseInput(
        concatinated_pos_indices_op->GetSynOutputs()[0]);
    index_select_op->SetSynapseInput(sort_op->GetSynOutputs()[1]);
    index_select_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();
    std::vector<int64_t> reshape_size2({permutation.sizes().vec()[0], 1});
    // auto update_locs =
    //     at::reshape(permutation, {permutation.sizes().vec()[0], 1});
    auto reshape_op =
        make_operator<ReshapeOperator>(device_id, indices_scalar_type);
    stack = {IValue(sort_op->GetOutputs()[1]), IValue(reshape_size2)};

    reshape_op->SetSynapseInput(sort_op->GetSynOutputs()[1]);
    reshape_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();
    // auto permutation = Reshape_op->GetOutputs()[0];
    scatter_op = make_operator<ScatterNdOperator>(device_id, self_scalar_type);

    stack = {
        IValue(self),
        IValue(concatenated_indices),
        IValue(index_select_op->GetOutputs()[0]),
        IValue(reshape_op->GetOutputs()[0]),
        IValue(reshape_or_identity_op->GetOutputs()[0])};

    scatter_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    scatter_op->SetSynapseInput(
        concatinated_pos_indices_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(index_select_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(reshape_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(reshape_or_identity_op->GetSynOutputs()[0]);

    scatter_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();
    auto add_op = make_operator<AddOperator>(device_id, self_scalar_type);
    stack = {IValue(self), IValue(scatter_op->GetOutputs()[0]), IValue(1)};
    add_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    add_op->SetSynapseInput(scatter_op->GetSynOutputs()[0]);

    add_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();
    p_context_->syn_outputs_.emplace_back(
        std::move(add_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(add_op->GetOutputs()[0]));
  }
  return;
}

void IndexPutOperator2::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto self = inputs[0].toTensor();
  auto where_tensor = inputs[1].toTensor();
  auto shape_tensor = inputs[2].toTensor();
  auto values = inputs[3].toTensor();
  auto value_dim_tensor = inputs[4].toTensor();
  auto zero_shape_tensor = inputs[5].toTensor();
  auto accumulate = inputs[6].toBool();

  auto device_id = this->p_context_->device_id_;
  Stack stack;
  auto values_scalar_type = values.scalar_type();
  auto bcastOp =
      make_operator<BroadcastOperator>(device_id, values_scalar_type);
  stack = {IValue(values), IValue(value_dim_tensor), IValue(false)};
  bcastOp->SetSynapseInput(p_context_->syn_inputs_[3]);
  bcastOp->SetSynapseInput(p_context_->syn_inputs_[4]);

  bcastOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

  auto broadcasted_values = bcastOp->GetOutputs()[0];
  stack.clear();

  std::shared_ptr<HabanaOperator> scatter_op;
  auto self_scalar_type = self.scalar_type();
  scatter_op =
      make_operator<ScatterNdONNXOperator>(device_id, self_scalar_type);

  if (!accumulate) {
    stack = {
        IValue(self),
        IValue(where_tensor),
        IValue(bcastOp->GetOutputs()[0]),
        IValue(shape_tensor)};

    scatter_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    scatter_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    scatter_op->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(p_context_->syn_inputs_[2]);

    scatter_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();
    p_context_->syn_outputs_.emplace_back(
        std::move(scatter_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(scatter_op->GetOutputs()[0]));
  } else if (
      self.scalar_type() == c10::ScalarType::Bool ||
      self.scalar_type() == c10::ScalarType::Char) {
    std::shared_ptr<HabanaOperator> castOp1;
    std::shared_ptr<HabanaOperator> castOp2;
    std::string node1_type = "cast_i8_to_i16";
    self_scalar_type = c10::ScalarType::Short;

    castOp1 =
        make_operator<CastOperator>(this->p_context_->device_id_, node1_type);
    castOp1->SetSynapseInput(p_context_->syn_inputs_[0]);
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    auto md = OutputMetaDataVector(1);
    md[0].dtype = stack[1].toScalarType();
    castOp1->AllocateAndAddSynapseNode(graph, stack, md);
    stack.clear();

    node1_type = "cast_i8_to_i16";
    castOp2 =
        make_operator<CastOperator>(this->p_context_->device_id_, node1_type);
    castOp2->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(broadcasted_values));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    md[0].dtype = stack[1].toScalarType();
    castOp2->AllocateAndAddSynapseNode(graph, stack, md);
    stack.clear();

    auto zero_op = make_operator<ConstantOperator>(device_id, self_scalar_type);
    stack = {IValue(zero_shape_tensor), IValue(0)};
    zero_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
    stack = {
        IValue(zero_op->GetOutputs()[0]),
        IValue(where_tensor),
        IValue(broadcasted_values),
        IValue(shape_tensor)};

    scatter_op->SetSynapseInput(zero_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    scatter_op->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(p_context_->syn_inputs_[2]);

    scatter_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto add_op = make_operator<AddOperator>(device_id, self_scalar_type);
    stack = {IValue(self), IValue(scatter_op->GetOutputs()[0]), IValue(1.0)};
    add_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    add_op->SetSynapseInput(scatter_op->GetSynOutputs()[0]);

    add_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto out_type = self.scalar_type();
    node1_type = "cast_i16_to_i8";
    std::shared_ptr<HabanaOperator> castOpOut =
        make_operator<CastOperator>(this->p_context_->device_id_, node1_type);
    castOpOut->SetSynapseInput(add_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(add_op->GetOutputs()[0]));
    stack.emplace_back(IValue(out_type));
    castOpOut->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();
    p_context_->syn_outputs_.emplace_back(
        std::move(castOpOut->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(castOpOut->GetOutputs()[0]));
  } else {
    auto zero_op = make_operator<ConstantOperator>(device_id, self_scalar_type);
    stack = {IValue(zero_shape_tensor), IValue(0)};
    zero_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
    scatter_op =
        make_operator<ScatterNdONNXOperator>(device_id, self_scalar_type);
    stack = {
        IValue(zero_op->GetOutputs()[0]),
        IValue(where_tensor),
        IValue(broadcasted_values),
        IValue(shape_tensor)};
    scatter_op->SetSynapseInput(zero_op->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    scatter_op->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    scatter_op->SetSynapseInput(p_context_->syn_inputs_[2]);

    scatter_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto add_op = make_operator<AddOperator>(device_id, self_scalar_type);
    stack = {IValue(self), IValue(scatter_op->GetOutputs()[0]), IValue(1)};
    add_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    add_op->SetSynapseInput(scatter_op->GetSynOutputs()[0]);

    add_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();
    p_context_->syn_outputs_.emplace_back(
        std::move(add_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(add_op->GetOutputs()[0]));
  }
  return;
}

void IndexPutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      (inputs.size() == 4 || inputs.size() == 5),
      "Incorrect size of inputs for index_put operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input 0 type expected to be Tensor for index_put operator");
  TORCH_CHECK(
      (inputs[1].isTensorList() || inputs[1].isOptionalTensorList()),
      "Input 1 type expected to be TensorList for index_put operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input 2 type expected to be Tensor for index_put operator");
  TORCH_CHECK(
      inputs[3].isBool(),
      "Input 3 type expected to be Bool for index_put operator");

  std::vector<at::Tensor> indices;

  // This is an incomplete change - but shows we can recieve list of optional
  // tensors via JIT The actual handling for such cases is not done here and
  // will fail
  if (inputs[1].isOptionalTensorList()) {
    PT_KERNEL_DEBUG("index_put: received list of optional tensors");
    auto opt_tensorlist_args = inputs[1].toOptionalTensorList();
    for (c10::optional<at::Tensor> input_ind : opt_tensorlist_args) {
      auto input = input_ind.value_or(at::Tensor());
      if (input.defined()) {
        PT_KERNEL_DEBUG(
            "indices tensor: ", input.scalar_type(), " size = ", input.sizes());
        indices.push_back(input);
      } else {
        PT_KERNEL_DEBUG("undefined indices tensor");
        HABANA_ASSERT(
            0 &&
            "index_put: unsupported case: None is not yet supported on HPU for c10::List<c10::optional<Tensor>>");
      }
    }
    HABANA_ASSERT(0, "index_put: OptionalTensorList is not handled in kernel");
  } else {
    indices = inputs[1].toTensorList().vec();
  }

  if (indices[0].scalar_type() == c10::ScalarType::Bool) {
    AllocateAndAddSynapseNodeBoolIndices(graph, inputs, output_metadata);
  } else {
    AllocateAndAddSynapseNodeNonBoolIndices(graph, inputs, output_metadata);
  }
}

bool ScatterNdONNXOperator::isInputValid(Stack& inputs) {
  auto inpSize = inputs[0].toTensor().sizes().vec();
  auto indxSize = inputs[1].toTensor().sizes().vec();
  int indxRank = indxSize.size();
  int indxFCD = indxSize[indxRank - 1];
  int64_t totalIndices = 1;
  int64_t totalScatters = 1;

  for (int i = 0; i < indxRank - 1; i++) {
    totalIndices *= std::max(indxSize[i], 1L);
  }

  for (int i = 0; i < indxFCD; i++) {
    totalScatters *= std::max(inpSize[i], 1L);
  }

  if (totalIndices > totalScatters) {
    return false;
  }

  return true;
}

habana::InferOutputMetaRetType ScatterNdONNXOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto inp = inputs[0].toTensor();
  auto shape_out = inp.sizes().vec();

  InferOutputMetaRetType out;
  // output tensor
  out.AddOutputTensor(TensorMetaData(
      shape_out,
      HabanaOperator::CalculateStrides(shape_out, inp.suggest_memory_format()),
      inp.scalar_type(),
      inp.suggest_memory_format()));
  return out;
}

void ScatterNdONNXOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() >= 3,
      "Incorrect number of inputs passed to ScatterNdONNXOperator");

  auto inp = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto values = inputs[2].toTensor();

  TORCH_CHECK(
      (indices.numel() / indices.sizes().vec()[1]) <= inp.numel(),
      "number of indices should be less than of self");
  TORCH_CHECK(
      isInputValid(inputs) == true, "Invalid inputs for scatter_nd_onnx");

  auto shape = DimVector(inp.sizes());
  auto output = habana::createPTTensor(
      inp,
      shape,
      inp.options(),
      inp.suggest_memory_format(),
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

habana::InferOutputMetaRetType ScatterNdOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;

  auto inp = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto grouped_indices = inputs[2].toTensor();
  auto update_locations = inputs[3].toTensor();
  auto updates = inputs[4].toTensor();

  auto shape = inp.sizes();

  auto tensor_meta_data = TensorMetaData(
      shape.vec(),
      HabanaOperator::CalculateStrides(shape, inp.suggest_memory_format()),
      inp.scalar_type(),
      inp.suggest_memory_format());

  out.AddOutputTensor(tensor_meta_data);
  out.AddShapeTensor(tensor_meta_data);

  return out;
}

void ScatterNdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect number of inputs passed to ScatterNdOperator");

  auto inp = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto grouped_indices = inputs[2].toTensor();
  auto update_locations = inputs[3].toTensor();
  auto updates = inputs[4].toTensor();

  auto shape = DimVector(inp.sizes());
  auto output = habana::createPTTensor(
      inp,
      shape,
      inp.options(),
      inp.suggest_memory_format(),
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[2];
  synapse_helpers::tensor& arg2_syn_tensor = p_context_->syn_inputs_[3];
  synapse_helpers::tensor& arg3_syn_tensor = p_context_->syn_inputs_[4];

  std::vector<synTensor> syn_inputs;
  syn_inputs.emplace_back(arg1_syn_tensor.get());
  syn_inputs.emplace_back(arg2_syn_tensor.get());
  syn_inputs.emplace_back(arg3_syn_tensor.get());

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, output);
    synapse_helpers::tensor& shape_syn_tensor = p_context_->syn_inputs_.back();
    syn_inputs.emplace_back(shape_syn_tensor.get());
  }

  synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

  ns_ScatterNDKernel::Params params{int(indices.ndimension()), {0}};
  auto indices_shape = indices.sizes().vec();
  // For Dynamic case fill index params with max size
  if (graph.is_dynamic_graph() && (!graph.is_dry_run())) {
    synapse_helpers::tensor& syn_input_tensor = p_context_->syn_inputs_[1];
    std::vector<int64_t> min, max;
    std::tie(min, max) =
        habana::ShapeInference::GetMinMaxShape(syn_input_tensor.id());
    indices_shape = max;
  }
  // Dims reversed between PT and synapse
  for (int i = indices_shape.size() - 1, j = 0; i >= 0; --i, ++j) {
    params.origIndicesShape[j] = indices_shape[i];
  }
  p_context_->params_.emplace<ns_ScatterNDKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      &params,
      sizeof(params),
      guid_,
      nullptr,
      nullptr,
      nullptr,
      deterministic,
      getContextHints());
}

void IndexSelectOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for IndexSelect operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be Tensor for IndexSelect operator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input type expected to be Int for IndexSelect operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input type expected to be Tensor for IndexSelect operator");

  auto index = inputs[2].toTensor();
  TORCH_CHECK(index.dim() <= 1, "index tensor cannot be more than 1D")
  bool sparse_grad = false;
  inputs.emplace_back(IValue(sparse_grad));
  GatherOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  // Revert input stack
  inputs.pop_back();
}

InferOutputMetaRetType IndexSelectOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  bool sparse_grad = false;
  inputs.emplace_back(IValue(sparse_grad));
  return GatherOperator::InferOutputMeta(inputs);
}

void IndexSelectOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto index = inputs[2].toTensor();
  TORCH_CHECK(index.dim() <= 1, "index tensor cannot be more than 1D")
  bool sparse_grad = false;
  inputs.emplace_back(IValue(sparse_grad));
  GatherOperator::SetPTOutputs(inputs);
}

void NarrowOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for narrow operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isInt(), "Input arg2 type expected to be integer");
  TORCH_CHECK(inputs[2].isInt(), "Input arg3 type expected to be integer");
  TORCH_CHECK(inputs[3].isInt(), "Input arg4 type expected to be integer");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto start = inputs[2].toInt();
  auto length = inputs[3].toInt();

  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.size(dim);
  if (start != cur_size) { // start being the end is valid, but not a valid dim
                           // specification.
    start = at::maybe_wrap_dim(start, cur_size);
  }
  TORCH_CHECK(
      length >= 0 && start <= cur_size - length,
      "start (",
      start,
      ") + length (",
      length,
      ") exceeds dimension size (",
      cur_size,
      ").");

  inputs.erase(inputs.cend() - 1, inputs.cend());
  inputs.emplace_back(IValue(start + length));
  inputs.emplace_back(IValue(1));
  SliceOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}

std::vector<std::vector<int64_t>> SliceOperator::compute_output_shape(
    const std::vector<int64_t>& self_size,
    int64_t& dim,
    int64_t& start_val,
    int64_t& end_val,
    int64_t& step) {
  // reuse the logic in at::native::slice
  int64_t ndim = self_size.size();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = at::maybe_wrap_dim(dim, ndim);
  std::vector<int64_t> sizes(self_size.begin(), self_size.end());

  // TODO: support negative strides
  TORCH_CHECK(step > 0, "slice step must be positive");

  // INT64_MAX stands for default value.
  if (start_val == INT64_MAX) {
    start_val = 0;
  }
  if (start_val < 0) {
    start_val += sizes[dim];
  }
  if (end_val < 0) {
    end_val += sizes[dim];
  }
  if (start_val < 0) {
    start_val = 0;
  } else if (start_val >= sizes[dim]) {
    start_val = sizes[dim];
  }
  if (end_val < start_val) {
    end_val = start_val;
  } else if (end_val >= sizes[dim]) {
    end_val = sizes[dim];
  }

  auto len = end_val - start_val;
  sizes[dim] = (len + step - 1) / step; // round-up

  return {sizes};
}

std::vector<int64_t> SliceOperator::compute_output_shape(
    const Tensor& self,
    int64_t& dim,
    int64_t& start_val,
    int64_t& end_val,
    int64_t& step) {
  // reuse the logic in at::native::slice
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = at::maybe_wrap_dim(dim, ndim);
  std::vector<int64_t> sizes(self.sizes().begin(), self.sizes().end());

  // TODO: support negative strides
  TORCH_CHECK(step > 0, "slice step must be positive");

  // INT64_MAX stands for default value.
  if (start_val == INT64_MAX) {
    start_val = 0;
  }
  if (start_val < 0) {
    start_val += sizes[dim];
  }
  if (end_val < 0) {
    end_val += sizes[dim];
  }
  if (start_val < 0) {
    start_val = 0;
  } else if (start_val >= sizes[dim]) {
    start_val = sizes[dim];
  }
  if (end_val < start_val) {
    end_val = start_val;
  } else if (end_val >= sizes[dim]) {
    end_val = sizes[dim];
  }

  auto len = end_val - start_val;
  sizes[dim] = (len + step - 1) / step; // round-up

  return sizes;
}

Tensor SliceOperator::AllocateOutputTensor(
    const Tensor& self,
    int64_t& dim,
    int64_t& start,
    int64_t& end,
    int64_t& step,
    const OutputMetaData& output_metadata) {
  auto shape = compute_output_shape(self, dim, start, end, step);

  // allocate output tensor
  auto output = habana::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      output_metadata.persistent);

  return output;
}

void SliceOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto start = inputs[2].toInt();
  auto end = inputs[3].toInt();
  auto step = inputs[4].toInt();
  OutputMetaData md;
  md.persistent = true;
  auto output = AllocateOutputTensor(self, dim, start, end, step, md);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

void SliceOperator::ValidateSliceInputs(
    std::vector<int64_t>& inp_shape,
    std::vector<int64_t>& out_shape,
    std::vector<int64_t>& step,
    std::vector<int64_t>& start) {
  for (unsigned i = 0; i < inp_shape.size(); i++) {
    // exclude ZST from shape validation check
    if (inp_shape[i]) {
      TORCH_CHECK(
          (start[i] < inp_shape[i]),
          "Slice invalid starts param, which is greater or equal to the dimension");
    }

    // original equation as per at::native::slice
    // sizes[dim] = (end_val - start_val + step - 1) / step; // round-up

    // inverse to find end
    // end_val = sizes[dim]*step + 1 - step + start_val
    auto end_val = out_shape[i] * step[i] + 1 - step[i] + start[i];

    TORCH_CHECK(
        (end_val <= inp_shape[i]),
        "Slice invalid end param, which is greater or equal to the dimension ",
        end_val,
        " ",
        inp_shape[i]);
  }

  PT_DYNAMIC_SHAPE_DEBUG(
      "SliceOperator validated for inp_shape::",
      inp_shape,
      ", out_shape::",
      out_shape,
      ", step::",
      step,
      ", start::",
      start);
}

InferOutputMetaRetType SliceOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  std::vector<int64_t> out_shape;
  bool has_shape_tensor = inputs[2].isTensor();

  if (has_shape_tensor) {
    std::vector<int64_t> inp_shape = inputs[0].toTensor().sizes().vec();
    out_shape = inputs[1].toTensor().sizes().vec();

    if ((habana::ShapeInference::GetCurrentPass() ==
         habana::ShapeInfo::InferencePass::MAX_SHAPE) &&
        (habana::ShapeInference::GetMaxPolicyInUse() ==
         habana_helpers::DynamicDimsPolicy::CALCULATED)) {
      for (uint64_t i = 0; i < inp_shape.size(); i++) {
        out_shape[i] =
            out_shape[i] < inp_shape[i] ? out_shape[i] : inp_shape[i];
      }
    }
  } else {
    int64_t dim, start, end, step;
    dim = inputs[1].toInt();
    start = inputs[2].toInt();
    end = inputs[3].isNone() ? INT64_MAX : inputs[3].toInt();
    step = inputs[4].toInt();
    out_shape = compute_output_shape(self, dim, start, end, step);
  }

  auto metaData = TensorMetaData(
      out_shape,
      HabanaOperator::CalculateStrides(out_shape, self.suggest_memory_format()),
      self.scalar_type(),
      self.suggest_memory_format());
  InferOutputMetaRetType out;
  out.AddOutputTensor(metaData);

  if (!has_shape_tensor) {
    out.AddShapeTensor(metaData);
  }

  return out;
}

std::vector<int64_t> SliceOperator::GetH2DTensorData(
    const at::Tensor& host_tensor,
    bool is_dry_run,
    bool is_min_shape_inference) {
  auto tmeta{get_tensor_extra_meta(host_tensor)};

  void* host_ptr = nullptr;
  if (is_dry_run) {
    host_ptr = tmeta->get_compile_host_ptr();
  } else {
    host_ptr = tmeta->get_host_ptr();
  }

  size_t h2d_data_size = tmeta->get_host_size();
  if (is_min_shape_inference) {
    size_t data_size = h2d_data_size * tmeta->get_host_el_size();
    host_ptr = static_cast<char*>(host_ptr) + data_size;
  }

  std::vector<int64_t> params;
  uint64_t* h2d_data = static_cast<uint64_t*>(host_ptr);
  for (size_t i = 0; i < h2d_data_size; i++) {
    params.push_back(*h2d_data++);
  }

  return params;
}

std::vector<int64_t> SliceOperator::ComputeParamsfromH2DTensor(
    const at::Tensor& host_tensor) {
  bool is_dry_run = false;
  if (habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MIN_SHAPE ||
      habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MAX_SHAPE) {
    is_dry_run = true;
  }

  bool is_min_shape_inference = false;
  if (habana::ShapeInference::GetCurrentPass() ==
      habana::ShapeInfo::InferencePass::MIN_SHAPE) {
    is_min_shape_inference = true;
  }

  return GetH2DTensorData(host_tensor, is_dry_run, is_min_shape_inference);
}

std::vector<int64_t> SliceOperator::get_start_tensor(
    std::vector<int64_t> h2d_vec) {
  std::vector<int64_t> start(
      h2d_vec.rbegin() + SYN_MAX_TENSOR_DIM - h2d_vec[0],
      h2d_vec.rbegin() + SYN_MAX_TENSOR_DIM);
  return start;
}

std::vector<int64_t> SliceOperator::get_step_tensor(
    std::vector<int64_t> h2d_vec) {
  std::vector<int64_t> step(
      h2d_vec.rend() - h2d_vec[0] - 1, h2d_vec.rend() - 1);
  return step;
}

void SliceOperator::UpdateMaxPassSliceInputs(
    std::vector<int64_t>& inp_shape,
    std::vector<int64_t>& out_shape,
    std::vector<int64_t>& step,
    std::vector<int64_t>& start,
    std::vector<int64_t>& min,
    std::vector<int64_t>& max) {
  for (uint64_t i = 0; i < inp_shape.size(); i++) {
    out_shape[i] = out_shape[i] < inp_shape[i] ? out_shape[i] : inp_shape[i];
    /*
    (end - start)/step = output
    Assuming end max is input
    (input - start)/step = output
    input - start = output * step
    start = input - output * step
    */

    // start shape tensor is updated, so change the buckets as well if
    // start is there in bucket
    if (out_shape[i] != 0) {
      auto old_start = start[i];
      start[i] = inp_shape[i] - (out_shape[i] * step[i]);
      if (old_start != start[i]) {
        // If the newly calculated value is less than current value keep the
        // current value Since the current value is not available in
        // AllocateAndAdd, used a hack to find it from the max value
        HABANA_ASSERT(min.size() == max.size());
        if (min.size() && (min[i] != max[i])) {
          auto curr_val = max[i] /
              habana_helpers::DynamicBucketInfo::default_max_multiplier_;
          if (start[i] < curr_val) {
            start[i] = curr_val;
          }
        }
      }
    }
  }
  TORCH_CHECK(
      min <= start, "SliceOperator Start tensor min is greater than max");
}

void SliceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  auto self = inputs[0].toTensor();
  int64_t dim, start, end, step;
  std::vector<int64_t> shape;

  bool has_shape_tensor = inputs[2].isTensor();
  if (has_shape_tensor && inputs.size() == 4) {
    TORCH_CHECK(
        (inputs.size() == 4 || inputs.size() == 5),
        "Incorrect size of inputs expected for slice operator");
    TORCH_CHECK(
        p_context_->syn_inputs_[1].ref().is_shape_tensor(),
        "Synapse input2 type expected to be shape tensor");
    TORCH_CHECK(
        p_context_->syn_inputs_[2].ref().is_shape_tensor(),
        "Synapse input3 type expected to be shape tensor");
    TORCH_CHECK(
        p_context_->syn_inputs_[3].ref().is_shape_tensor(),
        "Synapse input4 type expected to be shape tensor");
    shape = p_context_->syn_inputs_[1].ref().pt_shape();
    auto inp_shape = self.sizes().vec();
    auto out_shape = inputs[1].toTensor().sizes().vec();
    auto step = inputs[2].toTensor().sizes().vec();
    auto start = inputs[3].toTensor().sizes().vec();

    if ((habana::ShapeInference::GetCurrentPass() ==
         habana::ShapeInfo::InferencePass::MAX_SHAPE) &&
        (habana::ShapeInference::GetMaxPolicyInUse() ==
         habana_helpers::DynamicDimsPolicy::CALCULATED)) {
      std::vector<int64_t> min, max;
      synapse_helpers::tensor& syn_tensor_start = p_context_->syn_inputs_[3];
      std::tie(min, max) =
          habana::ShapeInference::GetMinMaxShape(syn_tensor_start.id());

      UpdateMaxPassSliceInputs(inp_shape, out_shape, step, start, min, max);

      // Modify the start and output shape in name shape map to create valid
      // ranges
      synapse_helpers::tensor& syn_tensor_output = p_context_->syn_inputs_[1];
      habana::ShapeInference::UpdateShapeInfo(
          graph, syn_tensor_output.id(), out_shape);
      habana::ShapeInference::UpdateShapeInfo(
          graph, syn_tensor_start.id(), start);
      shape = out_shape;
    }
    ValidateSliceInputs(inp_shape, out_shape, step, start);
  } else if (has_shape_tensor && inputs.size() == 3) {
    TORCH_CHECK(
        inputs.size() == 3,
        "Incorrect size of inputs expected for slice operator");
    TORCH_CHECK(
        p_context_->syn_inputs_[1].ref().is_shape_tensor(),
        "Synapse input2 type expected to be shape tensor");
    TORCH_CHECK(
        p_context_->syn_inputs_[2].ref().is_host_to_device_tensor(),
        "Synapse input3 type expected to be host to device tensor");
    shape = p_context_->syn_inputs_[1].ref().pt_shape();
    auto inp_shape = self.sizes().vec();
    auto out_shape = inputs[1].toTensor().sizes().vec();
    auto host_tensor = inputs[2].toTensor();
    auto params_vec = ComputeParamsfromH2DTensor(host_tensor);

    std::vector<int64_t> start, step;
    start = get_start_tensor(params_vec);
    step = get_step_tensor(params_vec);

    if ((habana::ShapeInference::GetCurrentPass() ==
         habana::ShapeInfo::InferencePass::MAX_SHAPE) &&
        (habana::ShapeInference::GetMaxPolicyInUse() ==
         habana_helpers::DynamicDimsPolicy::CALCULATED)) {
      PT_DYNAMIC_SHAPE_DEBUG(
          "SliceOperator max pass before update: inp_shape::",
          inp_shape,
          ", out_shape::",
          out_shape,
          ", step::",
          step,
          ", start::",
          start);

      auto max = start;
      auto min_shape_params = GetH2DTensorData(host_tensor, true, true);
      auto min = get_start_tensor(min_shape_params);

      UpdateMaxPassSliceInputs(inp_shape, out_shape, step, start, min, max);

      // Modify the start and output shape in name shape map to create valid
      // ranges
      synapse_helpers::tensor& syn_tensor_output = p_context_->syn_inputs_[1];
      habana::ShapeInference::UpdateShapeInfo(
          graph, syn_tensor_output.id(), out_shape);
      std::vector<uint64_t> new_params_vec(
          params_vec.begin(), params_vec.end());
      std::copy(start.rbegin(), start.rend(), new_params_vec.begin() + 6);
      auto tmeta{get_tensor_extra_meta(host_tensor)};
      tmeta->set_max<uint64_t>(new_params_vec);
      shape = out_shape;
      PT_DYNAMIC_SHAPE_DEBUG(
          "SliceOperator max pass after update: inp_shape::",
          inp_shape,
          ", out_shape::",
          out_shape,
          ", step::",
          step,
          ", start::",
          start);
    }
    ValidateSliceInputs(inp_shape, out_shape, step, start);
  } else {
    TORCH_CHECK(
        (inputs.size() == 5) || (inputs.size() == 6),
        "Incorrect size of inputs expected for slice operator");
    TORCH_CHECK(inputs[1].isInt(), "Input arg2 type expected to be integer");
    TORCH_CHECK(inputs[2].isInt(), "Input arg3 type expected to be integer");
    TORCH_CHECK(
        inputs[3].isNone() || inputs[3].isInt(),
        "Input arg4 type expected to be optional integer");
    TORCH_CHECK(inputs[4].isInt(), "Input arg5 type expected to be integer");
    dim = inputs[1].toInt();
    start = inputs[2].toInt();
    end = inputs[3].isNone() ? INT64_MAX : inputs[3].toInt();
    step = inputs[4].toInt();
    shape = compute_output_shape(self, dim, start, end, step);
  }

  Tensor output = habana::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  if (has_shape_tensor) {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      AllocateSynapseShapeTensor(graph, output);
    }
    synSliceParamsV2 params;
    // set defaults
    std::fill_n(params.axes, HABANA_DIM_MAX, 0);
    std::fill_n(params.starts, HABANA_DIM_MAX, 0);
    std::fill_n(params.ends, HABANA_DIM_MAX, 0);
    std::fill_n(params.steps, HABANA_DIM_MAX, 1);
    // slice triggered only on 1 dim, therefore use only index 0
    params.axes[0] = get_dim_in_tpc_order(dim, self.dim());
    params.starts[0] = start;
    params.ends[0] = end;
    params.steps[0] = step;

    bool needs_params_handling = false;
    if (graph.is_dynamic_graph() && (!graph.is_dry_run()) &&
        end > self.sizes().vec()[dim]) {
      needs_params_handling = true;
    }

    if (needs_params_handling) {
      synapse_helpers::tensor& syn_input_tensor = p_context_->syn_inputs_[0];
      auto tensor_id = syn_input_tensor.id();
      std::vector<int64_t> min, max;
      std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
      params.ends[0] = max[dim];
    }

    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

void ArangeOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto result = inputs[3].toTensor();
  HabanaOperator::SetPTOutput(result);
}

void ArangeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for Arange operator");
  TORCH_CHECK(
      inputs[0].isScalar(),
      "Input arg0 expected to be Scalar for Arange operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg1 expected to be Scalar for Arange operator");
  TORCH_CHECK(
      inputs[2].isScalar(),
      "Input arg2 expected to be Scalar for Arange operator");
  TORCH_CHECK(
      inputs[3].isTensor(),
      "Input arg3 expected to be tensor for Arange operator");
  auto start = inputs[0].toScalar();
  auto end = inputs[1].toScalar();
  auto step = inputs[2].toScalar();
  auto result = inputs[3].toTensor();

  // save to be used as input to cast operator if required
  synapse_helpers::tensor& range_syn_input =
      std::move(p_context_->syn_inputs_.at(0));
  bool cast_required =
      !(result.scalar_type() == ScalarType::Int ||
        result.scalar_type() == ScalarType::Float ||
        result.scalar_type() == ScalarType::BFloat16);
  if (!cast_required) {
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            range_syn_input, graph, output_metadata.at(0).external));
  }
  p_context_->pt_outputs_.emplace_back(result);

  // Adding a clear for inputs as arange TPC kernel expects no inputs
  // but graph mode call creates a syn tensor anyway, which causes a
  // synapse graph compilation failure
  p_context_->syn_inputs_.clear();

  ns_RangeKernel::Params param;
  if (result.scalar_type() == ScalarType::Float ||
      result.scalar_type() == ScalarType::BFloat16) {
    param.start.f = static_cast<float>(start.to<double>());
    param.limit.f = static_cast<float>(end.to<double>());
    param.delta.f = static_cast<float>(step.to<double>());
  } else {
    param.start.i = static_cast<int>(start.to<int>());
    param.limit.i = static_cast<int>(end.to<int>());
    param.delta.i = static_cast<int>(step.to<int>());
    SetGuid("range_i32");
  }

  // If datatype is int/bf16/fp32 , no cast node is required
  if (!cast_required) {
    AddNodeToSynapseGraph(graph, &param, sizeof(param));
  } else {
    // For datatypes Char, Bool one additional cast node is
    // required. Arange kernel return i32 output node Cast kernel will convert
    // i32 -> (i8)

    auto output_range = habana::createPTTensor(
        result,
        result.sizes(),
        result.options(),
        result.suggest_memory_format(),
        c10::ScalarType::Int,
        false);

    AllocateSynapseOutput(graph, output_range, OutputMetaData());
    // syn_output_[0] is the output of range node
    synapse_helpers::tensor& range_syn_output = p_context_->syn_outputs_[0];

    std::vector<synTensor> syn_in{};
    std::vector<synTensor> syn_out{range_syn_output.get()};

    // range_i32
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &param,
        sizeof(param),
        guid_,
        nullptr,
        nullptr,
        nullptr,
        deterministic,
        getContextHints());

    // respective cast node
    std::string node_type = "cast_i32_to_i8";

    // Create cast operator
    auto castOp =
        make_operator<CastOutOperator>(this->p_context_->device_id_, node_type);

    // Build Params for the graph
    torch::jit::Stack stack = {IValue(output_range), IValue(result)};

    castOp->SetSynapseInput(range_syn_output);
    // range_syn_input is the original Out result tensor
    castOp->SetSynapseInput(range_syn_input);
    castOp->AllocateAndAddSynapseNode(
        graph, stack, SelectVectorIndices(output_metadata, {0}));
    // replace arange syn output with cast op syn output
    p_context_->syn_outputs_.pop_back();
    p_context_->syn_outputs_.emplace_back(
        std::move(castOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.pop_back();
    p_context_->pt_outputs_[0] = std::move(castOp->GetOutputs()[0]);
  }
}

template <typename T>
std::vector<T> get_start_step_end(const IntArrayRef& shape) {
  HABANA_ASSERT(shape.size() == 1);
  std::vector<int32_t> data = {0, static_cast<int32_t>(shape[0]), 1};
  std::vector<T> d;
  for (size_t i = 0; i < 3; ++i) {
    d.emplace_back(static_cast<T>(data[i]));
  }
  return d;
}

InferOutputMetaRetType ArangeOperatorHT::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  if (inputs.size() == 3) {
    auto output_shape_tensor = inputs[2].toTensor();
    auto result = inputs[1].toTensor();
    at::Tensor host_tensor = inputs[0].toTensor();

    auto tmeta{get_tensor_extra_meta(host_tensor)};
    if (tmeta->get_host_dt_type() == habana::HostDataType::INT32_T) {
      out.AddOutputTensor(TensorMetaData(
          output_shape_tensor.sizes().vec(),
          HabanaOperator::CalculateStrides(
              output_shape_tensor.sizes(),
              output_shape_tensor.suggest_memory_format()),
          output_shape_tensor.scalar_type(),
          output_shape_tensor.suggest_memory_format()));
    } else if (tmeta->get_host_dt_type() == habana::HostDataType::FLOAT_T) {
      out.AddOutputTensor(TensorMetaData(
          result.sizes().vec(),
          HabanaOperator::CalculateStrides(
              result.sizes(), result.suggest_memory_format()),
          result.scalar_type(),
          result.suggest_memory_format()));
    }
  } else {
    auto result = inputs[3].toTensor();
    auto start = inputs[0].toScalar();
    auto end = inputs[1].toScalar();
    auto step = inputs[2].toScalar();

    if (!(result.scalar_type() == ScalarType::Float ||
          result.scalar_type() == ScalarType::BFloat16)) {
      std::vector<int64_t> sizes_vec{step.toInt(), end.toInt(), start.toInt()};
      IntArrayRef idst_sizes(sizes_vec.data(), sizes_vec.size());

      out.AddShapeTensor(TensorMetaData(
          idst_sizes.vec(),
          HabanaOperator::CalculateStrides(
              idst_sizes, result.suggest_memory_format()),
          c10::ScalarType::Int,
          result.suggest_memory_format()));
    }

    // If datatype is int/bf16/fp32 , no cast node is required
    // For datatypes Char, Bool one additional cast node is
    // required. Arange kernel return i32 output node Cast kernel will convert
    // i32 -> (i8)
    if (!(result.scalar_type() == ScalarType::Int ||
          result.scalar_type() == ScalarType::Float ||
          result.scalar_type() == ScalarType::BFloat16)) {
      auto output_range = habana::createPTTensor(
          result,
          result.sizes(),
          result.options(),
          result.suggest_memory_format(),
          c10::ScalarType::Int,
          false);

      out.AddOutputTensor(TensorMetaData(
          result.sizes().vec(),
          HabanaOperator::CalculateStrides(
              result.sizes(), result.suggest_memory_format()),
          c10::ScalarType::Int,
          result.suggest_memory_format()));

      // Create cast operator
      auto castOp = make_operator<CastOutOperator>(
          this->p_context_->device_id_, "cast_i32_to_i8");
      torch::jit::Stack stack = {IValue(output_range), IValue(result)};
      auto& castOp_out = out.call_InferOutputMeta(castOp, stack);
      auto out_tensor = castOp_out.GetOutputTensor(0);
      out.MoveToOutput(std::move(out_tensor));
    }
  }
  return out;
}

void ArangeOperatorHT::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4 || inputs.size() == 3,
      "Incorrect size of inputs expected for Arange operator");
  // inputs size == 2 when the idst tensor is added from frontend.
  if (inputs.size() == 3) {
    TORCH_CHECK(
        inputs[0].isTensor(),
        "Input arg0 expected to be tensor for Arange operator");
    TORCH_CHECK(
        inputs[1].isTensor(),
        "Input arg1 expected to be tensor for Arange operator");
    TORCH_CHECK(
        inputs[2].isTensor(),
        "Input arg2 expected to be tensor for Arange operator");
    TORCH_CHECK(p_context_->syn_inputs_[0].ref().is_host_to_device_tensor());
    kernel_meta_data_.tpc_input_order = {0};
    auto output_shape_tensor = inputs[2].toTensor();

    auto result = inputs[1].toTensor();
    if (result.scalar_type() == ScalarType::Float) {
      SetGuid("range_f32");
    } else {
      SetGuid("range_i32");
    }

    at::Tensor host_tensor = inputs[0].toTensor();
    auto tmeta{get_tensor_extra_meta(host_tensor)};

    if (tmeta->get_host_dt_type() == habana::HostDataType::INT32_T) {
      if (habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MIN_SHAPE) {
        auto data = get_start_step_end<int32_t>(output_shape_tensor.sizes());
        tmeta->set_min<int32_t>(data);
      } else if (
          habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MAX_SHAPE) {
        auto data = get_start_step_end<int32_t>(output_shape_tensor.sizes());
        tmeta->set_max<int32_t>(data);
      }
    } else if (tmeta->get_host_dt_type() == habana::HostDataType::FLOAT_T) {
      if (habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MIN_SHAPE) {
        auto data = get_start_step_end<float>(result.sizes());
        tmeta->set_min<float>(data);
      } else if (
          habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MAX_SHAPE) {
        auto data = get_start_step_end<float>(result.sizes());
        tmeta->set_max<float>(data);
      }
    }

    HABANA_ASSERT(
        result.scalar_type() == ScalarType::Int ||
        result.scalar_type() == ScalarType::Float ||
        result.scalar_type() == ScalarType::BFloat16);
    p_context_->syn_outputs_.emplace_back(
        std::move(p_context_->syn_inputs_[1]));
    p_context_->syn_inputs_.pop_back();
    p_context_->pt_outputs_.emplace_back(result);
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    TORCH_CHECK(
        inputs[3].isTensor(),
        "Input arg0 expected to be tensor for Arange operator");
    TORCH_CHECK(
        inputs[0].isScalar(),
        "Input arg1 expected to be Scalar for Arange operator");
    TORCH_CHECK(
        inputs[1].isScalar(),
        "Input arg2 expected to be Scalar for Arange operator");
    TORCH_CHECK(
        inputs[2].isScalar(),
        "Input arg3 expected to be Scalar for Arange operator");

    auto result = inputs[3].toTensor();
    auto start = inputs[0].toScalar();
    auto end = inputs[1].toScalar();
    auto step = inputs[2].toScalar();

    p_context_->syn_outputs_.emplace_back(
        std::move(p_context_->syn_inputs_[0]));
    p_context_->pt_outputs_.emplace_back(result);

    // Adding a clear for inputs as arange TPC kernel expects no inputs
    // but graph mode call creates a syn tensor anyway, which causes a
    // synapse graph compilation failure
    p_context_->syn_inputs_.clear();

    ns_RangeKernel::Params param;
    if (result.scalar_type() == ScalarType::Float ||
        result.scalar_type() == ScalarType::BFloat16) {
      param.start.f = static_cast<float>(start.to<double>());
      param.limit.f = static_cast<float>(end.to<double>());
      param.delta.f = static_cast<float>(step.to<double>());
    } else {
      param.start.i = static_cast<int>(start.to<int>());
      param.limit.i = static_cast<int>(end.to<int>());
      param.delta.i = static_cast<int>(step.to<int>());
      SetGuid("range_i32");
    }

    // If datatype is int/bf16/fp32 , no cast node is required
    if (result.scalar_type() == ScalarType::Int ||
        result.scalar_type() == ScalarType::Float ||
        result.scalar_type() == ScalarType::BFloat16) {
      AddNodeToSynapseGraph(graph, &param, sizeof(param));
    } else {
      // For datatypes Char, Bool one additional cast node is
      // required. Arange kernel return i32 output node Cast kernel will convert
      // i32 -> (i8)

      auto output_range = habana::createPTTensor(
          result,
          result.sizes(),
          result.options(),
          result.suggest_memory_format(),
          c10::ScalarType::Int,
          false);

      AllocateSynapseOutput(graph, output_range, OutputMetaData());
      synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[1];

      std::vector<synTensor> syn_in{};
      std::vector<synTensor> syn_out{synOutput.get()};

      // range_i32
      graph.add_node(
          std::move(syn_in),
          std::move(syn_out),
          &param,
          sizeof(param),
          guid_,
          nullptr,
          nullptr,
          nullptr,
          deterministic,
          getContextHints());

      // respective cast node
      std::string node_type = "cast_i32_to_i8";

      // Create cast operator
      auto castOp = make_operator<CastOutOperator>(
          this->p_context_->device_id_, node_type);

      // Build Params for the graph
      torch::jit::Stack stack = {IValue(output_range), IValue(result)};
      // syn_output_[1] is the output of range node
      castOp->SetSynapseInput(p_context_->syn_outputs_[1]);
      // syn_output_[0] is the original Out result tensor
      castOp->SetSynapseInput(p_context_->syn_outputs_[0]);

      castOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
      // There are 2 outputs {result, rangeOut}, we need only one {result}
      p_context_->syn_outputs_.pop_back();
      p_context_->pt_outputs_.pop_back();
      p_context_->pt_outputs_[0] = std::move(castOp->GetOutputs()[0]);
    }
  }
}

void Unique_Operator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(
      inputs.size() == 3 && "Incorrect size of inputs in Unique_Operator");
  HABANA_ASSERT(inputs[0].isTensor() && "Input 0 is expected to be tensor");
  HABANA_ASSERT(inputs[1].isBool() && "Input 1 is expected to be bool");
  HABANA_ASSERT(inputs[2].isBool() && "Input 2 is expected to be bool");

  bool sorted = inputs[1].toBool();
  if (sorted == true) {
    PT_KERNEL_WARN(
        "Recieved sorted=True, ignoring as TPC kernel does not support it");
  }

  auto self = inputs[0].toTensor();
  int elements = self.numel();
  auto output_shape = DimVector{elements};
  auto valid_shape = DimVector{1};

  // create output and valid shape tensors which are compulsory
  auto output_feature_map = habana::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      output_metadata.at(0).persistent);
  auto valid_count = habana::createPTTensor(
      self,
      valid_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(1).persistent);
  auto inverse_tensor = habana::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Long,
      output_metadata.at(2).persistent);

  ns_UniqueKernel::Params params;
  params.returnInverse = 1; // When set to 1 will return Inverse
  params.returnCounts = 0; // When set to 0 will not return Counts
  // dim = -5 returns flattened result(unique elements over all dimesions)
  params.dim = -5;

  p_context_->params_.emplace<ns_UniqueKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output_feature_map, output_metadata.at(0));
  synDataType synType = syn_type_int32;
  AllocateSynapseOutput(
      graph,
      valid_count,
      synType,
      output_metadata.at(1),
      graph.is_dynamic_graph());
  AllocateSynapseOutput(
      graph,
      inverse_tensor,
      synType,
      output_metadata.at(2),
      graph.is_dynamic_graph());
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void UniqueDimOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(
      inputs.size() == 5 && "Incorrect size of inputs in UniqueDimOperator");
  HABANA_ASSERT(inputs[0].isTensor() && "Input 0 is expected to be tensor");
  HABANA_ASSERT(inputs[1].isInt() && "Input 1 is expected to be int");
  HABANA_ASSERT(inputs[2].isBool() && "Input 2 is expected to be bool");
  HABANA_ASSERT(inputs[3].isBool() && "Input 3 is expected to be bool");
  HABANA_ASSERT(inputs[4].isBool() && "Input 4 is expected to be bool");

  bool sorted = inputs[2].toBool();
  if (sorted == true) {
    PT_KERNEL_WARN(
        "Recieved sorted=True, ignoring as TPC kernel does not support it");
  }
  int64_t dim = inputs[1].toInt();
  auto self = inputs[0].toTensor();
  if (dim < 0) {
    dim = self.dim() + dim;
  }
  auto output_shape = DimVector(self.sizes());
  auto valid_shape =
      DimVector{1}; // As valid tensor will be a 1D tensor with single value
  auto inverse_tensor_shape = DimVector{self.sizes().vec().at(dim)};
  auto counts_tensor_shape = DimVector{self.sizes().vec().at(dim)};

  // create output and valid shape tensors which are compulsory
  auto output_feature_map = habana::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      output_metadata.at(0).persistent);
  auto valid_count = habana::createPTTensor(
      self,
      valid_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(1).persistent);
  auto inverse_tensor = habana::createPTTensor(
      self,
      inverse_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Long,
      output_metadata.at(2).persistent);
  auto counts_tensor = habana::createPTTensor(
      self,
      counts_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Long,
      output_metadata.at(3).persistent);

  ns_UniqueKernel::Params params;
  params.returnInverse = 1; // When set to 1 will return Inverse
  params.returnCounts = 1; // When set to 1 will return Counts
  params.dim = self.dim() - dim - 1;

  p_context_->params_.emplace<ns_UniqueKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output_feature_map, output_metadata.at(0));
  synDataType synType = syn_type_int32;
  AllocateSynapseOutput(
      graph,
      valid_count,
      synType,
      output_metadata.at(1),
      graph.is_dynamic_graph());
  AllocateSynapseOutput(
      graph,
      inverse_tensor,
      synType,
      output_metadata.at(2),
      graph.is_dynamic_graph());
  AllocateSynapseOutput(
      graph,
      counts_tensor,
      synType,
      output_metadata.at(3),
      graph.is_dynamic_graph());

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void UniqueOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  int elements = self.numel();
  auto output_shape = DimVector{elements};
  auto valid_shape = DimVector{1};
  auto inverse_tensor_shape = DimVector{elements};
  auto counts_tensor_shape = DimVector{elements};

  // create output and valid shape tensors which are compulsory
  auto output_feature_map = habana::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      true);
  auto valid_count = habana::createPTTensor(
      self,
      valid_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);
  auto inverse_tensor = habana::createPTTensor(
      self,
      inverse_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Long,
      true);
  auto counts_tensor = habana::createPTTensor(
      self,
      counts_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Long,
      true);
  std::vector<at::Tensor> outputs{
      output_feature_map, valid_count, inverse_tensor, counts_tensor};
  HabanaOperator::SetPTOutputs(outputs);
}

void UniqueOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(
      inputs.size() == 4 && "Incorrect size of inputs in UniqueOperator");
  HABANA_ASSERT(inputs[0].isTensor() && "Input 0 is expected to be tensor");
  HABANA_ASSERT(inputs[1].isBool() && "Input 1 is expected to be bool");
  HABANA_ASSERT(inputs[2].isBool() && "Input 2 is expected to be bool");
  HABANA_ASSERT(inputs[3].isBool() && "Input 3 is expected to be bool");

  bool sorted = inputs[1].toBool();

  auto self = inputs[0].toTensor();
  int elements = self.numel();
  auto output_shape = DimVector{elements};
  auto valid_shape = DimVector{1};
  auto inverse_tensor_shape = DimVector{elements};
  auto counts_tensor_shape = DimVector{elements};

  // The first output tensor contains unique elements.
  // The second output tensor contains the number of unique elements.
  // The two optional tensors(Inverse index(1D), Counts(1D)) can be enabled by
  // setting the corresponding parameters in the structure(return_inverse,
  // return_counts)

  // create output and valid shape tensors which are compulsory
  auto output_feature_map = habana::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      output_metadata.at(0).persistent);
  auto valid_count = habana::createPTTensor(
      self,
      valid_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(1).persistent);
  auto inverse_tensor = habana::createPTTensor(
      self,
      inverse_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Long,
      output_metadata.at(2).persistent);
  auto counts_tensor = habana::createPTTensor(
      self,
      counts_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Long,
      output_metadata.at(3).persistent);

  ns_UniqueKernel::ParamsV2 params;
  params.returnInverse = 1;
  params.returnCounts = 1;
  params.sorted = 0;
  if (self.dim() <= 4) // TPC can support only upto 4D(1D to 4D)
    params.sorted = sorted;
  // dim = -5 returns flattened result(unique elements over all dimesions)
  params.dim = -5;

  p_context_->params_.emplace<ns_UniqueKernel::ParamsV2>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output_feature_map, output_metadata.at(0));
  synDataType synType = syn_type_uint32;
  AllocateSynapseOutput(
      graph,
      valid_count,
      synType,
      output_metadata.at(1),
      graph.is_dynamic_graph());
  AllocateSynapseOutput(graph, inverse_tensor, synType, output_metadata.at(2));
  AllocateSynapseOutput(graph, counts_tensor, synType, output_metadata.at(3));

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

InferOutputMetaRetType UniqueOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  int elements = self.numel();
  std::vector<int64_t> output_shape{elements};
  std::vector<int64_t> valid_shape{1};
  std::vector<int64_t> inverse_tensor_shape{elements};
  std::vector<int64_t> counts_tensor_shape{elements};

  InferOutputMetaRetType out;
  out.AddOutputTensor(habana::TensorMetaData(
      output_shape,
      HabanaOperator::CalculateStrides(
          output_shape, self.suggest_memory_format()),
      self.scalar_type(),
      self.suggest_memory_format()));
  out.AddOutputTensor(habana::TensorMetaData(
      valid_shape,
      HabanaOperator::CalculateStrides(
          valid_shape, self.suggest_memory_format()),
      c10::ScalarType::Int,
      self.suggest_memory_format()));
  out.AddOutputTensor(habana::TensorMetaData(
      inverse_tensor_shape,
      HabanaOperator::CalculateStrides(
          inverse_tensor_shape, self.suggest_memory_format()),
      c10::ScalarType::Long,
      self.suggest_memory_format()));
  out.AddOutputTensor(habana::TensorMetaData(
      counts_tensor_shape,
      HabanaOperator::CalculateStrides(
          counts_tensor_shape, self.suggest_memory_format()),
      c10::ScalarType::Long,
      self.suggest_memory_format()));
  return out;
}

std::vector<int64_t> SqueezeOperator::compute_output_shape(
    const at::Tensor& self,
    int64_t dim) {
  std::vector<int64_t> out_shape;
  auto dims = self.dim();
  TORCH_CHECK(dim <= HABANA_DIM_MAX, "incorrect dim for SqueezeOperator");
  if (dim < HABANA_DIM_MAX) {
    if (dims == 0 || (dims == 1 && self.numel() == 1)) {
      out_shape = {};
    } else if (dims == 1 || self.sizes()[dim] != 1) {
      out_shape = self.sizes().vec();
    } else {
      for (const auto d : c10::irange(dims)) {
        if (d != dim || self.sizes()[dim] != 1) {
          out_shape.push_back(self.sizes()[d]);
        }
      }
    }
  } else {
    for (const auto s : self.sizes()) {
      if (s != 1) {
        out_shape.push_back(s);
      }
    }
  }

  return out_shape;
}

InferOutputMetaRetType SqueezeOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto input = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto out_shape = SqueezeOperator::compute_output_shape(input, dim);
  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      out_shape,
      HabanaOperator::CalculateStrides(
          out_shape, input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));

  return out;
}

void SqueezeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2, "Incorrect size of inputs for squeeze operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isInt(), "Input arg2 type expected to be integer");

  auto input = inputs[0].toTensor();
  auto dim = inputs[1].toInt();

  auto shape = SqueezeOperator::compute_output_shape(input, dim);

  auto output = habana::createPTTensor(
      input,
      shape,
      input.options(),
      input.suggest_memory_format(),
      output_metadata.at(0).persistent);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  if (shape.empty() || shape == input.sizes()) {
    SetGuid("identity");
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else if (dim < HABANA_DIM_MAX) {
    const auto syn_axis = input.dim() - dim - 1;
    synAxisParams params{static_cast<unsigned int>(syn_axis)};

    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  } else {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  }
}

std::vector<int64_t> UnsqueezeOperator::compute_output_shape(
    const at::Tensor& self,
    int64_t dim) {
  std::vector<int64_t> out_shape(self.sizes().vec());
  dim = at::maybe_wrap_dim(dim, self.dim() + 1);
  out_shape.insert(out_shape.begin() + dim, 1);

  return out_shape;
}

InferOutputMetaRetType UnsqueezeOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto input = inputs[0].toTensor();
  auto dim = inputs[1].toInt();

  auto shape = UnsqueezeOperator::compute_output_shape(input, dim);

  auto metaData = TensorMetaData(
      shape,
      HabanaOperator::CalculateStrides(shape, input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format());

  InferOutputMetaRetType out;
  out.AddOutputTensor(metaData);

  return out;
}

void UnsqueezeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2, "Incorrect size of inputs for unsqueeze operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isInt(), "Input arg2 type expected to be integer");

  auto input = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  dim = at::maybe_wrap_dim(dim, input.dim() + 1);

  auto shape = UnsqueezeOperator::compute_output_shape(input, dim);

  auto output = habana::createPTTensor(
      input,
      shape,
      input.options(),
      input.suggest_memory_format(),
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  const auto syn_axis = input.dim() - dim;
  synAxisParams params{static_cast<unsigned int>(syn_axis)};

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

static auto& IndexKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::scatter_add", KERNEL_FN(ScatterAddOperator))
        .add("hpu::scatter_nd", KERNEL_FN(ScatterNdOperator))
        .add("hpu::scatter_nd_onnx", KERNEL_FN(ScatterNdONNXOperator))
        .add("aten::index_put", KERNEL_FN(IndexPutOperator))
        .add(
            "hpu::index_put_normal_and_neg_indices",
            KERNEL_FN(IndexPutOperator))
        .add("hpu::index_put", KERNEL_FN(IndexPutOperator2))
        .add("aten::slice.Tensor", KERNEL_FN(SliceOperator))
        .add("hpu::slice", KERNEL_FN(SliceOperator))
        .add("hpu::slice_ds", KERNEL_FN(SliceOperator))
        .add("hpu::slice_ht", KERNEL_FN(SliceOperator))
        .add("aten::index_add", KERNEL_FN(IndexAddOperator))
        .add("hpu::index_add", KERNEL_FN(IndexAddV2Operator))
        .add("hpu::_unique2", KERNEL_FN(UniqueOperator))
        .add("hpu::_unique", KERNEL_FN(Unique_Operator))
        .add("hpu::unique_dim", KERNEL_FN(UniqueDimOperator))
        .add("aten::squeeze.dim", KERNEL_FN(SqueezeOperator))
        .add("aten::unsqueeze", KERNEL_FN(UnsqueezeOperator));
