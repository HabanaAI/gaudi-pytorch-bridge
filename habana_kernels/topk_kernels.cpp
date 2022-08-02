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
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "absl/strings/string_view.h"

#include "backend/create_pt_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/recipe.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/repeat.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/topk_kernels.h"

using namespace torch;

using namespace habana;
// ensure we get good values and indices for topk
inline void _allocate_or_resize_output_with_indices(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    int64_t k,
    bool values_persistent,
    bool indices_persistent) {
  auto result_sizes = self.sizes().vec();
  if (result_sizes.size() > 0) {
    result_sizes[dim] = k;
  }
  if (values.defined()) {
    TORCH_CHECK(
        self.options().type_equal(values.options()),
        "output values must be of same type as input");
    auto tht_values = values.unsafeGetTensorImpl();
    if (values.numel() || values_persistent)
      THHTensor_resizeNd(tht_values, self.dim(), result_sizes.data(), nullptr);
    else {
      THHTensor_resizeNd_nonpersistent(
          tht_values, self.dim(), result_sizes.data(), nullptr);
    }
  } else {
    values = at::empty(result_sizes, self.options());
  }
  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == c10::ScalarType::Int,
        "output indices must be of scalar type Int");
    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    auto tht_indices = indices.unsafeGetTensorImpl();
    if (indices.numel() || indices_persistent)
      THHTensor_resizeNd(tht_indices, self.dim(), result_sizes.data(), nullptr);
    else {
      THHTensor_resizeNd_nonpersistent(
          tht_indices, self.dim(), result_sizes.data(), nullptr);
    }
  } else {
    indices =
        at::empty(result_sizes, self.options().dtype(c10::ScalarType::Int));
  }
}
OutputShapeInfRetType TopkOutOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  OutputShapeInfRetType out;
  if (!(GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_TOPK_USING_CGUID))) {
    out.set_empty(true);
    return out;
  }

  auto self = inputs[0].toTensor();
  int64_t dim_ = inputs[2].toInt();
  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto values = inputs[5].toTensor();
  auto indices = inputs[6].toTensor();

  int64_t k;
  Tensor k_tensor = inputs[1].toTensor();
  k = k_tensor.sizes().vec().at(0);

  auto result_sizes = self.sizes().vec();
  if (result_sizes.size() > 0) {
    result_sizes[dim] = k;
  }

  out.AddOutputTensor(TensorMetaData(
      result_sizes,
      HabanaOperator::CalculateStrides(
          self.sizes(), self.suggest_memory_format()),
      self.scalar_type(),
      self.suggest_memory_format()));
  out.AddOutputTensor(TensorMetaData(
      result_sizes,
      HabanaOperator::CalculateStrides(
          self.sizes(), self.suggest_memory_format()),
      c10::ScalarType::Int,
      self.suggest_memory_format()));
  return out;
}
void TopkOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs expected for topk operator");

  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 expected to be tensor for topkout operator");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isInt(),
      "Input arg1 expected to be of type Int or Tensor for topkout operator");
  TORCH_CHECK(
      inputs[2].isInt(),
      "Input arg2 expected to be of type Int for topkout operator");
  TORCH_CHECK(
      inputs[3].isBool(),
      "Input arg3 expected to be of type Bool for topkout operator");
  TORCH_CHECK(
      inputs[4].isBool(),
      "Input arg4 expected to be of type Bool for topkout operator");
  TORCH_CHECK(
      inputs[5].isTensor(),
      "Input arg5 expected to be tensor for topkout operator");
  TORCH_CHECK(
      inputs[6].isTensor(),
      "Input arg6 expected to be tensor for topkout operator");
  TORCH_CHECK(
      output_metadata.size() == 2,
      "TopkOutOperator: #output_metadata should be 2");

  auto self = inputs[0].toTensor();
  int64_t dim_ = inputs[2].toInt();
  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto values = inputs[5].toTensor();
  auto indices = inputs[6].toTensor();

  int64_t k;
  // Get k value
  if (inputs[1].isTensor()) {
    TORCH_CHECK(
        (p_context_->syn_inputs_.size() == 2) ||
        (p_context_->syn_inputs_.size() == 4));
    TORCH_CHECK(p_context_->syn_inputs_.at(1).ref().is_shape_tensor());
    Tensor k_tensor = inputs[1].toTensor();
    k = k_tensor.sizes().vec().at(
        0); // Get the first element which holds the dynamic value of k
  } else {
    k = inputs[1].toInt();
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      Tensor k_tensor = habana::createPTTensor(
          self, k, self.options(), self.suggest_memory_format(), false);
      AllocateSynapseShapeTensor(graph, k_tensor);
    }
  }

  /*
     To support dynamic shape, the TPC kernel inputs needs to be {values_tensor,
     indices_tensor, null, k_tensor}. The following code create 3 additional ops
     to create the indices tensor: arrnage op -> reshape op -> repeat op
  */
  synapse_helpers::tensor& syn_in_self = p_context_->syn_inputs_[0];
  std::vector<synTensor> syn_inputs{syn_in_self.get()};
  auto enable_topk_in_cguid =
      GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_TOPK_USING_CGUID);
  if (graph.is_dynamic_graph()) {
    if (enable_topk_in_cguid) {
      syn_inputs.emplace_back(nullptr);
      syn_inputs.emplace_back(nullptr);
      synapse_helpers::tensor& syn_in_tensor_k = p_context_->syn_inputs_[1];
      syn_inputs.emplace_back(syn_in_tensor_k.get());
    } else {
      torch::jit::Stack temp_stack;
      const auto input_shape = self.sizes();
      const int start = 0;
      const int limit = input_shape[dim];
      const int step = 1;

      // Add arange op - the input tensor for the arange op is also the output
      // tensor
      auto arange_input_output_scalar_type = c10::ScalarType::Int;
      int input_output_depth =
          ArangeOperator::GetOutputSize(start, limit, step);
      std::vector<int64_t> input_output_sizes_vec{input_output_depth};
      IntArrayRef input_output_shape(
          input_output_sizes_vec.data(), input_output_sizes_vec.size());
      auto arangeInputOutput = habana::createPTTensor(
          self,
          input_output_shape,
          self.options(),
          self.suggest_memory_format(),
          arange_input_output_scalar_type,
          false);
      auto arangeOp = make_operator<ArangeOperator>(
          this->p_context_->device_id_, arange_input_output_scalar_type);
      arangeOp->AllocateSynapseInput(graph, arangeInputOutput, false);
      temp_stack = {
          IValue(start),
          IValue(limit),
          IValue(step),
          IValue(arangeInputOutput)};
      arangeOp->AllocateAndAddSynapseNode(
          graph, temp_stack, OutputMetaDataVector(1));
      temp_stack.clear();

      // Add reshape op
      auto reshaped_shape = std::vector<int64_t>(self.ndimension(), 1);
      reshaped_shape[dim] = limit;
      auto reshapeOp = make_operator<ReshapeOperator>(
          this->p_context_->device_id_, arange_input_output_scalar_type);
      temp_stack = {IValue(arangeOp->GetOutputs()[0]), IValue(reshaped_shape)};
      reshapeOp->SetSynapseInput(arangeOp->GetSynOutputs()[0]);
      reshapeOp->AllocateAndAddSynapseNode(
          graph, temp_stack, OutputMetaDataVector(1));
      temp_stack.clear();

      // Add repeat op
      auto repeatOp = make_operator<RepeatOperator>(
          this->p_context_->device_id_, arange_input_output_scalar_type);
      std::vector<int64_t> repeats = input_shape.vec();
      repeats[dim] = 1;
      temp_stack = {IValue(reshapeOp->GetOutputs()[0]), IValue(repeats)};
      repeatOp->SetSynapseInput(reshapeOp->GetSynOutputs()[0]);
      repeatOp->AllocateAndAddSynapseNode(
          graph, temp_stack, OutputMetaDataVector(1));
      temp_stack.clear();

      // Add relevant syn inputs to support dynamic shape
      synapse_helpers::tensor& syn_in_tensor_k = p_context_->syn_inputs_[1];
      synapse_helpers::tensor& syn_in_indices = repeatOp->GetSynOutputs()[0];
      syn_inputs.emplace_back(syn_in_indices.get());
      syn_inputs.emplace_back(nullptr);
      syn_inputs.emplace_back(syn_in_tensor_k.get());
    }
  }

  bool largest = inputs[3].toBool();
  bool sorted = inputs[4].toBool();

  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");
  // TPC doen't support unsorted or ascending order - but that applies only for
  // tensors with more than 1 element
  if (self.numel() > 1) {
    TORCH_CHECK(sorted == true, "unsorted output not supported")
  }

  _allocate_or_resize_output_with_indices(
      values,
      indices,
      self,
      dim,
      k,
      output_metadata.at(0).persistent,
      output_metadata.at(1).persistent);

  std::vector<at::Tensor> outputs{values, indices};
  AllocateSynapseOutputs(graph, outputs, output_metadata);

  synapse_helpers::tensor& syn_out0 = p_context_->syn_outputs_[0];
  synapse_helpers::tensor& syn_out1 = p_context_->syn_outputs_[1];
  std::vector<synTensor> syn_outputs{syn_out0.get(), syn_out1.get()};

  if (enable_topk_in_cguid) {
    ns_TopkNodeV2::ParamsV4 params{};
    params.axis = self.dim() - dim - 1;
    params.bottomK = !largest;
    params.isVcData = false;
    if (graph.is_dynamic_graph()) {
      params.kType = K_TENSOR_SHAPE;
    } else {
      params.bsw = k;
      params.kType = K_TENSOR_NONE;
    }

    // add topk node
    graph.add_node(
        std::move(syn_inputs),
        std::move(syn_outputs),
        &params,
        sizeof(params),
        guid_,
        nullptr,
        nullptr,
        nullptr,
        deterministic);
  } else {
    synBeamParams params;
    params.bsw = k;
    params.axis = self.dim() - dim - 1;
    params.bottomK = !largest;

    // add topk node
    graph.add_node(
        std::move(syn_inputs),
        std::move(syn_outputs),
        &params,
        sizeof(params),
        guid_,
        nullptr,
        nullptr,
        nullptr,
        deterministic);
  }
}

void TopkOutOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  int64_t k = inputs[1].toInt();
  int64_t dim_ = inputs[2].toInt();
  auto values = inputs[5].toTensor();
  auto indices = inputs[6].toTensor();

  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  _allocate_or_resize_output_with_indices(
      values, indices, self, dim, k, true, true);
  std::vector<at::Tensor> v{values, indices};
  HabanaOperator::SetPTOutputs(v);
}

OutputShapeInfRetType TopkOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  if (inputs.size() == 5) {
    Tensor self = inputs[0].toTensor();
    auto values = habana::createPTTensor(
        self, {0}, self.options(), self.suggest_memory_format(), false);
    auto indices = habana::createPTTensor(
        self,
        {0},
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Int,
        false);
    inputs.push_back(IValue(values));
    inputs.push_back(IValue(indices));
  }
  auto out = TopkOutOperator::ComputeOutputShape(inputs);
  if (inputs.size() == 7) {
    inputs.erase(inputs.cbegin() + 5);
    inputs.erase(inputs.cbegin() + 6);
  }
  return out;
}
void TopkOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for topk operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for topk operator");
  TORCH_CHECK(
      output_metadata.size() == 2,
      "TopkOperator: #output_metadata should be 2");

  Tensor self = inputs[0].toTensor();
  auto values = habana::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);
  auto indices = habana::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(1).persistent);
  inputs.push_back(IValue(values));
  inputs.push_back(IValue(indices));

  TopkOutOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}

void TopkOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  auto values = habana::createPTTensor(
      self, {0}, self.options(), self.suggest_memory_format(), true);
  auto indices = habana::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);

  inputs.push_back(IValue(values));
  inputs.push_back(IValue(indices));

  TopkOutOperator::SetPTOutputs(inputs);
}

void SortOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for sort operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for sort operator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input arg2 expected to be of type Int for sort operator");
  TORCH_CHECK(
      inputs[2].isBool(),
      "Input arg3 expected to be of type Bool for sort operator");
  TORCH_CHECK(
      output_metadata.size() == 2,
      "SortOperator: #output_metadata should be 2");

  Tensor self = inputs[0].toTensor();
  int64_t dim_ = inputs[1].toInt();
  bool sorted = true; // topk supports only sorted output

  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  inputs.insert(inputs.begin() + 1, IValue(self.size(dim)));
  inputs.emplace_back(IValue(sorted));

  auto values = habana::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);
  auto indices = habana::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(1).persistent);

  inputs.push_back(IValue(values));
  inputs.push_back(IValue(indices));

  TopkOutOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}

void SortOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  int64_t dim_ = inputs[1].toInt();
  bool sorted = true; // topk supports only sorted output

  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  inputs.insert(inputs.begin() + 1, IValue(self.size(dim)));
  inputs.emplace_back(IValue(sorted));

  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(c10::ScalarType::Int));

  inputs.push_back(IValue(values));
  inputs.push_back(IValue(indices));

  TopkOutOperator::SetPTOutputs(inputs);
}