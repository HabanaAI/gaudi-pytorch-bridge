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
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "absl/strings/string_view.h"

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/recipe.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/repeat.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/topk_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

using namespace torch;

using namespace habana;

namespace {
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
} // namespace

InferOutputMetaRetType TopkOutOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;

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
  if (graph.is_dynamic_graph()) {
    // Using topk cguid
    syn_inputs.emplace_back(nullptr);
    syn_inputs.emplace_back(nullptr);
    synapse_helpers::tensor& syn_in_tensor_k = p_context_->syn_inputs_[1];
    syn_inputs.emplace_back(syn_in_tensor_k.get());
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

  auto enable_topk_in_cguid =
      GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_TOPK_USING_CGUID);
  if (enable_topk_in_cguid) {
    ns_TopkNodeV2::ParamsV4 params{};
    params.axis = get_dim_in_tpc_order(dim, self.dim());
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
        deterministic,
        getContextHints());
  } else {
    synBeamParams params;
    params.bsw = k;
    params.axis = get_dim_in_tpc_order(dim, self.dim());
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
        deterministic,
        getContextHints());
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

InferOutputMetaRetType TopkOperator::InferOutputMeta(
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
  auto out = TopkOutOperator::InferOutputMeta(inputs);
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
