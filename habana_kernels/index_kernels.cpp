/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;

/*************************************************************************
 * @brief This helper function makes the size of index tensor to be same as
 * value tensor, with broadcast of indices (within index tensor)
 ************************************************************************/
int GetOutputSize(Scalar start_, Scalar end_, Scalar step_) {
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
  depth = depth == 0 ? 1 : depth;
  return depth;
}
Tensor GatherOperator::AllocateOutput(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto dim_ = inputs[1].toInt();
  auto index = inputs[2].toTensor();

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto shape = DimVector(self.sizes());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, index.numel());
  auto output = at::empty(shape, self.options(), self.suggest_memory_format());
  return output;
}

void GatherOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto output = AllocateOutput(inputs);
  HabanaOperator::SetPTOutput(output);
}

void GatherOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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
    index.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  auto output = AllocateOutput(inputs);

  ns_GatherKernel::Params params;
  params.axis = self.dim() - dim - 1;

  p_context_->params_.emplace<ns_GatherKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/*************************************************************************
 * @brief Kernel implementation for torch.gather
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param index - Tensor used to index into self
 * @param sparse_grad - Boolean to indicate if sparse grad is supported
 ************************************************************************/
Tensor gather_src_hpu(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad) {
  PT_KERNEL_BEGIN;

  auto index_int = habana_helpers::cast_tensor_to_integer(index);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "gather_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  GatherOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim_), IValue(index_int), IValue(sparse_grad)};
  size_t key = Op.GetRecipeKey(node_type, stack);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, index_int};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}

Tensor ScatterWrapperOperator::AllocateOutput(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto output = at::empty(
      self.sizes().vec(), self.options(), self.suggest_memory_format());
  return output;
}

void ScatterWrapperOperator::SetPTOutput(torch::jit::Stack& inputs) {
  auto output = AllocateOutput(inputs);
  HabanaOperator::SetPTOutputs({output});
}

void ScatterWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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
    index.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  auto output = AllocateOutput(inputs);

  ns_ScatterKernel::Params params;
  params.axis = self.dim() - dim - 1;

  p_context_->params_.emplace<ns_ScatterKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/*************************************************************************
 * @brief Kernel implementation for torch.scatter
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param index - Tensor used to index into self
 * @param src - Tensor with values to be updated (of same type as self)
 ************************************************************************/
Tensor scatter_src_hpu(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_KERNEL_BEGIN;

  auto index_int = habana_helpers::cast_tensor_to_integer(index);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "scatter_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  ScatterOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim_), IValue(index_int), IValue(src)};
  size_t key = Op.GetRecipeKey(node_type, stack);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, index_int, src};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}

/*************************************************************************
 * @brief Kernel implementation for scatter_.src(Tensor(a!) self, int dim,
 *Tensor index, Tensor src) -> Tensor(a!)
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param index - Tensor used to index into self
 * @param src -Tensor with values to be updated (of same type as self)
 ************************************************************************/
Tensor& scatter_inplace_src_hpu(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_KERNEL_BEGIN;
  auto out = scatter_src_hpu(self, dim_, index, src);
  self.copy_(out);
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for torch.scatter_add
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param index - Tensor used to index into self
 * @param src - Tensor with values to be updated (of same type as self)
 ************************************************************************/
Tensor scatter_add_src_hpu(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_KERNEL_BEGIN;

  auto index_int = habana_helpers::cast_tensor_to_integer(index);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "scatter_add_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  ScatterAddOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim_), IValue(index_int), IValue(src)};
  size_t key = Op.GetRecipeKey(node_type, stack);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, index_int, src};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}

/*************************************************************************
 * @brief Kernel implementation for scatter_add_(Tensor(a!) self, int dim,
 * Tensor index, Tensor src) -> Tensor(a!)
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param index - Tensor used to index into self
 * @param src -Tensor with values to be updated (of same type as self)
 ************************************************************************/
Tensor& scatter_add_inplace_src_hpu(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_KERNEL_BEGIN;

  auto out = scatter_add_src_hpu(self, dim_, index, src);
  self.copy_(out);

  PT_KERNEL_END;
  return self;
}

void IndexAddOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4, "Incorrect size of inputs for index_add operator");
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

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto index = inputs[2].toTensor();
  auto value = inputs[3].toTensor();

  std::vector<synapse_helpers::tensor_or_ref> addSynOutput;
  torch::jit::Stack temp_stack;

  // Create MemCopy operator to copy value into value_acc
  MemCopyOperator memcpyOp(this->p_context_->device_id_, value.scalar_type());
  // No need for output PT tensor as it's non persistent
  temp_stack = {IValue(value), IValue(value)};
  auto& syn_memcpyIn =
      memcpyOp.SetSynapseInput(std::move(p_context_->syn_inputs_[2]));
  memcpyOp.AllocateAndAddSynapseNode(graph, temp_stack, false);
  p_context_->syn_inputs_[2] = std::move(syn_memcpyIn);
  temp_stack.clear();

  ////auto slice = at::index_select(self, 0, indices[0]);
  IndexSelectOperator index_selectOp(
      this->p_context_->device_id_, self.scalar_type());
  temp_stack = {IValue(self), IValue(dim), IValue(index)};
  auto& syn_isSelf =
      index_selectOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  auto& syn_isIndex =
      index_selectOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  index_selectOp.AllocateAndAddSynapseNode(graph, temp_stack, false);
  p_context_->syn_inputs_[0] = std::move(syn_isSelf);
  p_context_->syn_inputs_[1] = std::move(syn_isIndex);
  temp_stack.clear();

  ////value_acc += slice;
  AddOperator addOp(this->p_context_->device_id_, value.scalar_type());
  temp_stack = {
      IValue(value),
      IValue(index_selectOp.GetOutputs()[0]),
      IValue(Scalar(1.0))};
  addOp.SetSynapseInput(std::move(memcpyOp.GetSynOutputs()[0]));
  addOp.SetSynapseInput(std::move(index_selectOp.GetSynOutputs()[0]));
  addOp.AllocateAndAddSynapseNode(graph, temp_stack, false);
  addSynOutput.push_back(std::move(addOp.GetSynOutputs()[0]));
  temp_stack.clear();

  // Expand 1D index tensor to same number of dimensions as value tensor
  auto expanded_sizes = std::vector<int64_t>(value.ndimension(), 1);
  expanded_sizes[dim] = index.sizes()[0];

  ////auto index_expanded = index.view(expanded_sizes)
  ReshapeOperator reshapeOp(this->p_context_->device_id_, index.scalar_type());
  temp_stack = {IValue(index), IValue(expanded_sizes)};
  auto& syn_reshape =
      reshapeOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  reshapeOp.AllocateAndAddSynapseNode(graph, temp_stack, false);
  p_context_->syn_inputs_[1] = std::move(syn_reshape);
  temp_stack.clear();

  // Broadcast index tensor to same shape as value tensor
  bool implicit =
      false; // The value of implicit is currently ignored in broadcast kernel
  BroadcastOperator bcastOp(
      this->p_context_->device_id_, reshapeOp.GetOutputs()[0].scalar_type());
  temp_stack = {
      IValue(reshapeOp.GetOutputs()[0]),
      IValue(value.sizes()),
      IValue(implicit)};
  bcastOp.SetSynapseInput(std::move(reshapeOp.GetSynOutputs()[0]));
  bcastOp.AllocateAndAddSynapseNode(graph, temp_stack, false);
  temp_stack.clear();

  ////auto temp  = scatter_src_hpu(self, dim, index_broadcast, value_acc);
  ScatterOperator scatterOp(this->p_context_->device_id_, self.scalar_type());
  temp_stack = {
      IValue(self),
      IValue(dim),
      IValue(bcastOp.GetOutputs()[0]),
      IValue(value)};

  auto& syn_scatter1 =
      scatterOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  UNUSED auto& syn_scatter2 =
      scatterOp.SetSynapseInput(std::move(bcastOp.GetSynOutputs()[0]));
  UNUSED auto& syn_scatter3 =
      scatterOp.SetSynapseInput(std::move(addSynOutput[0]));

  scatterOp.AllocateAndAddSynapseNode(graph, temp_stack, is_output_persistent);
  p_context_->syn_inputs_[0] = std::move(syn_scatter1);

  p_context_->syn_outputs_.emplace_back(
      std::move(scatterOp.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(scatterOp.GetOutputs()[0]));
}

Tensor index_add_hpu(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source) {
  PT_KERNEL_BEGIN;
  TORCH_CHECK(indices.dim() <= 1, "index tensor cannot be more than 1D")
  // Convert index tensor from 0D to 1D if required
  if (indices.dim() == 0) {
    indices.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto index_int = habana_helpers::cast_tensor_to_integer(indices);

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "index_add_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  IndexAddOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(index_int), IValue(source)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, index_int, source};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

/*************************************************************************
 * @brief Kernel implementation for index_add(dim, index, tensor) → Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param indices - Tensor used to index into self
 * @param source -Tensor with values to be updated (of same type as self)
 ************************************************************************/
Tensor& index_add_hpu_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source) {
  PT_KERNEL_BEGIN;
  self.copy_(index_add_hpu(self, dim_, indices, source));
  PT_KERNEL_END;
  return self;
}

void IndexPutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4, "Incorrect size of inputs for index_put operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input 0 type expected to be Tensor for index_put operator");
  TORCH_CHECK(
      inputs[1].isTensorList(),
      "Input 1 type expected to be TensorList for index_put operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input 2 type expected to be Tensor for index_put operator");
  TORCH_CHECK(
      inputs[3].isBool(),
      "Input 3 type expected to be Bool for index_put operator");
  TORCH_CHECK(
      inputs[1].toTensorList().get(0).dim() == 1,
      "index tensor should be 1D for index_put operator");
  TORCH_CHECK(
      inputs[1].toTensorList().size() == 1,
      "Input 1 is expected to be a TensorList having only 1 member");

  auto self = inputs[0].toTensor();
  auto index = inputs[1].toTensorList().get(0);
  auto value = inputs[2].toTensor();
  auto accumulate = inputs[3].toBool();
  int64_t dim = 0;
  // Following last_index is calculated in case inputs[1] has more than
  // one members - note that we are interested only in the first member
  int64_t last_index = inputs[1].toTensorList().size() + 1;
  std::vector<synapse_helpers::tensor_or_ref> addSynOutput;
  torch::jit::Stack temp_stack;

  if (accumulate) {
    // Create MemCopy operator to copy value into value_acc
    MemCopyOperator memcpyOp(this->p_context_->device_id_, value.scalar_type());
    // No need for output PT tensor as it's non persistent
    temp_stack = {IValue(value), IValue(value)};
    auto& syn_memcpyIn = memcpyOp.SetSynapseInput(
        std::move(p_context_->syn_inputs_[last_index]));
    memcpyOp.AllocateAndAddSynapseNode(graph, temp_stack, false);
    p_context_->syn_inputs_[last_index] = std::move(syn_memcpyIn);
    temp_stack.clear();

    ////auto slice = at::index_select(self, 0, indices[0]);
    IndexSelectOperator index_selectOp(
        this->p_context_->device_id_, self.scalar_type());
    temp_stack = {IValue(self), IValue(dim), IValue(index)};
    auto& syn_isSelf =
        index_selectOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    auto& syn_isIndex =
        index_selectOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
    index_selectOp.AllocateAndAddSynapseNode(graph, temp_stack, false);
    p_context_->syn_inputs_[0] = std::move(syn_isSelf);
    p_context_->syn_inputs_[1] = std::move(syn_isIndex);
    temp_stack.clear();

    ////value_acc += slice;
    AddOperator addOp(this->p_context_->device_id_, value.scalar_type());
    temp_stack = {IValue(value),
                  IValue(index_selectOp.GetOutputs()[0]),
                  IValue(Scalar(1.0))};
    addOp.SetSynapseInput(std::move(memcpyOp.GetSynOutputs()[0]));
    addOp.SetSynapseInput(std::move(index_selectOp.GetSynOutputs()[0]));
    addOp.AllocateAndAddSynapseNode(graph, temp_stack, false);
    addSynOutput.push_back(std::move(addOp.GetSynOutputs()[0]));
    temp_stack.clear();
  }

  // Expand 1D index tensor to same number of dimensions as value tensor
  auto expanded_sizes = std::vector<int64_t>(value.ndimension(), 1);
  expanded_sizes[0] = index.sizes()[0];

  ////auto index_expanded = index.view(expanded_sizes)
  ReshapeOperator reshapeOp(this->p_context_->device_id_, index.scalar_type());
  temp_stack = {IValue(index), IValue(expanded_sizes)};
  auto& syn_reshape =
      reshapeOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  reshapeOp.AllocateAndAddSynapseNode(graph, temp_stack, false);
  p_context_->syn_inputs_[1] = std::move(syn_reshape);
  temp_stack.clear();

  // Broadcast index tensor to same shape as value tensor
  bool implicit =
      false; // The value of implicit is currently ignored in broadcast kernel
  BroadcastOperator bcastOp(
      this->p_context_->device_id_, reshapeOp.GetOutputs()[0].scalar_type());
  temp_stack = {IValue(reshapeOp.GetOutputs()[0]),
                IValue(value.sizes()),
                IValue(implicit)};
  bcastOp.SetSynapseInput(std::move(reshapeOp.GetSynOutputs()[0]));
  bcastOp.AllocateAndAddSynapseNode(graph, temp_stack, false);
  temp_stack.clear();

  ////auto temp  = scatter_src_hpu(self, dim, index_broadcast, value_acc);
  ScatterOperator scatterOp(this->p_context_->device_id_, self.scalar_type());
  temp_stack = {IValue(self),
                IValue(dim),
                IValue(bcastOp.GetOutputs()[0]),
                IValue(value)};

  auto& syn_scatter1 =
      scatterOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  UNUSED auto& syn_scatter2 =
      scatterOp.SetSynapseInput(std::move(bcastOp.GetSynOutputs()[0]));
  auto& syn_scatter3 = accumulate
      ? scatterOp.SetSynapseInput(std::move(addSynOutput[0]))
      : scatterOp.SetSynapseInput(
            std::move(p_context_->syn_inputs_[last_index]));
  scatterOp.AllocateAndAddSynapseNode(graph, temp_stack, is_output_persistent);
  p_context_->syn_inputs_[0] = std::move(syn_scatter1);
  if (!accumulate) {
    p_context_->syn_inputs_[last_index] = std::move(syn_scatter3);
  }

  p_context_->syn_outputs_.emplace_back(
      std::move(scatterOp.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(scatterOp.GetOutputs()[0]));
}

/*************************************************************************
 * @brief Kernel implementation for index_put(indices, value, accumulate=False)
 *→ Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param indices - Tensors used to index into self
 * @param value - Tensor with values to be updated (of same type as self)
 * @param accumulate - Flag to indicate whether to accumulate into self
 ************************************************************************/
Tensor index_put_hpu(
    const Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  PT_KERNEL_BEGIN;

  // Convert index tensor from 0D to 1D if required
  if (indices[0].dim() == 0) {
    indices[0].unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  // For index_put kernel, indices is expected to have a single member
  auto index_int = habana_helpers::cast_tensor_to_integer(indices[0]);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "index_put_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  IndexPutOperator Op(device_id, scalar_type);

  // Assign inputs for the Operator
  // Note that although indices is passed as TensorList in stack,
  // it's unrolled to individual Tensors for pt_inputs
  std::vector<at::Tensor> pt_inputs{self, index_int, value};
  torch::jit::Stack stack = {
      IValue(self), IValue(indices), IValue(value), IValue(accumulate)};

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty_like(self);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs({output});
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);

    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}

/*************************************************************************
 * @brief Kernel implementation for index_put(indices, value, accumulate=False)
 *→ Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param indices - Tensors used to index into self
 * @param value - Tensor with values to be updated (of same type as self)
 * @param accumulate - Flag to indicate whether to accumulate into self
 ************************************************************************/
Tensor& index_put_hpu_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  PT_KERNEL_BEGIN;

  auto temp = index_put_hpu(self, indices, value, accumulate);
  self.copy_(temp);

  PT_KERNEL_END;
  return self;
}

void IndexSelectOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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
  GatherOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void IndexSelectOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto index = inputs[2].toTensor();
  TORCH_CHECK(index.dim() <= 1, "index tensor cannot be more than 1D")
  bool sparse_grad = false;
  inputs.emplace_back(IValue(sparse_grad));
  GatherOperator::SetPTOutputs(inputs);
}
/*************************************************************************
 * @brief Kernel implementation for torch.index_select(input, dim, index) →
 *Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - The dimension in which we index
 * @param index - 1D tensor containing the indices to index
 ************************************************************************/
Tensor index_select_hpu(const Tensor& self, int64_t dim, const Tensor& index) {
  PT_KERNEL_BEGIN;

  auto index_int = habana_helpers::cast_tensor_to_integer(index);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "index_select_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  IndexSelectOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {IValue(self), IValue(dim), IValue(index)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, index_int};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}

void Gather2dOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for gather2d operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isInt(), "Input arg4 type expected to be integer");

  auto input = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto validCount = inputs[2].toInt();

  TORCH_CHECK(indices.dim() <= 1, "index tensor cannot be more than 1D")

  TORCH_CHECK(
      indices.numel() >= validCount,
      "validCount cannot be greater than number of indices provided")
  TORCH_CHECK(input.dim() == 2, "Input tensor should be 2D")

  auto shape = DimVector(input.sizes());
  shape.erase(shape.begin() + 0);
  shape.insert(shape.begin() + 0, std::min(indices.numel(), validCount));
  auto output = habana_helpers::createPTTensor(
      input,
      shape,
      input.options(),
      input.suggest_memory_format(),
      is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/*************************************************************************
 * @brief Kernel implementation for gather2d custom OP
 * @param self - Input tensor 2D fp32
 * @param indices - 1D tensor containing the indices to index
 * @param validCount - number of valid indices in indices tensor
 ************************************************************************/
Tensor gather2d_hpu(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  PT_KERNEL_BEGIN;

  // Convert index tensor from 0D to 1D if required
  if (indices.dim() == 0) {
    indices.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto indices_int = habana_helpers::cast_tensor_to_integer(indices);

  // This conversion from scalar to tensor not done within
  // AllocateAndAddSynapseNode because graph mode does not have
  // support for DMA handling.
  auto validCount_int = at::empty({1}, indices_int.options());
  validCount_int.fill_(static_cast<int32_t>(validCount));

  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type = "gather_with_valid_count_2d_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = input.device().index();

  Gather2dOperator Op(device_id, node_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{input, indices_int, validCount_int};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(input), IValue(indices_int), IValue(validCount)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

void NarrowOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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
  SliceOperator::AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
}

Tensor SliceOperator::AllocateOutputTensor(
    const Tensor& self,
    int64_t& dim,
    int64_t& start,
    int64_t& end,
    int64_t& step,
    bool is_output_persistent) {
  // convert dim to positive value if required
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  auto sizes = self.sizes().vec();
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }

  // compute output shape
  auto len = 0;
  for (auto i = start; i < end; i += step) {
    len++;
  }
  auto shape = DimVector(self.sizes());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, len);

  // allocate output tensor
  auto output = habana_helpers::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);

  return output;
}

void SliceOperator::SetPTOutputs(const torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto start = inputs[2].toInt();
  auto end = inputs[3].toInt();
  auto step = inputs[4].toInt();
  auto output = AllocateOutputTensor(self, dim, start, end, step, true);
  HabanaOperator::SetPTOutputs({output});
}

void SliceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for slice operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isInt(), "Input arg2 type expected to be integer");
  TORCH_CHECK(inputs[2].isInt(), "Input arg3 type expected to be integer");
  TORCH_CHECK(inputs[3].isInt(), "Input arg4 type expected to be integer");
  TORCH_CHECK(inputs[4].isInt(), "Input arg5 type expected to be integer");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto start = inputs[2].toInt();
  auto end = inputs[3].toInt();
  auto step = inputs[4].toInt();

  if (dim == self.dim() - 1) {
    // check required due to GC limitation. Strided slice is possible on FCD
    // only if there is another dimension with size 1 in the tensor.
    TORCH_CHECK(step <= 1, "strided slice not supported on FCD");
  }

  auto output =
      AllocateOutputTensor(self, dim, start, end, step, is_output_persistent);
  std::vector<const at::Tensor*> pt_outputs{&output};

  synSliceParams params;
  // set defaults
  std::fill_n(params.axes, MAX_DIMENSIONS_NUM, 0);
  std::fill_n(params.starts, MAX_DIMENSIONS_NUM, 0);
  std::fill_n(params.ends, MAX_DIMENSIONS_NUM, 0);
  std::fill_n(params.steps, MAX_DIMENSIONS_NUM, 1);
  // slice triggered only on 1 dim, therefore use only index 0
  params.axes[0] = self.dim() - dim - 1;
  params.starts[0] = start;
  params.ends[0] = end;
  params.steps[0] = step;

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/*************************************************************************
 * @brief Kernel implementation for torch slice operator
 * @param self - Input tensor
 * @param dim - Axis to slice
 * @param start - index of first element in given axis
 * @param end - index of last element in given axis
 * @param steps - number of elements to stride in given axis
 ************************************************************************/
Tensor slice_hpu(
    const Tensor& in_self,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  PT_KERNEL_BEGIN;

  Tensor self;
  if (in_self.scalar_type() == c10::ScalarType::Long) {
    self = habana_helpers::cast_tensor_to_integer(in_self);
  } else {
    self = in_self;
  }

  // for handling trivial cases, fall-back to simple tensor meta-data
  // manipulation done in CPU implementation. This was added because
  // distributed MNIST stops working if run synapse version of slice
  // which creates new storage for output storage
  if ((self.dim() <= 1) && (step == 1)) {
    PT_KERNEL_END;
    return at::native::slice(self, dim, start, end, step);
  }

  at::ScalarType scalar_type = self.scalar_type();

  std::string node_type = "slice";
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  SliceOperator Op(device_id, scalar_type);
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(start), IValue(end), IValue(step)};

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  Tensor cast_out;
  if (in_self.scalar_type() == c10::ScalarType::Long) {
    cast_out = habana_helpers::cast_tensor_to_long(out.at(0));
  } else {
    cast_out = out.at(0);
  }
  PT_KERNEL_END;
  return cast_out;
}

void SelectOperator::SetPTOutputs(const torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto index = inputs[2].toInt();
  auto start = index;
  auto end = index + 1;
  int64_t step = 1;
  SliceOperator slice_op(self.device().index(), self.scalar_type());
  auto slice_output =
      slice_op.AllocateOutputTensor(self, dim, start, end, step, false);

  // case for select op where tensor dimension is reduced
  // only rank 4 tensor can have channels last format
  at::MemoryFormat memory_format = at::MemoryFormat::Contiguous;

  // allocate output tensor
  auto shape = slice_output.sizes().vec();
  shape.erase(shape.begin() + dim);
  auto output = habana_helpers::createPTTensor(
      self, shape, self.options(), memory_format, true);

  HabanaOperator::SetPTOutputs({output});
}

void SelectOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  HABANA_ASSERT(inputs.size() == 3);
  HABANA_ASSERT(inputs[0].isTensor());
  HABANA_ASSERT(inputs[1].isInt());
  HABANA_ASSERT(inputs[2].isInt());

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto index = inputs[2].toInt();

  auto start = index;
  auto end = index + 1;
  int64_t step = 1;

  SliceOperator slice_op(self.device().index(), self.scalar_type());
  auto& syn_in_slice =
      slice_op.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  std::vector<c10::IValue> stack1 = {
      IValue(self), IValue(dim), IValue(start), IValue(end), IValue(step)};
  slice_op.AllocateAndAddSynapseNode(graph, stack1, false);
  p_context_->syn_inputs_[0] = std::move(syn_in_slice);

  // Add Reshape node to graph
  ReshapeOperator reshape_op(self.device().index(), self.scalar_type());
  UNUSED auto& syn_in_reshape =
      reshape_op.SetSynapseInput(std::move(slice_op.GetSynOutputs()[0]));
  auto slice_out_tensor = slice_op.GetOutputs()[0];
  auto shape = slice_out_tensor.sizes().vec();
  shape.erase(shape.begin() + dim);
  torch::jit::Stack stack2 = {c10::IValue(slice_out_tensor),
                              c10::IValue(shape)};
  reshape_op.AllocateAndAddSynapseNode(graph, stack2, is_output_persistent);

  p_context_->syn_outputs_.emplace_back(
      std::move(reshape_op.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(reshape_op.GetOutputs()[0]));
}

/*************************************************************************
 * @brief Kernel implementation for torch select operator
 * @param self - Input tensor
 * @param dim - Axis to slice
 * @param index - index the element in given axis
 ************************************************************************/

Tensor select_hpu(const Tensor& in_self, int64_t dim, int64_t index) {
  PT_KERNEL_BEGIN;

  Tensor self;
  if (in_self.scalar_type() == c10::ScalarType::Long) {
    self = habana_helpers::cast_tensor_to_integer(in_self);
  } else {
    self = in_self;
  }

  at::ScalarType scalar_type = self.scalar_type();
  HABANA_ASSERT(
      ((scalar_type == c10::ScalarType::Int) ||
       (scalar_type == c10::ScalarType::Float) ||
       (scalar_type == c10::ScalarType::BFloat16)));

  std::string node_type = "slice";
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  SelectOperator Op(device_id, scalar_type);
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {IValue(self), IValue(dim), IValue(index)};

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  HABANA_ASSERT(out.size() == 1);

  Tensor cast_out;
  if (in_self.scalar_type() == c10::ScalarType::Long) {
    cast_out = habana_helpers::cast_tensor_to_long(out.at(0));
  } else {
    cast_out = out.at(0);
  }
  PT_KERNEL_END;
  return cast_out;
}

void ArangeOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto result = inputs[3].toTensor();

  if (result.scalar_type() == ScalarType::Long) {
    auto output_int = habana_helpers::createPTTensor(
        result,
        result.sizes(),
        result.options(),
        result.suggest_memory_format(),
        c10::ScalarType::Int,
        true);
    HabanaOperator::SetPTOutput(output_int);
  } else {
    HabanaOperator::SetPTOutput(result);
  }
}

void ArangeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for Arange operator");
  TORCH_CHECK(
      inputs[3].isTensor(),
      "Input arg3 expected to be tensor for Arange operator");
  TORCH_CHECK(
      inputs[0].isScalar(),
      "Input arg1 expected to be Scalar for Arange operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be Scalar for Arange operator");
  TORCH_CHECK(
      inputs[2].isScalar(),
      "Input arg3 expected to be Scalar for Arange operator");

  auto start = inputs[0].toScalar();
  auto end = inputs[1].toScalar();
  auto step = inputs[2].toScalar();
  auto result = inputs[3].toTensor();

  // Adding a clear for inputs as arange TPC kernel expects no inputs
  // but graph mode call creates a syn tensor anyway, which causes a
  // synapse graph compilation failure
  p_context_->syn_inputs_.clear();

  ns_RangeKernel::Params param;
  param.start.f = static_cast<float>(start.to<double>());
  param.limit.f = static_cast<float>(end.to<double>());
  param.delta.f = static_cast<float>(step.to<double>());

  // Set Guid here again because in graph mode we may have set guid to
  // range_i32 which is not supported by TPC kernel
  if (result.scalar_type() == ScalarType::BFloat16) {
    SetGuid("range_bf16");
  } else {
    SetGuid("range_f32");
  }

  // TPC kernel support only bf16/f32,
  // If datatype is bf16/fp32 , no cast node is required
  if (result.scalar_type() == ScalarType::Float ||
      result.scalar_type() == ScalarType::BFloat16) {
    AllocateSynapseOutput(graph, result, is_output_persistent);
    AddNodeToSynapseGraph(graph, &param, sizeof(param));
  } else {
    // For datatypes Int, Long, Char, Bool one additional cast node is required.
    // Arange kernel return f32 output node
    // Cast kernel will convert f32 -> (i32/i8)

    auto output_range = habana_helpers::createPTTensor(
        result,
        result.sizes(),
        result.options(),
        result.suggest_memory_format(),
        c10::ScalarType::Float,
        false);

    AllocateSynapseOutput(graph, output_range, false);
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    std::vector<synTensor> syn_in{};
    std::vector<synTensor> syn_out{synOutput.get()};

    // range_f32
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &param,
        sizeof(param),
        std::move(guid_));

    // respective cast node
    std::string node_type;
    if (start.type() == ScalarType::Bool || start.type() == ScalarType::Char) {
      node_type = "cast_f32_to_i8";
    } else {
      node_type = "cast_f32_to_i32";
    }

    // Create cast operator
    CastOutOperator castOp(this->p_context_->device_id_, node_type);
    castOp.SetSynapseInput(std::move(p_context_->syn_outputs_[0]));

    // Build Params for the graph
    torch::jit::Stack stack;
    stack.emplace_back(IValue(output_range));

    // cast is not supported for Long. It has to be cast first to Int
    // The Int value will be converted to long on CPU
    // That converted value will be copied to output tensor.
    // For this we have to create one extra Int tensor
    if (result.scalar_type() == ScalarType::Long) {
      auto output_int = habana_helpers::createPTTensor(
          result,
          result.sizes(),
          result.options(),
          result.suggest_memory_format(),
          c10::ScalarType::Int,
          is_output_persistent);
      stack.emplace_back(IValue(output_int));
    } else {
      stack.emplace_back(IValue(result));
    }
    castOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    p_context_->syn_outputs_[0] = std::move(castOp.GetSynOutputs()[0]);
    p_context_->pt_outputs_[0] = std::move(castOp.GetOutputs()[0]);
  }
}

/*************************************************************************
 * @brief Kernel implementation for torch.arange operator
 * @param output - output tensor
 * @param start - start index of the sequence
 * @param end - end index of the sequence
 * @param step - step value of the sequence
 ************************************************************************/

Tensor& arange_hpu(Tensor& output, Scalar start, Scalar end, Scalar step) {
  PT_KERNEL_BEGIN;

  // resizing the output as it is coming as empty from model
  int depth = GetOutputSize(start, end, step);
  auto shape = DimVector({depth});
  auto tht_result = output.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);
  at::ScalarType scalar_type;
  if (output.scalar_type() == ScalarType::BFloat16) {
    scalar_type = c10::ScalarType::BFloat16;
  } else {
    scalar_type = c10::ScalarType::Float;
  }

  std::string node_type =
      "range_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = output.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  ArangeOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(start), IValue(end), IValue(step), IValue(output)};

  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (output.scalar_type() == ScalarType::Long) {
    output.copy_(habana_helpers::cast_tensor_to_long(out.at(0)));
    PT_KERNEL_END;
    return output;
  } else if (output.scalar_type() == ScalarType::Bool) {
    out.at(0).to(c10::ScalarType::Bool);
    PT_KERNEL_END;
    return out.at(0);
  } else {
    PT_KERNEL_END;
    return out.at(0);
  }
}

static auto& KernelRegistry =
    ::habana::KernelRegistry()
        .add(
            "aten::index_select",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<IndexSelectOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::gather",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<GatherOperator>(device_id, node_type);
            })
        .add(
            "aten::scatter",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ScatterOperator>(device_id, node_type);
            })
        .add(
            "aten::scatter_add",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ScatterAddOperator>(device_id, node_type);
            })
        .add(
            "aten::select",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SelectOperator>(device_id, node_type);
            })
        .add(
            "aten::index_put",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<IndexPutOperator>(device_id, node_type);
            })
        .add(
            "aten::arange",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ArangeOperator>(device_id, node_type);
            })
        .add(
            "aten::slice",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SliceOperator>(device_id, node_type);
            })
        .add(
            "aten::index_add",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<IndexAddOperator>(device_id, node_type);
            });
