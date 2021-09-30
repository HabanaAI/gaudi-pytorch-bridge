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

#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/aten_hpu_type_default.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "synapse_helpers/tensor_builder_base.h"

using namespace torch;
using namespace habana;
using tensor_name_generator = synapse_helpers::detail::tensor_name_generator;

void LinspaceOutOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto result = inputs[3].toTensor();
  HabanaOperator::SetPTOutput(result);
}

void LinspaceOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  const unsigned short constExpectedNoOfInput = 4;
  TORCH_CHECK(
      inputs.size() == constExpectedNoOfInput,
      "Expected ",
      constExpectedNoOfInput,
      " inputs for LinspaceOutOperator operator but received ",
      inputs.size(),
      " inputs.");

  // Upper bound extended to include upper bound with
  // range TPC kernel which support [start, limit)
  float upperBoundExtension = 0.000001;

  TORCH_CHECK(inputs[0].isScalar(), "Input 1 type expected to be a scalar");
  TORCH_CHECK(inputs[1].isScalar(), "Input 2 type expected to be a scalar");

  TORCH_CHECK(inputs[3].isTensor(), "Input 4 type expected to be a tensor");

  auto start = inputs[0].toScalar().toFloat();
  auto end = inputs[1].toScalar().toFloat();
  auto stepCount = inputs[2].toOptional<int64_t>();
  auto out = inputs[3].toTensor();

  int64_t arange_step = stepCount.value();

  float delta = (end - start);
  if (1.0 != arange_step) {
    delta /= (arange_step - 1.0);
  }
  if ((end - start) < 0.f) {
    upperBoundExtension *= -1.0;
  }
  end += upperBoundExtension;

  auto device_id = this->p_context_->device_id_;

  ArangeOperator Op(device_id, ScalarType::Float);

  Op.SetSynapseInput(p_context_->syn_inputs_[0]);

  std::vector<c10::IValue> stack{
      IValue(start), IValue(end), IValue(delta), IValue(out)};
  Op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

  p_context_->syn_outputs_.emplace_back(std::move(Op.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(Op.GetOutputs()[0]));
}

Tensor& linspace_out_hpu(
    const Scalar& start,
    const Scalar& end,
    c10::optional<int64_t> step,
    Tensor& output) {
  PT_KERNEL_BEGIN;
  // If step value is not provided, set it 100, following
  // the CPU implementtaion....
  // pytorch-fork/aten/src/ATen/native/RangeFactories.cpp
  // Tensor& linspace_cpu_out(...
  // ...
  // const auto steps = optional_steps.value_or(100);
  int64_t step_corrected = step.value_or(100);

  auto shape = DimVector({step_corrected});
  auto tht_result = output.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);
  output.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(shape));

  Tensor output_int;
  if (output.scalar_type() == ScalarType::Long) {
    output_int = habana_helpers::createPTTensor(
        output,
        output.sizes(),
        output.options(),
        output.suggest_memory_format(),
        c10::ScalarType::Int,
        true);
  }

  at::ScalarType scalar_type = output.scalar_type();

  std::string node_type = "linspce_out_" + NO_TPC +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = output.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  LinspaceOutOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<at::Tensor> pt_inputs;
  std::vector<c10::IValue> stack = {
      IValue(start), IValue(end), IValue(step_corrected)};

  stack.push_back(IValue(output));
  pt_inputs.emplace_back(output);

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
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
  return output;
}

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
  depth = depth == 0 ? 1 : depth;
  return depth;
}

std::vector<int64_t> GatherOperator::compute_output_shape(
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

Tensor GatherOperator::AllocateOutput(
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  auto self = inputs[0].toTensor();
  auto dim_ = inputs[1].toInt();
  auto index = inputs[2].toTensor();

  auto shape = GatherOperator::compute_output_shape(self, dim_, index);

  auto output = habana_helpers::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
  return output;
}

void GatherOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto output = AllocateOutput(inputs, true);
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

  auto output = AllocateOutput(inputs, is_output_persistent);

  ns_GatherKernel::Params params;
  params.axis = self.dim() - dim - 1;

  p_context_->params_.emplace<ns_GatherKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
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
    bool is_output_persistent) {
  auto self = inputs[0].toTensor();
  auto dim_ = inputs[3].toInt();
  auto index = inputs[1].toTensor();

  auto shape = GatherElemOperator::compute_output_shape(self, dim_, index);

  auto output = habana_helpers::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
  return output;
}

void GatherElemOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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

  auto output = AllocateOutput(inputs, is_output_persistent);

  ns_GatherElementsKernel::Params params;
  params.axis = self.dim() - dim - 1;

  p_context_->params_.emplace<ns_GatherElementsKernel::Params>(params);
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

std::vector<int64_t> ScatterWrapperOperator::compute_output_shape(
    const Tensor& self) {
  return self.sizes().vec();
}

Tensor ScatterWrapperOperator::AllocateOutput(
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  auto self = inputs[0].toTensor();
  auto output = habana_helpers::createPTTensor(self, is_output_persistent);
  return output;
}

void ScatterWrapperOperator::SetPTOutput(torch::jit::Stack& inputs) {
  auto output = AllocateOutput(inputs, true);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
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

  auto output = AllocateOutput(inputs, is_output_persistent);

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

void ScatterValueOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs for scatter_value operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input 0 type expected to be Tensor for scatter-value operator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input 1 type expected to be int64_t for scatter_value operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input 2 type expected to be Tensor for scatter_value operator");
  TORCH_CHECK(
      inputs[3].isScalar(),
      "Input 3 type expected to be Scalar for scatter_value operator");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto index = inputs[2].toTensor();
  auto value = inputs[3].toScalar();
  at::ScalarType scalar_type = self.scalar_type();

  torch::jit::Stack stack;
  Tensor src = habana_helpers::createPTTensor(
      self,
      index.sizes(),
      self.options(),
      self.suggest_memory_format(),
      scalar_type,
      false);

  // Create Constant Operator to convert scalar to tensor
  auto constOp = make_operator<ConstantOperator>(
      this->p_context_->device_id_, scalar_type);
  stack = {IValue(src), IValue(value)};
  constOp->AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  auto scatterOp =
      make_operator<ScatterOperator>(this->p_context_->device_id_, scalar_type);
  stack = {
      IValue(self),
      IValue(dim),
      IValue(index),
      IValue(constOp->GetOutputs()[0])};

  scatterOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  scatterOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  scatterOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
  scatterOp->SetOutputMetadata(output_metadata_);
  scatterOp->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  p_context_->syn_outputs_.emplace_back(
      std::move(scatterOp->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(scatterOp->GetOutputs()[0]));
}

/*************************************************************************
 * @brief Kernel implementation for torch.scatter
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param index - Tensor used to index into self
 * @param value - Scalar with value to be updated (of same type as self)
 ************************************************************************/
Tensor scatter_value_hpu(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    Scalar value) {
  PT_KERNEL_BEGIN;

  auto index_int = habana_helpers::cast_tensor_to_integer(index);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();

  std::string node_type =
      "scatter_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  ScatterValueOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim_), IValue(index_int), IValue(value)};
  size_t key = Op.GetRecipeKey(node_type, stack);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, index_int};

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
 * @brief Kernel implementation for scatter_.value(Tensor(a!) self, int dim,
 *Tensor index, Tensor src) -> Tensor(a!)
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param index - Tensor used to index into self
 * @param value - Scalar with values to be updated (of same type as self)
 ************************************************************************/
Tensor& scatter_inplace_value_hpu(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Scalar& value) {
  PT_KERNEL_BEGIN;
  auto out = scatter_value_hpu(self, dim_, index, value);
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

  ////auto slice = at::index_select(self, 0, indices[0]);
  auto index_selectOp = make_operator<IndexSelectOperator>(
      this->p_context_->device_id_, self.scalar_type());
  temp_stack = {IValue(self), IValue(dim), IValue(index)};
  index_selectOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  index_selectOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  index_selectOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
  temp_stack.clear();

  ////value_acc += slice;
  auto addOp = make_operator<AddOperator>(
      this->p_context_->device_id_, value.scalar_type());
  temp_stack = {
      IValue(value),
      IValue(index_selectOp->GetOutputs()[0]),
      IValue(Scalar(1.0))};
  addOp->SetSynapseInput(p_context_->syn_inputs_[2]);
  addOp->SetSynapseInput(index_selectOp->GetSynOutputs()[0]);
  addOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
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
  reshapeOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
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
  bcastOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
  temp_stack.clear();

  ////auto temp  = scatter_src_hpu(self, dim, index_broadcast, value_acc);
  auto scatterOp = make_operator<ScatterOperator>(
      this->p_context_->device_id_, self.scalar_type());
  temp_stack = {
      IValue(self),
      IValue(dim),
      IValue(bcastOp->GetOutputs()[0]),
      IValue(value)};

  scatterOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  scatterOp->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
  scatterOp->SetSynapseInput(addSynOutput[0]);
  scatterOp->SetOutputMetadata(output_metadata_);

  scatterOp->AllocateAndAddSynapseNode(graph, temp_stack, is_output_persistent);

  p_context_->syn_outputs_.emplace_back(
      std::move(scatterOp->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(scatterOp->GetOutputs()[0]));
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
    ////auto slice = at::index_select(self, 0, indices[0]);
    auto index_selectOp = make_operator<IndexSelectOperator>(
        this->p_context_->device_id_, self.scalar_type());
    temp_stack = {IValue(self), IValue(dim), IValue(index)};
    index_selectOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    index_selectOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    index_selectOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
    temp_stack.clear();

    ////value_acc += slice;
    auto addOp = make_operator<AddOperator>(
        this->p_context_->device_id_, value.scalar_type());
    temp_stack = {
        IValue(value),
        IValue(index_selectOp->GetOutputs()[0]),
        IValue(Scalar(1.0))};
    addOp->SetSynapseInput(p_context_->syn_inputs_[last_index]);
    addOp->SetSynapseInput(index_selectOp->GetSynOutputs()[0]);
    addOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
    addSynOutput.push_back(std::move(addOp->GetSynOutputs()[0]));
    temp_stack.clear();
  }

  // Expand 1D index tensor to same number of dimensions as value tensor
  auto expanded_sizes = std::vector<int64_t>(value.ndimension(), 1);
  expanded_sizes[0] = index.sizes()[0];

  ////auto index_expanded = index.view(expanded_sizes)
  auto reshapeOp = make_operator<ReshapeOperator>(
      this->p_context_->device_id_, index.scalar_type());
  temp_stack = {IValue(index), IValue(expanded_sizes)};
  reshapeOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  reshapeOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
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
  bcastOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
  temp_stack.clear();

  ////auto temp  = scatter_src_hpu(self, dim, index_broadcast, value_acc);
  auto scatterOp = make_operator<ScatterOperator>(
      this->p_context_->device_id_, self.scalar_type());
  temp_stack = {
      IValue(self),
      IValue(dim),
      IValue(bcastOp->GetOutputs()[0]),
      IValue(value)};

  scatterOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  scatterOp->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
  accumulate ? scatterOp->SetSynapseInput(addSynOutput[0])
             : scatterOp->SetSynapseInput(p_context_->syn_inputs_[last_index]);
  scatterOp->SetOutputMetadata(output_metadata_);

  scatterOp->AllocateAndAddSynapseNode(graph, temp_stack, is_output_persistent);

  p_context_->syn_outputs_.emplace_back(
      std::move(scatterOp->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(scatterOp->GetOutputs()[0]));
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
    std::vector<at::Tensor> v{output};
    Op.SetPTOutputs(v);
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
    at::Tensor& self,
    TensorList indices,
    const at::Tensor& value,
    bool accumulate) {
  PT_KERNEL_BEGIN;

  // We need a Scatter-ND TPC kernel to support all input configurations
  // possible for this operator. Also Boolean indexing needs "nonzero"
  // operation. Until TPC supports all these
  // https://jira.habana-labs.com/browse/SW-37171, fallback to CPU
  c10::List<c10::optional<at::Tensor>> indices_list{};
  auto tensorlist = indices.vec();
  indices_list.reserve(tensorlist.size());
  for (size_t i = 0; i < tensorlist.size(); i++) {
    indices_list.push_back(c10::make_optional(tensorlist[i]));
  }
  if ((indices[0].scalar_type() == c10::ScalarType::Bool) ||
      (value.dim() == 0) || (self.scalar_type() == c10::ScalarType::Bool)) {
    AtenHpuTypeDefault::index_put_(self, indices_list, value, accumulate);
    PT_KERNEL_END;
    return self;
  }

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
std::vector<int64_t> SliceOperator::compute_output_shape(
    const Tensor& self,
    int64_t& dim,
    int64_t& start,
    int64_t& end,
    int64_t& step) {
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
  auto shape = self.sizes().vec();
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, len);

  return shape;
}
Tensor SliceOperator::AllocateOutputTensor(
    const Tensor& self,
    int64_t& dim,
    int64_t& start,
    int64_t& end,
    int64_t& step,
    bool is_output_persistent) {
  auto shape = compute_output_shape(self, dim, start, end, step);

  // allocate output tensor
  auto output = habana_helpers::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);

  return output;
}

void SliceOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto start = inputs[2].toInt();
  auto end = inputs[3].toInt();
  auto step = inputs[4].toInt();
  auto output = AllocateOutputTensor(self, dim, start, end, step, true);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
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

  bool needs_params_handling = false;
  if (graph.is_dynamic_graph() && (!graph.is_dry_run()) &&
      end > self.sizes().vec()[dim]) {
    needs_params_handling = true;
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
  // Allocate Shape tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, output);
  }

  if (needs_params_handling) {
    synapse_helpers::tensor& syn_input_tensor = p_context_->syn_inputs_[0];
    std::string tensor_name = syn_input_tensor.name();
    std::vector<int64_t> min, max;
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_name);
    params.ends[0] = static_cast<int>(max[dim]);
  }

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
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  PT_KERNEL_BEGIN;

  // Handling the case where slice recieves NULL in_self tensor.
  // Although tensor is NULL, still output is expected of correct size
  if (in_self.numel() == 0) {
    SliceOperator slice_op(in_self.device().index(), in_self.scalar_type());
    auto slice_output = slice_op.AllocateOutputTensor(
        in_self, dim, start.value(), end.value(), step, true);
    PT_KERNEL_END;
    return slice_output;
  }

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
  if (self.dim() <= 1) {
    PT_KERNEL_END;
    return at::native::slice(in_self, dim, start, end, step);
  }

  // WA for https://jira.habana-labs.com/browse/SW-37197
  auto dim_orig = dim;
  if ((dim == self.dim() - 1) && (step > 1)) {
    self = self.transpose(self.dim() - 1, self.dim() - 2);
    dim = self.dim() - 2;
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

  // WA for https://jira.habana-labs.com/browse/SW-37197
  if ((dim_orig == self.dim() - 1) && (step > 1)) {
    cast_out = cast_out.transpose(self.dim() - 1, self.dim() - 2);
  }

  PT_KERNEL_END;
  return cast_out;
}

std::vector<int64_t> SelectOperator::compute_output_shape(
    const Tensor& self,
    int64_t& dim) {
  // convert dim to positive value if required
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  auto shape = self.sizes().vec();
  shape.erase(shape.begin() + dim);
  return shape;
}

void SelectOperator::SetPTOutputs(torch::jit::Stack& inputs) {
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
  auto shape = compute_output_shape(self, dim);
  auto output = habana_helpers::createPTTensor(
      self, shape, self.options(), memory_format, true);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
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
  auto input_shape = self.sizes();
  auto dimensions = input_shape.size();

  auto start = index;
  auto end = index + 1;
  int64_t step = 1;

  bool is_slice_output = false;
  is_slice_output = (dimensions != 1) ? false : true;

  auto slice_op =
      make_operator<SliceOperator>(self.device().index(), self.scalar_type());
  slice_op->SetSynapseInput(p_context_->syn_inputs_[0]);
  if (is_slice_output) {
    slice_op->SetOutputMetadata(output_metadata_);
  }
  std::vector<c10::IValue> stack1 = {
      IValue(self), IValue(dim), IValue(start), IValue(end), IValue(step)};
  slice_op->AllocateAndAddSynapseNode(graph, stack1, is_slice_output);

  if (is_slice_output == false) {
    // Add Reshape node to graph
    auto reshape_op = make_operator<ReshapeOperator>(
        self.device().index(), self.scalar_type());
    UNUSED auto& syn_in_reshape =
        reshape_op->SetSynapseInput(slice_op->GetSynOutputs()[0]);
    reshape_op->SetOutputMetadata(output_metadata_);

    auto slice_out_tensor = slice_op->GetOutputs()[0];
    auto shape = slice_out_tensor.sizes().vec();
    shape.erase(shape.begin() + dim);
    torch::jit::Stack stack2 = {
        c10::IValue(slice_out_tensor), c10::IValue(shape)};
    reshape_op->AllocateAndAddSynapseNode(graph, stack2, is_output_persistent);
    p_context_->syn_outputs_.emplace_back(
        std::move(reshape_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(reshape_op->GetOutputs()[0]));
  } else {
    p_context_->syn_outputs_.emplace_back(
        std::move(slice_op->GetSynOutputs()[0]));

    p_context_->pt_outputs_.emplace_back(slice_op->GetOutputs()[0]);
  }
}

/*************************************************************************
 * @brief Kernel implementation for torch select operator
 * @param self - Input tensor
 * @param dim - Axis to slice
 * @param index - index the element in given axis
 ************************************************************************/

Tensor select_hpu(const Tensor& in_self, int64_t dim, int64_t index) {
  PT_KERNEL_BEGIN;

  // Handling the case where select recieves NULL in_self tensor.
  // Although tensor is NULL, still output is expected of correct size
  if (in_self.numel() == 0) {
    auto start = index;
    auto end = index + 1;
    int64_t step = 1;
    SliceOperator slice_op(in_self.device().index(), in_self.scalar_type());
    auto slice_output =
        slice_op.AllocateOutputTensor(in_self, dim, start, end, step, false);
    // case for select op where tensor dimension is reduced
    // only rank 4 tensor can have channels last format
    at::MemoryFormat memory_format = at::MemoryFormat::Contiguous;
    // allocate output tensor
    auto shape = slice_output.sizes().vec();
    shape.erase(shape.begin() + dim);
    auto output = habana_helpers::createPTTensor(
        in_self, shape, in_self.options(), memory_format, true);
    PT_KERNEL_END;
    return output;
  }
  Tensor self;
  if (in_self.scalar_type() == c10::ScalarType::Long) {
    self = habana_helpers::cast_tensor_to_integer(in_self);
  } else {
    self = in_self;
  }

  at::ScalarType scalar_type = self.scalar_type();

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
  HabanaOperator::SetPTOutput(result);
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

  p_context_->syn_outputs_.emplace_back(std::move(p_context_->syn_inputs_[0]));
  p_context_->pt_outputs_.emplace_back(result);
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

    AllocateSynapseOutput(graph, output_range, false, false, false);
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[1];

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
    auto castOp =
        make_operator<CastOutOperator>(this->p_context_->device_id_, node_type);

    // Build Params for the graph
    torch::jit::Stack stack = {IValue(output_range), IValue(result)};
    // syn_output_[1] is the output of range node
    castOp->SetSynapseInput(p_context_->syn_outputs_[1]);
    // syn_output_[0] is the original Out result tensor
    castOp->SetSynapseInput(p_context_->syn_outputs_[0]);
    castOp->SetOutputMetadata(output_metadata_);

    castOp->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    // There are 2 outputs {result, rangeOut}, we need only one {result}
    p_context_->syn_outputs_.pop_back();
    p_context_->pt_outputs_.pop_back();
    p_context_->pt_outputs_[0] = std::move(castOp->GetOutputs()[0]);
  }
}

/*************************************************************************
 * @brief Kernel implementation for torch.arange operator
 * @param output - output tensor
 * @param start - start index of the sequence
 * @param end - end index of the sequence
 * @param step - step value of the sequence
 ************************************************************************/

Tensor& arange_hpu(Tensor& output, const Scalar& start, const Scalar& end, const Scalar& step) {
  PT_KERNEL_BEGIN;

  // resizing the output as it is coming as empty from model
  int depth = ArangeOperator::GetOutputSize(start, end, step);
  auto shape = DimVector({depth});
  auto tht_result = output.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);
  Tensor output_int;
  if (output.scalar_type() == ScalarType::Long) {
    output_int = habana_helpers::createPTTensor(
        output,
        output.sizes(),
        output.options(),
        output.suggest_memory_format(),
        c10::ScalarType::Int,
        true);
  }
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
  std::vector<at::Tensor> pt_inputs;
  std::vector<c10::IValue> stack = {IValue(start), IValue(end), IValue(step)};

  if (output.scalar_type() == ScalarType::Long) {
    stack.push_back(IValue(output_int));
    pt_inputs.emplace_back(output_int);
  } else {
    stack.push_back(IValue(output));
    pt_inputs.emplace_back(output);
  }

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
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

  if (output.scalar_type() == ScalarType::Long) {
    output.copy_(habana_helpers::cast_tensor_to_long(out.at(0)));
  } else if (output.scalar_type() == ScalarType::Bool) {
    out.at(0).to(c10::ScalarType::Bool);
  }
  PT_KERNEL_END;
  return output;
}

// brodcast index tensor shape and get the correct shape and size
std::vector<int64_t> broadcast_size(at::TensorList indices) {
  auto size = indices[0].sizes().vec();
  for (size_t i = 1; i < indices.size(); i++) {
    size = infer_size(size, indices[i].sizes());
  }
  return size;
}

// get the first index tensor shape and size
std::vector<int64_t> indices_size(at::TensorList indices) {
  auto first_size = broadcast_size(indices);

  int64_t in_tensor_count = indices.size(); // num input tensors

  std::vector<int64_t> out_size{in_tensor_count};
  out_size.insert(out_size.end(), first_size.begin(), first_size.end());

  return out_size;
}
// index is implemented using mxnet_gatherNd, refer below for output shape
// computation
// ref:https://github.com/apache/incubator-mxnet/blob/master/src/operator/tensor/indexing_op.h#L1319
std::vector<int64_t> IndexOperator::compute_output_shape(
    const Tensor& input,
    at::TensorList indices) {
  auto input_shape = input.sizes();
  auto indices_shape = indices_size(indices);

  auto output_rank = static_cast<int64_t>(
      indices_shape.size() + input.ndimension() - indices_shape[0] - 1);

  std::vector<int64_t> output_shape(output_rank, -1);

  for (size_t i = 0; i < indices_shape.size() - 1; i++) {
    output_shape[i] = indices_shape[i + 1];
  }

  for (int64_t i = 0;
       i < static_cast<int64_t>(input.ndimension() - indices_shape[0]);
       i++) {
    output_shape[indices_shape.size() - 1 + i] =
        input_shape[indices_shape[0] + i];
  }
  return output_shape;
}

void IndexOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for gather2d operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(
      inputs[1].isTensorList(),
      "Input 1 type expected to be TensorList for [] operator");

  auto input = inputs[0].toTensor();
  auto tensorlist = inputs[1].toTensorList().vec();

  auto max_size = broadcast_size(tensorlist);
  auto device_id = this->p_context_->device_id_;
  auto scalar_type = tensorlist[0].scalar_type();

  std::vector<Tensor> cat_input;
  auto cat_indices = make_operator<CatOperator>(device_id, scalar_type);

  for (size_t i = 0; i < tensorlist.size(); i++) {
    // broadcast index tensor to largest index tensor size
    auto bcastOp = make_operator<BroadcastOperator>(device_id, scalar_type);
    Stack stack = {IValue(tensorlist[i]), IValue(max_size), IValue(false)};
    bcastOp->SetSynapseInput(p_context_->syn_inputs_[i + 1]);
    bcastOp->AllocateAndAddSynapseNode(graph, stack, false);

    stack.clear();

    std::vector<int64_t> expanded_size{1};
    for (auto s : bcastOp->GetOutputs()[0].sizes()) {
      expanded_size.push_back(s);
    }
    stack = {IValue(bcastOp->GetOutputs()[0]), IValue(expanded_size)};
    auto ReshapeOp = make_operator<ReshapeOperator>(device_id, scalar_type);
    ReshapeOp->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    ReshapeOp->AllocateAndAddSynapseNode(graph, stack, false);

    cat_input.emplace_back(ReshapeOp->GetOutputs()[0]);
    cat_indices->SetSynapseInput(ReshapeOp->GetSynOutputs()[0]);
  }
  // index is implemented using mxnet_gatherNd, where indices needs to be
  // single tensor, wherease we get tensorlist. so we stack the tensors
  // from tensorlist by reshape followed by cat

  Stack stack = {IValue(cat_input), IValue(0)};
  cat_indices->AllocateAndAddSynapseNode(graph, stack, false);

  auto shape = compute_output_shape(input, tensorlist);

  auto output = habana_helpers::createPTTensor(
      input,
      IntArrayRef(shape.data(), shape.size()),
      input.options(),
      input.suggest_memory_format(),
      is_output_persistent);

  AllocateSynapseOutput(graph, output, is_output_persistent);

  synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[0];
  synapse_helpers::tensor& arg2_syn_tensor =
      std::move(cat_indices->GetSynOutputs()[0]);

  std::vector<synTensor> syn_inputs;
  syn_inputs.emplace_back(arg1_syn_tensor.get());
  syn_inputs.emplace_back(arg2_syn_tensor.get());

  synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      nullptr,
      0,
      std::move(guid_));
}

/*************************************************************************
 * @brief Kernel implementation for index
 * @param input - Input tensor 2D fp32
 * @param indices - TensorList for indices
 ************************************************************************/
Tensor index_hpu(const at::Tensor& input, TensorList indices) {
  PT_KERNEL_BEGIN;
  // fallback to cpu for boolean indexing
  if (indices[0].scalar_type() == c10::ScalarType::Bool) {
    c10::List<c10::optional<at::Tensor>> indices_list{};
    auto tensorlist = indices.vec();
    indices_list.reserve(tensorlist.size());
    for (size_t i = 0; i < tensorlist.size(); i++) {
      indices_list.push_back(c10::make_optional(tensorlist[i]));
    }
    return AtenHpuTypeDefault::index(input, indices_list);
  }

  // if there is only 1 indices tensor, then operation is equivalent to gather
  // 1d. Since gather_nd_mxnet is throwing a TPC error for 4d input in such
  // cases, therefore call "gather" TPC kernel instead
  // Note that we can remove these work-arounds once TPC kernel for index
  // operation is available https://jira.habana-labs.com/browse/SW-37171
  if (indices.size() == 1 && indices[0].dim() == 1) {
    Tensor output;
    if (input.scalar_type() == c10::ScalarType::Long) {
      auto input_i32 = habana_helpers::cast_tensor_to_integer(input);
      output = gather_src_hpu(input_i32, 0, indices[0], false);
      output = habana_helpers::cast_tensor_to_long(output);
    } else {
      output = gather_src_hpu(input, 0, indices[0], false);
    }
    PT_KERNEL_END;
    return output;
  }

  // cast input to fp32 int32 not supported yet
  Tensor input_cast;
  if (input.scalar_type() == c10::ScalarType::Long ||
      input.scalar_type() == c10::ScalarType::Int) {
    auto input_i32 = habana_helpers::cast_tensor_to_integer(input);
    input_cast = habana_helpers::hpu_cast_tensor(
        input_i32, at::scalarTypeToTypeMeta(c10::ScalarType::Float));
  } else {
    input_cast = input;
  }
  at::ScalarType scalar_type = input_cast.scalar_type();
  std::string node_type = "gather_nd_mxnet_fwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = input_cast.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  IndexOperator Op(device_id, scalar_type);

  std::vector<at::Tensor> pt_inputs{input_cast};

  std::vector<at::Tensor> indices_cast;

  for (const auto& t : indices) {
    if (t.scalar_type() == c10::ScalarType::Long) {
      indices_cast.emplace_back(habana_helpers::cast_tensor_to_integer(t));
    } else {
      indices_cast.emplace_back(t);
    }
  }

  pt_inputs.insert(pt_inputs.end(), indices_cast.begin(), indices_cast.end());
  TensorList new_indices_list{indices_cast};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(input_cast), IValue(new_indices_list)};

  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    auto shape =
        IndexOperator::compute_output_shape(input_cast, new_indices_list);
    auto output = habana_helpers::createPTTensor(
        input_cast,
        IntArrayRef(shape.data(), shape.size()),
        input_cast.options(),
        input_cast.suggest_memory_format(),
        true);
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

  if (input.scalar_type() == c10::ScalarType::Long ||
      input.scalar_type() == c10::ScalarType::Int) {
    auto output = habana_helpers::hpu_cast_tensor(
        out.at(0), at::scalarTypeToTypeMeta(c10::ScalarType::Int));
    if (input.scalar_type() == c10::ScalarType::Long) {
      output = habana_helpers::cast_tensor_to_long(output);
    }
    PT_KERNEL_END;
    return output;
  }

  PT_KERNEL_END;
  return out.at(0);
}

void UniqueOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  int elements = self.numel();
  auto output_shape = DimVector{elements};
  auto valid_shape = DimVector{1};

  // create output and valid shape tensors which are compulsory
  auto output_feature_map = habana_helpers::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      true);
  auto valid_count = habana_helpers::createPTTensor(
      self,
      valid_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);
  std::vector<at::Tensor> outputs{output_feature_map, valid_count};
  HabanaOperator::SetPTOutputs(outputs);
}

void UniqueOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  HABANA_ASSERT(
      inputs.size() == 4 && "Incorrect size of inputs in UniqueOperator");
  HABANA_ASSERT(inputs[0].isTensor() && "Input 0 is expected to be tensor");
  HABANA_ASSERT(inputs[1].isBool() && "Input 1 is expected to be bool");
  HABANA_ASSERT(inputs[2].isBool() && "Input 2 is expected to be bool");
  HABANA_ASSERT(inputs[3].isBool() && "Input 3 is expected to be bool");

  bool sorted = inputs[1].toBool();
  bool return_inverse = inputs[2].toBool();
  bool return_counts = inputs[3].toBool();
  if (sorted == true) {
    PT_KERNEL_WARN(
        "Recieved sorted=True, ignoring as TPC kernel does not support it");
  }
  // Assert for return_inverse, return_counts as function expects extra output
  // if set
  HABANA_ASSERT((!return_inverse) && "return_inverse not supported in unique2");
  HABANA_ASSERT((!return_counts) && "return_counts not supported in unique2");

  auto self = inputs[0].toTensor();
  int elements = self.numel();
  auto output_shape = DimVector{elements};
  auto valid_shape = DimVector{1};

  // The first output tensor contains unique elements.
  // The second output tensor contains the number of unique elements.
  // The two optional tensors(Inverse index(1D), Counts(1D)) can be enabled by
  // setting the corresponding parameters in the structure(return_inverse,
  // return_counts) Currently this implementation supports with both
  // return_inverse and return_counts as false

  // create output and valid shape tensors which are compulsory
  auto output_feature_map = habana_helpers::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      is_output_persistent[0]);
  auto valid_count = habana_helpers::createPTTensor(
      self,
      valid_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      is_output_persistent[1]);

  ns_UniqueKernel::Params params;
  params.returnInverse = 0;
  params.returnCounts = 0;
  // dim = -5 returns flattened result(unique elements over all dimesions)
  params.dim = -5;

  p_context_->params_.emplace<ns_UniqueKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output_feature_map, is_output_persistent[0]);
  synDataType synType = syn_type_uint32;
  AllocateSynapseOutput(
      graph,
      valid_count,
      synType,
      is_output_persistent[1],
      graph.is_dynamic_graph() ? true : false);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/*************************************************************************
 * @brief Kernel implementation for torch.unique operator
 * @param self - Input tensor
 * @param sorted - Whether to sort the unique elements
 * @param return_inverse - To return the indices for where elements in the
 *  original input end in result
 * @param return_counts - To return counts for each unique element
 ************************************************************************/
std::tuple<Tensor, Tensor, Tensor> unique2_hpu(
    const Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  PT_KERNEL_BEGIN;

  Tensor input_cast = self;
  if (self.scalar_type() == c10::ScalarType::Long) {
    input_cast = habana_helpers::cast_tensor_to_integer(self);
  }

  std::string node_type = "unique2";
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  UniqueOperator Op(device_id, self.scalar_type());
  std::vector<at::Tensor> pt_inputs{input_cast};
  std::vector<c10::IValue> stack = {
      IValue(input_cast),
      IValue(sorted),
      IValue(return_inverse),
      IValue(return_counts)};

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
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect Size of outputs returned for unique");
  auto end = out.at(1).item<int64_t>();
  auto result = out.at(0).slice(0, 0, end, 1);

  Tensor cast_out = result;
  if (self.scalar_type() == c10::ScalarType::Long) {
    cast_out = habana_helpers::cast_tensor_to_long(result);
  }

  // These are optional tensors which shall be populated only when we start
  // supporting return_inverse and return_counts
  Tensor inverse_indices;
  Tensor counts;
  PT_KERNEL_END;
  return std::make_tuple(cast_out, inverse_indices, counts);
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::index_select",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<IndexSelectOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::gather_elements",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<GatherElemOperator>(device_id, node_type);
            })
        .add(
            "aten::gather",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<GatherOperator>(device_id, node_type);
            })
        .add(
            "aten::scatter.src",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ScatterOperator>(device_id, node_type);
            })
        .add(
            "aten::scatter_add",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ScatterAddOperator>(device_id, node_type);
            })
        .add(
            "hpu::scatter_value",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ScatterValueOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::select.int",
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
            "aten::slice.Tensor",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SliceOperator>(device_id, node_type);
            })
        .add(
            "aten::index_add",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<IndexAddOperator>(device_id, node_type);
            })
        .add(
            "hpu::_unique2",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<UniqueOperator>(device_id, node_type);
            })
        .add(
            "hpu::arange_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ArangeOperator>(device_id, node_type);
            })
        .add(
            "aten::index.Tensor_hacked_twin",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<IndexOperator>(device_id, node_type);
            })
        .add(
            "aten::linspace.out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<LinspaceOutOperator>(
                  device_id, node_type);
            });
